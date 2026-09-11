"""The pipeline of §6.1 — user input to scored dashboard in one request cycle.

    paraphrase -> fire at N models in parallel -> extract -> score

Everything left of the dashboard is stateless, exactly as Figure 2 specifies:
no background jobs, no persistent crawl, no stored graph. The only durable
artefact is the response cache, which is an optimisation and can be deleted at
any time without changing behaviour.

The run is exposed as an async generator of events rather than a single return
value, so the grid fills in cell by cell while the remaining calls are still in
flight. On ~50-60 calls that is the difference between a demo that looks alive
and one that shows a spinner for eight seconds.

Failure policy, in order of preference: degrade, isolate, then report.
A failed re-check falls back to deterministic extraction. A failed measurement
call fails one cell, never the run. A failed paraphrase call falls back to
deterministic templates. Only a total roster failure ends the run.
"""

from __future__ import annotations

import asyncio
import random
import time
import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any

from . import paraphrase as paraphrase_mod
from .cache import DiskCache
from .config import Settings, get_settings
from .domain import (
    Cell,
    CellState,
    EntityFinding,
    ModelRef,
    Paraphrase,
    RunMeta,
    RunResult,
    RunSpec,
    Verdict,
)
from .extract import (
    RECHECK_SCHEMA,
    RECHECK_SYSTEM,
    build_findings,
    build_recheck_prompt,
    cell_state_from_findings,
    parse_recheck,
)
from .jsonutil import parse_json_object
from .matching import EntityMatcher
from .providers import CompletionRequest, CompletionResult, ProviderError, ProviderRegistry
from .score import score_run

#: Room for a normal consumer-length answer. Deliberately not larger: we are
#: measuring what a model volunteers, and an unbounded budget invites an
#: exhaustive catalogue that names every vendor and flattens the grid to "both".
_ANSWER_MAX_TOKENS = 1200
_RECHECK_MAX_TOKENS = 1400
_PARAPHRASE_MAX_TOKENS = 2000

_MAX_ATTEMPTS = 3
_BACKOFF_BASE = 0.6


class Pipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        cache: DiskCache | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._cache = cache or DiskCache(
            self._settings.cache_dir, enabled=self._settings.cache_enabled
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self, spec: RunSpec, fresh: bool = False) -> AsyncIterator[dict[str, Any]]:
        run_id = uuid.uuid4().hex[:12]
        started_wall = time.perf_counter()
        meta = RunMeta(
            run_id=run_id,
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        models, warnings = self._settings.build_roster(spec.models)
        meta.warnings = list(warnings)
        meta.simulated = all(m.provider == "simulated" for m in models)

        families = {m.family for m in models}
        if len(families) == 1 and len(models) > 1:
            meta.single_family = next(iter(families))
            # build_roster() raises this for the auto-widened case, but an
            # explicit roster override bypasses that path. Attach it here so
            # every surface — dashboard, CLI, JSON API — says the same thing.
            #
            # Not for a simulated roster: the fixtures banner already invalidates
            # every figure, and claiming "per-model Trust and RSI remain valid"
            # would be straightforwardly false there.
            if not meta.simulated and not any(
                "within-family" in w for w in meta.warnings
            ):
                meta.warnings.append(
                    f"All {len(models)} columns are {meta.single_family}-family "
                    "models. Per-model Trust and RSI remain valid, but the "
                    "consensus figures measure within-family agreement and do "
                    "not support the cross-family AITC claim in §3.5."
                )

        if not models:
            yield {"type": "error", "message": "no model columns available"}
            return

        registry = ProviderRegistry(self._settings, seed=spec.seed)
        semaphore = asyncio.Semaphore(self._settings.max_concurrency)

        # A column that cannot honour temperature=0 mixes sampling noise into
        # the variance that RSI attributes to phrasing. Say so rather than let
        # the number quietly mean something different per column.
        no_temp = [
            m.label
            for m in models
            if not registry.accepts_temperature(m.provider, m.model)
        ]
        if no_temp:
            meta.warnings.append(
                "These columns do not accept a sampling temperature: "
                + ", ".join(no_temp)
                + ". Their measured variance includes provider-side sampling "
                "noise in addition to phrasing sensitivity, so treat their RSI "
                "as a lower bound on stability."
            )

        count = spec.paraphrase_count or self._settings.paraphrase_count
        recheck_enabled = (
            spec.llm_recheck if spec.llm_recheck is not None else self._settings.llm_recheck
        )

        yield {
            "type": "run_started",
            "run_id": run_id,
            "spec": spec.model_dump(),
            "models": [m.model_dump() for m in models],
            "meta": meta.model_dump(),
            "planned_calls": count * len(models) * (2 if recheck_enabled else 1),
        }

        try:
            paraphrases, para_note = await self._paraphrases(
                spec, count, registry, semaphore, models, fresh, meta
            )
            if para_note:
                meta.warnings.append(para_note)

            yield {
                "type": "paraphrases",
                "paraphrases": [p.model_dump() for p in paraphrases],
                "warnings": meta.warnings,
            }

            matcher = EntityMatcher(spec.tracked_names())
            total = len(paraphrases) * len(models)
            cells: list[Cell] = []

            tasks = [
                asyncio.create_task(
                    self._cell(
                        spec,
                        model,
                        para,
                        matcher,
                        registry,
                        semaphore,
                        recheck_enabled,
                        fresh,
                        meta,
                    )
                )
                for para in paraphrases
                for model in models
            ]

            done = 0
            for future in asyncio.as_completed(tasks):
                cell = await future
                cells.append(cell)
                done += 1
                yield {
                    "type": "cell",
                    "cell": cell.model_dump(),
                    "done": done,
                    "total": total,
                }

            scores = score_run(spec, models, cells, len(paraphrases))
            meta.finished_at = datetime.now(timezone.utc).isoformat()
            meta.duration_ms = int((time.perf_counter() - started_wall) * 1000)

            cells.sort(key=lambda c: (c.query_index, c.model_key))
            result = RunResult(
                meta=meta,
                spec=spec,
                models=models,
                paraphrases=paraphrases,
                cells=cells,
                scores=scores,
            )

            yield {"type": "scores", "scores": scores.model_dump()}
            yield {"type": "run_complete", "result": result.model_dump()}

        except Exception as exc:  # noqa: BLE001 — surfaced to the client
            yield {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
        finally:
            await registry.aclose()

    async def collect(self, spec: RunSpec, fresh: bool = False) -> RunResult:
        """Run to completion and return the result (used by the CLI and tests)."""
        result: RunResult | None = None
        error: str | None = None
        async for event in self.run(spec, fresh=fresh):
            if event["type"] == "run_complete":
                result = RunResult.model_validate(event["result"])
            elif event["type"] == "error":
                error = event["message"]
        if result is None:
            raise RuntimeError(error or "run produced no result")
        return result

    # ------------------------------------------------------------------
    # Stage 1 — paraphrases
    # ------------------------------------------------------------------

    async def _paraphrases(
        self,
        spec: RunSpec,
        count: int,
        registry: ProviderRegistry,
        semaphore: asyncio.Semaphore,
        models: list[ModelRef],
        fresh: bool,
        meta: RunMeta,
    ) -> tuple[list[Paraphrase], str | None]:
        utility = self._settings.utility_model(models)
        if utility is None:
            return paraphrase_mod.fallback(spec.category, count), (
                "No utility model available — used the built-in query templates "
                "instead of generated paraphrases."
            )

        provider_name, model_name = utility
        request = CompletionRequest(
            model=model_name,
            prompt=paraphrase_mod.build_prompt(spec.category, count),
            system=paraphrase_mod.PARAPHRASE_SYSTEM,
            max_tokens=_PARAPHRASE_MAX_TOKENS,
            temperature=0.0,
            json_schema=paraphrase_mod.PARAPHRASE_SCHEMA,
            sim_context={"category": spec.category, "count": count},
        )

        try:
            async with semaphore:
                result = await self._call(
                    provider_name, request, registry, "paraphrase", fresh, meta
                )
            payload = parse_json_object(result.text)
            generated = paraphrase_mod.parse(payload, count)
        except (ProviderError, ValueError) as exc:
            return paraphrase_mod.fallback(spec.category, count), (
                f"Paraphrase generation failed ({exc}); used the built-in query "
                "templates instead."
            )

        if not generated:
            return paraphrase_mod.fallback(spec.category, count), (
                "Paraphrase generation returned nothing usable; used the built-in "
                "query templates instead."
            )

        topped = paraphrase_mod.ensure_coverage(generated, spec.category, count)
        note = None
        if len(generated) < count:
            note = (
                f"Only {len(generated)} of {count} paraphrases were generated; "
                "the remainder came from the built-in templates."
            )
        return topped, note

    # ------------------------------------------------------------------
    # Stage 2-4 — fire, extract, derive one cell
    # ------------------------------------------------------------------

    async def _cell(
        self,
        spec: RunSpec,
        model: ModelRef,
        para: Paraphrase,
        matcher: EntityMatcher,
        registry: ProviderRegistry,
        semaphore: asyncio.Semaphore,
        recheck_enabled: bool,
        fresh: bool,
        meta: RunMeta,
    ) -> Cell:
        names = spec.tracked_names()

        # No system prompt: we are measuring what the model volunteers to a bare
        # consumer question. Any persona we injected would be measuring our own
        # prompt instead of the model's disposition.
        request = CompletionRequest(
            model=model.model,
            prompt=para.text,
            system=None,
            max_tokens=_ANSWER_MAX_TOKENS,
            temperature=0.0,
            sim_context={
                "entity": spec.entity,
                "competitors": spec.competitors,
                "category": spec.category,
            },
        )

        try:
            async with semaphore:
                answer = await self._call(
                    model.provider, request, registry, "answer", fresh, meta
                )
        except ProviderError as exc:
            meta.failed_calls += 1
            return self._error_cell(para, model, names, str(exc))
        except Exception as exc:  # noqa: BLE001
            meta.failed_calls += 1
            return self._error_cell(para, model, names, f"{type(exc).__name__}: {exc}")

        recheck = None
        if recheck_enabled:
            recheck = await self._recheck(
                para, answer.text, names, registry, semaphore, fresh, meta, fallback_model=model
            )

        findings = build_findings(answer.text, matcher, recheck)
        state, target, competitors = cell_state_from_findings(findings, spec.entity)

        return Cell(
            query_index=para.index,
            model_key=model.key,
            state=state,
            target=target,
            competitors=competitors,
            response_text=answer.text,
            target_mentioned_only=target.verdict is Verdict.MENTIONED,
            competitors_mentioned_only=[
                f.name for f in competitors if f.verdict is Verdict.MENTIONED
            ],
            latency_ms=answer.latency_ms,
            cached=answer.cached,
        )

    async def _recheck(
        self,
        para: Paraphrase,
        response_text: str,
        names: list[str],
        registry: ProviderRegistry,
        semaphore: asyncio.Semaphore,
        fresh: bool,
        meta: RunMeta,
        fallback_model: ModelRef,
    ) -> dict[str, dict[str, Any]] | None:
        utility = self._settings.utility_model([fallback_model])
        if utility is None:
            return None
        provider_name, model_name = utility

        request = CompletionRequest(
            model=model_name,
            prompt=build_recheck_prompt(para.text, response_text, names),
            system=RECHECK_SYSTEM,
            max_tokens=_RECHECK_MAX_TOKENS,
            temperature=0.0,
            json_schema=RECHECK_SCHEMA,
            sim_context={"names": names, "response_text": response_text},
        )

        try:
            async with semaphore:
                result = await self._call(
                    provider_name, request, registry, "recheck", fresh, meta
                )
            return parse_recheck(parse_json_object(result.text))
        except (ProviderError, ValueError):
            # Degrade to deterministic-only extraction for this cell. The cell
            # still scores; its `agreed` fields simply stay null and drop out of
            # the extraction-agreement statistic.
            return None

    # ------------------------------------------------------------------
    # Provider call with cache + retry
    # ------------------------------------------------------------------

    async def _call(
        self,
        provider_name: str,
        request: CompletionRequest,
        registry: ProviderRegistry,
        namespace: str,
        fresh: bool,
        meta: RunMeta,
    ) -> CompletionResult:
        cache_key = DiskCache.key(
            namespace,
            {
                "provider": provider_name,
                "model": request.model,
                "prompt": request.prompt,
                "system": request.system,
                "temperature": request.temperature,
                "schema": bool(request.json_schema),
                "sim": request.sim_context,
            },
        )

        if not fresh:
            hit = self._cache.get(cache_key)
            if hit and isinstance(hit.get("text"), str):
                meta.cached_calls += 1
                return CompletionResult(
                    text=hit["text"],
                    provider=provider_name,
                    model=request.model,
                    latency_ms=int(hit.get("latency_ms") or 0),
                    usage=hit.get("usage") or {},
                    cached=True,
                )

        provider = registry.get(provider_name)

        last: ProviderError | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                result = await provider.complete(request)
            except ProviderError as exc:
                last = exc
                if not exc.retryable or attempt == _MAX_ATTEMPTS - 1:
                    raise
                await asyncio.sleep(
                    _BACKOFF_BASE * (2**attempt) + random.uniform(0, 0.3)
                )
                continue

            meta.provider_calls += 1
            self._cache.set(
                cache_key,
                {
                    "text": result.text,
                    "usage": result.usage,
                    "latency_ms": result.latency_ms,
                },
            )
            return result

        raise last or ProviderError("provider call failed")

    # ------------------------------------------------------------------

    @staticmethod
    def _error_cell(
        para: Paraphrase, model: ModelRef, names: list[str], message: str
    ) -> Cell:
        findings = [
            EntityFinding(name=name, verdict=Verdict.ABSENT) for name in names
        ]
        return Cell(
            query_index=para.index,
            model_key=model.key,
            state=CellState.NEITHER,
            target=findings[0],
            competitors=findings[1:],
            response_text="",
            error=message,
        )
