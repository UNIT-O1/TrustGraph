"""FastAPI application: one screen, one streaming endpoint.

The run endpoint streams Server-Sent Events rather than returning a single
payload, because a full run is ~50-60 provider calls and the paper's hero
element is a grid of 15-20 x 3 cells. Streaming lets the grid populate as
results land, which is both a better demo and a genuinely better diagnostic —
you see the price-framing band forming before the run finishes.

SSE is served over POST (not the EventSource default of GET) so the run spec
travels in a body instead of a query string; the frontend reads it with
``fetch`` + a stream reader.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .cache import DiskCache
from .config import PROVIDER_LABELS, PROVIDER_MODELS, get_settings
from .domain import RunSpec
from .pipeline import Pipeline
from .providers import ProviderError, ProviderRegistry

WEB_DIR = Path(__file__).parent / "web"

app = FastAPI(
    title="TrustGraph — Recommendation Stability Tool",
    description=(
        "Section 6 of *The AI Trust Graph* working paper: paraphrase a category "
        "query, fire it at several production LLMs, and report Trust, RSI and "
        "head-to-head rank for a named entity."
    ),
    version="0.1.0",
)


class RunRequest(RunSpec):
    fresh: bool = Field(
        default=False,
        description="Bypass the response cache and issue live provider calls.",
    )


# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------

if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    target = WEB_DIR / "index.html"
    if not target.exists():  # pragma: no cover
        raise HTTPException(status_code=500, detail="frontend not installed")
    return FileResponse(target)


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------


@app.get("/api/health")
async def health() -> dict[str, Any]:
    settings = get_settings()
    roster, warnings = settings.build_roster()
    return {
        "ok": True,
        "providers": {
            name: {
                "label": PROVIDER_LABELS.get(name, name),
                "configured": bool((settings.api_key(name) or "").strip()),
                "default_models": [m for m, _ in models],
            }
            for name, models in PROVIDER_MODELS.items()
        },
        "roster": [m.model_dump() for m in roster],
        "warnings": warnings,
        "settings": {
            "paraphrase_count": settings.paraphrase_count,
            "max_concurrency": settings.max_concurrency,
            "llm_recheck": settings.llm_recheck,
            "cache_enabled": settings.cache_enabled,
        },
    }


@app.get("/api/models")
async def models() -> dict[str, Any]:
    """Live model discovery.

    Vendor model IDs drift, and a stale hardcoded default is the most likely
    reason a demo fails at the worst moment. This asks each configured provider
    what it will actually serve.
    """
    settings = get_settings()
    registry = ProviderRegistry(settings)
    out: dict[str, Any] = {}
    try:
        for provider in PROVIDER_MODELS:
            if not (settings.api_key(provider) or "").strip():
                out[provider] = {"configured": False, "models": []}
                continue
            try:
                listed = await registry.get(provider).list_models()
                out[provider] = {"configured": True, "models": listed}
            except ProviderError as exc:
                out[provider] = {"configured": True, "error": str(exc), "models": []}
    finally:
        await registry.aclose()
    return out


@app.post("/api/cache/clear")
async def clear_cache() -> dict[str, Any]:
    settings = get_settings()
    removed = DiskCache(settings.cache_dir, enabled=True).clear()
    return {"removed": removed}


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, default=str)}\n\n"


@app.post("/api/run")
async def run_stream(request: RunRequest) -> StreamingResponse:
    spec = RunSpec.model_validate(request.model_dump(exclude={"fresh"}))
    pipeline = Pipeline()

    async def generator() -> AsyncIterator[str]:
        # Tell any intermediary not to buffer, or the whole point of streaming
        # is lost behind a proxy.
        yield ": trustgraph stream open\n\n"
        try:
            async for event in pipeline.run(spec, fresh=request.fresh):
                yield _sse(event)
        except asyncio.CancelledError:  # client navigated away
            raise
        except Exception as exc:  # noqa: BLE001
            yield _sse({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        yield "data: {\"type\": \"stream_end\"}\n\n"

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post("/api/run/sync")
async def run_sync(request: RunRequest) -> JSONResponse:
    """Blocking variant, for scripting and tests."""
    spec = RunSpec.model_validate(request.model_dump(exclude={"fresh"}))
    try:
        result = await Pipeline().collect(spec, fresh=request.fresh)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return JSONResponse(result.model_dump())
