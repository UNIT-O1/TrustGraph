"""End-to-end pipeline, roster degradation, and API surface."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from trustgraph.cache import DiskCache
from trustgraph.config import Settings
from trustgraph.domain import CellState, RunSpec
from trustgraph.jsonutil import parse_json_object
from trustgraph.paraphrase import ensure_coverage, fallback, parse
from trustgraph.pipeline import Pipeline
from trustgraph.providers import CompletionRequest, ProviderError, ProviderRegistry


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(_env_file=None, TRUSTGRAPH_CACHE_DIR=str(tmp_path / "cache"))


@pytest.fixture
def pipeline(settings, tmp_path) -> Pipeline:
    return Pipeline(settings=settings, cache=DiskCache(tmp_path / "cache", enabled=True))


SPEC = RunSpec(
    entity="Razorpay",
    category="payment gateway for an Indian D2C startup",
    competitors=["Stripe", "PayU"],
    paraphrase_count=8,
    seed=42,
)


# ---------------------------------------------------------------------------
# Roster degradation
# ---------------------------------------------------------------------------


def test_no_keys_falls_back_to_simulated(settings):
    roster, warnings = settings.build_roster()
    assert [m.provider for m in roster] == ["simulated"] * 3
    assert any("simulated" in w for w in warnings)


def test_single_provider_widens_and_warns_about_family(tmp_path):
    settings = Settings(_env_file=None, GEMINI_API_KEY="x")
    roster, warnings = settings.build_roster()
    assert len(roster) == 3
    assert {m.provider for m in roster} == {"gemini"}
    # The honesty requirement: a one-family roster must not be presented as
    # cross-family consensus.
    assert any("within-family" in w for w in warnings)
    assert any("3.5" in w for w in warnings)


def test_two_providers_use_one_column_each():
    settings = Settings(_env_file=None, GEMINI_API_KEY="x", OPENAI_API_KEY="y")
    roster, warnings = settings.build_roster()
    assert {m.provider for m in roster} == {"gemini", "openai"}
    assert len(roster) == 2


def test_explicit_roster_override_warns_about_missing_keys():
    settings = Settings(_env_file=None)
    roster, warnings = settings.build_roster(["gemini:gemini-2.5-pro"])
    assert roster[0].model == "gemini-2.5-pro"
    assert any("no API key" in w for w in warnings)


# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------


async def test_run_produces_a_complete_grid(pipeline):
    result = await pipeline.collect(SPEC)

    assert len(result.models) == 3
    assert len(result.paraphrases) == 8
    # Every (phrasing, model) pair must produce exactly one cell.
    assert len(result.cells) == 8 * 3
    seen = {(c.query_index, c.model_key) for c in result.cells}
    assert len(seen) == 8 * 3

    assert result.meta.simulated is True
    assert result.scores.headline.startswith("Razorpay is recommended in")
    assert 0.0 <= result.scores.aitc <= 1.0

    composition = result.scores.consensus
    assert composition.n == 8
    assert (
        composition.unanimous_win + composition.split + composition.unanimous_loss
    ) == 8


async def test_simulated_run_warns_about_fixtures_and_nothing_redundant(pipeline):
    """The fixtures banner already invalidates every figure.

    Adding a within-family caveat on top would claim "Trust and RSI remain
    valid", which is false for fixtures.
    """
    result = await pipeline.collect(SPEC)
    warnings = result.meta.warnings
    assert any("simulated provider" in w for w in warnings)
    assert not any("within-family" in w for w in warnings)
    assert len(warnings) == 1


async def test_single_family_real_roster_does_warn():
    settings = Settings(_env_file=None, GEMINI_API_KEY="x")
    roster, warnings = settings.build_roster()
    assert len(roster) == 3
    assert any("within-family" in w for w in warnings)


async def test_run_is_deterministic_for_a_fixed_seed(pipeline):
    first = await pipeline.collect(SPEC, fresh=True)
    second = await pipeline.collect(SPEC, fresh=True)
    assert [c.state for c in first.cells] == [c.state for c in second.cells]
    assert first.scores.aitc == second.scores.aitc


async def test_events_arrive_in_a_usable_order(pipeline):
    types = []
    cells = 0
    async for event in pipeline.run(SPEC):
        types.append(event["type"])
        if event["type"] == "cell":
            cells += 1
            # Cells must stream before scoring, or the grid cannot fill live.
            assert "scores" not in types

    assert types[0] == "run_started"
    assert types[1] == "paraphrases"
    assert types[-1] == "run_complete"
    assert types[-2] == "scores"
    assert cells == 8 * 3


async def test_cache_is_used_on_a_second_identical_run(pipeline):
    await pipeline.collect(SPEC, fresh=True)
    second = await pipeline.collect(SPEC)
    assert second.meta.cached_calls > 0
    assert all(c.cached for c in second.cells)


async def test_fresh_bypasses_the_cache(pipeline):
    await pipeline.collect(SPEC, fresh=True)
    second = await pipeline.collect(SPEC, fresh=True)
    assert second.meta.cached_calls == 0


async def test_all_four_states_are_reachable(pipeline):
    """The four-state encoding is only worth having if the pipeline can produce
    all four from real prose."""
    spec = SPEC.model_copy(update={"paraphrase_count": 20, "seed": 3})
    result = await pipeline.collect(spec)
    assert {c.state for c in result.cells} == set(CellState)


async def test_run_without_competitors_still_scores(pipeline):
    spec = RunSpec(entity="Razorpay", category="payment gateway", competitors=[], paraphrase_count=6)
    result = await pipeline.collect(spec)
    assert result.scores.head_to_head == []
    # With no competitors, no cell can ever be Loss or Both.
    assert {c.state for c in result.cells} <= {CellState.WIN, CellState.NEITHER}


async def test_failing_provider_isolates_to_its_own_cells(pipeline, monkeypatch):
    """A provider outage must cost cells, never the run."""
    from trustgraph.providers.simulated import SimulatedProvider

    original = SimulatedProvider.complete
    calls = {"n": 0}

    async def flaky(self, request):
        # Fail the answer calls for one model only; leave utility calls alone.
        if request.json_schema is None and request.model == "sim-beta":
            calls["n"] += 1
            raise ProviderError("simulated outage")
        return await original(self, request)

    monkeypatch.setattr(SimulatedProvider, "complete", flaky)

    result = await pipeline.collect(SPEC, fresh=True)
    assert calls["n"] > 0
    failed = [c for c in result.cells if c.error]
    assert len(failed) == 8
    assert all(c.model_key.endswith("sim-beta") for c in failed)

    beta = next(s for s in result.scores.per_model if s.model_key.endswith("sim-beta"))
    assert beta.n == 0
    assert beta.errors == 8
    # The other columns are unaffected and still produce a score.
    others = [s for s in result.scores.per_model if not s.model_key.endswith("sim-beta")]
    assert all(s.n == 8 for s in others)


async def test_recheck_failure_degrades_to_deterministic(pipeline, monkeypatch):
    from trustgraph.providers.simulated import SimulatedProvider

    original = SimulatedProvider.complete

    async def no_recheck(self, request):
        if request.json_schema and "entities" in (request.json_schema.get("properties") or {}):
            raise ProviderError("recheck down")
        return await original(self, request)

    monkeypatch.setattr(SimulatedProvider, "complete", no_recheck)

    result = await pipeline.collect(SPEC, fresh=True)
    assert len(result.cells) == 24
    assert all(c.error is None for c in result.cells)
    # No second opinion means no agreement statistic — reported as null, not faked.
    assert result.scores.extraction_agreement is None


async def test_paraphrase_failure_falls_back_to_templates(pipeline, monkeypatch):
    from trustgraph.providers.simulated import SimulatedProvider

    original = SimulatedProvider.complete

    async def no_paraphrase(self, request):
        if request.json_schema and "queries" in (request.json_schema.get("properties") or {}):
            raise ProviderError("paraphrase down")
        return await original(self, request)

    monkeypatch.setattr(SimulatedProvider, "complete", no_paraphrase)

    result = await pipeline.collect(SPEC, fresh=True)
    assert len(result.paraphrases) == 8
    assert any("templates" in w for w in result.meta.warnings)


# ---------------------------------------------------------------------------
# Paraphrase helpers
# ---------------------------------------------------------------------------


def test_paraphrase_parse_dedupes_and_trims():
    payload = {
        "queries": [
            {"text": "best gateway", "intent": "Superlative"},
            {"text": "  best   gateway ", "intent": "dupe"},
            {"text": "", "intent": "empty"},
            {"text": "x" * 400, "intent": "toolong"},
            "bare string form",
        ]
    }
    result = parse(payload, 10)
    assert [p.text for p in result] == ["best gateway", "bare string form"]
    assert result[0].intent == "superlative"
    assert [p.index for p in result] == [0, 1]


def test_ensure_coverage_guarantees_a_price_framing():
    generated = parse({"queries": [{"text": "best gateway", "intent": "superlative"}]}, 5)
    topped = ensure_coverage(generated, "gateway", 5)
    assert len(topped) == 5
    assert any((p.intent or "") == "price" for p in topped)
    assert [p.index for p in topped] == list(range(5))


def test_fallback_is_the_requested_length():
    assert len(fallback("gateway", 12)) == 12


# ---------------------------------------------------------------------------
# JSON tolerance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        'Here you go:\n{"a": 1}\nHope that helps.',
        '{"a": 1,}',
    ],
)
def test_parse_json_object_recovers_wrapped_payloads(raw):
    assert parse_json_object(raw)["a"] == 1


def test_parse_json_object_rejects_garbage():
    with pytest.raises(ValueError):
        parse_json_object("no json at all")


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


async def test_registry_refuses_unconfigured_provider(settings):
    registry = ProviderRegistry(settings)
    with pytest.raises(ProviderError, match="no API key"):
        registry.get("gemini")
    await registry.aclose()


async def test_simulated_provider_is_reproducible(settings):
    registry = ProviderRegistry(settings, seed=1)
    provider = registry.get("simulated")
    request = CompletionRequest(
        model="sim-alpha",
        prompt="best payment gateway",
        sim_context={"entity": "Razorpay", "competitors": ["Stripe"], "category": "x"},
    )
    a = await provider.complete(request)
    b = await provider.complete(request)
    assert a.text == b.text
    await registry.aclose()


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


@pytest.fixture
def client() -> TestClient:
    from trustgraph.api import app

    return TestClient(app)


def test_index_serves_the_dashboard(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "TrustGraph" in response.text


def test_health_reports_roster_and_providers(client):
    payload = client.get("/api/health").json()
    assert payload["ok"] is True
    assert set(payload["providers"]) == {"gemini", "anthropic", "openai"}
    assert payload["roster"]


def test_run_sync_returns_a_scored_result(client):
    response = client.post(
        "/api/run/sync",
        json={
            "entity": "Razorpay",
            "category": "payment gateway",
            "competitors": ["Stripe"],
            "paraphrase_count": 4,
            "seed": 5,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["cells"]) == 4 * len(payload["models"])
    assert "headline" in payload["scores"]


def test_run_stream_emits_sse_events(client):
    with client.stream(
        "POST",
        "/api/run",
        json={
            "entity": "Razorpay",
            "category": "payment gateway",
            "competitors": ["Stripe"],
            "paraphrase_count": 4,
            "seed": 5,
        },
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())

    assert '"type": "run_started"' in body
    assert '"type": "cell"' in body
    assert '"type": "run_complete"' in body


def test_run_rejects_a_blank_entity(client):
    response = client.post(
        "/api/run/sync", json={"entity": "   ", "category": "payment gateway"}
    )
    assert response.status_code == 422
