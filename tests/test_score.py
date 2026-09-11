"""The scoring maths — Eq. (3), Eq. (5), AITC, consensus, head-to-head."""

from __future__ import annotations

import math

import pytest

from trustgraph.domain import (
    Cell,
    CellState,
    EntityFinding,
    ModelRef,
    RunSpec,
    Verdict,
)
from trustgraph.score import (
    MAX_VARIANCE,
    consensus_composition,
    cross_model_consensus,
    head_to_head,
    pearson,
    population_variance,
    rsi,
    score_run,
    stability_band,
)


def model(key: str, weight: float = 1.0) -> ModelRef:
    return ModelRef(
        key=key, provider="simulated", model=key, family="simulated", label=key, weight=weight
    )


def cell(
    q: int,
    model_key: str,
    *,
    target: Verdict = Verdict.ABSENT,
    competitors: dict[str, Verdict] | None = None,
    error: str | None = None,
) -> Cell:
    competitors = competitors or {}
    target_finding = EntityFinding(name="T", verdict=target)
    comp_findings = [EntityFinding(name=n, verdict=v) for n, v in competitors.items()]
    from trustgraph.domain import derive_cell_state

    return Cell(
        query_index=q,
        model_key=model_key,
        state=derive_cell_state(target, competitors),
        target=target_finding,
        competitors=comp_findings,
        error=error,
    )


# ---------------------------------------------------------------------------
# RSI (Eq. 5)
# ---------------------------------------------------------------------------


def test_max_variance_is_the_bernoulli_maximum():
    assert MAX_VARIANCE == 0.25
    assert population_variance([0, 1] * 8) == pytest.approx(0.25)


def test_rsi_is_one_when_perfectly_consistent():
    assert rsi([1.0] * 10) == pytest.approx(1.0)
    # Consistently absent is equally *consistent*. Eq. (5) has no direction —
    # which is exactly why the UI always pairs RSI with a direction label.
    assert rsi([0.0] * 10) == pytest.approx(1.0)


def test_rsi_is_zero_at_maximum_disagreement():
    assert rsi([1.0, 0.0] * 6) == pytest.approx(0.0)


def test_rsi_matches_the_closed_form():
    indicators = [1.0] * 7 + [0.0] * 3
    p = 0.7
    assert rsi(indicators) == pytest.approx(1 - 4 * p * (1 - p))


def test_rsi_is_bounded():
    for n_ones in range(0, 13):
        value = rsi([1.0] * n_ones + [0.0] * (12 - n_ones))
        assert 0.0 <= value <= 1.0


def test_rsi_of_empty_is_zero():
    assert rsi([]) == 0.0


def test_stability_bands():
    assert stability_band(1.0) == "stable"
    assert stability_band(0.3) == "mixed"
    assert stability_band(0.0) == "fragile"


# ---------------------------------------------------------------------------
# Pearson
# ---------------------------------------------------------------------------


def test_pearson_perfect_and_inverse():
    assert pearson([0, 1, 0, 1], [0, 1, 0, 1]) == pytest.approx(1.0)
    assert pearson([0, 1, 0, 1], [1, 0, 1, 0]) == pytest.approx(-1.0)


def test_pearson_is_undefined_for_a_constant_series():
    """Undefined, not zero. Returning 0 would invent disagreement."""
    assert pearson([1, 1, 1, 1], [0, 1, 0, 1]) is None
    assert pearson([1, 1], [1, 1]) is None


def test_pearson_needs_two_points():
    assert pearson([1], [0]) is None


# ---------------------------------------------------------------------------
# Head-to-head (§7.4)
# ---------------------------------------------------------------------------


def test_head_to_head_excludes_uncontested_phrasings():
    cells = [
        cell(0, "m", target=Verdict.RECOMMENDED, competitors={"C": Verdict.ABSENT}),
        cell(1, "m", target=Verdict.ABSENT, competitors={"C": Verdict.RECOMMENDED}),
        cell(2, "m", target=Verdict.RECOMMENDED, competitors={"C": Verdict.RECOMMENDED}),
        # Off-axis: neither recommended, so it must not count either way.
        cell(3, "m", target=Verdict.ABSENT, competitors={"C": Verdict.ABSENT}),
    ]
    record = head_to_head(cells, "C", "m")
    assert (record.eligible, record.wins, record.losses, record.ties) == (3, 1, 1, 1)
    assert record.win_rate == pytest.approx(0.5)
    assert record.decisive_win_rate == pytest.approx(0.5)


def test_head_to_head_with_no_contest():
    record = head_to_head([cell(0, "m")], "C", "m")
    assert record.eligible == 0
    assert record.win_rate is None
    assert record.decisive_win_rate is None


def test_head_to_head_ignores_failed_cells():
    cells = [
        cell(0, "m", target=Verdict.RECOMMENDED, competitors={"C": Verdict.ABSENT}),
        cell(1, "m", error="boom"),
    ]
    assert head_to_head(cells, "C", "m").eligible == 1


# ---------------------------------------------------------------------------
# Consensus composition (§7.3)
# ---------------------------------------------------------------------------


def test_composition_partitions_every_phrasing():
    models = [model("a"), model("b")]
    cells = [
        cell(0, "a", target=Verdict.RECOMMENDED),
        cell(0, "b", target=Verdict.RECOMMENDED),  # unanimous win
        cell(1, "a", target=Verdict.RECOMMENDED),
        cell(1, "b", target=Verdict.ABSENT),  # split
        cell(2, "a", target=Verdict.ABSENT),
        cell(2, "b", target=Verdict.ABSENT),  # unanimous loss
    ]
    composition = consensus_composition(cells, models, 3)
    assert (composition.unanimous_win, composition.split, composition.unanimous_loss) == (1, 1, 1)
    assert composition.n == 3
    total = (
        composition.unanimous_win_frac
        + composition.split_frac
        + composition.unanimous_loss_frac
    )
    assert total == pytest.approx(1.0)


def test_composition_skips_phrasings_with_no_valid_response():
    models = [model("a")]
    composition = consensus_composition([cell(0, "a", error="x")], models, 1)
    assert composition.n == 0


# ---------------------------------------------------------------------------
# Cross-model correlation (§3.5)
# ---------------------------------------------------------------------------


def test_cross_model_correlation_is_reported_with_coverage():
    models = [model("a"), model("b")]
    cells = []
    pattern = [1, 0, 1, 0, 1, 0]
    for q, hit in enumerate(pattern):
        v = Verdict.RECOMMENDED if hit else Verdict.ABSENT
        cells.append(cell(q, "a", target=v))
        cells.append(cell(q, "b", target=v))
    result = cross_model_consensus(cells, models)
    assert result.coefficient == pytest.approx(1.0)
    assert (result.pairs_used, result.pairs_total) == (1, 1)


def test_cross_model_correlation_undefined_when_all_constant():
    models = [model("a"), model("b")]
    cells = [
        cell(q, key, target=Verdict.RECOMMENDED)
        for q in range(4)
        for key in ("a", "b")
    ]
    result = cross_model_consensus(cells, models)
    assert result.coefficient is None
    assert result.pairs_used == 0
    assert "undefined" in result.note


def test_cross_model_correlation_single_column():
    result = cross_model_consensus([cell(0, "a", target=Verdict.RECOMMENDED)], [model("a")])
    assert result.pairs_total == 0
    assert "not defined" in result.note


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


def test_aitc_is_a_weighted_mean_on_the_unit_interval():
    """The paper writes a weighted sum; we normalise so AITC stays comparable
    to Trust. Without normalising, three models at Trust=1.0 would give 3.0."""
    spec = RunSpec(entity="T", category="c", competitors=[])
    models = [model("a", weight=1.0), model("b", weight=3.0)]
    cells = [
        cell(0, "a", target=Verdict.RECOMMENDED),
        cell(0, "b", target=Verdict.ABSENT),
    ]
    scores = score_run(spec, models, cells, 1)
    # trust_a = 1.0, trust_b = 0.0 -> weighted mean = 1*1 / (1+3) = 0.25
    assert scores.aitc == pytest.approx(0.25)
    assert scores.aitc_unweighted == pytest.approx(0.5)
    assert 0.0 <= scores.aitc <= 1.0


def test_score_run_counts_states_and_errors():
    spec = RunSpec(entity="T", category="c", competitors=["C"])
    models = [model("a")]
    cells = [
        cell(0, "a", target=Verdict.RECOMMENDED, competitors={"C": Verdict.ABSENT}),
        cell(1, "a", target=Verdict.ABSENT, competitors={"C": Verdict.RECOMMENDED}),
        cell(2, "a", error="boom"),
    ]
    scores = score_run(spec, models, cells, 3)
    per_model = scores.per_model[0]
    assert per_model.n == 2
    assert per_model.errors == 1
    assert per_model.counts[CellState.WIN] == 1
    assert per_model.counts[CellState.LOSS] == 1
    assert per_model.trust == pytest.approx(0.5)
    assert per_model.competitor_trust["C"] == pytest.approx(0.5)


def test_headline_mentions_entity_and_a_competitor():
    spec = RunSpec(entity="Razorpay", category="c", competitors=["Stripe"])
    models = [model("a")]
    cells = [
        cell(q, "a", target=Verdict.RECOMMENDED, competitors={"Stripe": Verdict.ABSENT})
        for q in range(4)
    ]
    scores = score_run(spec, models, cells, 4)
    assert "Razorpay" in scores.headline
    assert "Stripe" in scores.headline
    assert "100%" in scores.headline


def test_all_failed_run_does_not_divide_by_zero():
    spec = RunSpec(entity="T", category="c", competitors=["C"])
    models = [model("a")]
    cells = [cell(q, "a", error="boom") for q in range(3)]
    scores = score_run(spec, models, cells, 3)
    assert scores.aitc == 0.0
    assert scores.per_model[0].n == 0
    assert not math.isnan(scores.per_model[0].rsi)


def test_a_column_with_no_data_is_unmeasured_not_fragile():
    """An outage must not be reported as a finding about the entity.

    RSI of an empty series is 0, which would otherwise render as the most
    alarming badge available.
    """
    spec = RunSpec(entity="T", category="c", competitors=[])
    models = [model("a")]
    scores = score_run(spec, models, [cell(q, "a", error="boom") for q in range(3)], 3)
    per_model = scores.per_model[0]
    assert per_model.stability == "unmeasured"
    assert "failed" in per_model.stability_note
    assert per_model.errors == 3


def test_headline_says_so_when_nothing_was_measured():
    spec = RunSpec(entity="T", category="c", competitors=["C"])
    models = [model("a")]
    scores = score_run(spec, models, [cell(0, "a", error="boom")], 1)
    assert "No measurement completed" in scores.headline
    assert "0%" not in scores.headline


def test_no_data_is_distinguished_from_zero_variance():
    models = [model("a"), model("b")]
    empty = cross_model_consensus([cell(0, "a", error="x"), cell(0, "b", error="x")], models)
    assert empty.coefficient is None
    assert "nothing to correlate" in empty.note

    constant = cross_model_consensus(
        [cell(q, k, target=Verdict.RECOMMENDED) for q in range(3) for k in ("a", "b")],
        models,
    )
    assert "zero variance" in constant.note
