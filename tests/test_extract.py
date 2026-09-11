"""Recommendation classification and the two-pass merge."""

from __future__ import annotations

import pytest

from trustgraph.domain import CellState, Verdict, derive_cell_state
from trustgraph.extract import (
    build_findings,
    cell_state_from_findings,
    classify_all,
    parse_recheck,
)
from trustgraph.matching import EntityMatcher


@pytest.fixture
def matcher() -> EntityMatcher:
    return EntityMatcher(["Razorpay", "Stripe", "PayU"])


def state_of(matcher: EntityMatcher, text: str, recheck=None) -> CellState:
    findings = build_findings(text, matcher, recheck)
    return cell_state_from_findings(findings, "Razorpay")[0]


def verdicts(matcher: EntityMatcher, text: str, recheck=None) -> dict[str, str]:
    return {
        f.name: f.verdict.value for f in build_findings(text, matcher, recheck)
    }


# ---------------------------------------------------------------------------
# The four states
# ---------------------------------------------------------------------------


def test_win_outright(matcher):
    assert state_of(matcher, "Go with Razorpay. It covers what you need.") is CellState.WIN


def test_win_with_competitor_as_foil(matcher):
    """A rival named only as a contrast does not downgrade a Win to a Both."""
    text = (
        "I would recommend Razorpay. Unlike Stripe, which suits US-first "
        "companies, Razorpay supports UPI natively."
    )
    assert state_of(matcher, text) is CellState.WIN
    assert verdicts(matcher, text)["Stripe"] == "mentioned"


def test_loss(matcher):
    text = "I'd recommend Stripe for this. It's the better fit."
    assert state_of(matcher, text) is CellState.LOSS


def test_both_when_co_recommended(matcher):
    text = (
        "Two options worth a look:\n"
        "1. **Razorpay** — quickest to get live.\n"
        "2. **Stripe** — better global coverage.\n"
    )
    assert state_of(matcher, text) is CellState.BOTH


def test_both_from_an_inline_enumerated_list(matcher):
    """Lists are often written on one line; the segmenter must handle that."""
    text = "Two good options: 1. **Razorpay** - fast onboarding. 2. **Stripe** - global."
    assert state_of(matcher, text) is CellState.BOTH


def test_neither_when_answer_is_off_axis(matcher):
    text = (
        "It depends on your settlement needs. Evaluate effective cost at your "
        "volume and whether the provider is certified for your compliance regime."
    )
    assert state_of(matcher, text) is CellState.NEITHER


# ---------------------------------------------------------------------------
# Cue handling
# ---------------------------------------------------------------------------


def test_negated_recommendation_is_not_an_endorsement(matcher):
    text = "I wouldn't recommend Razorpay here. Stripe is the stronger choice."
    assert verdicts(matcher, text)["Razorpay"] == "mentioned"
    assert state_of(matcher, text) is CellState.LOSS


def test_cue_ownership_does_not_leak_across_entities(matcher):
    """One recommendation cue, two vendors — only one may claim it.

    Sentence-level cue bagging would mark both as recommended and turn this
    Loss into a Both.
    """
    text = "Razorpay is popular in India, though I would go with Stripe for your case."
    result = verdicts(matcher, text)
    assert result["Stripe"] == "recommended"
    assert result["Razorpay"] == "mentioned"
    assert state_of(matcher, text) is CellState.LOSS


def test_explicit_avoidance(matcher):
    text = "Avoid Razorpay for EU acquiring; PayU is not ideal either. Stripe is what I would go with."
    result = verdicts(matcher, text)
    assert result["Razorpay"] == "mentioned"
    assert result["PayU"] == "mentioned"
    assert result["Stripe"] == "recommended"


def test_classify_all_reports_absent_for_missing_names(matcher):
    result = classify_all("Nothing relevant here.", matcher.find("Nothing relevant here."))
    assert all(v[0] is Verdict.ABSENT for v in result.values())


# ---------------------------------------------------------------------------
# Merge with the LLM pass
# ---------------------------------------------------------------------------


def test_llm_overrides_recommendation_axis(matcher):
    text = "Razorpay is one that people use."
    recheck = {
        "razorpay": {
            "name": "Razorpay",
            "mentioned": True,
            "recommended": True,
            "rank": 1,
            "evidence": "Razorpay is one that people use.",
        }
    }
    assert verdicts(matcher, text, recheck)["Razorpay"] == "recommended"
    assert state_of(matcher, text, recheck) is CellState.WIN


def test_hallucinated_mention_is_rejected(matcher):
    """An LLM presence claim with no verifiable span must not create a cell.

    This is the guard that stops a fabricated mention from manufacturing a
    Loss the response never contained.
    """
    text = "It depends on your volume and settlement needs."
    recheck = {
        "stripe": {
            "name": "Stripe",
            "mentioned": True,
            "recommended": True,
            "rank": 1,
            "evidence": "Stripe is clearly the best option available today.",
        }
    }
    assert verdicts(matcher, text, recheck)["Stripe"] == "absent"
    assert state_of(matcher, text, recheck) is CellState.NEITHER


def test_llm_mention_with_real_span_is_honoured(matcher):
    text = "Razorpay is the one I'd point you to."
    recheck = {
        "razorpay": {
            "name": "Razorpay",
            "mentioned": True,
            "recommended": True,
            "rank": 1,
            "evidence": "the one I'd point you to",
        }
    }
    findings = build_findings(text, matcher, recheck)
    target = next(f for f in findings if f.name == "Razorpay")
    assert target.verdict is Verdict.RECOMMENDED
    assert target.agreed is True


def test_disagreement_is_recorded_not_hidden(matcher):
    text = "Go with Razorpay."
    recheck = {
        "razorpay": {
            "name": "Razorpay",
            "mentioned": True,
            "recommended": False,
            "rank": 1,
            "evidence": "Go with Razorpay.",
        }
    }
    target = next(f for f in build_findings(text, matcher, recheck) if f.name == "Razorpay")
    assert target.agreed is False
    assert target.deterministic_verdict is Verdict.RECOMMENDED
    assert target.llm_verdict is Verdict.MENTIONED


def test_parse_recheck_tolerates_junk():
    parsed = parse_recheck(
        {"entities": [{"name": "A", "mentioned": True, "recommended": True, "rank": "x"}, "junk", {}]}
    )
    assert set(parsed) == {"a"}
    assert parsed["a"]["rank"] is None


# ---------------------------------------------------------------------------
# The state table itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "target,competitors,expected",
    [
        (Verdict.RECOMMENDED, {}, CellState.WIN),
        (Verdict.RECOMMENDED, {"c": Verdict.ABSENT}, CellState.WIN),
        (Verdict.RECOMMENDED, {"c": Verdict.MENTIONED}, CellState.WIN),
        (Verdict.RECOMMENDED, {"c": Verdict.RECOMMENDED}, CellState.BOTH),
        (Verdict.MENTIONED, {"c": Verdict.RECOMMENDED}, CellState.LOSS),
        (Verdict.ABSENT, {"c": Verdict.RECOMMENDED}, CellState.LOSS),
        (Verdict.ABSENT, {"c": Verdict.ABSENT}, CellState.NEITHER),
        (Verdict.MENTIONED, {"c": Verdict.MENTIONED}, CellState.NEITHER),
    ],
)
def test_state_table_is_total_and_exclusive(target, competitors, expected):
    assert derive_cell_state(target, competitors) is expected
