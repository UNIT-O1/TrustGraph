"""Matcher precision — the failure modes that would silently fabricate cells."""

from __future__ import annotations

import pytest

from trustgraph.matching import (
    EntityMatcher,
    build_aliases,
    normalize,
    rank_by_first_mention,
    squash,
)


def names(matcher: EntityMatcher, text: str) -> set[str]:
    return {name for name, spans in matcher.find(text).items() if spans}


@pytest.fixture
def matcher() -> EntityMatcher:
    return EntityMatcher(["Razorpay", "Stripe", "Cashfree Payments", "PayU"])


def test_normalize_and_squash():
    assert normalize("  Razor  Pay ") == "razor pay"
    assert squash("Razor-Pay!") == "razorpay"
    assert squash("Café") == "cafe"


def test_exact_and_case_insensitive(matcher):
    assert names(matcher, "I'd use razorpay here.") == {"Razorpay"}


def test_squashed_spacing_variant(matcher):
    """"Razor Pay" is the same vendor as "Razorpay"."""
    assert names(matcher, "Try Razor Pay for this.") == {"Razorpay"}
    assert names(matcher, "razor-pay works too") == {"Razorpay"}


def test_suffix_alias(matcher):
    assert names(matcher, "Cashfree is fine.") == {"Cashfree Payments"}
    assert names(matcher, "Cashfree Payments Pvt Ltd is fine.") == {"Cashfree Payments"}


def test_punctuation_boundary_is_a_match(matcher):
    assert names(matcher, "See razorpay.com for docs.") == {"Razorpay"}


@pytest.mark.parametrize(
    "text",
    [
        "The fabric had stripes on it.",
        "It was striped and worn.",
        "razorpays",
        "unstripe the wire",
    ],
)
def test_morphological_variants_are_rejected(matcher, text):
    """The prefix fence is what makes fuzzy matching safe on brand names.

    Without it "stripes" scores ~92 against "Stripe" and would manufacture a
    Loss out of a sentence about fabric.
    """
    assert names(matcher, text) == set()


def test_longest_match_wins():
    matcher = EntityMatcher(["Pay", "PayU"])
    # "Pay" is below the minimum alias length, so it cannot shadow "PayU".
    assert build_aliases("Pay") == []
    assert names(matcher, "PayU is the one.") == {"PayU"}


def test_no_double_counting_of_one_mention(matcher):
    found = matcher.find("Cashfree Payments is good.")
    assert len(found["Cashfree Payments"]) == 1


def test_ranks_follow_first_appearance(matcher):
    ranks = rank_by_first_mention(
        matcher.find("First Stripe, then Razorpay, and PayU last.")
    )
    assert ranks["Stripe"] == 1
    assert ranks["Razorpay"] == 2
    assert ranks["PayU"] == 3
    assert ranks["Cashfree Payments"] is None


def test_empty_text_is_safe(matcher):
    assert names(matcher, "") == set()
