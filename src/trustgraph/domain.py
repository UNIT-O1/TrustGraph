"""Core domain types for the Recommendation Stability Tool.

The vocabulary here maps 1:1 onto the working paper:

- ``Verdict``     — per (query, model, entity) extraction outcome.
- ``CellState``   — the four-state grid cell encoding of Table 2 (§7.2).
- ``Cell``        — one square of the hero grid: rows are paraphrases, columns
                    are models (§7.1).
- ``ModelScore``  — Trust (Eq. 3) and RSI (Eq. 5) for one model.
- ``RunResult``   — everything the dashboard needs, in one payload.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Extraction vocabulary
# ---------------------------------------------------------------------------


class Verdict(str, Enum):
    """How one entity fared in one model response.

    The distinction between ``RECOMMENDED`` and ``MENTIONED`` is the whole
    point of §7.2: a name appearing in a caveat ("unlike Stripe, which charges
    more") is not the same event as a name being put forward as the answer.
    """

    RECOMMENDED = "recommended"
    MENTIONED = "mentioned"
    ABSENT = "absent"

    @property
    def is_present(self) -> bool:
        return self is not Verdict.ABSENT


class CellState(str, Enum):
    """The four-state cell encoding of Table 2."""

    WIN = "win"
    LOSS = "loss"
    BOTH = "both"
    NEITHER = "neither"


# ---------------------------------------------------------------------------
# Cell-state derivation
# ---------------------------------------------------------------------------
#
# SPEC RESOLUTION (documented deliberately, not glossed over).
#
# Table 2's four rows overlap as literally written. "Win" is defined on the
# *recommendation* axis ("no named competitor is [recommended]") while "Both"
# is defined on the *presence* axis ("both mentioned/recommended"). A response
# that recommends the target while merely name-dropping a competitor therefore
# satisfies both rows, and a response that merely mentions the target and no
# competitor at all satisfies none of the four.
#
# We resolve this by making `recommended` the primary axis. That is the axis
# Eq. (3) actually counts, so deriving the grid from it keeps the picture and
# the number describing the same event:
#
#     BOTH    <- target recommended AND >=1 competitor recommended
#     WIN     <- target recommended AND no competitor recommended
#     LOSS    <- >=1 competitor recommended AND target not recommended
#     NEITHER <- no tracked entity recommended at all
#
# This partition is total and mutually exclusive, and NEITHER lands exactly on
# its gloss in Table 2 — "the model answered off-axis, e.g. recommending a
# category of solution rather than a named vendor".
#
# The presence nuance is not discarded, it is demoted: every cell keeps
# `target_mentioned_only` / `competitors_mentioned_only`, which the grid renders
# as a marker and the evidence drawer spells out. So "recommended the target but
# name-dropped a rival" is still visible — it just doesn't silently collapse a
# Win into a Both.


def derive_cell_state(
    target: Verdict,
    competitors: dict[str, Verdict],
) -> CellState:
    """Map extraction verdicts onto the four-state encoding of Table 2."""
    target_rec = target is Verdict.RECOMMENDED
    comp_rec = any(v is Verdict.RECOMMENDED for v in competitors.values())

    if target_rec and comp_rec:
        return CellState.BOTH
    if target_rec:
        return CellState.WIN
    if comp_rec:
        return CellState.LOSS
    return CellState.NEITHER


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


class RunSpec(BaseModel):
    """The entire input surface, per §6.2."""

    entity: str = Field(..., min_length=1, max_length=120)
    category: str = Field(..., min_length=1, max_length=400)
    competitors: list[str] = Field(default_factory=list, max_length=5)
    paraphrase_count: int | None = Field(default=None, ge=4, le=40)
    llm_recheck: bool | None = None
    models: list[str] | None = None
    seed: int | None = None

    @field_validator("entity", "category")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = " ".join(v.split())
        if not v:
            raise ValueError("must not be blank")
        return v

    @field_validator("competitors")
    @classmethod
    def _clean_competitors(cls, v: list[str]) -> list[str]:
        seen: dict[str, str] = {}
        for raw in v:
            name = " ".join(raw.split())
            if name and name.casefold() not in seen:
                seen[name.casefold()] = name
        return list(seen.values())

    def tracked_names(self) -> list[str]:
        """Target first, then competitors — the canonical column order."""
        return [self.entity, *self.competitors]


# ---------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------


class ModelRef(BaseModel):
    """A model column in the grid."""

    key: str  # "gemini:gemini-2.5-pro"
    provider: str  # "gemini"
    model: str  # "gemini-2.5-pro"
    family: str  # "gemini" — the *vendor family*, for the consensus caveat
    label: str  # "Gemini 2.5 Pro"
    weight: float = 1.0  # w_m in the AITC sum (§3.5)


# ---------------------------------------------------------------------------
# Per-response extraction record
# ---------------------------------------------------------------------------


class Match(BaseModel):
    """One textual hit for a tracked name, kept for the evidence drawer."""

    name: str
    start: int
    end: int
    text: str
    via: Literal["exact", "alias", "fuzzy"]


class EntityFinding(BaseModel):
    name: str
    verdict: Verdict
    rank: int | None = None  # 1-based order of first appearance
    matches: list[Match] = Field(default_factory=list)
    evidence: str | None = None  # the span that justifies `recommended`
    # Provenance of the verdict — surfaced so the score is inspectable (§7.2).
    deterministic_verdict: Verdict | None = None
    llm_verdict: Verdict | None = None
    agreed: bool | None = None


class Cell(BaseModel):
    """One square of the hero grid."""

    query_index: int
    model_key: str
    state: CellState
    target: EntityFinding
    competitors: list[EntityFinding] = Field(default_factory=list)
    response_text: str = ""
    target_mentioned_only: bool = False
    competitors_mentioned_only: list[str] = Field(default_factory=list)
    error: str | None = None
    latency_ms: int | None = None
    cached: bool = False

    @property
    def target_recommended(self) -> bool:
        return self.target.verdict is Verdict.RECOMMENDED


class Paraphrase(BaseModel):
    index: int
    text: str
    # Tags let the UI surface the §7.1 finding — e.g. a red band across every
    # price-framed phrasing while category-framed phrasings stay green.
    intent: str | None = None


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------


class HeadToHead(BaseModel):
    competitor: str
    model_key: str | None = None  # None == aggregated across models
    eligible: int
    wins: int
    losses: int
    ties: int
    win_rate: float | None = None  # (wins + 0.5*ties) / eligible
    decisive_win_rate: float | None = None  # wins / (wins + losses)


class ModelScore(BaseModel):
    model_key: str
    n: int  # paraphrases scored (excludes hard errors)
    trust: float  # Eq. (3)
    rsi: float  # Eq. (5)
    variance: float
    #: "unmeasured" is distinct from "fragile": a column whose every call failed
    #: has no variance to report, and labelling that instability would invent a
    #: finding out of an outage.
    stability: Literal["stable", "mixed", "fragile", "unmeasured"]
    stability_note: str
    counts: dict[CellState, int]
    competitor_trust: dict[str, float] = Field(default_factory=dict)
    head_to_head: list[HeadToHead] = Field(default_factory=list)
    errors: int = 0


class ConsensusComposition(BaseModel):
    """§7.3 — one composition bar, not per-model cards."""

    n: int
    unanimous_win: int
    split: int
    unanimous_loss: int
    unanimous_win_frac: float
    split_frac: float
    unanimous_loss_frac: float


class CrossModelConsensus(BaseModel):
    """The stronger AITC formulation of §3.5.

    Mean pairwise Pearson correlation of the per-query recommendation indicator
    vectors. Pairs where either vector is constant have undefined correlation
    and are excluded — `pairs_used` / `pairs_total` reports that honestly rather
    than silently imputing zero.
    """

    coefficient: float | None
    pairs_used: int
    pairs_total: int
    pairwise: dict[str, float] = Field(default_factory=dict)
    note: str = ""


class Scores(BaseModel):
    per_model: list[ModelScore]
    aitc: float  # §3.5, weighted cross-model trust
    aitc_unweighted: float
    consensus: ConsensusComposition
    cross_model: CrossModelConsensus
    head_to_head: list[HeadToHead]  # aggregated across models
    extraction_agreement: float | None = None
    headline: str = ""


class RunMeta(BaseModel):
    run_id: str
    started_at: str
    finished_at: str | None = None
    duration_ms: int | None = None
    provider_calls: int = 0
    cached_calls: int = 0
    failed_calls: int = 0
    simulated: bool = False
    single_family: str | None = None  # set when every column shares one family
    warnings: list[str] = Field(default_factory=list)


class RunResult(BaseModel):
    meta: RunMeta
    spec: RunSpec
    models: list[ModelRef]
    paraphrases: list[Paraphrase]
    cells: list[Cell]
    scores: Scores
