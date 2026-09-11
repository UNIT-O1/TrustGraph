"""Scoring: Trust (Eq. 3), RSI (Eq. 5), AITC (§3.5), and consensus (§7.3).

Every formula below is the paper's, with two places where the paper's notation
needs a decision made explicitly rather than silently. Both are marked.
"""

from __future__ import annotations

import math

from .domain import (
    Cell,
    CellState,
    ConsensusComposition,
    CrossModelConsensus,
    HeadToHead,
    ModelRef,
    ModelScore,
    RunSpec,
    Scores,
    Verdict,
)

#: Maximum variance of a Bernoulli indicator, i.e. the ``Max_Variance``
#: denominator in Eq. (5). An indicator on {0,1} has variance p(1-p), maximised
#: at p = 0.5. Using anything else here would make RSI unbounded.
MAX_VARIANCE = 0.25

#: RSI bands for the badge. RSI = 1 - 4p(1-p), so these are tighter than they
#: look: "stable" needs a recommendation rate above ~85% or below ~15%.
_STABLE_AT = 0.5
_MIXED_AT = 0.2


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def population_variance(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def rsi(indicators: list[float]) -> float:
    """Eq. (5): 1 - Var / Max_Variance, clamped to [0, 1]."""
    if not indicators:
        return 0.0
    return max(0.0, min(1.0, 1.0 - population_variance(indicators) / MAX_VARIANCE))


def pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation, or ``None`` when either series is constant."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        # A constant vector has zero variance, so correlation is undefined
        # rather than zero. Reporting 0 here would fake disagreement.
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sxx * syy)


def stability_band(value: float) -> str:
    if value >= _STABLE_AT:
        return "stable"
    if value >= _MIXED_AT:
        return "mixed"
    return "fragile"


def stability_note(band: str, trust: float) -> str:
    """Give RSI a direction.

    Eq. (5) is a pure consistency measure: an entity recommended in 0% of
    phrasings is exactly as "stable" as one recommended in 100%. That is
    mathematically correct and practically misleading, so the badge always
    carries the direction alongside it.
    """
    if band == "stable":
        return (
            "consistently recommended regardless of phrasing"
            if trust >= 0.5
            else "consistently absent regardless of phrasing"
        )
    if band == "mixed":
        return "phrasing changes the answer on a minority of variants"
    return "highly phrasing-sensitive — the recommendation flips with wording"


# ---------------------------------------------------------------------------
# Per-model scores
# ---------------------------------------------------------------------------


def score_model(
    model: ModelRef,
    cells: list[Cell],
    spec: RunSpec,
) -> ModelScore:
    valid = [c for c in cells if c.error is None]
    errors = len(cells) - len(valid)

    indicators = [1.0 if c.target_recommended else 0.0 for c in valid]
    n = len(valid)
    trust = (sum(indicators) / n) if n else 0.0
    variance = population_variance(indicators)
    value = rsi(indicators)

    if n == 0:
        # No valid responses means no measurement. Calling that "fragile" would
        # report an outage as a finding about the entity.
        band = "unmeasured"
        note = f"no valid responses — all {errors} call(s) for this column failed"
    else:
        band = stability_band(value)
        note = stability_note(band, trust)

    counts = {state: 0 for state in CellState}
    for cell in valid:
        counts[cell.state] += 1

    competitor_trust: dict[str, float] = {}
    for competitor in spec.competitors:
        hits = sum(
            1
            for c in valid
            if any(
                f.name == competitor and f.verdict is Verdict.RECOMMENDED
                for f in c.competitors
            )
        )
        competitor_trust[competitor] = (hits / n) if n else 0.0

    return ModelScore(
        model_key=model.key,
        n=n,
        trust=trust,
        rsi=value,
        variance=variance,
        stability=band,  # type: ignore[arg-type]
        stability_note=note,
        counts=counts,
        competitor_trust=competitor_trust,
        head_to_head=[
            head_to_head(valid, competitor, model.key) for competitor in spec.competitors
        ],
        errors=errors,
    )


# ---------------------------------------------------------------------------
# Head-to-head (§7.4)
# ---------------------------------------------------------------------------


def head_to_head(
    cells: list[Cell], competitor: str, model_key: str | None = None
) -> HeadToHead:
    """Normalised win rate against one named competitor.

    Restricted, per §7.4, to phrasings where at least one of the two was
    actually recommended — otherwise off-axis answers would dilute the rate
    toward 50% and hide the real result.
    """
    wins = losses = ties = eligible = 0

    for cell in cells:
        if cell.error is not None:
            continue
        target_rec = cell.target.verdict is Verdict.RECOMMENDED
        comp_rec = any(
            f.name == competitor and f.verdict is Verdict.RECOMMENDED
            for f in cell.competitors
        )
        if not (target_rec or comp_rec):
            continue
        eligible += 1
        if target_rec and comp_rec:
            ties += 1
        elif target_rec:
            wins += 1
        else:
            losses += 1

    win_rate = ((wins + 0.5 * ties) / eligible) if eligible else None
    decisive = (wins / (wins + losses)) if (wins + losses) else None

    return HeadToHead(
        competitor=competitor,
        model_key=model_key,
        eligible=eligible,
        wins=wins,
        losses=losses,
        ties=ties,
        win_rate=win_rate,
        decisive_win_rate=decisive,
    )


# ---------------------------------------------------------------------------
# Consensus (§7.3) and cross-model correlation (§3.5)
# ---------------------------------------------------------------------------


def consensus_composition(
    cells: list[Cell], models: list[ModelRef], n_queries: int
) -> ConsensusComposition:
    """One composition bar: unanimous win / split / unanimous loss.

    Partitions phrasings by how many models recommended the target. This is
    total and exhaustive over phrasings that produced at least one valid
    response, which is why it can be a single stacked bar rather than three
    unrelated numbers.
    """
    by_query: dict[int, list[Cell]] = {}
    for cell in cells:
        if cell.error is None:
            by_query.setdefault(cell.query_index, []).append(cell)

    unanimous_win = split = unanimous_loss = 0
    for query_cells in by_query.values():
        recommending = sum(1 for c in query_cells if c.target_recommended)
        if recommending == len(query_cells):
            unanimous_win += 1
        elif recommending == 0:
            unanimous_loss += 1
        else:
            split += 1

    n = len(by_query)
    denom = n or 1
    return ConsensusComposition(
        n=n,
        unanimous_win=unanimous_win,
        split=split,
        unanimous_loss=unanimous_loss,
        unanimous_win_frac=unanimous_win / denom,
        split_frac=split / denom,
        unanimous_loss_frac=unanimous_loss / denom,
    )


def cross_model_consensus(
    cells: list[Cell], models: list[ModelRef]
) -> CrossModelConsensus:
    """Mean pairwise Pearson correlation of per-query indicator vectors.

    This is the "stronger formulation" of §3.5: an entity trusted consistently
    across model families scores higher than a one-model favourite with the same
    average. Pairs where either model's vector is constant are excluded, since
    correlation is undefined there — ``pairs_used``/``pairs_total`` reports that
    rather than imputing zero and inventing disagreement.
    """
    vectors: dict[str, dict[int, float]] = {m.key: {} for m in models}
    for cell in cells:
        if cell.error is None and cell.model_key in vectors:
            vectors[cell.model_key][cell.query_index] = (
                1.0 if cell.target_recommended else 0.0
            )

    pairwise: dict[str, float] = {}
    pairs_total = 0
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a, b = models[i], models[j]
            pairs_total += 1
            shared = sorted(set(vectors[a.key]) & set(vectors[b.key]))
            if len(shared) < 2:
                continue
            r = pearson(
                [vectors[a.key][q] for q in shared],
                [vectors[b.key][q] for q in shared],
            )
            if r is not None:
                pairwise[f"{a.key}|{b.key}"] = r

    coefficient = (sum(pairwise.values()) / len(pairwise)) if pairwise else None
    have_data = any(vectors[m.key] for m in models)

    if pairs_total == 0:
        note = "Only one model column — cross-model correlation is not defined."
    elif not have_data:
        # Distinct from zero variance: there were no valid responses at all.
        note = (
            "No valid responses were recorded, so there is nothing to correlate. "
            "Check the provider errors before reading any figure on this page."
        )
    elif not pairwise:
        note = (
            "Every model gave an identical verdict on every phrasing, so "
            "correlation is undefined (zero variance). Read the composition bar "
            "instead: perfect agreement shows there directly."
        )
    elif len(pairwise) < pairs_total:
        note = (
            f"{len(pairwise)} of {pairs_total} model pairs had enough variance "
            "to correlate; the rest were constant."
        )
    else:
        note = "All model pairs correlated."

    return CrossModelConsensus(
        coefficient=coefficient,
        pairs_used=len(pairwise),
        pairs_total=pairs_total,
        pairwise=pairwise,
        note=note,
    )


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


def extraction_agreement(cells: list[Cell]) -> float | None:
    """Share of entity judgements where both extraction passes agreed."""
    total = agreed = 0
    for cell in cells:
        if cell.error is not None:
            continue
        for finding in [cell.target, *cell.competitors]:
            if finding.agreed is None:
                continue
            total += 1
            agreed += 1 if finding.agreed else 0
    return (agreed / total) if total else None


def build_headline(spec: RunSpec, aitc: float, per_model: list[ModelScore]) -> str:
    """The one sentence §6.3 asks for."""
    n_models = len([m for m in per_model if m.n > 0])
    if n_models == 0:
        return (
            "No measurement completed — every provider call failed. "
            "The figures below are not a result."
        )

    pct = round(aitc * 100)
    sentence = (
        f"{spec.entity} is recommended in {pct}% of responses across "
        f"{n_models} model{'s' if n_models != 1 else ''}"
    )

    aggregated = [
        head_to_head_total(per_model, competitor) for competitor in spec.competitors
    ]
    scored = [h for h in aggregated if h.win_rate is not None]
    if scored:
        best = max(scored, key=lambda h: h.win_rate or 0.0)
        sentence += (
            f", beating {best.competitor} "
            f"{round((best.win_rate or 0) * 100)}% of the time"
        )
    return sentence + "."


def head_to_head_total(per_model: list[ModelScore], competitor: str) -> HeadToHead:
    """Sum one competitor's head-to-head across every model column."""
    wins = losses = ties = eligible = 0
    for model in per_model:
        for record in model.head_to_head:
            if record.competitor == competitor:
                wins += record.wins
                losses += record.losses
                ties += record.ties
                eligible += record.eligible

    return HeadToHead(
        competitor=competitor,
        model_key=None,
        eligible=eligible,
        wins=wins,
        losses=losses,
        ties=ties,
        win_rate=((wins + 0.5 * ties) / eligible) if eligible else None,
        decisive_win_rate=(wins / (wins + losses)) if (wins + losses) else None,
    )


def score_run(
    spec: RunSpec,
    models: list[ModelRef],
    cells: list[Cell],
    n_queries: int,
) -> Scores:
    by_model = {m.key: [c for c in cells if c.model_key == m.key] for m in models}
    per_model = [score_model(m, by_model[m.key], spec) for m in models]

    scored = [m for m in per_model if m.n > 0]

    # DECISION: the paper writes AITC as the weighted *sum* Σ w_m·Trust(e,c,m).
    # Taken literally that is unbounded above and not comparable to Trust, which
    # is a probability. We normalise by Σ w_m, making AITC a weighted *mean* on
    # [0,1] — the same units as Trust, so the headline percentage is meaningful.
    # `aitc_unweighted` is the plain mean, for anyone who wants the weights out.
    weights = {m.key: m.weight for m in models}
    total_weight = sum(weights[m.model_key] for m in scored)
    aitc = (
        sum(weights[m.model_key] * m.trust for m in scored) / total_weight
        if total_weight > 0
        else 0.0
    )
    aitc_unweighted = (
        sum(m.trust for m in scored) / len(scored) if scored else 0.0
    )

    return Scores(
        per_model=per_model,
        aitc=aitc,
        aitc_unweighted=aitc_unweighted,
        consensus=consensus_composition(cells, models, n_queries),
        cross_model=cross_model_consensus(cells, models),
        head_to_head=[
            head_to_head_total(per_model, competitor) for competitor in spec.competitors
        ],
        extraction_agreement=extraction_agreement(cells),
        headline=build_headline(spec, aitc, per_model),
    )
