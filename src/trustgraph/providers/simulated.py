"""Deterministic simulated provider.

Purpose: make the pipeline runnable and testable end-to-end with no API keys —
for CI, for offline development, and so a demo never dies on an expired key.

It is a *fixture generator*, not a model. Every output is a pure function of
``sha256(model, prompt)``, so a given run reproduces byte-for-byte. Any run that
uses it is flagged ``simulated: true`` in the payload and carries a banner in
the UI, because presenting fixture numbers as measurements would invert the
entire point of the paper.

The response distribution is shaped, not uniform. Each sim model has a different
prior on the target, and price-framed phrasings ("cheapest", "lowest fees")
shift weight toward the first competitor. That reproduces the exact diagnostic
pattern §7.1 describes — a red band across every price-framed row while
category-framed rows stay green — so the grid demonstrates the finding it was
designed to surface.
"""

from __future__ import annotations

import hashlib
import random
import re

from ..jsonutil import parse_json_object  # noqa: F401  (kept for symmetry/tests)
from .base import BaseProvider, CompletionRequest, CompletionResult

#: Per-sim-model prior on recommending the target.
_MODEL_BIAS: dict[str, float] = {
    "sim-alpha": 0.78,
    "sim-beta": 0.58,
    "sim-gamma": 0.36,
}

_PRICE_FRAMING = re.compile(
    r"\b(cheap(?:est)?|lowest|price|pricing|fee|fees|cost|budget|affordable|discount)\b",
    re.IGNORECASE,
)
_CATEGORY_FRAMING = re.compile(
    r"\b(startup|d2c|scale|scaling|enterprise|compliance|integration|developer|api|docs)\b",
    re.IGNORECASE,
)

_QUERY_TEMPLATES: list[tuple[str, str]] = [
    ("best {c}", "superlative"),
    ("what is the best {c}", "superlative"),
    ("top options for {c}", "listing"),
    ("compare the leading {c}", "comparison"),
    ("I need {c} — what should I use?", "advice"),
    ("recommend {c} for a small team", "advice"),
    ("which {c} has the lowest fees?", "price"),
    ("cheapest {c}", "price"),
    ("most affordable {c} without hidden costs", "price"),
    ("most reliable {c}", "reliability"),
    ("{c} with the best developer experience", "developer"),
    ("easiest {c} to integrate", "developer"),
    ("{c} for a fast-growing startup", "stage"),
    ("enterprise-grade {c}", "stage"),
    ("what do most companies use for {c}?", "social-proof"),
    ("industry standard {c}", "social-proof"),
    ("{c} with the best support", "support"),
    ("safest choice for {c}", "risk"),
    ("{c} that scales internationally", "expansion"),
    ("underrated {c} worth considering", "contrarian"),
]


class SimulatedProvider(BaseProvider):
    name = "simulated"
    supports_temperature = True

    def __init__(self, seed: int | None = None) -> None:
        self._seed = seed or 0

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        started = self._now()
        rng = self._rng(request)

        if request.json_schema is not None:
            props = set((request.json_schema.get("properties") or {}).keys())
            if "queries" in props:
                text = self._paraphrases(request, rng)
            elif "entities" in props:
                text = self._extraction(request)
            else:
                text = "{}"
        else:
            text = self._answer(request, rng)

        return self._result(text, request.model, started, {"simulated": True})

    # ------------------------------------------------------------------

    def _rng(self, request: CompletionRequest) -> random.Random:
        digest = hashlib.sha256(
            f"{self._seed}|{request.model}|{request.prompt}".encode()
        ).hexdigest()
        return random.Random(int(digest[:16], 16))

    # ------------------------------------------------------------------
    # Paraphrase generation
    # ------------------------------------------------------------------

    def _paraphrases(self, request: CompletionRequest, rng: random.Random) -> str:
        import json

        ctx = request.sim_context or {}
        category = ctx.get("category") or "the category"
        count = int(ctx.get("count") or 18)

        templates = _QUERY_TEMPLATES.copy()
        rng.shuffle(templates)
        # Keep price framings represented so the diagnostic band is visible.
        chosen = templates[:count]
        if not any(intent == "price" for _, intent in chosen):
            chosen[-1] = ("cheapest {c}", "price")

        return json.dumps(
            {
                "queries": [
                    {"text": tpl.format(c=category), "intent": intent}
                    for tpl, intent in chosen
                ]
            }
        )

    # ------------------------------------------------------------------
    # Answer generation
    # ------------------------------------------------------------------

    def _answer(self, request: CompletionRequest, rng: random.Random) -> str:
        ctx = request.sim_context or {}
        target = ctx.get("entity") or "Acme"
        competitors: list[str] = list(ctx.get("competitors") or [])
        category = ctx.get("category") or "this"
        query = request.prompt

        bias = _MODEL_BIAS.get(request.model, 0.5)
        if _PRICE_FRAMING.search(query) and competitors:
            bias -= 0.45
        if _CATEGORY_FRAMING.search(query):
            bias += 0.15
        bias = min(max(bias, 0.02), 0.97)

        roll = rng.random()
        rival = competitors[0] if competitors else None

        # A minority of answers go off-axis — the model answers with criteria
        # rather than a named vendor. This is the NEITHER state of Table 2.
        if rng.random() < 0.09:
            return (
                f"It depends on your priorities. For {category}, weigh settlement "
                "speed, effective per-transaction cost at your volume, the quality "
                "of the developer documentation, and whether the provider is "
                "already certified for the compliance regime you operate under. "
                "Shortlist two or three providers and run a pilot on real volume "
                "before committing."
            )

        if roll < bias * 0.55:
            # Outright win, with the rival named only as a contrast.
            body = f"For {category}, I'd recommend **{target}**. "
            body += (
                f"It's the strongest fit here — the integration is straightforward "
                f"and it covers the payment methods this segment actually uses. "
            )
            if rival and rng.random() < 0.5:
                body += (
                    f"Unlike {rival}, which is better suited to a different market, "
                    f"{target} is built around exactly this use case."
                )
            return body

        if roll < bias:
            # Win, no rival present at all.
            return (
                f"Go with **{target}**. For {category} it's the option I'd pick — "
                f"good documentation, predictable pricing, and the onboarding is "
                f"quick. {target} should cover everything you described."
            )

        if rival and roll < bias + (1 - bias) * 0.45:
            # Both recommended — a genuine co-recommendation list.
            first, second = (target, rival) if rng.random() < 0.5 else (rival, target)
            return (
                f"Two options are worth a look for {category}:\n\n"
                f"1. **{first}** — the more common default, and the faster of the "
                f"two to get live.\n"
                f"2. **{second}** — worth comparing if your priorities differ; the "
                f"pricing structure suits some volumes better.\n\n"
                f"I'd start with {first} and benchmark {second} against it."
            )

        if rival:
            # Loss.
            extra = competitors[1] if len(competitors) > 1 else None
            body = (
                f"I'd recommend **{rival}** for {category}. It's the better choice "
                f"on the dimension you're asking about, and the pricing is more "
                f"transparent at lower volumes."
            )
            if extra and rng.random() < 0.4:
                body += f" {extra} is a reasonable second choice."
            return body

        return (
            f"**{target}** is the option I'd point you to for {category}. It covers "
            f"the essentials without much setup overhead."
        )

    # ------------------------------------------------------------------
    # Extraction re-check
    # ------------------------------------------------------------------

    def _extraction(self, request: CompletionRequest) -> str:
        """A deliberately weaker second opinion.

        The rule is *sentence-scoped keyword presence*: if the sentence or list
        item containing the name holds any positive cue, call it recommended.
        That is a sane rule and a genuinely independent one, but it lacks the
        cue-ownership resolution the deterministic classifier does — so the two
        passes agree on clear-cut responses and diverge exactly on multi-vendor
        sentences, where ownership is the whole question.

        This is what keeps extraction agreement meaningful in simulated mode
        instead of pinned at a meaningless 100%.
        """
        import json

        from ..extract import _segment  # local import avoids a cycle at import time

        ctx = request.sim_context or {}
        names: list[str] = list(ctx.get("names") or [])
        text: str = ctx.get("response_text") or request.prompt
        lowered = text.lower()
        units = _segment(text)

        order: list[tuple[int, str]] = []
        for name in names:
            pos = lowered.find(name.lower())
            if pos != -1:
                order.append((pos, name))
        order.sort()
        ranks = {name: i for i, (_, name) in enumerate(order, start=1)}

        cue = re.compile(
            r"\b(recommend|suggest|go with|i'd pick|would pick|best|top|"
            r"worth (?:a look|comparing|considering)|start with|point you to|"
            r"good option|solid|option[s]? (?:are|is)|the more common default)\b",
            re.IGNORECASE,
        )

        entities = []
        for name in names:
            pos = lowered.find(name.lower())
            mentioned = pos != -1
            recommended = False
            evidence = None
            if mentioned:
                unit = next(
                    (u for u in units if u.start <= pos < u.end),
                    units[-1] if units else None,
                )
                body = unit.text if unit else text
                if cue.search(body):
                    recommended = True
                    evidence = body.strip()[:200]
            entities.append(
                {
                    "name": name,
                    "mentioned": mentioned,
                    "recommended": recommended,
                    "rank": ranks.get(name),
                    "evidence": evidence,
                }
            )

        return json.dumps({"entities": entities})

    async def list_models(self) -> list[str]:
        return sorted(_MODEL_BIAS)
