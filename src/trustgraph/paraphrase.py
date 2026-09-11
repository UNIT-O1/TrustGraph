"""Paraphrase generation (§6.2).

One LLM call turns the category string into 15-20 differently-phrased natural
queries, spanning formality and constraint framing. The framing spread is the
point, not decoration: §7.1's headline finding is a visible band of losses
across every price-framed phrasing, and that band can only appear if the query
set actually contains price framings. So the prompt asks for named intents and
the fallback guarantees coverage of the important ones.

If generation fails or returns garbage, a deterministic template set takes over.
A run must never die at step one because a utility call hiccuped.
"""

from __future__ import annotations

from typing import Any

from .domain import Paraphrase

PARAPHRASE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["queries"],
    "properties": {
        "queries": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "intent"],
                "properties": {
                    "text": {"type": "string"},
                    "intent": {"type": "string"},
                },
            },
        }
    },
}

PARAPHRASE_SYSTEM = (
    "You generate realistic search-style questions that a real person would "
    "type to an AI assistant. You output JSON only."
)

_PROMPT = """A user is researching this category or use case:

<category>{category}</category>

Write {count} distinct questions a real person might ask an AI assistant when
they are trying to decide which product or vendor to pick in this category.

Requirements:
- Do NOT name any specific brand, product, or company. The questions must be
  brand-neutral, because they are used to test which brands an AI recommends
  unprompted.
- Vary the framing deliberately. Cover at least: a plain superlative ("best X
  for Y"), a listing request, a direct-advice request in first person, at least
  two cost- or price-framed questions, a reliability or trust framing, a
  developer- or integration-experience framing, and a company-stage framing.
- Vary register: some terse and keyword-like, some full polite sentences, some
  casual with a stated constraint.
- Each question must still be answerable by naming vendors in this category.

For each question, give a short lowercase `intent` tag naming its framing — for
example: superlative, listing, advice, price, reliability, developer, stage,
social-proof, risk, expansion.

Return JSON: {{"queries": [{{"text": "...", "intent": "..."}}]}}"""


def build_prompt(category: str, count: int) -> str:
    return _PROMPT.format(category=category, count=count)


def parse(payload: dict[str, Any], count: int) -> list[Paraphrase]:
    """Turn the model payload into paraphrases, de-duplicated and trimmed."""
    seen: set[str] = set()
    out: list[Paraphrase] = []

    for raw in payload.get("queries") or []:
        if isinstance(raw, str):
            text, intent = raw, None
        elif isinstance(raw, dict):
            text = str(raw.get("text") or "").strip()
            intent = (str(raw.get("intent")).strip().lower() or None) if raw.get("intent") else None
        else:
            continue

        text = " ".join(text.split())
        if not text or len(text) > 300:
            continue
        fingerprint = text.casefold()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        out.append(Paraphrase(index=len(out), text=text, intent=intent))
        if len(out) >= count:
            break

    return out


# ---------------------------------------------------------------------------
# Deterministic fallback
# ---------------------------------------------------------------------------

_FALLBACK: list[tuple[str, str]] = [
    ("best {c}", "superlative"),
    ("what is the best {c}?", "superlative"),
    ("top options for {c}", "listing"),
    ("compare the leading choices for {c}", "comparison"),
    ("I need {c} — what should I use?", "advice"),
    ("what would you recommend for {c}?", "advice"),
    ("which option for {c} has the lowest fees?", "price"),
    ("cheapest {c}", "price"),
    ("most affordable {c} with no hidden costs", "price"),
    ("most reliable {c}", "reliability"),
    ("{c} with the best developer experience", "developer"),
    ("easiest {c} to integrate", "developer"),
    ("{c} for an early-stage startup", "stage"),
    ("enterprise-grade {c}", "stage"),
    ("what do most companies use for {c}?", "social-proof"),
    ("industry standard for {c}", "social-proof"),
    ("{c} with the best customer support", "support"),
    ("safest choice for {c}", "risk"),
    ("{c} that scales internationally", "expansion"),
    ("underrated {c} worth considering", "contrarian"),
]


def fallback(category: str, count: int) -> list[Paraphrase]:
    chosen = _FALLBACK[:count]
    return [
        Paraphrase(index=i, text=template.format(c=category), intent=intent)
        for i, (template, intent) in enumerate(chosen)
    ]


def ensure_coverage(
    paraphrases: list[Paraphrase], category: str, count: int
) -> list[Paraphrase]:
    """Top up with fallbacks if generation came back short or too uniform.

    Also guarantees at least one price framing survives, since that is the
    framing most likely to expose a fragile recommendation and the one §7.1
    builds its example around.
    """
    out = list(paraphrases)
    have = {p.text.casefold() for p in out}

    if not any((p.intent or "") == "price" for p in out):
        for template, intent in _FALLBACK:
            if intent == "price":
                text = template.format(c=category)
                if text.casefold() not in have:
                    out.append(Paraphrase(index=0, text=text, intent=intent))
                    have.add(text.casefold())
                    break

    for template, intent in _FALLBACK:
        if len(out) >= count:
            break
        text = template.format(c=category)
        if text.casefold() not in have:
            out.append(Paraphrase(index=0, text=text, intent=intent))
            have.add(text.casefold())

    out = out[:count]
    return [
        Paraphrase(index=i, text=p.text, intent=p.intent) for i, p in enumerate(out)
    ]
