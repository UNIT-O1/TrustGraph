"""Mention parsing and recommendation classification.

§6.2 calls this "the one place worth spending real build time", and it is the
direct implementation of the paper's extraction problem: turning free-form model
prose into the four-state verdict that Table 2 needs.

Two independent passes, then a merge:

**Pass 1 — deterministic.** :mod:`trustgraph.matching` locates the names; this
module classifies each occurrence on the recommendation axis by scoring
recommendation cues over the enclosing sentence or list item.

The non-obvious part is **cue ownership**. A sentence-level bag of cues is not
good enough, because "Razorpay is popular in India, though I would go with
Stripe" contains one recommendation cue and two vendors — crediting the cue to
both turns a Loss into a Both. So each cue is attributed to a single entity by
grammatical direction: verb cues that govern an object ("go with X", "unlike X")
attach to the name that *follows* them, predicate cues ("X is the stronger
choice") attach to the name that *precedes* them. Negations are resolved before
the positive form they contain, so "I wouldn't recommend X" never scores as an
endorsement.

**Pass 2 — structured LLM re-check.** One JSON-schema call per response asks a
model the same question directly.

**Merge.** The LLM is authoritative on *recommended vs merely mentioned* (the
judgement call), the matcher is authoritative on *presence* (the factual claim),
and any LLM presence claim must be backed by a verbatim span that actually
occurs in the response — otherwise a hallucinated mention would fabricate a Win
or a Loss out of nothing. Per-entity agreement is recorded and aggregated, so
the dashboard reports how much of the grid the two passes actually agreed on
rather than asserting the merged answer is simply correct.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from rapidfuzz import fuzz

from .domain import CellState, EntityFinding, Match, Verdict, derive_cell_state
from .matching import EntityMatcher, rank_by_first_mention

# ---------------------------------------------------------------------------
# Cue lexicons
# ---------------------------------------------------------------------------

_NEGATED_RECOMMEND = re.compile(
    r"\b(?:would\s?n[o']?t|wouldn't|do\s?n[o']?t|don't|does\s?n[o']?t|never|not|"
    r"avoid|hesitate to|steer clear of)\s+(?:\w+\s+){0,2}"
    r"(?:recommend|suggest|use|pick|choose|go with)\b",
    re.IGNORECASE,
)

_STRONG_NEGATIVE = re.compile(
    r"\b(?:avoid|steer clear|not ideal|not a (?:good|great) fit|not the right|"
    r"overkill|too expensive|prohibitively|lacks|missing|no longer|deprecated|"
    r"downside|drawback|weaker|worse|poor(?:ly)?|struggles?|falls short|"
    r"less suitable|skip|rule(?:d)? out)\b",
    re.IGNORECASE,
)

_STRONG_POSITIVE = re.compile(
    r"\b(?:i(?:'d| would)? (?:recommend|suggest|pick|choose|go with|use)|"
    r"we (?:recommend|suggest)|recommend|suggest|go with|stick with|opt for|"
    r"your best bet|i'd pick|would choose|start with|point you to)\b",
    re.IGNORECASE,
)

_WEAK_POSITIVE = re.compile(
    r"\b(?:best|top|strong(?:er|est)|better|ideal|excellent|"
    r"great (?:choice|option|fit)|solid|robust|worth (?:a look|considering)|"
    r"leading|most popular|industry standard|de facto|winner|good fit|"
    r"well[- ]suited|the default|safe(?:st)? (?:choice|bet)|reliable|mature|"
    r"(?:stronger|better) choice)\b",
    re.IGNORECASE,
)

_CONTRAST = re.compile(
    r"\b(?:unlike|instead of|rather than|as opposed to|whereas|in contrast to)\b",
    re.IGNORECASE,
)

_LIST_MARKER = re.compile(r"^\s*(?:[-*•–]|\d+[.)])\s+")

#: Inline enumerators, for lists written on a single line
#: ("Two options: 1. Razorpay ... 2. Stripe ...").
_INLINE_ENUM = re.compile(r"(?:(?<=\s)|(?<=^))(?:\d+[.)]|[•–])\s+")

#: Sentence boundary that refuses to break after a numeric enumerator, so
#: "1. Razorpay" stays one unit instead of splitting at "1.".
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])(?<![0-9]\.)\s+")

#: Cue score at or above which an occurrence counts as a recommendation.
_RECOMMEND_THRESHOLD = 2

#: How far a cue will reach to find its owning entity.
_ATTRIBUTION_WINDOW = 80

#: How far into a list item a name can appear and still be the item's subject.
_LIST_SUBJECT_WINDOW = 70


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Unit:
    start: int
    end: int
    text: str
    is_list_item: bool


def _segment(text: str) -> list[_Unit]:
    """Split into sentences and list items, preserving absolute offsets."""
    units: list[_Unit] = []

    for line_match in re.finditer(r"[^\n]+", text):
        line = line_match.group(0)
        base = line_match.start()

        if _LIST_MARKER.match(line):
            units.append(_Unit(base, base + len(line), line, True))
            continue

        # A single line carrying two or more enumerators is a list written
        # inline; split it at the enumerators rather than at sentence ends.
        enums = list(_INLINE_ENUM.finditer(line))
        if len(enums) >= 2:
            cuts = [0] + [m.start() for m in enums] + [len(line)]
            for i in range(len(cuts) - 1):
                chunk = line[cuts[i] : cuts[i + 1]]
                if chunk.strip():
                    units.append(
                        _Unit(
                            base + cuts[i],
                            base + cuts[i + 1],
                            chunk,
                            i > 0,  # the leading fragment is the preamble
                        )
                    )
            continue

        cursor = 0
        for part in _SENTENCE_SPLIT.split(line):
            if not part:
                continue
            idx = line.find(part, cursor)
            if idx == -1:
                idx = cursor
            units.append(_Unit(base + idx, base + idx + len(part), part, False))
            cursor = idx + len(part)

    return units or [_Unit(0, len(text), text, False)]


# ---------------------------------------------------------------------------
# Cue attribution
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Cue:
    start: int
    end: int
    weight: int
    direction: str  # "after" | "before" | "nearest"
    label: str


#: Ordered so that negations are found before the positive forms they contain.
_LEXICON: tuple[tuple[re.Pattern[str], int, str, str], ...] = (
    (_NEGATED_RECOMMEND, -4, "nearest", "negated-recommendation"),
    (_STRONG_NEGATIVE, -3, "nearest", "negative"),
    (_CONTRAST, -3, "after", "contrast"),
    (_STRONG_POSITIVE, 3, "after", "recommendation"),
    (_WEAK_POSITIVE, 2, "before", "endorsement"),
)


def _find_cues(unit: _Unit) -> list[_Cue]:
    cues: list[_Cue] = []
    suppressed: list[tuple[int, int]] = []

    for pattern, weight, direction, label in _LEXICON:
        for m in pattern.finditer(unit.text):
            span = (m.start(), m.end())
            # A positive cue sitting inside a negated construction is not a
            # separate signal — it is the thing being negated.
            if weight > 0 and any(
                span[0] < s_end and span[1] > s_start for s_start, s_end in suppressed
            ):
                continue
            cues.append(
                _Cue(
                    start=unit.start + m.start(),
                    end=unit.start + m.end(),
                    weight=weight,
                    direction=direction,
                    label=label,
                )
            )
            if label == "negated-recommendation":
                suppressed.append(span)

    return cues


def _attribute(
    cue: _Cue, occurrences: list[tuple[str, Match]]
) -> tuple[str, int] | None:
    """Pick the single entity occurrence a cue belongs to."""
    if not occurrences:
        return None
    if len(occurrences) == 1:
        name, match = occurrences[0]
        return name, match.start

    def key(item: tuple[str, Match]) -> tuple[str, int]:
        return item[0], item[1].start

    if cue.direction == "after":
        following = [
            o for o in occurrences if o[1].start >= cue.end and o[1].start - cue.end <= _ATTRIBUTION_WINDOW
        ]
        if following:
            return key(min(following, key=lambda o: o[1].start))
    elif cue.direction == "before":
        preceding = [
            o for o in occurrences if o[1].end <= cue.start and cue.start - o[1].end <= _ATTRIBUTION_WINDOW
        ]
        if preceding:
            return key(max(preceding, key=lambda o: o[1].end))

    # Fallback: whichever occurrence sits closest to the cue.
    def distance(item: tuple[str, Match]) -> int:
        _, match = item
        if match.end <= cue.start:
            return cue.start - match.end
        if match.start >= cue.end:
            return match.start - cue.end
        return 0

    return key(min(occurrences, key=distance))


# ---------------------------------------------------------------------------
# Deterministic classification
# ---------------------------------------------------------------------------


def classify_all(
    text: str, matches_by_name: dict[str, list[Match]]
) -> dict[str, tuple[Verdict, str | None, int]]:
    """Classify every tracked entity jointly.

    Joint rather than per-entity because cue ownership is inherently a
    competition between the names in a sentence.

    Returns ``{name: (verdict, evidence, score)}``.
    """
    units = _segment(text)
    occurrences = [(name, m) for name, ms in matches_by_name.items() for m in ms]

    best: dict[str, tuple[int, str | None]] = {}

    for unit in units:
        local = [
            (name, m) for name, m in occurrences if unit.start <= m.start < unit.end
        ]
        if not local:
            continue

        scores: dict[tuple[str, int], int] = {}
        for name, match in local:
            score = 0
            if unit.is_list_item:
                marker = _LIST_MARKER.match(unit.text) or _INLINE_ENUM.match(unit.text)
                content_start = marker.end() if marker else 0
                if (match.start - unit.start) - content_start <= _LIST_SUBJECT_WINDOW:
                    score += 2

            around = text[max(0, match.start - 3) : match.end + 4]
            if "**" in around or "__" in around:
                score += 1
            elif re.match(r"\s*(?:—|–|-{1,2}|:)\s", text[match.end : match.end + 4]):
                score += 1

            scores[(name, match.start)] = score

        for cue in _find_cues(unit):
            owner = _attribute(cue, local)
            if owner is not None and owner in scores:
                scores[owner] += cue.weight

        evidence = unit.text.strip()[:280]
        for (name, _), score in scores.items():
            current = best.get(name)
            if current is None or score > current[0]:
                best[name] = (score, evidence)

    out: dict[str, tuple[Verdict, str | None, int]] = {}
    for name in matches_by_name:
        if not matches_by_name[name]:
            out[name] = (Verdict.ABSENT, None, 0)
            continue
        score, evidence = best.get(name, (0, None))
        verdict = (
            Verdict.RECOMMENDED if score >= _RECOMMEND_THRESHOLD else Verdict.MENTIONED
        )
        out[name] = (verdict, evidence, score)

    return out


# ---------------------------------------------------------------------------
# LLM re-check
# ---------------------------------------------------------------------------

RECHECK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["entities"],
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "mentioned", "recommended", "rank", "evidence"],
                "properties": {
                    "name": {"type": "string"},
                    "mentioned": {"type": "boolean"},
                    "recommended": {"type": "boolean"},
                    "rank": {"type": ["integer", "null"]},
                    "evidence": {"type": ["string", "null"]},
                },
            },
        }
    },
}

RECHECK_SYSTEM = (
    "You are a strict extraction function, not an assistant. You classify how "
    "named vendors were treated in an AI assistant's answer. You never add "
    "opinions and you never invent text."
)

_RECHECK_TEMPLATE = """A user asked an AI assistant this question:

<question>
{query}
</question>

The assistant replied:

<response>
{response}
</response>

For each of these names, classify how the response treated it:

{names}

Definitions:
- "mentioned": the name, or an unmistakable variant of it, physically appears in
  the response.
- "recommended": the response puts it forward as something the user should use
  or seriously consider — including as one item in a list of suggested options.
  It is NOT recommended if it appears only as a counterexample, a caveat, a
  thing to avoid, a rejected alternative, or a passing comparison.
- "rank": 1-based position among the named vendors, in the order the response
  presents them. null if not mentioned.
- "evidence": the shortest span copied VERBATIM from the response that justifies
  your "recommended" decision. Copy it character-for-character from the response
  above. null if not mentioned.

Return one entry per name, in the order given."""


def build_recheck_prompt(query: str, response: str, names: list[str]) -> str:
    listed = "\n".join(f"- {n}" for n in names)
    return _RECHECK_TEMPLATE.format(query=query, response=response, names=listed)


def _evidence_supported(evidence: str | None, response: str) -> bool:
    """Guard against a fabricated quote.

    An LLM presence claim is honoured only if its evidence span genuinely occurs
    in the response. Exact containment after whitespace normalisation is the
    primary test; a high partial-ratio match is allowed so trivial quoting
    differences (smart quotes, dropped markdown) don't reject a real span.
    """
    if not evidence:
        return False
    needle = " ".join(evidence.split()).casefold()
    haystack = " ".join(response.split()).casefold()
    if len(needle) < 8:
        return False
    if needle in haystack:
        return True
    return fuzz.partial_ratio(needle, haystack) >= 92.0


def parse_recheck(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Normalise the re-check payload into ``{casefolded name: record}``."""
    out: dict[str, dict[str, Any]] = {}
    for raw in payload.get("entities") or []:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or "").strip()
        if not name:
            continue
        rank = raw.get("rank")
        out[name.casefold()] = {
            "name": name,
            "mentioned": bool(raw.get("mentioned")),
            "recommended": bool(raw.get("recommended")),
            "rank": int(rank) if isinstance(rank, (int, float)) else None,
            "evidence": (str(raw["evidence"]) if raw.get("evidence") else None),
        }
    return out


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


def build_findings(
    response: str,
    matcher: EntityMatcher,
    recheck: dict[str, dict[str, Any]] | None = None,
) -> list[EntityFinding]:
    """Merge both passes into one finding per tracked name."""
    matches = matcher.find(response)
    ranks = rank_by_first_mention(matches)
    deterministic = classify_all(response, matches)

    findings: list[EntityFinding] = []
    for name in matcher.names:
        spans = matches[name]
        det_verdict, det_evidence, _ = deterministic[name]

        llm = (recheck or {}).get(name.casefold())
        llm_verdict: Verdict | None = None
        llm_evidence_ok = False
        if llm is not None:
            llm_evidence_ok = _evidence_supported(llm.get("evidence"), response)
            if llm["mentioned"]:
                llm_verdict = (
                    Verdict.RECOMMENDED if llm["recommended"] else Verdict.MENTIONED
                )
            else:
                llm_verdict = Verdict.ABSENT

        # Presence: the matcher is authoritative. An LLM-only claim survives
        # only when its verbatim span is genuinely in the response.
        present = bool(spans) or (
            llm is not None and llm["mentioned"] and llm_evidence_ok
        )

        if not present:
            verdict = Verdict.ABSENT
            evidence = None
        elif llm_verdict in (Verdict.RECOMMENDED, Verdict.MENTIONED):
            # The recommendation axis is a judgement call — defer to the model.
            verdict = llm_verdict
            evidence = llm.get("evidence") if llm_evidence_ok else det_evidence
        else:
            verdict = det_verdict if spans else Verdict.MENTIONED
            evidence = det_evidence

        agreed: bool | None = None
        if llm_verdict is not None and (spans or llm["mentioned"]):
            agreed = llm_verdict == det_verdict

        findings.append(
            EntityFinding(
                name=name,
                verdict=verdict,
                rank=ranks.get(name) or (llm.get("rank") if llm and present else None),
                matches=spans,
                evidence=evidence,
                deterministic_verdict=det_verdict,
                llm_verdict=llm_verdict,
                agreed=agreed,
            )
        )

    return findings


def cell_state_from_findings(
    findings: list[EntityFinding], target_name: str
) -> tuple[CellState, EntityFinding, list[EntityFinding]]:
    """Split findings into target/competitors and derive the Table 2 state."""
    target = next(f for f in findings if f.name == target_name)
    competitors = [f for f in findings if f.name != target_name]
    state = derive_cell_state(target.verdict, {f.name: f.verdict for f in competitors})
    return state, target, competitors
