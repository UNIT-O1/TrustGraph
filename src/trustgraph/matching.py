"""Deterministic entity-name matching over free-form model prose.

This is the high-precision half of §6.2's "regex + fuzzy-match pass". It answers
only one question — *does this name physically occur in this text, and where* —
and deliberately leaves "was it recommended" to :mod:`trustgraph.extract`.

Precision matters more than recall here, because a false positive silently
manufactures a Win or a Loss and there is no downstream check that would catch
it. The three stages are ordered by how much they can go wrong:

1. **Boundary-anchored literal match** on the raw text (aliases included).
2. **Squashed match** — punctuation and whitespace removed from both sides, so
   "Razor Pay" and "razor-pay" find "Razorpay". This is still an *exact* match
   in a normalised space, not a similarity score, so it cannot drift.
3. **Bounded fuzzy match**, only for names with no hit yet, and structurally
   fenced against the failure mode that makes naive fuzzy matching unusable on
   brand names: a prefix relationship is rejected outright, so "Stripe" never
   matches "stripes" or "striped" no matter how high the ratio is. Genuine
   typos are mid-token edits, which survive the fence.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from .domain import Match

#: Trailing corporate/functional words that brands are cited with or without.
#: Stripping these yields an extra alias ("Cashfree Payments" -> "Cashfree").
_SUFFIXES = {
    "inc",
    "inc.",
    "llc",
    "ltd",
    "ltd.",
    "limited",
    "plc",
    "corp",
    "corp.",
    "corporation",
    "co",
    "co.",
    "company",
    "gmbh",
    "bv",
    "nv",
    "sa",
    "ag",
    "pvt",
    "pte",
    "private",
    "technologies",
    "technology",
    "tech",
    "software",
    "solutions",
    "systems",
    "labs",
    "group",
    "holdings",
    "payments",
    "payment",
    "pay",
    "platform",
    "io",
    "ai",
    "app",
    "com",
}

#: Aliases shorter than this are rejected — a 2-3 character token matches far
#: too much English prose to be worth the recall.
_MIN_ALIAS_LEN = 4

#: Fuzzy stage thresholds. Tight on purpose.
_FUZZY_MIN_RATIO = 92.0
_FUZZY_MAX_LEN_DELTA = 2
_FUZZY_MIN_LEN = 6


def normalize(text: str) -> str:
    """Casefold, strip diacritics, collapse whitespace."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(stripped.casefold().split())


def squash(text: str) -> str:
    """Normalise, then keep only alphanumerics."""
    return re.sub(r"[^a-z0-9]+", "", normalize(text))


def _squash_with_offsets(text: str) -> tuple[str, list[int]]:
    """Squashed text plus, for each squashed char, its index in ``text``."""
    out: list[str] = []
    offsets: list[int] = []
    decomposed_cache = {}
    for i, ch in enumerate(text):
        if ch not in decomposed_cache:
            d = unicodedata.normalize("NFKD", ch)
            decomposed_cache[ch] = "".join(
                c for c in d if not unicodedata.combining(c)
            ).casefold()
        for c in decomposed_cache[ch]:
            if c.isalnum() and c.isascii():
                out.append(c)
                offsets.append(i)
    return "".join(out), offsets


def _is_boundary(text: str, start: int, end: int) -> bool:
    """True if ``text[start:end]`` is not glued to surrounding alphanumerics.

    Used instead of ``\\b`` so names containing punctuation ("Pay-U", "Razorpay.")
    still anchor correctly.
    """
    before = text[start - 1] if start > 0 else ""
    after = text[end] if end < len(text) else ""
    return not (before.isalnum() or after.isalnum())


def build_aliases(name: str) -> list[str]:
    """Alias set for one tracked name, longest first."""
    aliases: set[str] = set()

    def add(candidate: str) -> None:
        candidate = " ".join(candidate.split())
        if len(squash(candidate)) >= _MIN_ALIAS_LEN:
            aliases.add(candidate)

    add(name)

    tokens = name.split()
    # Progressively drop trailing corporate/functional words.
    while len(tokens) > 1 and normalize(tokens[-1]).strip(".") in _SUFFIXES:
        tokens = tokens[:-1]
        add(" ".join(tokens))

    # A single-token name glued from two words ("PayU") is already covered by
    # the squashed stage; no need to invent split variants.
    return sorted(aliases, key=lambda a: (-len(squash(a)), a))


@dataclass
class _NameSpec:
    name: str
    aliases: list[str]
    squashed: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.squashed = {a: squash(a) for a in self.aliases}


class EntityMatcher:
    """Finds occurrences of a fixed set of tracked names in model responses."""

    def __init__(self, names: list[str]) -> None:
        self._specs = [_NameSpec(name=n, aliases=build_aliases(n)) for n in names]

    @property
    def names(self) -> list[str]:
        return [s.name for s in self._specs]

    def find(self, text: str) -> dict[str, list[Match]]:
        """Return non-overlapping matches per tracked name.

        Longer aliases win overlapping spans, so tracking both "Razorpay" and a
        hypothetical "Razorpay X" cannot double-count one mention.
        """
        if not text:
            return {s.name: [] for s in self._specs}

        candidates: list[Match] = []
        candidates.extend(self._literal_candidates(text))
        candidates.extend(self._squashed_candidates(text))

        accepted = self._resolve_overlaps(candidates)

        hit_names = {m.name for m in accepted}
        for spec in self._specs:
            if spec.name not in hit_names:
                accepted.extend(self._fuzzy_candidates(text, spec, accepted))

        accepted = self._resolve_overlaps(accepted)

        result: dict[str, list[Match]] = {s.name: [] for s in self._specs}
        for m in sorted(accepted, key=lambda m: m.start):
            result[m.name].append(m)
        return result

    # ------------------------------------------------------------------
    # Stage 1 — boundary-anchored literal
    # ------------------------------------------------------------------

    def _literal_candidates(self, text: str) -> list[Match]:
        out: list[Match] = []
        for spec in self._specs:
            for alias in spec.aliases:
                pattern = re.compile(re.escape(alias), re.IGNORECASE)
                for m in pattern.finditer(text):
                    if _is_boundary(text, m.start(), m.end()):
                        out.append(
                            Match(
                                name=spec.name,
                                start=m.start(),
                                end=m.end(),
                                text=m.group(0),
                                via="exact" if alias == spec.name else "alias",
                            )
                        )
        return out

    # ------------------------------------------------------------------
    # Stage 2 — squashed exact
    # ------------------------------------------------------------------

    def _squashed_candidates(self, text: str) -> list[Match]:
        squashed, offsets = _squash_with_offsets(text)
        if not squashed:
            return []

        out: list[Match] = []
        for spec in self._specs:
            for alias, needle in spec.squashed.items():
                if not needle:
                    continue
                pos = squashed.find(needle)
                while pos != -1:
                    start = offsets[pos]
                    end = offsets[pos + len(needle) - 1] + 1
                    if _is_boundary(text, start, end):
                        out.append(
                            Match(
                                name=spec.name,
                                start=start,
                                end=end,
                                text=text[start:end],
                                via="exact" if alias == spec.name else "alias",
                            )
                        )
                    pos = squashed.find(needle, pos + 1)
        return out

    # ------------------------------------------------------------------
    # Stage 3 — bounded fuzzy
    # ------------------------------------------------------------------

    def _fuzzy_candidates(
        self, text: str, spec: _NameSpec, taken: list[Match]
    ) -> list[Match]:
        needle = squash(spec.name)
        if len(needle) < _FUZZY_MIN_LEN:
            return []

        word_count = max(1, len(spec.name.split()))
        tokens = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        occupied = [(m.start, m.end) for m in taken]

        out: list[Match] = []
        for width in range(1, word_count + 2):
            for i in range(len(tokens) - width + 1):
                start = tokens[i][1]
                end = tokens[i + width - 1][2]
                if any(start < o_end and end > o_start for o_start, o_end in occupied):
                    continue
                candidate = squash(text[start:end])
                if not candidate or abs(len(candidate) - len(needle)) > _FUZZY_MAX_LEN_DELTA:
                    continue
                # Structural fence: a prefix relationship means a morphological
                # variant ("stripe"/"stripes"), not a typo. Reject regardless of
                # how similar the strings score.
                if candidate.startswith(needle) or needle.startswith(candidate):
                    continue
                if fuzz.ratio(candidate, needle) >= _FUZZY_MIN_RATIO and _is_boundary(
                    text, start, end
                ):
                    out.append(
                        Match(
                            name=spec.name,
                            start=start,
                            end=end,
                            text=text[start:end],
                            via="fuzzy",
                        )
                    )
        return out

    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_overlaps(matches: list[Match]) -> list[Match]:
        """Greedy longest-match-wins over overlapping spans."""
        ordered = sorted(matches, key=lambda m: (m.start, -(m.end - m.start)))
        accepted: list[Match] = []
        for m in ordered:
            clash = next(
                (a for a in accepted if m.start < a.end and m.end > a.start), None
            )
            if clash is None:
                accepted.append(m)
            elif (m.end - m.start) > (clash.end - clash.start):
                accepted.remove(clash)
                accepted.append(m)
        return accepted


def rank_by_first_mention(matches: dict[str, list[Match]]) -> dict[str, int | None]:
    """1-based ordinal of each name's first appearance; ``None`` if absent."""
    firsts = [
        (name, spans[0].start) for name, spans in matches.items() if spans
    ]
    firsts.sort(key=lambda pair: pair[1])
    ranks: dict[str, int | None] = {name: None for name in matches}
    for i, (name, _) in enumerate(firsts, start=1):
        ranks[name] = i
    return ranks
