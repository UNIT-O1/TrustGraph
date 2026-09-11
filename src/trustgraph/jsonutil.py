"""Tolerant JSON extraction from model output.

Even with structured-output modes enabled, providers occasionally wrap JSON in a
fenced code block or prepend a sentence. Because the extraction re-check is on
the critical path for every cell, a parse failure would silently degrade the
grid — so parsing is deliberately forgiving here rather than strict.
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")


def parse_json_object(text: str) -> dict[str, Any]:
    """Best-effort parse of a single JSON object out of ``text``.

    Raises ``ValueError`` if nothing object-shaped can be recovered.
    """
    if not text or not text.strip():
        raise ValueError("empty response")

    for candidate in _candidates(text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                parsed = json.loads(_TRAILING_COMMA.sub(r"\1", candidate))
            except json.JSONDecodeError:
                continue
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            # A bare array is a common near-miss; wrap it so callers see a dict.
            return {"items": parsed}

    raise ValueError(f"no JSON object found in response: {text[:200]!r}")


def _candidates(text: str) -> list[str]:
    out: list[str] = []
    stripped = text.strip()
    out.append(stripped)

    for m in _FENCE.finditer(text):
        out.append(m.group(1).strip())

    # Widest brace/bracket span — survives leading and trailing prose.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            out.append(text[start : end + 1])

    seen: set[str] = set()
    unique: list[str] = []
    for c in out:
        if c and c not in seen:
            seen.add(c)
            unique.append(c)
    return unique
