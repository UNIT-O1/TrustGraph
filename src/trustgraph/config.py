"""Configuration and model-roster construction.

The roster is the set of grid columns. It is built from whichever provider keys
are actually present, because the paper explicitly allows a two-provider demo
fallback — and because a tool that hard-fails on a missing key is useless at
demo time.

Provider model IDs drift. Rather than pretend a hardcoded list stays correct,
every default here is overridable via ``TRUSTGRAPH_ROSTER`` and discoverable at
runtime through ``GET /api/models``.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .domain import ModelRef

# ---------------------------------------------------------------------------
# Provider catalogue
# ---------------------------------------------------------------------------

#: Preference-ordered candidate models per provider. The first entry is the
#: provider's primary grid column; the rest are used only to widen a
#: single-provider roster so the grid still has more than one column.
PROVIDER_MODELS: dict[str, list[tuple[str, str]]] = {
    "gemini": [
        ("gemini-2.5-pro", "Gemini 2.5 Pro"),
        ("gemini-2.5-flash", "Gemini 2.5 Flash"),
        ("gemini-2.0-flash", "Gemini 2.0 Flash"),
    ],
    "anthropic": [
        ("claude-opus-5", "Claude Opus 5"),
        ("claude-sonnet-5", "Claude Sonnet 5"),
        ("claude-haiku-4-5", "Claude Haiku 4.5"),
    ],
    "openai": [
        ("gpt-4o", "GPT-4o"),
        ("gpt-4o-mini", "GPT-4o mini"),
    ],
}

#: Cheapest/fastest model per provider, used for the two utility calls
#: (paraphrase generation and the structured extraction re-check).
UTILITY_MODELS: dict[str, str] = {
    "gemini": "gemini-2.5-flash",
    "anthropic": "claude-haiku-4-5",
    "openai": "gpt-4o-mini",
}

#: Rough market-share weights for w_m in the AITC sum (§3.5). These are a
#: stated prior, not a measurement — the paper says "e.g. market share".
FAMILY_WEIGHTS: dict[str, float] = {
    "openai": 1.0,
    "gemini": 0.9,
    "anthropic": 0.7,
    "simulated": 1.0,
}

PROVIDER_LABELS: dict[str, str] = {
    "gemini": "Google",
    "anthropic": "Anthropic",
    "openai": "OpenAI",
    "simulated": "Simulated",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="",
    )

    gemini_api_key: str | None = None
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None

    trustgraph_roster: str | None = None
    trustgraph_utility_model: str | None = None

    paraphrase_count: int = Field(default=18, ge=4, le=40, alias="TRUSTGRAPH_PARAPHRASE_COUNT")
    max_concurrency: int = Field(default=12, ge=1, le=64, alias="TRUSTGRAPH_MAX_CONCURRENCY")
    llm_recheck: bool = Field(default=True, alias="TRUSTGRAPH_LLM_RECHECK")
    cache_dir: str = Field(default=".cache", alias="TRUSTGRAPH_CACHE_DIR")
    cache_enabled: bool = Field(default=True, alias="TRUSTGRAPH_CACHE_ENABLED")
    request_timeout: float = Field(default=90.0, alias="TRUSTGRAPH_REQUEST_TIMEOUT")

    # ------------------------------------------------------------------
    # Key discovery
    # ------------------------------------------------------------------

    def api_key(self, provider: str) -> str | None:
        return {
            "gemini": self.gemini_api_key,
            "anthropic": self.anthropic_api_key,
            "openai": self.openai_api_key,
        }.get(provider)

    def available_providers(self) -> list[str]:
        """Providers with a non-empty key, in AITC-weight order."""
        found = [p for p in PROVIDER_MODELS if (self.api_key(p) or "").strip()]
        return sorted(found, key=lambda p: -FAMILY_WEIGHTS.get(p, 0.5))

    # ------------------------------------------------------------------
    # Roster
    # ------------------------------------------------------------------

    def build_roster(self, override: list[str] | None = None) -> tuple[list[ModelRef], list[str]]:
        """Return ``(roster, warnings)``.

        Degradation ladder, in order:

        1. An explicit override (request body or ``TRUSTGRAPH_ROSTER``).
        2. One primary column per provider that has a key — the paper's intent.
        3. Only one provider has a key: widen to up to three of *its* models so
           the grid keeps a model axis, and warn loudly that cross-model
           consensus is now within-family and cannot support the §3.5 claim.
        4. No keys at all: the deterministic simulated provider, clearly marked.
        """
        warnings: list[str] = []
        spec = override or (
            [s.strip() for s in self.trustgraph_roster.split(",") if s.strip()]
            if self.trustgraph_roster
            else None
        )

        if spec:
            roster = [_ref_from_spec(s) for s in spec]
            missing = {
                r.provider
                for r in roster
                if r.provider != "simulated" and not (self.api_key(r.provider) or "").strip()
            }
            if missing:
                warnings.append(
                    "Roster names provider(s) with no API key configured: "
                    + ", ".join(sorted(missing))
                    + ". Those columns will report errors."
                )
            return roster, warnings

        providers = self.available_providers()

        if not providers:
            warnings.append(
                "No provider API keys found — running the deterministic simulated "
                "provider. Numbers are reproducible fixtures, not measurements of "
                "live model behaviour."
            )
            return _simulated_roster(), warnings

        if len(providers) == 1:
            provider = providers[0]
            candidates = PROVIDER_MODELS[provider][:3]
            warnings.append(
                f"Only the {PROVIDER_LABELS[provider]} key is configured, so all "
                f"{len(candidates)} columns are {provider}-family models. Paraphrase "
                "stability (RSI) remains valid per model, but the cross-model "
                "consensus figures measure within-family agreement and do not "
                "support the cross-family AITC claim in §3.5."
            )
            return [_ref(provider, m, label) for m, label in candidates], warnings

        if len(providers) == 2:
            warnings.append(
                "Two provider families configured — the paper's stated demo "
                "fallback. Consensus is a 2-way comparison."
            )

        return [
            _ref(p, *PROVIDER_MODELS[p][0]) for p in providers
        ], warnings

    def utility_model(self, roster: list[ModelRef]) -> tuple[str, str] | None:
        """``(provider, model)`` for paraphrase generation and the re-check."""
        if self.trustgraph_utility_model:
            provider, _, model = self.trustgraph_utility_model.partition(":")
            return (provider.strip(), model.strip()) if model else None

        for provider in self.available_providers():
            return provider, UTILITY_MODELS[provider]

        # No keys: reuse whatever the roster is (simulated).
        return (roster[0].provider, roster[0].model) if roster else None


def _ref(provider: str, model: str, label: str | None = None) -> ModelRef:
    return ModelRef(
        key=f"{provider}:{model}",
        provider=provider,
        model=model,
        family=provider,
        label=label or model,
        weight=FAMILY_WEIGHTS.get(provider, 0.5),
    )


def _ref_from_spec(spec: str) -> ModelRef:
    provider, _, model = spec.partition(":")
    provider = provider.strip() or "simulated"
    model = model.strip() or spec.strip()
    label = next(
        (lbl for m, lbl in PROVIDER_MODELS.get(provider, []) if m == model),
        model,
    )
    return _ref(provider, model, label)


def _simulated_roster() -> list[ModelRef]:
    return [
        ModelRef(
            key=f"simulated:{name}",
            provider="simulated",
            model=name,
            family="simulated",
            label=label,
            weight=1.0,
        )
        for name, label in (
            ("sim-alpha", "Sim Alpha"),
            ("sim-beta", "Sim Beta"),
            ("sim-gamma", "Sim Gamma"),
        )
    ]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
