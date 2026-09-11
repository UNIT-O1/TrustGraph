"""Lazy provider construction and lifecycle."""

from __future__ import annotations

from ..config import Settings
from .base import Provider, ProviderError
from .simulated import SimulatedProvider


class ProviderRegistry:
    """Builds and caches one provider client per vendor for a run's lifetime.

    Clients are constructed lazily so a missing optional dependency or an
    unconfigured vendor only fails the columns that actually need it, rather
    than the whole run.
    """

    def __init__(self, settings: Settings, seed: int | None = None) -> None:
        self._settings = settings
        self._seed = seed
        self._cache: dict[str, Provider] = {}

    def get(self, provider: str) -> Provider:
        if provider in self._cache:
            return self._cache[provider]

        instance = self._build(provider)
        self._cache[provider] = instance
        return instance

    def _build(self, provider: str) -> Provider:
        if provider == "simulated":
            return SimulatedProvider(seed=self._seed)

        key = (self._settings.api_key(provider) or "").strip()
        if not key:
            raise ProviderError(f"no API key configured for provider {provider!r}")

        timeout = self._settings.request_timeout

        if provider == "gemini":
            from .gemini import GeminiProvider

            return GeminiProvider(api_key=key, timeout=timeout)
        if provider == "anthropic":
            from .anthropic import AnthropicProvider

            return AnthropicProvider(api_key=key, timeout=timeout)
        if provider == "openai":
            from .openai import OpenAIProvider

            return OpenAIProvider(api_key=key, timeout=timeout)

        raise ProviderError(f"unknown provider {provider!r}")

    def accepts_temperature(self, provider: str, model: str) -> bool:
        try:
            instance = self.get(provider)
        except ProviderError:
            return True
        checker = getattr(instance, "accepts_temperature", None)
        if callable(checker):
            return bool(checker(model))
        return bool(getattr(instance, "supports_temperature", True))

    async def aclose(self) -> None:
        for instance in self._cache.values():
            try:
                await instance.aclose()
            except Exception:  # pragma: no cover — shutdown must not raise
                pass
        self._cache.clear()
