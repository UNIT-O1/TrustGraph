"""Provider abstraction.

Every provider is reduced to a single operation — prompt in, text out, with an
optional JSON schema for structured extraction — because that is genuinely all
§6.2 needs. Keeping the surface this narrow is what makes the roster swappable
and the simulated provider a real drop-in rather than a special case.

One methodological note that lives here rather than in the docs, because it
changes the numbers: ``temperature`` defaults to ``0.0``. RSI (Eq. 5) is defined
as variance across *semantically equivalent phrasings*. If sampling temperature
were left at each provider's default, the measured variance would mix phrasing
sensitivity with sampling noise, and RSI would no longer mean what §3.4 says it
means. Providers that do not expose sampling controls declare
``supports_temperature = False`` so the run can say so out loud.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


class ProviderError(RuntimeError):
    """A provider call failed."""

    def __init__(self, message: str, *, retryable: bool = False, status: int | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.status = status


@dataclass(slots=True)
class CompletionRequest:
    model: str
    prompt: str
    system: str | None = None
    max_tokens: int = 2048
    temperature: float | None = 0.0
    #: When set, the provider must return a single JSON object matching this
    #: JSON Schema. Used only by the extraction re-check.
    json_schema: dict[str, Any] | None = None
    #: Test-double affordance, ignored by every real provider. The simulated
    #: provider reads it to generate prose that is on-topic for the run, which
    #: is what makes a keyless demo produce a meaningful grid instead of a wall
    #: of "neither". Deliberately named so it can never be mistaken for a
    #: vendor API parameter.
    sim_context: dict[str, Any] | None = None


@dataclass(slots=True)
class CompletionResult:
    text: str
    provider: str
    model: str
    latency_ms: int
    usage: dict[str, Any] = field(default_factory=dict)
    cached: bool = False


@runtime_checkable
class Provider(Protocol):
    name: str
    supports_temperature: bool

    async def complete(self, request: CompletionRequest) -> CompletionResult: ...

    async def list_models(self) -> list[str]: ...

    async def aclose(self) -> None: ...


class BaseProvider:
    """Shared plumbing: timing and a default no-op close."""

    name: str = "base"
    supports_temperature: bool = True

    async def complete(self, request: CompletionRequest) -> CompletionResult:  # pragma: no cover
        raise NotImplementedError

    async def list_models(self) -> list[str]:
        return []

    async def aclose(self) -> None:
        return None

    # -- helpers ------------------------------------------------------

    @staticmethod
    def _now() -> float:
        return time.perf_counter()

    def _result(
        self,
        text: str,
        model: str,
        started: float,
        usage: dict[str, Any] | None = None,
    ) -> CompletionResult:
        return CompletionResult(
            text=text,
            provider=self.name,
            model=model,
            latency_ms=int((self._now() - started) * 1000),
            usage=usage or {},
        )
