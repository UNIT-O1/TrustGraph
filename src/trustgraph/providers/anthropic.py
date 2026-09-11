"""Anthropic adapter (official ``anthropic`` SDK, 1.x)."""

from __future__ import annotations

import re

from .base import BaseProvider, CompletionRequest, CompletionResult, ProviderError

#: Models on which sampling controls were removed — sending ``temperature``
#: returns a 400. Everything else (Haiku 4.5, the 4.6 generation, older) still
#: accepts it.
_NO_SAMPLING = re.compile(
    r"^claude-(?:opus-(?:5|4-8|4-7)|sonnet-5|fable-5|mythos-5)", re.IGNORECASE
)

#: Models where adaptive thinking is on unless disabled. Thinking tokens are
#: drawn from the same ``max_tokens`` budget as the visible answer, so a tight
#: budget would truncate the prose we are trying to measure.
_THINKING_DEFAULT_ON = re.compile(
    r"^claude-(?:opus-5|fable-5|mythos-5)", re.IGNORECASE
)

_MIN_TOKENS_WITH_THINKING = 6000


class AnthropicProvider(BaseProvider):
    name = "anthropic"
    #: Reported as False because the roster's flagship models reject sampling
    #: controls. The run surfaces this: on those columns, measured variance
    #: includes provider-side sampling noise as well as phrasing sensitivity.
    supports_temperature = False

    def __init__(self, api_key: str, timeout: float = 90.0) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise ProviderError(
                "anthropic is not installed — `pip install 'trustgraph[anthropic]'`"
            ) from exc

        self._sdk = anthropic
        self._client = anthropic.AsyncAnthropic(api_key=api_key, timeout=timeout, max_retries=2)
        self._structured_unsupported = False

    def accepts_temperature(self, model: str) -> bool:
        return not _NO_SAMPLING.match(model)

    def build_kwargs(self, request: CompletionRequest) -> dict:
        """Assemble the request payload.

        Split out from :meth:`complete` so the parameter rules — which are the
        easiest thing to get wrong here — are testable without a key.

        Note the temperature handling. ``temperature`` was removed from the 1.x
        SDK's ``messages.create`` signature entirely, so passing it by name
        raises ``TypeError`` rather than a catchable API error. For the older
        models that do still honour sampling server-side it therefore travels in
        ``extra_body``; for current models it is not sent at all.
        """
        max_tokens = request.max_tokens
        if _THINKING_DEFAULT_ON.match(request.model):
            max_tokens = max(max_tokens, _MIN_TOKENS_WITH_THINKING)

        system = request.system
        if request.json_schema is not None:
            # Belt and braces: the schema is also stated in the prompt, so the
            # non-structured fallback path below still returns parseable JSON.
            system = (system + "\n\n" if system else "") + (
                "Reply with a single JSON object and nothing else — no prose, no "
                "code fences."
            )

        kwargs: dict = {
            "model": request.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        if system:
            kwargs["system"] = system
        if request.temperature is not None and self.accepts_temperature(request.model):
            kwargs["extra_body"] = {"temperature": request.temperature}
        if request.json_schema is not None and not self._structured_unsupported:
            kwargs["output_config"] = {
                "format": {
                    "type": "json_schema",
                    "schema": request.json_schema,
                }
            }
        return kwargs

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        started = self._now()
        kwargs = self.build_kwargs(request)

        try:
            message = await self._client.messages.create(**kwargs)
        except self._sdk.BadRequestError as exc:
            if "output_config" in kwargs:
                # This SDK/model pairing does not take the structured-output
                # shape; fall back to prompt-instructed JSON permanently.
                self._structured_unsupported = True
                kwargs.pop("output_config", None)
                try:
                    message = await self._client.messages.create(**kwargs)
                except Exception as inner:
                    raise ProviderError(f"anthropic call failed: {inner}") from inner
            else:
                raise ProviderError(f"anthropic rejected the request: {exc}") from exc
        except self._sdk.RateLimitError as exc:
            raise ProviderError("anthropic rate limited", retryable=True, status=429) from exc
        except self._sdk.APIStatusError as exc:
            raise ProviderError(
                f"anthropic returned {exc.status_code}",
                retryable=exc.status_code >= 500,
                status=exc.status_code,
            ) from exc
        except self._sdk.APIConnectionError as exc:
            raise ProviderError("anthropic connection error", retryable=True) from exc
        except TypeError as exc:
            # An SDK signature change (a removed parameter) surfaces here rather
            # than as an API error. Fail this cell with a legible reason instead
            # of letting a bare TypeError escape as an unexplained crash.
            raise ProviderError(f"anthropic SDK rejected a parameter: {exc}") from exc

        if getattr(message, "stop_reason", None) == "refusal":
            details = getattr(message, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise ProviderError(f"anthropic declined the request (category={category})")

        text = "".join(
            block.text for block in message.content if getattr(block, "type", "") == "text"
        )
        if not text.strip():
            raise ProviderError(
                f"anthropic returned no text (stop_reason={getattr(message, 'stop_reason', None)})"
            )

        usage = getattr(message, "usage", None)
        return self._result(
            text,
            request.model,
            started,
            {
                "input_tokens": getattr(usage, "input_tokens", None),
                "output_tokens": getattr(usage, "output_tokens", None),
            }
            if usage
            else {},
        )

    async def list_models(self) -> list[str]:
        try:
            page = await self._client.models.list()
        except Exception as exc:
            raise ProviderError(f"anthropic model listing failed: {exc}") from exc
        return sorted(m.id for m in page.data)

    async def aclose(self) -> None:
        await self._client.close()
