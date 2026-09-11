"""OpenAI adapter (official ``openai`` SDK)."""

from __future__ import annotations

import re

from .base import BaseProvider, CompletionRequest, CompletionResult, ProviderError

#: Reasoning-family models that reject a custom ``temperature``.
_NO_SAMPLING = re.compile(r"^(?:gpt-5|o[1-9])", re.IGNORECASE)


class OpenAIProvider(BaseProvider):
    name = "openai"
    supports_temperature = True

    def __init__(self, api_key: str, timeout: float = 90.0) -> None:
        try:
            import openai
        except ImportError as exc:  # pragma: no cover
            raise ProviderError(
                "openai is not installed — `pip install 'trustgraph[openai]'`"
            ) from exc

        self._sdk = openai
        self._client = openai.AsyncOpenAI(api_key=api_key, timeout=timeout, max_retries=2)

    def accepts_temperature(self, model: str) -> bool:
        return not _NO_SAMPLING.match(model)

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        started = self._now()

        messages: list[dict] = []
        system = request.system
        if request.json_schema is not None:
            system = (system + "\n\n" if system else "") + (
                "Reply with a single JSON object and nothing else."
            )
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": request.prompt})

        kwargs: dict = {
            "model": request.model,
            "messages": messages,
            "max_completion_tokens": request.max_tokens,
        }
        if request.temperature is not None and self.accepts_temperature(request.model):
            kwargs["temperature"] = request.temperature
        if request.json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "extraction",
                    "strict": True,
                    "schema": request.json_schema,
                },
            }

        try:
            completion = await self._client.chat.completions.create(**kwargs)
        except self._sdk.BadRequestError as exc:
            message = str(exc)
            # Older deployments take `max_tokens`; newer ones take
            # `max_completion_tokens`. Swap once rather than guess up front.
            if "max_completion_tokens" in message and "max_tokens" in message:
                kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
                try:
                    completion = await self._client.chat.completions.create(**kwargs)
                except Exception as inner:
                    raise ProviderError(f"openai call failed: {inner}") from inner
            else:
                raise ProviderError(f"openai rejected the request: {exc}") from exc
        except self._sdk.RateLimitError as exc:
            raise ProviderError("openai rate limited", retryable=True, status=429) from exc
        except self._sdk.APIStatusError as exc:
            raise ProviderError(
                f"openai returned {exc.status_code}",
                retryable=exc.status_code >= 500,
                status=exc.status_code,
            ) from exc
        except self._sdk.APIConnectionError as exc:
            raise ProviderError("openai connection error", retryable=True) from exc

        choice = completion.choices[0]
        if choice.finish_reason == "content_filter":
            raise ProviderError("openai content filter blocked the response")

        text = choice.message.content or ""
        if not text.strip():
            raise ProviderError(
                f"openai returned no text (finish_reason={choice.finish_reason})"
            )

        usage = completion.usage
        return self._result(
            text,
            request.model,
            started,
            {
                "input_tokens": getattr(usage, "prompt_tokens", None),
                "output_tokens": getattr(usage, "completion_tokens", None),
            }
            if usage
            else {},
        )

    async def list_models(self) -> list[str]:
        try:
            page = await self._client.models.list()
        except Exception as exc:
            raise ProviderError(f"openai model listing failed: {exc}") from exc
        return sorted(m.id for m in page.data)

    async def aclose(self) -> None:
        await self._client.close()
