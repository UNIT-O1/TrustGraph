"""Google Gemini adapter (``google-genai`` SDK)."""

from __future__ import annotations

import asyncio

from .base import BaseProvider, CompletionRequest, CompletionResult, ProviderError


class GeminiProvider(BaseProvider):
    name = "gemini"
    supports_temperature = True

    def __init__(self, api_key: str, timeout: float = 90.0) -> None:
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover
            raise ProviderError(
                "google-genai is not installed — `pip install 'trustgraph[gemini]'`"
            ) from exc

        self._genai = genai
        self._client = genai.Client(api_key=api_key)
        self._timeout = timeout
        #: Set once if this SDK build rejects a raw JSON-Schema dict, so we stop
        #: paying for the failed attempt on every subsequent call.
        self._schema_unsupported = False

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        from google.genai import types

        started = self._now()
        cfg: dict = {
            "max_output_tokens": request.max_tokens,
            # We never declare tools, but the SDK emits an "AFC not recommended"
            # advisory unless automatic function calling is explicitly off. Turn
            # it off so a clean run produces clean output.
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        }
        if request.temperature is not None:
            cfg["temperature"] = request.temperature
        if request.system:
            cfg["system_instruction"] = request.system
        if request.json_schema is not None:
            cfg["response_mime_type"] = "application/json"
            if not self._schema_unsupported:
                cfg["response_schema"] = request.json_schema

        try:
            response = await asyncio.wait_for(
                self._client.aio.models.generate_content(
                    model=request.model,
                    contents=request.prompt,
                    config=types.GenerateContentConfig(**cfg),
                ),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError as exc:
            raise ProviderError(
                f"gemini timed out after {self._timeout:.0f}s", retryable=True
            ) from exc
        except Exception as exc:
            message = str(exc)
            # A schema the SDK build won't accept: drop it and retry once with
            # JSON mime-type only, which every version supports.
            if "response_schema" in cfg and _is_schema_error(message):
                self._schema_unsupported = True
                cfg.pop("response_schema", None)
                return await self._retry_without_schema(request, cfg, started)
            raise ProviderError(
                f"gemini call failed: {message}", retryable=_is_retryable(message)
            ) from exc

        return self._result(_text_of(response), request.model, started, _usage_of(response))

    async def _retry_without_schema(
        self, request: CompletionRequest, cfg: dict, started: float
    ) -> CompletionResult:
        from google.genai import types

        try:
            response = await asyncio.wait_for(
                self._client.aio.models.generate_content(
                    model=request.model,
                    contents=request.prompt,
                    config=types.GenerateContentConfig(**cfg),
                ),
                timeout=self._timeout,
            )
        except Exception as exc:
            raise ProviderError(
                f"gemini call failed: {exc}", retryable=_is_retryable(str(exc))
            ) from exc
        return self._result(_text_of(response), request.model, started, _usage_of(response))

    async def list_models(self) -> list[str]:
        try:
            page = await self._client.aio.models.list()
        except Exception as exc:
            raise ProviderError(f"gemini model listing failed: {exc}") from exc

        names: list[str] = []
        async for model in page:
            actions = getattr(model, "supported_actions", None) or []
            if actions and "generateContent" not in actions:
                continue
            name = (getattr(model, "name", "") or "").removeprefix("models/")
            if name:
                names.append(name)
        return sorted(names)


def _text_of(response) -> str:
    text = getattr(response, "text", None)
    if text:
        return text

    # `.text` is None when the candidate carried no text part — most often a
    # safety block or a max-token stop. Surface which, instead of an empty cell.
    feedback = getattr(response, "prompt_feedback", None)
    blocked = getattr(feedback, "block_reason", None) if feedback else None
    if blocked:
        raise ProviderError(f"gemini blocked the prompt: {blocked}")

    candidates = getattr(response, "candidates", None) or []
    if candidates:
        reason = getattr(candidates[0], "finish_reason", None)
        parts = getattr(getattr(candidates[0], "content", None), "parts", None) or []
        joined = "".join(getattr(p, "text", "") or "" for p in parts)
        if joined:
            return joined
        raise ProviderError(f"gemini returned no text (finish_reason={reason})")

    raise ProviderError("gemini returned an empty response")


def _usage_of(response) -> dict:
    usage = getattr(response, "usage_metadata", None)
    if not usage:
        return {}
    return {
        "input_tokens": getattr(usage, "prompt_token_count", None),
        "output_tokens": getattr(usage, "candidates_token_count", None),
        "total_tokens": getattr(usage, "total_token_count", None),
    }


def _is_schema_error(message: str) -> bool:
    lowered = message.lower()
    return "response_schema" in lowered or (
        "schema" in lowered and ("invalid" in lowered or "unsupported" in lowered)
    )


def _is_retryable(message: str) -> bool:
    lowered = message.lower()
    return any(
        token in lowered
        for token in ("429", "rate", "quota", "500", "503", "unavailable", "deadline", "timeout")
    )
