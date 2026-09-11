"""Adapter parameter rules, checked against the installed SDK signatures.

These are pure-construction tests — no network, no keys. They exist because the
parameter rules are the easiest thing here to get silently wrong, and a wrong
one fails an entire model column at demo time rather than raising at import.
"""

from __future__ import annotations

import inspect

import pytest

from trustgraph.providers import CompletionRequest
from trustgraph.providers.anthropic import AnthropicProvider
from trustgraph.providers.openai import OpenAIProvider


@pytest.fixture
def anthropic_provider() -> AnthropicProvider:
    return AnthropicProvider(api_key="test-key-not-used")


def request(model: str, **kwargs) -> CompletionRequest:
    return CompletionRequest(model=model, prompt="best payment gateway", **kwargs)


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    ["claude-opus-5", "claude-opus-4-8", "claude-opus-4-7", "claude-sonnet-5"],
)
def test_current_models_are_not_sent_a_temperature(anthropic_provider, model):
    kwargs = anthropic_provider.build_kwargs(request(model, temperature=0.0))
    assert "temperature" not in kwargs
    assert "extra_body" not in kwargs
    assert anthropic_provider.accepts_temperature(model) is False


@pytest.mark.parametrize("model", ["claude-haiku-4-5", "claude-opus-4-6", "claude-sonnet-4-6"])
def test_sampling_models_get_temperature_via_extra_body(anthropic_provider, model):
    kwargs = anthropic_provider.build_kwargs(request(model, temperature=0.0))
    assert anthropic_provider.accepts_temperature(model) is True
    assert kwargs["extra_body"] == {"temperature": 0.0}


def test_temperature_is_never_a_named_kwarg(anthropic_provider):
    """`temperature` was removed from the 1.x signature; by name it is a TypeError.

    This asserts the invariant against the *installed* SDK rather than against a
    remembered API shape, so an SDK upgrade that reinstates or further changes
    the parameter surfaces here.
    """
    import anthropic

    signature = inspect.signature(anthropic.AsyncAnthropic(api_key="x").messages.create)
    accepted = set(signature.parameters)
    assert "temperature" not in accepted, (
        "SDK now accepts `temperature` by name — revisit build_kwargs()"
    )

    for model in ("claude-opus-5", "claude-haiku-4-5"):
        kwargs = anthropic_provider.build_kwargs(request(model, temperature=0.0))
        assert set(kwargs) <= accepted, f"unsupported kwargs: {set(kwargs) - accepted}"


def test_thinking_models_get_headroom_beyond_the_requested_budget(anthropic_provider):
    """Thinking tokens share the max_tokens budget, so a tight budget would
    truncate the prose being measured."""
    tight = request("claude-opus-5", max_tokens=1200)
    assert anthropic_provider.build_kwargs(tight)["max_tokens"] >= 6000
    # A non-thinking-default model keeps the caller's budget.
    assert anthropic_provider.build_kwargs(request("claude-haiku-4-5", max_tokens=1200))[
        "max_tokens"
    ] == 1200


def test_structured_output_shape_and_prompt_fallback(anthropic_provider):
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    kwargs = anthropic_provider.build_kwargs(
        request("claude-opus-5", json_schema=schema, system="be strict")
    )
    assert kwargs["output_config"]["format"]["schema"] is schema
    # The JSON instruction is also in the system prompt so the non-structured
    # fallback path still yields parseable output.
    assert "JSON object" in kwargs["system"]
    assert "be strict" in kwargs["system"]


def test_structured_output_omitted_once_marked_unsupported(anthropic_provider):
    anthropic_provider._structured_unsupported = True
    kwargs = anthropic_provider.build_kwargs(
        request("claude-opus-5", json_schema={"type": "object"})
    )
    assert "output_config" not in kwargs
    assert "JSON object" in kwargs["system"]


def test_no_system_key_when_none_given(anthropic_provider):
    assert "system" not in anthropic_provider.build_kwargs(request("claude-opus-5"))


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["gpt-5", "gpt-5-mini", "o1", "o3-mini"])
def test_openai_reasoning_models_reject_sampling(model):
    provider = OpenAIProvider(api_key="test-key-not-used")
    assert provider.accepts_temperature(model) is False


@pytest.mark.parametrize("model", ["gpt-4o", "gpt-4o-mini", "gpt-4.1"])
def test_openai_chat_models_accept_sampling(model):
    provider = OpenAIProvider(api_key="test-key-not-used")
    assert provider.accepts_temperature(model) is True


def test_openai_signature_still_takes_the_params_we_send():
    import openai

    signature = inspect.signature(
        openai.AsyncOpenAI(api_key="x").chat.completions.create
    )
    accepted = set(signature.parameters)
    for name in ("model", "messages", "max_completion_tokens", "temperature", "response_format"):
        assert name in accepted, f"openai SDK no longer accepts `{name}`"


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------


def test_gemini_config_fields_exist():
    """Guards the config keys the adapter builds."""
    from google.genai import types

    fields = set(types.GenerateContentConfig.model_fields)
    for name in (
        "max_output_tokens",
        "temperature",
        "system_instruction",
        "response_mime_type",
        "response_schema",
    ):
        assert name in fields, f"google-genai no longer accepts `{name}`"


def test_gemini_client_exposes_the_async_surface():
    from google import genai

    client = genai.Client(api_key="test-key-not-used")
    assert hasattr(client.aio.models, "generate_content")
    assert hasattr(client.aio.models, "list")
