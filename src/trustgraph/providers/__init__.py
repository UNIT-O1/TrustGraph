"""Provider adapters for the multi-model firing stage (§6.2)."""

from .base import (
    BaseProvider,
    CompletionRequest,
    CompletionResult,
    Provider,
    ProviderError,
)
from .registry import ProviderRegistry
from .simulated import SimulatedProvider

__all__ = [
    "BaseProvider",
    "CompletionRequest",
    "CompletionResult",
    "Provider",
    "ProviderError",
    "ProviderRegistry",
    "SimulatedProvider",
]
