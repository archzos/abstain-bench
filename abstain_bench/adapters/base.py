"""Common adapter interface that decouples provider transport from scoring."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from abstain_bench.models import GenerationResult


class ModelAdapter(ABC):
    """Abstract model adapter interface for provider-specific generation."""

    def __init__(self, name: str, model_id: str, config: dict[str, Any] | None = None) -> None:
        self.name = name
        self.model_id = model_id
        self.config = config or {}

    @abstractmethod
    def generate(self, prompt: str) -> GenerationResult:
        """Generate one answer for a single prompt."""
