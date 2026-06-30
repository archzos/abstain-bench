"""OpenAI adapter with optional self-reported confidence extraction."""

from __future__ import annotations

import re
from typing import Any

from abstain_bench.adapters.base import ModelAdapter
from abstain_bench.models import GenerationResult


def _extract_confidence(text: str) -> float | None:
    match = re.search(r"confidence\s*[:=]\s*(\d{1,3})\s*%?", text, flags=re.IGNORECASE)
    if not match:
        return None
    value = float(match.group(1))
    return min(100.0, max(0.0, value))


class OpenAIAdapter(ModelAdapter):
    """Adapter for OpenAI-hosted frontier models."""

    def __init__(self, name: str, model_id: str, config: dict[str, Any] | None = None) -> None:
        super().__init__(name, model_id, config)
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise RuntimeError("OpenAI dependency missing. Install with: pip install -e .[openai]") from exc
        self._client = OpenAI()

    def generate(self, prompt: str) -> GenerationResult:
        system = (
            "Answer briefly. If uncertain, you may abstain. "
            "Always include 'Confidence: <0-100>' at the end."
        )
        response = self._client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=float(self.config.get("temperature", 0.0)),
        )

        text = response.choices[0].message.content or ""
        confidence = _extract_confidence(text)
        return GenerationResult(
            model_name=self.name,
            text=text,
            confidence=confidence,
            raw_response={"id": response.id},
        )
