"""AWS Bedrock adapter with confidence prompt fallback for non-logprob models."""

from __future__ import annotations

import json
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


class BedrockAdapter(ModelAdapter):
    """Adapter for Bedrock-hosted models using invoke_model."""

    def __init__(self, name: str, model_id: str, config: dict[str, Any] | None = None) -> None:
        super().__init__(name, model_id, config)
        try:
            import boto3  # type: ignore
        except ImportError as exc:
            raise RuntimeError("boto3 dependency missing. Install with: pip install -e .[bedrock]") from exc

        region = str(self.config.get("region", "ap-south-1"))
        self._client = boto3.client("bedrock-runtime", region_name=region)

    def generate(self, prompt: str) -> GenerationResult:
        system = (
            "Answer briefly. If uncertain, you may abstain. "
            "Always include 'Confidence: <0-100>' at the end."
        )
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": int(self.config.get("max_tokens", 256)),
            "messages": [
                {"role": "user", "content": f"{system}\n\nQuestion: {prompt}"},
            ],
        }

        response = self._client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        payload = json.loads(response["body"].read())
        text = payload.get("content", [{}])[0].get("text", "")
        confidence = _extract_confidence(text)

        return GenerationResult(
            model_name=self.name,
            text=text,
            confidence=confidence,
            raw_response={"usage": payload.get("usage", {})},
        )
