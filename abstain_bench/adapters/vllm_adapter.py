"""vLLM adapter using OpenAI-compatible chat completion endpoint."""

from __future__ import annotations

from typing import Any

import requests

from abstain_bench.adapters.base import ModelAdapter
from abstain_bench.models import GenerationResult


class VLLMAdapter(ModelAdapter):
    """Adapter for open-weight models served via vLLM."""

    def __init__(self, name: str, model_id: str, config: dict[str, Any] | None = None) -> None:
        super().__init__(name, model_id, config)
        base_url = str(self.config.get("base_url", "http://localhost:8000/v1"))
        self._endpoint = f"{base_url.rstrip('/')}/chat/completions"
        self._api_key = self.config.get("api_key")

    def generate(self, prompt: str) -> GenerationResult:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "system",
                    "content": "Answer briefly. If uncertain, abstain explicitly.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": float(self.config.get("temperature", 0.0)),
            "logprobs": True,
            "top_logprobs": 1,
        }

        response = requests.post(self._endpoint, headers=headers, json=payload, timeout=90)
        response.raise_for_status()
        data = response.json()

        choice = data["choices"][0]
        text = choice["message"]["content"]

        logprobs: list[float] = []
        content = choice.get("logprobs", {}).get("content", [])
        for token in content:
            logprob = token.get("logprob")
            if isinstance(logprob, (float, int)):
                logprobs.append(float(logprob))

        confidence = None
        if logprobs:
            # Convert average negative logprob to rough percentage proxy.
            avg_lp = sum(logprobs) / len(logprobs)
            confidence = max(0.0, min(100.0, 100.0 + (avg_lp * 20.0)))

        return GenerationResult(
            model_name=self.name,
            text=text,
            confidence=confidence,
            logprobs=logprobs or None,
            raw_response={"id": data.get("id")},
        )
