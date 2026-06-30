"""Provider adapters for unified model generation interface."""

from abstain_bench.adapters.base import ModelAdapter
from abstain_bench.adapters.bedrock_adapter import BedrockAdapter
from abstain_bench.adapters.openai_adapter import OpenAIAdapter
from abstain_bench.adapters.vllm_adapter import VLLMAdapter

__all__ = [
    "ModelAdapter",
    "OpenAIAdapter",
    "BedrockAdapter",
    "VLLMAdapter",
]
