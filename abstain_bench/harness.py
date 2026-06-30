"""Benchmark harness orchestrating dataset loading, generation, scoring, and persistence."""

from __future__ import annotations

import importlib.util
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from abstain_bench.adapters import BedrockAdapter, OpenAIAdapter, VLLMAdapter
from abstain_bench.adapters.base import ModelAdapter
from abstain_bench.datasets.loaders import load_dataset
from abstain_bench.decoding import detect_abstention
from abstain_bench.leaderboard.store import write_run_results
from abstain_bench.metrics import classify_response, summarize_scored_responses
from abstain_bench.models import BCSWeights, ModelScoreSummary, ScoredResponse


@dataclass(slots=True)
class BenchmarkRunResult:
    """Top-level run payload returned by harness execution."""

    run_id: str
    dataset_name: str
    summaries: list[ModelScoreSummary]
    total_rows: int


def _load_config(config_path: str) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_weights(config: dict[str, Any]) -> BCSWeights:
    weights = config.get("weights", {})
    return BCSWeights(
        lambda_confident_wrong=float(weights.get("lambda_confident_wrong", 2.0)),
        lambda_unwarranted_abstain=float(weights.get("lambda_unwarranted_abstain", 0.5)),
        lambda_correct_abstain=float(weights.get("lambda_correct_abstain", 1.0)),
    )


def _build_adapter(model_cfg: dict[str, Any]) -> ModelAdapter:
    provider = model_cfg["provider"].lower()
    name = model_cfg["name"]
    model_id = model_cfg["model_id"]

    if provider == "openai":
        return OpenAIAdapter(name=name, model_id=model_id, config=model_cfg)
    if provider == "bedrock":
        return BedrockAdapter(name=name, model_id=model_id, config=model_cfg)
    if provider == "vllm":
        return VLLMAdapter(name=name, model_id=model_id, config=model_cfg)

    raise ValueError(f"Unsupported provider: {provider}")


def _run_optional_lm_eval_baseline(config: dict[str, Any]) -> dict[str, Any] | None:
    """Return baseline metadata when lm-eval is available and enabled."""
    lm_eval_config = config.get("harness", {}).get("lm_eval_baseline", {})
    if not lm_eval_config.get("enabled", False):
        return None

    if importlib.util.find_spec("lm_eval") is None:
        return {"enabled": True, "status": "skipped", "reason": "lm_eval_not_installed"}

    tasks = lm_eval_config.get("tasks", [])
    return {"enabled": True, "status": "ready", "tasks": tasks}


def _select_models(config: dict[str, Any], model_names: list[str] | None) -> list[dict[str, Any]]:
    all_models: list[dict[str, Any]] = config.get("models", [])
    if not model_names:
        return all_models
    wanted = {name.strip() for name in model_names}
    selected = [m for m in all_models if m.get("name") in wanted]
    missing = sorted(wanted - {m.get("name") for m in selected})
    if missing:
        raise ValueError(f"Models not found in config: {', '.join(missing)}")
    return selected


def run_benchmark(
    *,
    config_path: str,
    output_db: str,
    dataset_name: str | None = None,
    model_names: list[str] | None = None,
    dry_run: bool = False,
) -> BenchmarkRunResult:
    """Execute benchmark pipeline end-to-end."""
    config = _load_config(config_path)
    weights = _resolve_weights(config)

    selected_models = _select_models(config, model_names)
    if not selected_models:
        raise ValueError("No models selected for run.")

    default_dataset = dataset_name or "simpleqa-with-unanswerable"
    dataset_limit = int(config.get("harness", {}).get("max_questions", 300))
    questions = load_dataset(default_dataset, limit=dataset_limit)

    abstention_cfg = config.get("abstention", {})
    threshold = float(abstention_cfg.get("confidence_threshold", 40.0))
    hedge_patterns = list(abstention_cfg.get("hedge_patterns", []))

    baseline = _run_optional_lm_eval_baseline(config)

    if dry_run:
        if baseline and baseline.get("status") == "skipped":
            # Dry-run should expose optional baseline readiness but not fail.
            pass
        return BenchmarkRunResult(
            run_id="dry-run",
            dataset_name=default_dataset,
            summaries=[],
            total_rows=0,
        )

    run_id = str(uuid.uuid4())
    all_scored: list[ScoredResponse] = []
    summaries: list[ModelScoreSummary] = []

    for model_cfg in selected_models:
        adapter = _build_adapter(model_cfg)
        per_model_rows: list[ScoredResponse] = []

        for question in questions:
            prompt = question.question
            generation = adapter.generate(prompt)
            abstention = detect_abstention(
                response_text=generation.text,
                confidence=generation.confidence,
                confidence_threshold=threshold,
                hedge_patterns=hedge_patterns,
            )
            category = classify_response(
                example=question,
                generation=generation,
                abstention=abstention,
            )
            is_correct = category.value in {"correct", "correct_abstain"}

            row = ScoredResponse(
                run_id=run_id,
                model_name=model_cfg["name"],
                dataset_name=default_dataset,
                question_id=question.question_id,
                question=question.question,
                prediction=generation.text,
                ground_truth=question.answers,
                is_answerable=question.is_answerable,
                confidence=generation.confidence,
                category=category,
                is_correct=is_correct,
                abstained=abstention.is_abstention,
            )
            per_model_rows.append(row)

        summary = summarize_scored_responses(
            run_id=run_id,
            model_name=model_cfg["name"],
            dataset_name=default_dataset,
            scored_responses=per_model_rows,
            weights=weights,
        )
        all_scored.extend(per_model_rows)
        summaries.append(summary)

    write_run_results(
        db_path=output_db,
        run_id=run_id,
        dataset_name=default_dataset,
        weights=weights,
        summaries=summaries,
        scored_rows=all_scored,
    )

    return BenchmarkRunResult(
        run_id=run_id,
        dataset_name=default_dataset,
        summaries=summaries,
        total_rows=len(all_scored),
    )
