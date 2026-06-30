"""Behavioral Calibration Score (BCS) classification and aggregation logic."""

from __future__ import annotations

from collections import Counter

from abstain_bench.decoding import AbstentionResult, is_correct_answer
from abstain_bench.models import (
    BCSWeights,
    GenerationResult,
    ModelScoreSummary,
    QAExample,
    ResponseCategory,
    ScoredResponse,
)


def classify_response(
    *,
    example: QAExample,
    generation: GenerationResult,
    abstention: AbstentionResult,
) -> ResponseCategory:
    """Classify a response into one and only one BCS category."""
    if abstention.is_abstention:
        return (
            ResponseCategory.CORRECT_ABSTAIN
            if not example.is_answerable
            else ResponseCategory.UNWARRANTED_ABSTAIN
        )

    if is_correct_answer(generation.text, example.answers):
        return ResponseCategory.CORRECT

    return ResponseCategory.CONFIDENT_WRONG


def compute_bcs(
    *,
    total_questions: int,
    correct_count: int,
    confident_wrong_count: int,
    unwarranted_abstain_count: int,
    correct_abstain_count: int,
    weights: BCSWeights,
) -> float:
    """Compute BCS from category counts and configured lambdas."""
    if total_questions <= 0:
        return 0.0

    numerator = (
        correct_count
        - (weights.lambda_confident_wrong * confident_wrong_count)
        - (weights.lambda_unwarranted_abstain * unwarranted_abstain_count)
        + (weights.lambda_correct_abstain * correct_abstain_count)
    )
    return numerator / total_questions


def summarize_scored_responses(
    *,
    run_id: str,
    model_name: str,
    dataset_name: str,
    scored_responses: list[ScoredResponse],
    weights: BCSWeights,
) -> ModelScoreSummary:
    """Aggregate per-question outputs into model-level summary metrics."""
    total = len(scored_responses)
    category_counts = Counter(item.category for item in scored_responses)

    correct_count = category_counts[ResponseCategory.CORRECT]
    confident_wrong_count = category_counts[ResponseCategory.CONFIDENT_WRONG]
    correct_abstain_count = category_counts[ResponseCategory.CORRECT_ABSTAIN]
    unwarranted_abstain_count = category_counts[ResponseCategory.UNWARRANTED_ABSTAIN]

    bcs = compute_bcs(
        total_questions=total,
        correct_count=correct_count,
        confident_wrong_count=confident_wrong_count,
        unwarranted_abstain_count=unwarranted_abstain_count,
        correct_abstain_count=correct_abstain_count,
        weights=weights,
    )

    accuracy = (correct_count / total) if total else 0.0
    confident_wrong_rate = (confident_wrong_count / total) if total else 0.0
    abstention_count = correct_abstain_count + unwarranted_abstain_count
    abstention_rate = (abstention_count / total) if total else 0.0

    return ModelScoreSummary(
        run_id=run_id,
        model_name=model_name,
        dataset_name=dataset_name,
        total_questions=total,
        correct_count=correct_count,
        confident_wrong_count=confident_wrong_count,
        correct_abstain_count=correct_abstain_count,
        unwarranted_abstain_count=unwarranted_abstain_count,
        bcs=bcs,
        accuracy=accuracy,
        confident_wrong_rate=confident_wrong_rate,
        abstention_rate=abstention_rate,
    )
