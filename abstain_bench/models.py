"""Core data models for abstention-aware generation and scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ResponseCategory(str, Enum):
    """Mutually exclusive response classes used by BCS."""

    CORRECT = "correct"
    CONFIDENT_WRONG = "confident_wrong"
    CORRECT_ABSTAIN = "correct_abstain"
    UNWARRANTED_ABSTAIN = "unwarranted_abstain"


@dataclass(slots=True)
class QAExample:
    """Single benchmark question record."""

    question_id: str
    question: str
    answers: list[str]
    is_answerable: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GenerationResult:
    """Provider-normalized generation output."""

    model_name: str
    text: str
    confidence: float | None = None
    logprobs: list[float] | None = None
    raw_response: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AbstentionResult:
    """Output of abstention detector."""

    is_abstention: bool
    explicit_abstention: bool
    low_confidence: bool
    has_concrete_answer: bool
    reason: str


@dataclass(slots=True)
class ScoredResponse:
    """Per-question scored output used in persistence and analytics."""

    run_id: str
    model_name: str
    dataset_name: str
    question_id: str
    question: str
    prediction: str
    ground_truth: list[str]
    is_answerable: bool
    confidence: float | None
    category: ResponseCategory
    is_correct: bool
    abstained: bool


@dataclass(slots=True)
class BCSWeights:
    """Weight controls for Behavioral Calibration Score."""

    lambda_confident_wrong: float = 2.0
    lambda_unwarranted_abstain: float = 0.5
    lambda_correct_abstain: float = 1.0


@dataclass(slots=True)
class ModelScoreSummary:
    """Aggregate score summary for one model on one run/dataset."""

    run_id: str
    model_name: str
    dataset_name: str
    total_questions: int
    correct_count: int
    confident_wrong_count: int
    correct_abstain_count: int
    unwarranted_abstain_count: int
    bcs: float
    accuracy: float
    confident_wrong_rate: float
    abstention_rate: float
