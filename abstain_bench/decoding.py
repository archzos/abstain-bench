"""Abstention detection and answer normalization utilities."""

from __future__ import annotations

import re
import string
from dataclasses import dataclass


_DEFAULT_HEDGE_PATTERNS = [
    r"\bi don't know\b",
    r"\bi am not certain\b",
    r"\bi'm not certain\b",
    r"\bi cannot determine\b",
    r"\binsufficient information\b",
    r"\bnot enough information\b",
    r"\bcannot answer\b",
    r"\bunsure\b",
]


@dataclass(slots=True)
class AbstentionResult:
    """Intermediate abstention detection payload used by scoring."""

    is_abstention: bool
    explicit_abstention: bool
    low_confidence: bool
    has_concrete_answer: bool
    reason: str


def _normalize_whitespace(value: str) -> str:
    return " ".join(value.split())


def _strip_confidence_suffix(text: str) -> str:
    return re.sub(r"confidence\s*[:=]\s*\d{1,3}%?", "", text, flags=re.IGNORECASE).strip()


def normalize_answer(value: str) -> str:
    """Normalize textual answers for lightweight exact-match comparison."""
    value = _normalize_whitespace(value).lower()
    translator = str.maketrans("", "", string.punctuation)
    value = value.translate(translator)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return _normalize_whitespace(value)


def is_correct_answer(prediction: str, ground_truths: list[str]) -> bool:
    """Return true when normalized prediction exactly matches any ground-truth alias."""
    pred = normalize_answer(_strip_confidence_suffix(prediction))
    if not pred:
        return False
    normalized_truth = {normalize_answer(item) for item in ground_truths if item.strip()}
    return pred in normalized_truth


def has_concrete_answer(text: str, hedge_patterns: list[str] | None = None) -> bool:
    """Heuristic for whether a response provides substantive answer content."""
    cleaned = _strip_confidence_suffix(text).strip()
    if not cleaned:
        return False

    patterns = hedge_patterns or _DEFAULT_HEDGE_PATTERNS
    lowered = cleaned.lower()
    for pattern in patterns:
        if re.search(pattern, lowered, flags=re.IGNORECASE):
            # Hedge + answer should still count as concrete answer if there is
            # explicit answer marker or enough residual content.
            answer_marker = re.search(r"\b(answer|final answer)\s*[:\-]", lowered)
            if answer_marker:
                return True
            if len(lowered.split()) >= 7:
                return True
            return False

    return True


def detect_abstention(
    *,
    response_text: str,
    confidence: float | None,
    confidence_threshold: float,
    hedge_patterns: list[str] | None = None,
) -> AbstentionResult:
    """Detect abstention using explicit hedge pass + confidence fallback."""
    patterns = hedge_patterns or _DEFAULT_HEDGE_PATTERNS
    lowered = response_text.lower()
    explicit = any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in patterns)

    concrete = has_concrete_answer(response_text, patterns)
    low_confidence = confidence is not None and confidence < confidence_threshold

    abstain = (explicit and not concrete) or (low_confidence and not concrete)

    if explicit and not concrete:
        reason = "explicit_hedge_no_answer"
    elif low_confidence and not concrete:
        reason = "low_confidence_no_answer"
    else:
        reason = "answered"

    return AbstentionResult(
        is_abstention=abstain,
        explicit_abstention=explicit,
        low_confidence=bool(low_confidence),
        has_concrete_answer=concrete,
        reason=reason,
    )
