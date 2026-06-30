from abstain_bench.decoding import detect_abstention, is_correct_answer, normalize_answer


def test_normalize_answer_removes_articles_and_punctuation() -> None:
    assert normalize_answer("The, Eiffel Tower!") == "eiffel tower"


def test_correct_answer_handles_confidence_suffix() -> None:
    assert is_correct_answer("Paris. Confidence: 91", ["paris"]) is True


def test_low_confidence_without_answer_triggers_abstention() -> None:
    result = detect_abstention(
        response_text="",
        confidence=12.0,
        confidence_threshold=40.0,
    )
    assert result.is_abstention is True
    assert result.reason == "low_confidence_no_answer"


def test_explicit_hedge_with_final_answer_is_not_abstention() -> None:
    result = detect_abstention(
        response_text="I am not certain, but final answer: Mars.",
        confidence=67.0,
        confidence_threshold=40.0,
    )
    assert result.is_abstention is False
