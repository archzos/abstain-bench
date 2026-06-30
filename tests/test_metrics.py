from abstain_bench.decoding import detect_abstention
from abstain_bench.metrics import classify_response, compute_bcs
from abstain_bench.models import BCSWeights, GenerationResult, QAExample, ResponseCategory


def test_classifies_correct_answer() -> None:
    example = QAExample(question_id="q1", question="Capital of France?", answers=["Paris"], is_answerable=True)
    generation = GenerationResult(model_name="m", text="Paris", confidence=92.0)
    abstention = detect_abstention(
        response_text=generation.text,
        confidence=generation.confidence,
        confidence_threshold=40.0,
    )
    assert classify_response(example=example, generation=generation, abstention=abstention) == ResponseCategory.CORRECT


def test_classifies_confident_wrong() -> None:
    example = QAExample(question_id="q2", question="Capital of France?", answers=["Paris"], is_answerable=True)
    generation = GenerationResult(model_name="m", text="Lyon", confidence=88.0)
    abstention = detect_abstention(
        response_text=generation.text,
        confidence=generation.confidence,
        confidence_threshold=40.0,
    )
    assert classify_response(example=example, generation=generation, abstention=abstention) == ResponseCategory.CONFIDENT_WRONG


def test_classifies_correct_abstain() -> None:
    example = QAExample(question_id="q3", question="Unknown private value?", answers=[], is_answerable=False)
    generation = GenerationResult(model_name="m", text="I don't know.", confidence=20.0)
    abstention = detect_abstention(
        response_text=generation.text,
        confidence=generation.confidence,
        confidence_threshold=40.0,
    )
    assert classify_response(example=example, generation=generation, abstention=abstention) == ResponseCategory.CORRECT_ABSTAIN


def test_classifies_unwarranted_abstain() -> None:
    example = QAExample(question_id="q4", question="2+2?", answers=["4"], is_answerable=True)
    generation = GenerationResult(model_name="m", text="I cannot determine.", confidence=25.0)
    abstention = detect_abstention(
        response_text=generation.text,
        confidence=generation.confidence,
        confidence_threshold=40.0,
    )
    assert classify_response(example=example, generation=generation, abstention=abstention) == ResponseCategory.UNWARRANTED_ABSTAIN


def test_empty_response_is_low_confidence_abstain() -> None:
    abstention = detect_abstention(
        response_text="",
        confidence=10.0,
        confidence_threshold=40.0,
    )
    assert abstention.is_abstention is True
    assert abstention.reason == "low_confidence_no_answer"


def test_answer_and_hedge_is_not_abstain_when_concrete_answer_present() -> None:
    abstention = detect_abstention(
        response_text="I'm not certain, but final answer: Paris. Confidence: 62",
        confidence=62.0,
        confidence_threshold=40.0,
    )
    assert abstention.is_abstention is False


def test_multi_part_question_correctness_path() -> None:
    example = QAExample(
        question_id="q5",
        question="Name the capital of France and one official language.",
        answers=["Paris French"],
        is_answerable=True,
    )
    generation = GenerationResult(model_name="m", text="Paris French", confidence=70.0)
    abstention = detect_abstention(
        response_text=generation.text,
        confidence=generation.confidence,
        confidence_threshold=40.0,
    )
    category = classify_response(example=example, generation=generation, abstention=abstention)
    assert category == ResponseCategory.CORRECT


def test_bcs_decreases_monotonically_with_more_confident_wrong() -> None:
    weights = BCSWeights(lambda_confident_wrong=2.0, lambda_unwarranted_abstain=0.5, lambda_correct_abstain=1.0)
    total = 100
    fixed_correct = 60
    fixed_unwarranted_abstain = 5
    fixed_correct_abstain = 5

    prev_score = float("inf")
    for confident_wrong in [0, 5, 10, 20, 30]:
        score = compute_bcs(
            total_questions=total,
            correct_count=fixed_correct,
            confident_wrong_count=confident_wrong,
            unwarranted_abstain_count=fixed_unwarranted_abstain,
            correct_abstain_count=fixed_correct_abstain,
            weights=weights,
        )
        assert score <= prev_score
        prev_score = score
