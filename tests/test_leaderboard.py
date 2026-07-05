from pathlib import Path

from abstain_bench.leaderboard.store import fetch_language_leaderboard, fetch_leaderboard, fetch_model_details, write_run_results
from abstain_bench.models import BCSWeights, ModelScoreSummary, ResponseCategory, ScoredResponse


def test_leaderboard_write_and_fetch(tmp_path: Path) -> None:
    db_path = tmp_path / "leaderboard.duckdb"
    run_id = "run-1"

    summary = ModelScoreSummary(
        run_id=run_id,
        model_name="model-a",
        dataset_name="simpleqa",
        total_questions=2,
        correct_count=1,
        confident_wrong_count=1,
        correct_abstain_count=0,
        unwarranted_abstain_count=0,
        bcs=-0.5,
        accuracy=0.5,
        confident_wrong_rate=0.5,
        abstention_rate=0.0,
    )

    rows = [
        ScoredResponse(
            run_id=run_id,
            model_name="model-a",
            dataset_name="simpleqa",
            question_id="q1",
            question="Q1",
            prediction="A1",
            ground_truth=["A1"],
            is_answerable=True,
            confidence=90.0,
            category=ResponseCategory.CORRECT,
            is_correct=True,
            abstained=False,
        ),
        ScoredResponse(
            run_id=run_id,
            model_name="model-a",
            dataset_name="simpleqa",
            question_id="q2",
            question="Q2",
            prediction="A2",
            ground_truth=["A3"],
            is_answerable=True,
            confidence=80.0,
            category=ResponseCategory.CONFIDENT_WRONG,
            is_correct=False,
            abstained=False,
        ),
    ]

    write_run_results(
        db_path=str(db_path),
        run_id=run_id,
        dataset_name="simpleqa",
        weights=BCSWeights(),
        summaries=[summary],
        scored_rows=rows,
    )

    board = fetch_leaderboard(str(db_path))
    assert len(board) == 1
    assert board[0]["model_name"] == "model-a"

    details = fetch_model_details(str(db_path), run_id=run_id, model_name="model-a")
    assert len(details) == 2
    assert details[0]["question_id"] == "q1"
    assert details[0]["language"] == "en"

    language_rows = fetch_language_leaderboard(str(db_path), run_id=run_id, model_name="model-a")
    assert len(language_rows) == 1
    assert language_rows[0]["language"] == "en"
