"""DuckDB persistence layer for benchmark runs and leaderboard queries."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import duckdb

from abstain_bench.models import BCSWeights, ModelScoreSummary, ScoredResponse


def initialize_db(db_path: str) -> None:
    """Create required tables if absent."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                dataset_name TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                weights_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS model_results (
                run_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                dataset_name TEXT NOT NULL,
                total_questions INTEGER NOT NULL,
                correct_count INTEGER NOT NULL,
                confident_wrong_count INTEGER NOT NULL,
                correct_abstain_count INTEGER NOT NULL,
                unwarranted_abstain_count INTEGER NOT NULL,
                bcs DOUBLE NOT NULL,
                accuracy DOUBLE NOT NULL,
                confident_wrong_rate DOUBLE NOT NULL,
                abstention_rate DOUBLE NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS question_results (
                run_id TEXT NOT NULL,
                model_name TEXT NOT NULL,
                dataset_name TEXT NOT NULL,
                question_id TEXT NOT NULL,
                question TEXT NOT NULL,
                prediction TEXT NOT NULL,
                ground_truth_json TEXT NOT NULL,
                is_answerable BOOLEAN NOT NULL,
                confidence DOUBLE,
                language TEXT NOT NULL DEFAULT 'en',
                category TEXT NOT NULL,
                is_correct BOOLEAN NOT NULL,
                abstained BOOLEAN NOT NULL
            )
            """
        )
        conn.execute("ALTER TABLE question_results ADD COLUMN IF NOT EXISTS language TEXT DEFAULT 'en'")


def write_run_results(
    *,
    db_path: str,
    run_id: str,
    dataset_name: str,
    weights: BCSWeights,
    summaries: list[ModelScoreSummary],
    scored_rows: list[ScoredResponse],
) -> None:
    """Persist one run and associated model/question results."""
    initialize_db(db_path)
    with duckdb.connect(db_path) as conn:
        conn.execute(
            "DELETE FROM question_results WHERE run_id = ?",
            [run_id],
        )
        conn.execute(
            "DELETE FROM model_results WHERE run_id = ?",
            [run_id],
        )
        conn.execute(
            "DELETE FROM runs WHERE run_id = ?",
            [run_id],
        )

        conn.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?)",
                [
                    run_id,
                    dataset_name,
                    datetime.now(UTC).replace(tzinfo=None),
                    json.dumps(asdict(weights)),
                ],
            )

        conn.executemany(
            """
            INSERT INTO model_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                [
                    s.run_id,
                    s.model_name,
                    s.dataset_name,
                    s.total_questions,
                    s.correct_count,
                    s.confident_wrong_count,
                    s.correct_abstain_count,
                    s.unwarranted_abstain_count,
                    s.bcs,
                    s.accuracy,
                    s.confident_wrong_rate,
                    s.abstention_rate,
                ]
                for s in summaries
            ],
        )

        conn.executemany(
            """
            INSERT INTO question_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                [
                    row.run_id,
                    row.model_name,
                    row.dataset_name,
                    row.question_id,
                    row.question,
                    row.prediction,
                    json.dumps(row.ground_truth),
                    row.is_answerable,
                    row.confidence,
                    row.language,
                    row.category.value,
                    row.is_correct,
                    row.abstained,
                ]
                for row in scored_rows
            ],
        )


def fetch_leaderboard(db_path: str) -> list[dict[str, object]]:
    """Return leaderboard rows sorted by BCS descending."""
    initialize_db(db_path)
    with duckdb.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
              mr.run_id,
              mr.model_name,
              mr.dataset_name,
              mr.bcs,
              mr.accuracy,
              mr.confident_wrong_rate,
              mr.abstention_rate,
              mr.total_questions,
              r.created_at
            FROM model_results mr
            JOIN runs r ON r.run_id = mr.run_id
            ORDER BY mr.bcs DESC, mr.accuracy DESC
            """
        ).fetchall()

    return [
        {
            "run_id": row[0],
            "model_name": row[1],
            "dataset_name": row[2],
            "bcs": row[3],
            "accuracy": row[4],
            "confident_wrong_rate": row[5],
            "abstention_rate": row[6],
            "total_questions": row[7],
            "created_at": str(row[8]),
        }
        for row in rows
    ]


def fetch_model_details(db_path: str, *, run_id: str, model_name: str) -> list[dict[str, object]]:
    """Return per-question records for one model and run."""
    initialize_db(db_path)
    with duckdb.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT question_id, question, prediction, category, confidence, language
            FROM question_results
            WHERE run_id = ? AND model_name = ?
            ORDER BY question_id ASC
            """,
            [run_id, model_name],
        ).fetchall()

    return [
        {
            "question_id": row[0],
            "question": row[1],
            "prediction": row[2],
            "category": row[3],
            "confidence": row[4],
            "language": row[5],
        }
        for row in rows
    ]


def fetch_language_leaderboard(
    db_path: str,
    *,
    run_id: str | None = None,
    model_name: str | None = None,
) -> list[dict[str, object]]:
    """Return per-language breakdown rows for leaderboard filters/charts."""
    initialize_db(db_path)
    clauses = []
    params: list[object] = []
    if run_id:
        clauses.append("run_id = ?")
        params.append(run_id)
    if model_name:
        clauses.append("model_name = ?")
        params.append(model_name)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    with duckdb.connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT
              run_id,
              model_name,
              COALESCE(language, 'en') AS language,
              COUNT(*) AS total_questions,
              AVG(CASE WHEN is_correct THEN 1.0 ELSE 0.0 END) AS accuracy,
              AVG(CASE WHEN category = 'confident_wrong' THEN 1.0 ELSE 0.0 END) AS confident_wrong_rate,
              AVG(CASE WHEN abstained THEN 1.0 ELSE 0.0 END) AS abstention_rate
            FROM question_results
            {where}
            GROUP BY run_id, model_name, language
            ORDER BY run_id, model_name, language
            """,
            params,
        ).fetchall()

    return [
        {
            "run_id": row[0],
            "model_name": row[1],
            "language": row[2],
            "total_questions": row[3],
            "accuracy": row[4],
            "confident_wrong_rate": row[5],
            "abstention_rate": row[6],
        }
        for row in rows
    ]
