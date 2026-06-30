# Leaderboard Data Model

`abstain-bench` stores benchmark outputs in DuckDB with three tables.

## `runs`

- `run_id` (TEXT, primary key)
- `dataset_name` (TEXT)
- `created_at` (TIMESTAMP)
- `weights_json` (TEXT)

## `model_results`

- `run_id` (TEXT)
- `model_name` (TEXT)
- `dataset_name` (TEXT)
- `total_questions` (INTEGER)
- `correct_count` (INTEGER)
- `confident_wrong_count` (INTEGER)
- `correct_abstain_count` (INTEGER)
- `unwarranted_abstain_count` (INTEGER)
- `bcs` (DOUBLE)
- `accuracy` (DOUBLE)
- `confident_wrong_rate` (DOUBLE)
- `abstention_rate` (DOUBLE)

## `question_results`

- `run_id` (TEXT)
- `model_name` (TEXT)
- `dataset_name` (TEXT)
- `question_id` (TEXT)
- `question` (TEXT)
- `prediction` (TEXT)
- `ground_truth_json` (TEXT)
- `is_answerable` (BOOLEAN)
- `confidence` (DOUBLE, nullable)
- `category` (TEXT)
- `is_correct` (BOOLEAN)
- `abstained` (BOOLEAN)

This schema is intentionally simple so leaderboard queries stay reproducible and
portable without external services.
