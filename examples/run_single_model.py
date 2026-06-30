"""Run one model against the benchmark pipeline."""

from abstain_bench.harness import run_benchmark


if __name__ == "__main__":
    result = run_benchmark(
        config_path="config/eval_config.json",
        output_db="leaderboard/results.duckdb",
        dataset_name="simpleqa-with-unanswerable",
        model_names=["gpt-4o"],
        dry_run=True,
    )
    print(result)
