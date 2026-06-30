"""Run multiple models and print compact score summaries."""

from abstain_bench.harness import run_benchmark


if __name__ == "__main__":
    result = run_benchmark(
        config_path="config/eval_config.json",
        output_db="leaderboard/results.duckdb",
        dataset_name="simpleqa-with-unanswerable",
        model_names=["claude-sonnet-4-6", "gpt-4o", "llama-3.1-70b"],
        dry_run=True,
    )
    for summary in result.summaries:
        print(summary.model_name, summary.bcs)
