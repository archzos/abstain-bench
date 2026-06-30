"""Command-line interface for running benchmarks and serving leaderboard UI."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from abstain_bench.harness import run_benchmark
from abstain_bench.leaderboard.store import fetch_leaderboard


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="abstain-bench",
        description="Benchmark LLMs on confident-error and abstention calibration.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run benchmark and persist results.")
    run_parser.add_argument("--config", default="config/eval_config.json")
    run_parser.add_argument("--models", default="")
    run_parser.add_argument("--dataset", default="simpleqa-with-unanswerable")
    run_parser.add_argument("--output-db", default="leaderboard/results.duckdb")
    run_parser.add_argument("--dry-run", action="store_true")

    lb_parser = sub.add_parser("leaderboard", help="Inspect or serve leaderboard.")
    lb_parser.add_argument("--db", default="leaderboard/results.duckdb")
    lb_parser.add_argument("--serve", action="store_true")

    return parser


def main() -> None:
    """Entrypoint for abstain-bench CLI."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "run":
        model_names = [item.strip() for item in args.models.split(",") if item.strip()] or None
        result = run_benchmark(
            config_path=args.config,
            output_db=args.output_db,
            dataset_name=args.dataset,
            model_names=model_names,
            dry_run=bool(args.dry_run),
        )
        print(
            f"run_id={result.run_id} dataset={result.dataset_name} total_rows={result.total_rows}"
        )
        for summary in result.summaries:
            print(
                f"model={summary.model_name} bcs={summary.bcs:.4f} "
                f"accuracy={summary.accuracy:.4f} "
                f"confident_wrong_rate={summary.confident_wrong_rate:.4f}"
            )
        return

    if args.command == "leaderboard":
        if args.serve:
            app_path = Path(__file__).parent / "leaderboard" / "app.py"
            subprocess.run(
                ["streamlit", "run", str(app_path), "--", "--db", args.db],
                check=True,
            )
            return

        rows = fetch_leaderboard(args.db)
        if not rows:
            print("No leaderboard rows found.")
            return
        for row in rows:
            print(
                f"run={row['run_id']} model={row['model_name']} "
                f"bcs={row['bcs']:.4f} acc={row['accuracy']:.4f}"
            )


if __name__ == "__main__":
    main()
