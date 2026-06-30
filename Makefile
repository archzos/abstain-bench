.PHONY: install test dry-run run leaderboard

install:
	pip install -e '.[dev]'

test:
	python3.12 -m pytest

dry-run:
	python3.12 -m abstain_bench.cli run --config config/eval_config.json --dry-run

run:
	python3.12 -m abstain_bench.cli run \
		--config config/eval_config.json \
		--dataset simpleqa-with-unanswerable \
		--output-db leaderboard/results.duckdb

leaderboard:
	python3.12 -m abstain_bench.cli leaderboard --db leaderboard/results.duckdb --serve
