# abstain-bench

[![CI](https://github.com/archzos/abstain-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/archzos/abstain-bench/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)

We've been grading AI to lie confidently. `abstain-bench` flips that incentive by
measuring confident-error rate and calibrated abstention, not just raw accuracy.

`abstain-bench` is a production-oriented starter benchmark that layers a
Behavioral Calibration Score (BCS) on top of standard QA evaluation flows and
publishes a queryable DuckDB-backed leaderboard.

## Why this project exists

Standard binary accuracy gives no direct reward to calibrated uncertainty.
`abstain-bench` evaluates four response outcomes:
- Correct
- Confident-Wrong
- Correct-Abstain
- Unwarranted-Abstain

BCS penalizes confident mistakes most heavily.

## Core metric

```text
BCS = (correct_count
      - λ1 * confident_wrong_count
      - λ2 * unwarranted_abstain_count
      + λ3 * correct_abstain_count) / total_questions
```

Default weights:
- `λ1 = 2.0`
- `λ2 = 0.5`
- `λ3 = 1.0`

All weights are configurable in `config/eval_config.json`.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Quick start

Dry run (config + adapter + dataset validation):

```bash
abstain-bench run --config config/eval_config.json --dry-run
```

Run a benchmark:

```bash
abstain-bench run \
  --config config/eval_config.json \
  --models claude-sonnet-4-6,gpt-4o,llama-3.1-70b \
  --dataset simpleqa-with-unanswerable \
  --output-db leaderboard/results.duckdb
```

Serve leaderboard:

```bash
abstain-bench leaderboard --db leaderboard/results.duckdb --serve
```

## Limitations

- Self-reported confidence is gameable and less reliable than token logprobs.
- Provider API differences can change confidence availability.
- Initial built-in dataset slices prioritize speed and reproducibility over scale.

## Open source and governance

- License: [MIT](./LICENSE)
- Contribution guide: [CONTRIBUTING.md](./CONTRIBUTING.md)
- Security policy: [SECURITY.md](./SECURITY.md)
- Architecture context: [docs/ARCHZOS_AGENT_ARCHITECTURE_CONTEXT.md](./docs/ARCHZOS_AGENT_ARCHITECTURE_CONTEXT.md)

## Credits and ethics

- Built on top of `lm-evaluation-harness` conventions from EleutherAI.
- This project is not affiliated with OpenAI.
