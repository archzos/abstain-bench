# Contributing to abstain-bench

Thanks for contributing. This project focuses on calibration-first LLM evaluation,
so changes should prioritize scoring correctness and reproducibility.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
```

## Contribution workflow

1. Fork the repo and create a branch from `main`.
2. Keep pull requests scoped and focused.
3. Add or update tests for behavior changes.
4. Run `pytest` locally before opening the PR.
5. Open a PR with a clear evaluation-impact note.

## Pull request checklist

- [ ] Tests added/updated for new behavior
- [ ] Backward compatibility considered
- [ ] Scoring behavior documented (if changed)
- [ ] README updated (if user-facing behavior changed)
- [ ] No secrets or credentials introduced

## Code style guidance

- Prefer deterministic scoring logic over implicit heuristics.
- Keep evaluation and leaderboard paths reproducible.
- Minimize side effects in adapters and harness orchestration.

## Reporting issues

- Use GitHub issues for bugs and feature requests.
- For vulnerabilities, do not open public issues. See `SECURITY.md`.
