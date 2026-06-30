# Roadmap (Day 1 / Day 2)

## Day 1 (Core IP and reliability)

- Finalize BCS category classification and formula behavior.
- Ship complete `tests/test_metrics.py` coverage for category and monotonicity guarantees.
- Validate end-to-end harness flow on a small slice with dry-run safety.
- Keep CI green and governance files complete for public OSS readiness.

## Day 2 (Demo and leaderboard)

- Execute benchmark pipeline across 3 model adapters (OpenAI, Bedrock, vLLM).
- Persist run artifacts into DuckDB and expose comparative leaderboard in Streamlit.
- Finalize README for public launch framing around confident-wrong rate.
- Capture a clean leaderboard screenshot/GIF for launch assets.

## Out of scope for v1

- RAG-grounded abstention variants
- Multilingual benchmark expansion
- Agent-tool abstention protocols
- Calibration fine-tuning loops

Track all post-v1 ideas as GitHub issues with label: `extension`.
