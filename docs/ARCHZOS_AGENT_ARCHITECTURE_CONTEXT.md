# ArchzOS Agent Architecture Context

`abstain-bench` fits ArchzOS as evaluation infrastructure for model behavior,
not model-serving infrastructure.

## 1) Calibration-first evaluation layer

ArchzOS security and reliability tooling already assumes deterministic policy
layers. `abstain-bench` extends that posture to benchmarking by scoring whether
models abstain when uncertainty is high instead of rewarding confident guessing.

## 2) Wrapper architecture over existing benchmark ecosystems

The system wraps external evaluation and inference stacks (lm-evaluation-harness,
provider APIs, vLLM) rather than forking them. This preserves compatibility
while allowing ArchzOS-specific metrics (BCS, confident-wrong rate).

## 3) Standardized adapter boundary

A common adapter interface (`ModelAdapter.generate`) isolates provider-specific
transport concerns from scoring logic. This keeps metrics deterministic even when
runtime providers differ in confidence primitives.

## 4) Queryable benchmark artifacts

DuckDB is used as the source of truth for runs and leaderboard outputs. This
keeps daily iteration lightweight while preserving reproducible comparisons.

## 5) Operational rollout pattern

1. Validate metric logic with unit tests.
2. Run one-model dry-runs for config safety.
3. Execute multi-model benchmark runs.
4. Publish leaderboard snapshots through Streamlit.
5. Expand benchmarks incrementally without changing scoring semantics.
