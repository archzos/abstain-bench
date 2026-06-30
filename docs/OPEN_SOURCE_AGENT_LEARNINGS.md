# Open-Source Agent Learnings

This document captures the release learnings that should remain in-repo for
future contributors and agents.

## Core lessons

1. BCS metric correctness is the trust anchor.
2. Abstention detection should not punish concrete answers with caveats.
3. Keep leaderboard artifacts queryable (DuckDB) instead of static-only outputs.
4. Require tests on scoring-path changes.
5. Keep OSS governance files complete from first public release.

## Related handoff context

A dedicated future-agent context pack is maintained at:

`_agent-workspace/context/abstain-bench-open-source`

That pack includes project state, timeline, runbook, and expanded learnings.
