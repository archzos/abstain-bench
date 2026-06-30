"""Streamlit leaderboard UI for BCS-based model comparison."""

from __future__ import annotations

import argparse

import streamlit as st

from abstain_bench.leaderboard.store import fetch_leaderboard, fetch_model_details


def render(db_path: str) -> None:
    """Render leaderboard and drilldown views from DuckDB."""
    st.set_page_config(page_title="abstain-bench leaderboard", layout="wide")
    st.title("abstain-bench leaderboard")
    st.caption("Confident-wrong aware model ranking using Behavioral Calibration Score (BCS).")

    rows = fetch_leaderboard(db_path)
    if not rows:
        st.info("No runs found. Execute `abstain-bench run` first.")
        return

    st.subheader("Model Rankings")
    st.dataframe(rows, use_container_width=True)

    run_choices = sorted({f"{row['run_id']} | {row['model_name']}" for row in rows})
    selected = st.selectbox("Inspect run/model", options=run_choices)
    run_id, model_name = [part.strip() for part in selected.split("|", 1)]

    st.subheader("Per-question Details")
    details = fetch_model_details(db_path, run_id=run_id, model_name=model_name)
    st.dataframe(details, use_container_width=True)


def main() -> None:
    """CLI passthrough for running this app directly."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="leaderboard/results.duckdb")
    args = parser.parse_args()
    render(args.db)


if __name__ == "__main__":
    main()
