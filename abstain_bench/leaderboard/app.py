"""Streamlit leaderboard UI for BCS-based model comparison."""

from __future__ import annotations

import argparse

import streamlit as st

from abstain_bench.leaderboard.store import fetch_language_leaderboard, fetch_leaderboard, fetch_model_details


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

    language_rows = fetch_language_leaderboard(db_path, run_id=run_id, model_name=model_name)
    language_options = ["all"] + sorted({row["language"] for row in language_rows})
    selected_language = st.selectbox("Language filter", options=language_options)

    filtered_language_rows = (
        [row for row in language_rows if row["language"] == selected_language]
        if selected_language != "all"
        else language_rows
    )

    st.subheader("Per-language breakdown")
    st.dataframe(filtered_language_rows, use_container_width=True)
    if filtered_language_rows:
        chart_data = {
            row["language"]: {
                "accuracy": row["accuracy"],
                "confident_wrong_rate": row["confident_wrong_rate"],
                "abstention_rate": row["abstention_rate"],
            }
            for row in filtered_language_rows
        }
        st.bar_chart(chart_data)

    st.subheader("Per-question Details")
    details = fetch_model_details(db_path, run_id=run_id, model_name=model_name)
    if selected_language != "all":
        details = [row for row in details if row.get("language") == selected_language]
    st.dataframe(details, use_container_width=True)


def main() -> None:
    """CLI passthrough for running this app directly."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="leaderboard/results.duckdb")
    args = parser.parse_args()
    render(args.db)


if __name__ == "__main__":
    main()
