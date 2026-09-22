import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.utils import (
    SEVERITY_ORDER,
    configure_page,
    list_app_log_dates,
    load_app_logs_by_date,
    render_page_header,
)


def render_log_page() -> None:
    render_page_header("Log")

    available_dates = list_app_log_dates()
    date_options = ["All dates", *available_dates]
    selected_date = st.selectbox("Date filter", date_options)
    selected_date_value = None if selected_date == "All dates" else selected_date

    log_entries = load_app_logs_by_date(selected_date_value)
    if not log_entries:
        st.info("No log entries found.")
        return

    key_log = sorted(
        log_entries,
        key=lambda item: (
            SEVERITY_ORDER.get(item["log_type"], 99),
            item.get("timestamp", ""),
        ),
    )[0]
    with st.container(border=True):
        st.subheader("Key Log")
        st.dataframe(
            pd.DataFrame([key_log])[["date", "time", "source", "log_type", "content", "request", "response"]],
            width="stretch",
            hide_index=True,
        )

    all_types = ["All log types"] + sorted(
        {entry["log_type"] for entry in log_entries},
        key=lambda value: SEVERITY_ORDER.get(value, 99),
    )

    filter_cols = st.columns(2, gap="large")
    with filter_cols[0]:
        st.text_input("Selected date", value=selected_date, disabled=True)
    with filter_cols[1]:
        selected_type = st.selectbox("Log type filter", all_types)

    filtered_logs = log_entries
    if selected_type != "All log types":
        filtered_logs = [entry for entry in filtered_logs if entry["log_type"] == selected_type]

    with st.container(border=True):
        st.subheader("Log History")
        if filtered_logs:
            log_frame = pd.DataFrame(filtered_logs)
            ordered_columns = ["date", "time", "source", "log_type", "content", "request", "response"]
            available_columns = [column for column in ordered_columns if column in log_frame.columns]
            st.dataframe(log_frame[available_columns], width="stretch", hide_index=True)
        else:
            st.info("No log entries found for this filter.")


configure_page("Log")
render_log_page()
