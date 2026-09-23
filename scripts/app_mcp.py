from __future__ import annotations

import json
from typing import Any

import streamlit as st

SUMMARY_PAGE = "pages/1_Summary.py"


def summary_download_report() -> dict[str, Any]:
    """Request generation of the Summary page PDF report and open the Summary page."""
    st.session_state["summary_download_report_requested"] = True
    st.session_state["mcp_last_action"] = "Summary report generation requested."
    return {
        "status": "ok",
        "tool": "summary_download_report",
        "message": "Summary 리포트 생성을 요청했습니다. Summary 페이지로 이동합니다.",
        "target_page": SUMMARY_PAGE,
        "requires_rerun": True,
    }


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None

    candidates = [stripped]
    first = stripped.find("{")
    last = stripped.rfind("}")
    if first >= 0 and last > first:
        candidates.append(stripped[first : last + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None
