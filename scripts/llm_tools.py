from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import streamlit as st

from scripts.app_mcp import summary_download_report

BASE_DIR = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Streamlit app-control tools: Dashboard query settings / Summary report &
# inference results / Log reads.
#
# These are pure business-logic functions — validation + session_state writes
# only. The interactive, multi-turn parts (asking for missing info, confirming
# ambiguous choices) live in scripts/agent_graph.py, which wraps these as
# LangGraph tools using interrupt(). Keeping that split means these functions
# never need to know whether they were called after a clarifying question or
# with everything supplied up front.
# ---------------------------------------------------------------------------

DASHBOARD_PAGE = "Dashboard.py"
SUMMARY_PAGE = "pages/1_Summary.py"
LOG_PAGE = "pages/5_Log.py"

# Mirrors Dashboard.py's CSV_DEFAULT_QUERY_START_DATE / CSV_DEFAULT_QUERY_END_DATE —
# the queryable data window. Requests outside this range are rejected up front.
DASHBOARD_QUERY_MIN_DATE = date(2026, 1, 1)
DASHBOARD_QUERY_MAX_DATE = date(2026, 5, 31)

APP_TOOL_NAMES = {
    "dashboard_set_query_setting",
    "summary_download_report",
    "summary_read_inference_results",
    "log_read",
    "list_defect_types",
}

_POSITIVE_ANSWERS = ("y", "yes", "네", "예", "맞아", "맞습니다", "맞아요", "응", "어", "그래", "좋아", "확인", "ok", "okay")
_NEGATIVE_ANSWERS = ("n", "no", "아니", "아니오", "아니요", "아뇨", "틀려", "다른", "노")
_ALL_LOG_TYPE_ANSWERS = ("전체", "전체 로그", "전체로그", "모두", "전부")


def _parse_yes_no_answer(text: str) -> bool | None:
    """Interpret a free-text yes/no answer. Returns None if it can't be read either way."""
    normalized = str(text or "").strip().lower()
    if not normalized:
        return None
    if any(normalized == token or normalized.startswith(token) for token in _POSITIVE_ANSWERS):
        return True
    if any(normalized == token or normalized.startswith(token) for token in _NEGATIVE_ANSWERS):
        return False
    return None


def _parse_log_type_answer(text: str) -> str | None:
    """Interpret a free-text answer to the log_type question. "all"/"전체"-ish -> None (no filter)."""
    normalized = str(text or "").strip()
    if not normalized:
        return None
    if normalized.lower() == "all" or normalized in _ALL_LOG_TYPE_ANSWERS:
        return None
    return normalized


def dashboard_set_query_setting(
    table: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """Update the Dashboard query settings (table / start date / end date) and reload data."""
    from scripts.utils import SUPABASE_IMAGE_TABLE, load_dashboard_data

    valid_tables = (SUPABASE_IMAGE_TABLE,)

    if table:
        requested_table = str(table).strip()
        if requested_table and requested_table not in valid_tables:
            return {
                "status": "error",
                "tool": "dashboard_set_query_setting",
                "message": f"해당 테이블이 존재 하지 않습니다. {', '.join(valid_tables)} 테이블 중에서 선택해주세요.",
                "requires_rerun": False,
            }

    updates: list[str] = []
    parsed_dates: dict[str, date] = {}

    for label, raw_value in (("start_date", start_date), ("end_date", end_date)):
        if not raw_value:
            continue
        try:
            parsed_dates[label] = date.fromisoformat(str(raw_value).strip())
        except ValueError:
            return {
                "status": "error",
                "tool": "dashboard_set_query_setting",
                "message": f"{label} 값이 올바르지 않습니다: {raw_value!r}. YYYY-MM-DD 형식으로 입력해주세요.",
                "requires_rerun": False,
            }

    for parsed in parsed_dates.values():
        if parsed < DASHBOARD_QUERY_MIN_DATE or parsed > DASHBOARD_QUERY_MAX_DATE:
            return {
                "status": "error",
                "tool": "dashboard_set_query_setting",
                "message": "해당 날짜는 조회가 불가능합니다. 26년 1월 1일 부터 5월 31일 사이의 날짜를 선택해주세요.",
                "requires_rerun": False,
            }

    if "start_date" in parsed_dates and "end_date" in parsed_dates and parsed_dates["start_date"] > parsed_dates["end_date"]:
        return {
            "status": "error",
            "tool": "dashboard_set_query_setting",
            "message": "시작일이 종료일보다 늦을 수 없습니다.",
            "requires_rerun": False,
        }

    if table:
        requested_table = str(table).strip()
        if requested_table:
            # Pre-seed both the stored value and the selectbox's own widget key —
            # once a widget key exists in session_state, its `index=` argument is
            # ignored on rerun, so the widget key must be set directly too.
            st.session_state["dashboard_query_table"] = requested_table
            st.session_state["dashboard_query_table_selector"] = requested_table
            updates.append(f"table={requested_table}")

    if "start_date" in parsed_dates:
        parsed_start = parsed_dates["start_date"]
        st.session_state["dashboard_query_date_start_pending"] = parsed_start.isoformat()
        st.session_state["dashboard_query_date_start"] = parsed_start.isoformat()
        st.session_state["dashboard_query_date_start_input"] = parsed_start
        updates.append(f"start_date={parsed_start.isoformat()}")

    if "end_date" in parsed_dates:
        parsed_end = parsed_dates["end_date"]
        st.session_state["dashboard_query_date_end_pending"] = parsed_end.isoformat()
        st.session_state["dashboard_query_date_end"] = parsed_end.isoformat()
        st.session_state["dashboard_query_date_end_input"] = parsed_end
        updates.append(f"end_date={parsed_end.isoformat()}")

    if not updates:
        return {
            "status": "error",
            "tool": "dashboard_set_query_setting",
            "message": "table, start_date, end_date 중 최소 하나는 입력해야 합니다.",
            "requires_rerun": False,
        }

    st.session_state["dashboard_data_loaded"] = True
    load_dashboard_data.clear()
    st.session_state["mcp_last_action"] = f"Dashboard query setting updated: {', '.join(updates)}"

    return {
        "status": "ok",
        "tool": "dashboard_set_query_setting",
        "message": f"대시보드 조회 설정이 변경되었습니다 ({', '.join(updates)}).",
        "target_page": DASHBOARD_PAGE,
        "requires_rerun": True,
        "clear_dashboard_cache": True,
    }


def summary_read_inference_results(
    start_date: str | None = None,
    end_date: str | None = None,
    label: str | None = None,
    metric: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """Read the latest model inference results — aggregate stats plus recent per-image records."""
    from scripts.utils import build_aggregate_run, load_dashboard_data

    _config, runs, image_records, _log_entries = load_dashboard_data(
        query_date_start=start_date or None,
        query_date_end=end_date or None,
    )
    summary_run = build_aggregate_run(runs)
    if not summary_run:
        return {
            "status": "ok",
            "tool": "summary_read_inference_results",
            "message": "요청하신 기간에 해당하는 추론 결과가 없습니다.",
            "target_page": SUMMARY_PAGE,
            "requires_rerun": False,
            "summary": None,
            "records": [],
        }

    records = image_records
    if label:
        normalized_label = str(label).strip().lower()
        records = [r for r in records if str(r.get("label", "")).strip().lower() == normalized_label]

    records = sorted(records, key=lambda r: str(r.get("timestamp") or ""), reverse=True)
    safe_limit = min(max(int(limit or 10), 1), 100)
    top_records = records[:safe_limit]

    record_summaries = [
        {
            "run_name": record.get("run_name"),
            "label": record.get("label"),
            "filename": record.get("filename"),
            "timestamp": str(record.get("timestamp")) if record.get("timestamp") else None,
        }
        for record in top_records
    ]

    normalized_metric = str(metric or "").strip().lower()
    if normalized_metric == "good":
        message = f"정상(OK) 데이터는 {summary_run['good_count']:,}장입니다."
    elif normalized_metric == "bad":
        message = f"불량(NG) 데이터는 {summary_run['bad_count']:,}장입니다."
    elif normalized_metric == "total":
        message = f"전체 데이터는 {summary_run['total_count']:,}장입니다."
    else:
        message = (
            f"전체 {summary_run['total_count']:,}장 중 정상(OK) {summary_run['good_count']:,}장, "
            f"불량(NG) {summary_run['bad_count']:,}장, 평균 추론 시간 {summary_run['average_inference_ms']:.2f} ms"
        )
    if label:
        message += f" | label={label} 필터 적용"

    st.session_state["mcp_last_action"] = f"Summary inference results read: {message}"

    return {
        "status": "ok",
        "tool": "summary_read_inference_results",
        "message": message,
        "target_page": SUMMARY_PAGE,
        "requires_rerun": False,
        "summary": {
            "total_count": summary_run["total_count"],
            "good_count": summary_run["good_count"],
            "bad_count": summary_run["bad_count"],
            "average_inference_ms": summary_run["average_inference_ms"],
        },
        "records": record_summaries,
    }


def list_defect_types() -> dict[str, Any]:
    """List the classification model's defect (NG) categories — every class label other than
    "Normal" (the OK class)."""
    from scripts.utils import CLASS_VISUALIZATION_ORDER

    defect_types = [label for label in CLASS_VISUALIZATION_ORDER if label != "Normal"]
    message = f"불량 유형은 총 {len(defect_types)}가지입니다: " + ", ".join(defect_types) + "."

    return {
        "status": "ok",
        "tool": "list_defect_types",
        "message": message,
        "target_page": SUMMARY_PAGE,
        "requires_rerun": False,
        "defect_types": defect_types,
    }


def _validate_log_date(requested_date: str) -> str | None:
    """Return a Korean error message if `requested_date` has no log file, else None."""
    from scripts.utils import list_app_log_dates

    available_dates = list_app_log_dates()
    if requested_date in available_dates:
        return None
    options = ", ".join(available_dates) if available_dates else "조회 가능한 날짜 없음"
    return f"해당 날짜는 조회가 불가능합니다. {options} 중에서 선택해주세요."


def _validate_log_type(requested_type: str) -> str | None:
    """Return a Korean error message if `requested_type` is not a known log type, else None."""
    from scripts.utils import SEVERITY_ORDER

    valid_types = list(SEVERITY_ORDER.keys())
    if any(requested_type.lower() == valid_type.lower() for valid_type in valid_types):
        return None
    return f"해당 로그 타입은 존재하지 않습니다. {', '.join(valid_types)} 중에서 선택해주세요."


def log_read(date_value: str | None = None, log_type: str | None = None, limit: int = 20) -> dict[str, Any]:
    """Read /log entries for one date, optionally filtered by log type."""
    requested_date = str(date_value or "").strip()
    if not requested_date:
        return {
            "status": "error",
            "tool": "log_read",
            "message": "date_value는 필수입니다.",
            "requires_rerun": False,
        }

    validation_message = _validate_log_date(requested_date)
    if validation_message:
        return {
            "status": "error",
            "tool": "log_read",
            "message": validation_message,
            "requires_rerun": False,
        }

    requested_type = str(log_type or "").strip()
    if requested_type:
        validation_message = _validate_log_type(requested_type)
        if validation_message:
            return {
                "status": "error",
                "tool": "log_read",
                "message": validation_message,
                "requires_rerun": False,
            }

    from scripts.utils import load_app_logs_by_date

    entries = load_app_logs_by_date(requested_date)
    if requested_type:
        entries = [entry for entry in entries if str(entry.get("log_type", "")).lower() == requested_type.lower()]

    st.session_state["log_date_filter_requested"] = requested_date
    if requested_type:
        st.session_state["log_type_filter_requested"] = requested_type

    if not entries:
        message = f"{requested_date} 날짜에 해당하는 로그가 없습니다" + (f" (type={requested_type})" if requested_type else "") + "."
        st.session_state["mcp_last_action"] = f"Log read: date={requested_date}, type={requested_type or 'all'}, 0 entries."
        return {
            "status": "ok",
            "tool": "log_read",
            "message": message,
            "target_page": LOG_PAGE,
            "requires_rerun": False,
            "entries": [],
        }

    safe_limit = min(max(int(limit or 20), 1), 200)
    entries_for_return = entries[:safe_limit]

    st.session_state["mcp_last_action"] = (
        f"Log read: date={requested_date}, type={requested_type or 'all'}, {len(entries)} entries."
    )

    return {
        "status": "ok",
        "tool": "log_read",
        "message": f"로그 {len(entries)}건을 찾았습니다 (date={requested_date}, type={requested_type or '전체'}).",
        "target_page": LOG_PAGE,
        "requires_rerun": False,
        "entries": entries_for_return,
    }
