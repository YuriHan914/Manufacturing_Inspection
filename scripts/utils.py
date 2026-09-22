from __future__ import annotations

import base64
import csv
import json
import logging
import math
import os
import random
import re
import sys
import textwrap
import time
from collections import Counter
from datetime import datetime, timedelta
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import toml
try:
    from st_supabase_connection import SupabaseConnection
except ImportError:
    SupabaseConnection = None

from scripts.detail_finetune_mcp import (
    CLASSIFICATION_MODEL_ROOT,
    CLASSIFIER_MODEL_DIR,
    DetailFineTunePlan,
    load_available_classes,
    resolve_base_model_dir,
    run_detail_finetune_plan,
)
from scripts.local_gemma_model import (
    MODEL_DIR as DEFAULT_LLM_MODEL_DIR,
    are_runtime_dependencies_available,
    is_model_downloaded,
    list_available_model_dirs,
)
from scripts.agent_graph import handle_user_command


BASE_DIR = Path(__file__).resolve().parents[1]
SECRETS_PATH = BASE_DIR / ".streamlit" / "secrets.toml"
OUTPUT_DIR = BASE_DIR / "output"
LOG_DIR = BASE_DIR / "log"
APP_LOG_FILE_PREFIX = "dashboard"
APP_LOG_FILE_GLOB = f"{APP_LOG_FILE_PREFIX}_*.log"
APP_LOG_LINE_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
APP_LOG_PAYLOAD_FIELD_ORDER = ("source", "log_type", "content", "request", "response")
APP_LOG_PAYLOAD_FIELD_SET = set(APP_LOG_PAYLOAD_FIELD_ORDER)
SEVERITY_ORDER = {
    "error": 0,
    "Emergency": 1,
    "Warning": 2,
    "start": 3,
    "done": 4,
    "System done": 5,
    "System start": 6,
    "Model update": 7,
}
SUMMARY_ANALYSIS_SYSTEM_PROMPT = """You are a quality analyst summarizing semiconductor inspection results.
Respond only in English.
Write a concise 4 to 6 sentence analysis comment based on the provided metrics, trends, and image information.
Include the overall quality status, notable defects, trend interpretation, and immediate recommended actions.
Minimize speculation and stay grounded in the data."""
PDF_FONT_CANDIDATES = (
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
)
CLASS_VISUALIZATION_ORDER = (
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Local",
    "Near-Full",
    "Normal",
    "Scratch",
)
CLASS_VISUALIZATION_COLORS = {
    "Center": "#FF3B30",
    "Donut": "#FF9500",
    "Edge-Loc": "#34C759",
    "Edge-Ring": "#00C7BE",
    "Local": "#32ADE6",
    "Near-Full": "#007AFF",
    "Normal": "#AF52DE",
    "Scratch": "#FFD60A",
}
REQUIRED_CLASSIFIER_MODEL_FILES = (
    "model.safetensors",
    "config.json",
    "preprocessor_config.json",
    "label2id.json",
)
DEFAULT_LLM_TEMPERATURE = 0.0
DEFAULT_LLM_MAX_NEW_TOKENS = 512
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
SUPABASE_CONNECTION_NAME = "supabase"
SUPABASE_IMAGE_TABLE = "semiconductor"
SUPABASE_IMAGE_COLUMNS = "id,file_path,class,trained,predict,type,create_date"
SUPABASE_QUERY_TTL = "10m"
CSV_FALLBACK_DATA_PATH = BASE_DIR / "data" / "data.csv"


def _looks_like_project_path(value: str | Path | None) -> bool:
    if value is None:
        return False
    raw_value = str(value).strip()
    if not raw_value:
        return False
    if raw_value.startswith((".", "/", "~")):
        return True
    normalized = raw_value.replace("\\", "/")
    known_prefixes = ("model/", "output/", "pages/", "scripts/", "log/")
    if normalized.startswith(known_prefixes):
        return True
    return (BASE_DIR / raw_value).exists()


def _resolve_project_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    raw_value = str(value).strip()
    if not raw_value:
        return None
    path = Path(raw_value).expanduser()
    if path.is_absolute():
        return path
    return (BASE_DIR / path).resolve()


def _to_project_relative_path(value: str | Path | None) -> str:
    if value is None:
        return "-"
    raw_value = str(value).strip()
    if not raw_value:
        return "-"
    path = Path(raw_value).expanduser()
    absolute_path = path if path.is_absolute() else (BASE_DIR / path).resolve()
    return os.path.relpath(str(absolute_path), str(BASE_DIR))


def _format_display_path(value: str | Path | None) -> str:
    if not _looks_like_project_path(value):
        return str(value) if value is not None else "-"
    return _to_project_relative_path(value)


def _normalize_image_path_key(value: str | Path | None) -> str:
    if value is None:
        return ""
    raw_value = str(value).strip()
    if not raw_value:
        return ""
    return str(Path(raw_value).expanduser().resolve())


def _build_log_entry(
    *,
    log_type: str,
    source: str,
    content: str,
    timestamp: datetime | None = None,
    request: str = "",
    response: str = "",
) -> dict[str, str]:
    entry_timestamp = timestamp or datetime.now()
    return {
        "timestamp": entry_timestamp.isoformat(timespec="seconds"),
        "date": entry_timestamp.strftime("%Y-%m-%d"),
        "time": entry_timestamp.strftime("%H:%M:%S"),
        "source": source,
        "log_type": log_type,
        "content": content.strip(),
        "request": request.strip(),
        "response": response.strip(),
    }


def _sort_log_entries(log_entries: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(
        log_entries,
        key=lambda entry: (
            entry.get("timestamp", ""),
            -SEVERITY_ORDER.get(entry.get("log_type", ""), 99),
        ),
        reverse=True,
    )


def _append_app_log(
    *,
    log_type: str,
    source: str,
    content: str,
    request: str = "",
    response: str = "",
) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    entry_timestamp = datetime.now()
    log_path = LOG_DIR / f"{APP_LOG_FILE_PREFIX}_{entry_timestamp.strftime('%Y-%m-%d')}.log"
    logger_name = f"manufacturing_dashboard.app.{entry_timestamp.strftime('%Y-%m-%d')}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    file_handler_exists = any(
        isinstance(handler, logging.FileHandler) and Path(getattr(handler, "baseFilename", "")) == log_path
        for handler in logger.handlers
    )
    if not file_handler_exists:
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s\t%(levelname)s\t%(message)s", APP_LOG_LINE_DATE_FORMAT))
        logger.addHandler(file_handler)

    def _sanitize_log_value(value: str) -> str:
        return str(value).replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n").strip()

    payload_values = {
        "source": source,
        "log_type": log_type,
        "content": content,
        "request": request,
        "response": response,
    }
    payload = "\t".join(
        f"{field}={_sanitize_log_value(payload_values.get(field, ''))}"
        for field in APP_LOG_PAYLOAD_FIELD_ORDER
    )

    normalized_type = str(log_type).strip().lower()
    if normalized_type in {"error", "emergency"}:
        logger.error(payload)
    elif normalized_type == "warning":
        logger.warning(payload)
    else:
        logger.info(payload)

    load_dashboard_data.clear()


def _load_app_logs() -> list[dict[str, str]]:
    log_entries: list[dict[str, str]] = []
    if not LOG_DIR.exists():
        return []

    log_paths = sorted(LOG_DIR.glob(APP_LOG_FILE_GLOB), reverse=True)
    for log_path in log_paths:
        for raw_line in log_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split("\t", 2)
            if len(parts) != 3:
                continue
            raw_timestamp, _level_name, raw_payload = parts
            try:
                entry_timestamp = datetime.strptime(raw_timestamp, APP_LOG_LINE_DATE_FORMAT)
            except ValueError:
                continue

            payload_map: dict[str, str] = {}
            for token in raw_payload.split("\t"):
                if "=" not in token:
                    continue
                key, value = token.split("=", 1)
                key = key.strip()
                if key not in APP_LOG_PAYLOAD_FIELD_SET:
                    continue
                decoded_value = value.replace("\\n", "\n").replace("\\t", "\t").replace("\\\\", "\\")
                payload_map[key] = decoded_value

            log_entries.append(
                _build_log_entry(
                    log_type=str(payload_map.get("log_type", "done")),
                    source=str(payload_map.get("source", "App")),
                    content=str(payload_map.get("content", "")),
                    timestamp=entry_timestamp,
                    request=str(payload_map.get("request", "")),
                    response=str(payload_map.get("response", "")),
                )
            )
    return _sort_log_entries(log_entries)


def list_app_log_dates() -> list[str]:
    if not LOG_DIR.exists():
        return []

    pattern = re.compile(rf"^{re.escape(APP_LOG_FILE_PREFIX)}_(\d{{4}}-\d{{2}}-\d{{2}})\.log$")
    dates: set[str] = set()
    for log_path in LOG_DIR.glob(APP_LOG_FILE_GLOB):
        matched = pattern.match(log_path.name)
        if matched:
            dates.add(matched.group(1))
    return sorted(dates, reverse=True)


def load_app_logs_by_date(selected_date: str | None = None) -> list[dict[str, str]]:
    logs = _load_app_logs()
    if not selected_date:
        return logs
    return [entry for entry in logs if entry.get("date") == selected_date]


def _suppress_transformers_path_alias_warning() -> None:
    logger = logging.getLogger("transformers")
    if any(getattr(current_filter, "_manufacture_path_alias_filter", False) for current_filter in logger.filters):
        return

    class _TransformersPathAliasFilter(logging.Filter):
        _manufacture_path_alias_filter = True

        def filter(self, record: logging.LogRecord) -> bool:
            message = record.getMessage()
            if "Accessing `__path__`" in message and "alias will be removed in future versions" in message:
                return False
            return True

    logger.addFilter(_TransformersPathAliasFilter())


def configure_page(page_title: str) -> None:
    st.set_page_config(
        page_title=page_title,
        layout="wide",
        initial_sidebar_state="expanded",
    )


def read_json_file(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def _parse_record_timestamp(value: Any, fallback: datetime) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None) if value.tzinfo else value

    raw_value = str(value).strip() if value is not None else ""
    if not raw_value:
        return fallback

    normalized_value = raw_value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized_value)
    except ValueError:
        return fallback

    if parsed.tzinfo is not None:
        return parsed.astimezone().replace(tzinfo=None)
    return parsed


def _normalize_db_label(value: Any, fallback: str) -> str:
    raw_value = str(value).strip() if value is not None else ""
    return raw_value or fallback


def _parse_bool_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    return normalized in {"1", "true", "t", "yes", "y"}


def _fetch_csv_fallback_rows(
    query_date_start: str | None = None,
    query_date_end: str | None = None,
) -> list[dict[str, Any]]:
    if not CSV_FALLBACK_DATA_PATH.exists():
        raise RuntimeError(f"The CSV fallback file does not exist: {CSV_FALLBACK_DATA_PATH}")

    date_start = None
    date_end = None
    if query_date_start and query_date_end:
        try:
            date_start = datetime.fromisoformat(query_date_start)
            date_end = datetime.fromisoformat(query_date_end) + timedelta(days=1)
        except ValueError:
            date_start = None
            date_end = None

    rows: list[dict[str, Any]] = []
    with CSV_FALLBACK_DATA_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not isinstance(row, dict):
                continue

            if _parse_bool_flag(row.get("trained", False)):
                continue

            row_timestamp = _parse_record_timestamp(row.get("created_at"), datetime.now())
            if date_start is not None and date_end is not None:
                if row_timestamp < date_start or row_timestamp >= date_end:
                    continue

            rows.append(
                {
                    "id": row.get("id"),
                    "image_path": row.get("image_path"),
                    "class": row.get("class"),
                    "trained": _parse_bool_flag(row.get("trained", False)),
                    "predict": row.get("predict"),
                    "type": row.get("type") or row.get("img_type"),
                    "created_at": row.get("created_at"),
                }
            )
    return rows


def _fetch_supabase_semiconductor_rows(query_date_start: str | None = None, query_date_end: str | None = None) -> list[dict[str, Any]]:
    if SupabaseConnection is None:
        raise RuntimeError("The `st_supabase_connection` package could not be found.")

    connection = st.connection(SUPABASE_CONNECTION_NAME, type=SupabaseConnection)
    last_error: Exception | None = None

    date_start = None
    date_end = None
    if query_date_start and query_date_end:
        try:
            date_start = datetime.fromisoformat(query_date_start)
            date_end = datetime.fromisoformat(query_date_end) + timedelta(days=1)
        except ValueError:
            date_start = None
            date_end = None

    try:
        query_builder = connection.query(
            SUPABASE_IMAGE_COLUMNS,
            table=SUPABASE_IMAGE_TABLE,
            ttl=SUPABASE_QUERY_TTL,
        )
        if hasattr(query_builder, "eq"):
            query_builder = query_builder.eq("trained", False)
        if date_start is not None and date_end is not None:
            if hasattr(query_builder, "gte"):
                query_builder = query_builder.gte("create_date", date_start.isoformat())
            if hasattr(query_builder, "lt"):
                query_builder = query_builder.lt("create_date", date_end.isoformat())
        if hasattr(query_builder, "order"):
            query_builder = query_builder.order("create_date", desc=False)
        result = query_builder.execute()
    except Exception as exc:
        last_error = exc
        client = getattr(connection, "client", None)
        if client is None:
            raise

        query_builder = client.table(SUPABASE_IMAGE_TABLE).select(SUPABASE_IMAGE_COLUMNS).eq("trained", False)
        if date_start is not None and date_end is not None:
            if hasattr(query_builder, "gte"):
                query_builder = query_builder.gte("create_date", date_start.isoformat())
            if hasattr(query_builder, "lt"):
                query_builder = query_builder.lt("create_date", date_end.isoformat())
        if hasattr(query_builder, "order"):
            query_builder = query_builder.order("create_date", desc=False)
        result = query_builder.execute()

    rows = getattr(result, "data", result)
    if not rows:
        return []

    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            normalized_rows.append(row)
            continue
        if hasattr(row, "items"):
            normalized_rows.append(dict(row.items()))
            continue
        normalized_rows.append(dict(row))

    if last_error is not None:
        print(f"Supabase query fallback applied after primary query failed: {last_error}")
    return normalized_rows


def _load_supabase_image_candidates(
    reference_time: datetime,
    query_date_start: str | None = None,
    query_date_end: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rows = _fetch_supabase_semiconductor_rows(query_date_start=query_date_start, query_date_end=query_date_end)
    discovered_images: list[dict[str, Any]] = []
    warning_logs: list[dict[str, str]] = []
    skipped_missing_paths = 0
    skipped_invalid_paths = 0

    for row in rows:
        raw_image_path = str(row.get("file_path") or "").strip()
        if not raw_image_path:
            skipped_invalid_paths += 1
            continue

        image_path = Path(raw_image_path).expanduser()
        if image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            skipped_invalid_paths += 1
            continue
        if not image_path.exists() or not image_path.is_file():
            skipped_missing_paths += 1
            continue

        source_label = _normalize_db_label(row.get("class"), image_path.parent.name)
        database_predict = str(row.get("predict") or "").strip() or None
        created_at = _parse_record_timestamp(row.get("create_date"), reference_time)
        discovered_images.append(
            {
                "record_id": row.get("id"),
                "path": image_path,
                "source_label": source_label,
                "database_predict": database_predict,
                "timestamp": created_at,
                "dataset_type": str(row.get("type") or "").strip() or None,
                "trained": bool(row.get("trained", False)),
            }
        )

    if skipped_missing_paths:
        warning_logs.append(
            _build_log_entry(
                log_type="Warning",
                source="Supabase",
                content=(
                    f"{skipped_missing_paths} image entries loaded from the semiconductor table were skipped "
                    "because the local absolute-path files were not found."
                ),
                timestamp=reference_time,
            )
        )
    if skipped_invalid_paths:
        warning_logs.append(
            _build_log_entry(
                log_type="Warning",
                source="Supabase",
                content=(
                    f"{skipped_invalid_paths} image entries loaded from the semiconductor table were skipped "
                    "because the `file_path` format was invalid."
                ),
                timestamp=reference_time,
            )
        )

    discovered_images.sort(key=lambda item: (item["timestamp"], str(item["path"])))
    return discovered_images, warning_logs


def _load_csv_image_candidates(
    reference_time: datetime,
    query_date_start: str | None = None,
    query_date_end: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rows = _fetch_csv_fallback_rows(query_date_start=query_date_start, query_date_end=query_date_end)
    discovered_images: list[dict[str, Any]] = []
    warning_logs: list[dict[str, str]] = []
    skipped_missing_paths = 0
    skipped_invalid_paths = 0

    for row in rows:
        raw_image_path = str(row.get("image_path") or "").strip()
        if not raw_image_path:
            skipped_invalid_paths += 1
            continue

        raw_path = Path(raw_image_path).expanduser()
        image_path = raw_path if raw_path.is_absolute() else (BASE_DIR / raw_path).resolve()

        if image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            skipped_invalid_paths += 1
            continue
        if not image_path.exists() or not image_path.is_file():
            skipped_missing_paths += 1
            continue

        source_label = _normalize_db_label(row.get("class"), image_path.parent.name)
        database_predict = str(row.get("predict") or "").strip() or None
        created_at = _parse_record_timestamp(row.get("created_at"), reference_time)
        discovered_images.append(
            {
                "record_id": row.get("id"),
                "path": image_path,
                "source_label": source_label,
                "database_predict": database_predict,
                "timestamp": created_at,
                "dataset_type": str(row.get("type") or "").strip() or None,
                "trained": bool(row.get("trained", False)),
            }
        )

    if skipped_missing_paths:
        warning_logs.append(
            _build_log_entry(
                log_type="Warning",
                source="CSV",
                content=(
                    f"{skipped_missing_paths} image entries loaded from the CSV fallback were skipped "
                    "because the local files were not found."
                ),
                timestamp=reference_time,
            )
        )
    if skipped_invalid_paths:
        warning_logs.append(
            _build_log_entry(
                log_type="Warning",
                source="CSV",
                content=(
                    f"{skipped_invalid_paths} image entries loaded from the CSV fallback were skipped "
                    "because the `image_path` format was invalid."
                ),
                timestamp=reference_time,
            )
        )

    discovered_images.sort(key=lambda item: (item["timestamp"], str(item["path"])))
    return discovered_images, warning_logs


def _get_available_llm_model_dirs() -> list[Path]:
    available_model_dirs = list_available_model_dirs()
    if available_model_dirs:
        return available_model_dirs
    if DEFAULT_LLM_MODEL_DIR.exists():
        return [DEFAULT_LLM_MODEL_DIR]
    return []


def _initialize_llm_runtime_state() -> dict[str, Any]:
    available_model_dirs = _get_available_llm_model_dirs()
    available_values = [str(path) for path in available_model_dirs]
    default_model_value = str(available_model_dirs[0] if available_model_dirs else DEFAULT_LLM_MODEL_DIR)

    selected_model_value = st.session_state.get("llm_model_dir", default_model_value)
    if available_values and selected_model_value not in available_values:
        selected_model_value = default_model_value
    elif not available_values:
        selected_model_value = default_model_value

    temperature = float(st.session_state.get("llm_temperature", DEFAULT_LLM_TEMPERATURE))
    max_new_tokens = int(st.session_state.get("llm_max_new_tokens", DEFAULT_LLM_MAX_NEW_TOKENS))
    st.session_state["llm_model_dir"] = selected_model_value
    st.session_state["llm_temperature"] = max(0.0, temperature)
    st.session_state["llm_max_new_tokens"] = max(1, max_new_tokens)

    if "llm_model_dir_pending" not in st.session_state:
        st.session_state["llm_model_dir_pending"] = selected_model_value
    elif available_values and st.session_state["llm_model_dir_pending"] not in available_values:
        st.session_state["llm_model_dir_pending"] = selected_model_value

    if "llm_temperature_pending" not in st.session_state:
        st.session_state["llm_temperature_pending"] = st.session_state["llm_temperature"]
    if "llm_max_new_tokens_pending" not in st.session_state:
        st.session_state["llm_max_new_tokens_pending"] = st.session_state["llm_max_new_tokens"]

    return {
        "available_model_dirs": available_model_dirs,
        "default_model_value": default_model_value,
    }


def _get_llm_runtime_settings() -> dict[str, Any]:
    runtime_state = _initialize_llm_runtime_state()
    available_model_dirs = runtime_state["available_model_dirs"]
    return {
        "model_dir": st.session_state["llm_model_dir"],
        "temperature": max(0.0, float(st.session_state["llm_temperature"])),
        "max_new_tokens": max(1, int(st.session_state["llm_max_new_tokens"])),
        "available_model_dirs": available_model_dirs,
    }


def _get_pending_llm_runtime_settings() -> dict[str, Any]:
    runtime_state = _initialize_llm_runtime_state()
    available_model_dirs = runtime_state["available_model_dirs"]
    return {
        "model_dir": st.session_state["llm_model_dir_pending"],
        "temperature": max(0.0, float(st.session_state["llm_temperature_pending"])),
        "max_new_tokens": max(1, int(st.session_state["llm_max_new_tokens_pending"])),
        "available_model_dirs": available_model_dirs,
    }


def _get_discrete_class_colors(labels: list[str]) -> dict[str, str]:
    label_set = set(labels)
    ordered_labels = [label for label in CLASS_VISUALIZATION_ORDER if label in label_set]
    ordered_labels.extend(sorted(label_set - set(ordered_labels)))

    fallback_colors = (
        "#FF2D55",
        "#FF9F0A",
        "#64D2FF",
        "#30D158",
        "#5E5CE6",
        "#BF5AF2",
        "#FF375F",
        "#66D4CF",
    )
    color_map: dict[str, str] = {}
    fallback_index = 0
    for label in ordered_labels:
        if label in CLASS_VISUALIZATION_COLORS:
            color_map[label] = CLASS_VISUALIZATION_COLORS[label]
        else:
            color_map[label] = fallback_colors[fallback_index % len(fallback_colors)]
            fallback_index += 1
    return color_map


def _is_valid_classifier_model_dir(model_dir: Path) -> bool:
    return model_dir.is_dir() and all((model_dir / filename).exists() for filename in REQUIRED_CLASSIFIER_MODEL_FILES)


def _list_available_classifier_model_dirs() -> list[Path]:
    if not CLASSIFICATION_MODEL_ROOT.exists():
        return []
    return sorted(
        (path for path in CLASSIFICATION_MODEL_ROOT.iterdir() if _is_valid_classifier_model_dir(path)),
        key=lambda path: path.name,
    )


def _get_default_detail_inference_model_dir(selected_records: list[dict[str, Any]]) -> Path:
    available_model_dirs = _list_available_classifier_model_dirs()
    preferred_candidates: list[Path] = [
        resolve_base_model_dir(record.get("model_dir"))
        for record in selected_records
    ]

    preferred_candidates.append(resolve_base_model_dir(CLASSIFIER_MODEL_DIR))

    for candidate in preferred_candidates:
        if _is_valid_classifier_model_dir(candidate):
            return candidate

    if available_model_dirs:
        return available_model_dirs[0]
    return resolve_base_model_dir(CLASSIFIER_MODEL_DIR)


def _render_classifier_model_selector(
    selected_records: list[dict[str, Any]],
    container: Any,
    selector_key: str,
    active_key: str,
    section_title: str | None = None,
    helper_text: str | None = None,
    label: str = "Inference model",
    add_divider: bool = False,
) -> tuple[Path | None, bool]:
    available_model_dirs = _list_available_classifier_model_dirs()
    if add_divider:
        container.divider()
    if section_title:
        container.subheader(section_title)
    if helper_text:
        container.caption(helper_text)

    if not available_model_dirs:
        container.warning(f"Classifier model folder not found: {_to_project_relative_path(CLASSIFICATION_MODEL_ROOT)}")
        return None, False

    default_model_dir = _get_default_detail_inference_model_dir(selected_records)
    available_values = [str(path) for path in available_model_dirs]
    if st.session_state.get(selector_key) not in available_values:
        st.session_state[selector_key] = str(default_model_dir)

    selected_value = container.selectbox(
        label,
        available_values,
        format_func=lambda value: Path(value).name,
        key=selector_key,
    )
    container.caption(f"Model path: {_to_project_relative_path(selected_value)}")

    previous_value = st.session_state.get(active_key)
    model_changed = previous_value is not None and previous_value != selected_value
    st.session_state[active_key] = selected_value
    return Path(selected_value), model_changed


def _get_detail_inference_signature(
    selected_records: list[dict[str, Any]],
    model_dir: Path,
) -> tuple[str, tuple[str, ...]]:
    resolved_model_dir = resolve_base_model_dir(model_dir)
    return (
        str(resolved_model_dir),
        tuple(record["path"] for record in selected_records),
    )


def _get_cached_detail_inference_result(
    selected_records: list[dict[str, Any]],
    model_dir: Path,
) -> tuple[list[dict[str, Any]] | None, list[str], dict[str, str] | None]:
    signature = _get_detail_inference_signature(selected_records, model_dir)
    if st.session_state.get("detail_inference_prediction_signature") != signature:
        return None, [], None

    artifact_paths: dict[str, str] | None = None
    output_dir = st.session_state.get("detail_inference_prediction_output_dir")
    results_path = st.session_state.get("detail_inference_prediction_results_path")
    timing_path = st.session_state.get("detail_inference_prediction_timing_path")
    if output_dir and results_path and timing_path:
        artifact_paths = {
            "output_dir": str(output_dir),
            "results_path": str(results_path),
            "timing_path": str(timing_path),
        }

    return (
        list(st.session_state.get("detail_inference_prediction_records", [])),
        list(st.session_state.get("detail_inference_prediction_errors", [])),
        artifact_paths,
    )


@st.cache_resource(show_spinner=False)
def _load_dashboard_classifier_runtime(model_dir_value: str) -> tuple[Any, str]:
    """Default classifier runtime used to auto-label freshly discovered images.

    Always serves the fixed GPU(TensorRT)/CPU(ONNX) MobileViT export from
    scripts/classifier_runtime.py, regardless of `model_dir_value`; the argument is
    kept only so the cache key/log messages stay attributable to `default_model_dir` the way the
    rest of load_dashboard_data expects.
    """
    from scripts.classifier_runtime import get_default_classifier_runtime

    runtime = get_default_classifier_runtime()
    return runtime, runtime.device


@st.cache_data(show_spinner=False)
def _predict_dashboard_labels(
    image_paths: tuple[str, ...],
    model_dir_value: str,
) -> tuple[dict[str, str], float, float]:
    if not image_paths:
        return {}, 0.0, 0.0

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("The Pillow package required for dashboard inference is not installed.") from exc

    runtime, _device = _load_dashboard_classifier_runtime(model_dir_value)

    predicted_labels: dict[str, str] = {}
    total_process_ms = 0.0
    total_inference_ms = 0.0

    for image_path in image_paths:
        process_start_ns = time.perf_counter_ns()

        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")

        pixel_values = runtime.preprocess(rgb_image)

        inference_start_ns = time.perf_counter_ns()
        predicted_labels[image_path] = runtime.infer_label(pixel_values)
        inference_ms = (time.perf_counter_ns() - inference_start_ns) / 1_000_000.0

        process_ms = (time.perf_counter_ns() - process_start_ns) / 1_000_000.0
        total_inference_ms += inference_ms
        total_process_ms += process_ms

    average_inference_ms = total_inference_ms / len(image_paths) if image_paths else 0.0
    return predicted_labels, average_inference_ms, total_process_ms


@st.cache_data(show_spinner=False, ttl=600)
def load_dashboard_data(
    query_date_start: str | None = None,
    query_date_end: str | None = None,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, str]],
]:
    runs: list[dict[str, Any]] = []
    image_records: list[dict[str, Any]] = []
    now = datetime.now()
    default_model_dir = resolve_base_model_dir(CLASSIFIER_MODEL_DIR)
    data_source = "supabase"
    source_warning_logs: list[dict[str, str]] = []

    effective_query_date_start = query_date_start
    effective_query_date_end = query_date_end
    if not effective_query_date_start:
        session_query_date_start = st.session_state.get("dashboard_query_date_start", "")
        effective_query_date_start = str(session_query_date_start).strip() or None
    if not effective_query_date_end:
        session_query_date_end = st.session_state.get("dashboard_query_date_end", "")
        effective_query_date_end = str(session_query_date_end).strip() or None

    try:
        discovered_images, source_warning_logs = _load_supabase_image_candidates(
            now,
            query_date_start=effective_query_date_start,
            query_date_end=effective_query_date_end,
        )
    except Exception as exc:
        try:
            discovered_images, csv_warning_logs = _load_csv_image_candidates(
                now,
                query_date_start=effective_query_date_start,
                query_date_end=effective_query_date_end,
            )
            data_source = "csv_fallback"
            source_warning_logs = [
                _build_log_entry(
                    log_type="Warning",
                    source="Supabase",
                    content=(
                        "Supabase semiconductor integration failed, so the app switched to the `data/data.csv` fallback. "
                        f"error={exc}"
                    ),
                    timestamp=now,
                ),
                *csv_warning_logs,
            ]
        except Exception as csv_exc:
            data_source = "supabase_unavailable"
            discovered_images = []
            source_warning_logs = [
                _build_log_entry(
                    log_type="Warning",
                    source="Supabase",
                    content=(
                        "Supabase semiconductor integration failed, and the CSV fallback also failed. "
                        f"supabase_error={exc} | csv_error={csv_exc}"
                    ),
                    timestamp=now,
                )
            ]

    config = {
        "data_source": data_source,
        "supabase_table": SUPABASE_IMAGE_TABLE,
        "supabase_filter": "trained = false",
        "query_date_start": effective_query_date_start or "all",
        "query_date_end": effective_query_date_end or "all",
        "model_name": _to_project_relative_path(default_model_dir),
    }
    prediction_warning: dict[str, str] | None = None
    predicted_labels_by_path: dict[str, str] = {}
    average_inference_ms = 0.0
    total_process_ms = 0.0

    if discovered_images:
        image_paths = tuple(str(image_record["path"]) for image_record in discovered_images)
        try:
            predicted_labels_by_path, average_inference_ms, total_process_ms = _predict_dashboard_labels(
                image_paths,
                str(default_model_dir),
            )
        except Exception as exc:
            prediction_warning = _build_log_entry(
                log_type="Warning",
                source="Inference",
                content=(
                    "Default classifier inference failed, so the stored label values were used instead. "
                    f"model={_to_project_relative_path(default_model_dir)} | error={exc}"
                ),
                timestamp=now,
            )

    predicted_run_counts: Counter[str] = Counter()
    predicted_good_counts: Counter[str] = Counter()
    predicted_bad_counts: Counter[str] = Counter()
    predicted_latest_timestamps: dict[str, datetime] = {}

    for candidate in discovered_images:
        image_path = Path(candidate["path"])
        source_label = str(candidate["source_label"])
        database_predict = str(candidate.get("database_predict") or "").strip()
        predicted_label = predicted_labels_by_path.get(str(image_path))
        prediction_source = "default_model"
        if not predicted_label:
            if database_predict:
                predicted_label = database_predict
                prediction_source = "supabase_predict_fallback"
            else:
                predicted_label = source_label
                prediction_source = "source_label_fallback"

        display_path = str(image_path)
        record_timestamp = candidate["timestamp"]

        image_records.append(
            {
                "run_name": predicted_label,
                "timestamp": record_timestamp,
                "date": record_timestamp.strftime("%Y-%m-%d"),
                "label": predicted_label,
                "source_label": source_label,
                "prediction_source": prediction_source,
                "path": str(image_path),
                "display_path": display_path,
                "filename": image_path.name,
                "record_id": candidate.get("record_id"),
                "trained": bool(candidate.get("trained", False)),
                "dataset_type": candidate.get("dataset_type"),
                "database_predict": database_predict or None,
                "data_source": data_source,
                "model_dir": str(default_model_dir),
                "model_dir_display": _to_project_relative_path(default_model_dir),
                "exists": True,
            }
        )

        predicted_run_counts[predicted_label] += 1
        previous_timestamp = predicted_latest_timestamps.get(predicted_label)
        if previous_timestamp is None or record_timestamp > previous_timestamp:
            predicted_latest_timestamps[predicted_label] = record_timestamp
        if predicted_label == "Normal":
            predicted_good_counts[predicted_label] += 1
        else:
            predicted_bad_counts[predicted_label] += 1

    per_image_process_ms = (total_process_ms / len(discovered_images)) if discovered_images else 0.0
    for predicted_label in sorted(predicted_run_counts):
        image_count = int(predicted_run_counts[predicted_label])
        runs.append(
            {
                "name": predicted_label,
                "timestamp": predicted_latest_timestamps.get(predicted_label, now),
                "path": "",
                "total_count": image_count,
                "label_counts": {predicted_label: image_count},
                "good_count": int(predicted_good_counts.get(predicted_label, 0)),
                "bad_count": int(predicted_bad_counts.get(predicted_label, 0)),
                "model_dir": str(default_model_dir),
                "average_inference_ms": float(average_inference_ms),
                "total_process_ms": float(per_image_process_ms * image_count),
            }
        )

    runs.sort(key=lambda x: x["name"])
    image_records.sort(key=lambda x: (x["run_name"], x["timestamp"], x["filename"]))

    base_logs = [*_load_app_logs()]
    base_logs.extend(source_warning_logs)
    if prediction_warning:
        base_logs.append(prediction_warning)
    logs = _sort_log_entries(base_logs)
    print(
        "Loaded "
        f"{len(runs)} predicted groups and {len(image_records)} images from {data_source} "
        f"using {default_model_dir.name}."
    )
   
    return config, runs, image_records, logs

def build_label_distribution_frame(latest_run: dict[str, Any] | None) -> pd.DataFrame:
    if not latest_run:
        return pd.DataFrame({"label": ["No data"], "count": [0]}).set_index("label")

    label_counts = latest_run["label_counts"]
    if not label_counts:
        return pd.DataFrame({"label": ["No data"], "count": [0]}).set_index("label")

    return pd.DataFrame(
        [{"label": label, "count": count} for label, count in label_counts.items()]
    ).set_index("label")


def render_class_distribution_chart(label_frame: pd.DataFrame, container: Any | None = None) -> None:
    target = container or st
    if label_frame.empty:
        target.info("No class distribution data available.")
        return

    labels = [str(index) for index in label_frame.index]
    counts = [int(row["count"]) for _, row in label_frame.iterrows()]
    if labels == ["No data"]:
        target.info("No class distribution data available.")
        return

    total_count = sum(max(count, 0) for count in counts)
    bar_text = [
        f"{count} ({(count / total_count * 100):.0f}%)" if total_count > 0 else f"{count} (0%)"
        for count in counts
    ]

    try:
        import plotly.graph_objects as go
    except ImportError:
        target.bar_chart(label_frame, width="stretch")
        return

    color_map = _get_discrete_class_colors(labels)
    figure = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=counts,
                marker_color=[color_map.get(label, "#5B8FF9") for label in labels],
                text=bar_text,
                textposition="inside",
                textangle=-90,
                insidetextanchor="middle",
                textfont=dict(size=14),
                hovertemplate="Class: %{x}<br>Count: %{y}<br>Ratio: %{text}<extra></extra>",
                showlegend=False,
            )
        ]
    )
    figure.update_layout(
        xaxis_title="Class",
        yaxis_title="Count",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    target.plotly_chart(figure, width="stretch")


def _summarize_label_counts(label_counts: dict[str, int], limit: int = 6) -> str:
    if not label_counts:
        return "-"

    ordered_items = sorted(label_counts.items(), key=lambda item: (-int(item[1]), item[0]))
    summary = ", ".join(f"{label}={count}" for label, count in ordered_items[:limit])
    if len(ordered_items) > limit:
        summary += ", ..."
    return summary


def _collect_records_for_paths(
    image_records: list[dict[str, Any]],
    selected_paths: list[str],
) -> list[dict[str, Any]]:
    if not selected_paths:
        return []

    selected_path_set = set(selected_paths)
    ordered_records: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for record in image_records:
        path = str(record.get("path", ""))
        if path in selected_path_set and path not in seen_paths:
            ordered_records.append(record)
            seen_paths.add(path)
    return ordered_records


def _submit_sidebar_prompt() -> None:
    """on_click for the sidebar Send button — runs before the script reruns, so this is the
    only safe place to clear the "gemma_sidebar_prompt" text_area's own session_state value.
    (Setting it later, after the widget has already been instantiated this run, raises a
    StreamlitAPIException.) The submitted text is stashed in a separate key for the
    `if send_clicked:` block below to pick up."""
    st.session_state["gemma_sidebar_pending_prompt"] = st.session_state.get("gemma_sidebar_prompt", "").strip()
    st.session_state["gemma_sidebar_prompt"] = ""


def render_sidebar_llm_panel(current_page_title: str | None = None) -> None:
    st.session_state.setdefault("gemma_sidebar_status", "idle")
    st.session_state.setdefault("gemma_sidebar_notice", "")
    st.session_state.setdefault("gemma_sidebar_response", "")
    st.session_state.setdefault("gemma_sidebar_prompt", "")
    st.session_state.setdefault("gemma_sidebar_pending_prompt", "")
    llm_settings = _get_llm_runtime_settings()
    selected_model_dir = Path(str(llm_settings["model_dir"]))
    selected_model_name = selected_model_dir.name

    with st.sidebar:
        # The app uses a pre-downloaded local Gemma model from 05_Manufacutre/model/google__gemma-4-E2B-it.
        st.divider()
        st.subheader(selected_model_name)
        st.caption(
            f"Temp {float(llm_settings['temperature']):.1f} | Max tokens {int(llm_settings['max_new_tokens'])}"
        )
        dependency_ready, dependency_message = are_runtime_dependencies_available()
        model_ready = is_model_downloaded(selected_model_dir)

        st.text_area(
            "Command",
            key="gemma_sidebar_prompt",
            height=140,
            placeholder="Enter the command or question to send to Gemma 4 E2B.",
        )
        send_clicked = st.button(
            "Send",
            key="gemma_sidebar_send",
            type="primary",
            width="stretch",
            on_click=_submit_sidebar_prompt,
        )

        if send_clicked:
            prompt_text = st.session_state.get("gemma_sidebar_pending_prompt", "").strip()
            if not prompt_text:
                st.session_state["gemma_sidebar_status"] = "error"
                st.session_state["gemma_sidebar_notice"] = "Please enter a question or command first."
                st.session_state["gemma_sidebar_response"] = ""
                _append_app_log(
                    log_type="error",
                    source="Sidebar LLM",
                    content="Sidebar LLM request failed because the command field was empty.",
                    request="",
                    response=st.session_state["gemma_sidebar_notice"],
                )
            else:
                st.session_state["gemma_sidebar_status"] = "running"
                st.session_state["gemma_sidebar_notice"] = f"{selected_model_name} is processing the command."
                st.session_state["gemma_sidebar_response"] = ""
                _append_app_log(
                    log_type="start",
                    source="Sidebar LLM",
                    content=f"Sidebar LLM request started with model `{_to_project_relative_path(selected_model_dir)}`.",
                    request=prompt_text,
                )

                try:
                    with st.spinner("Running the LLM..."):
                        if not dependency_ready:
                            st.session_state["gemma_sidebar_status"] = "error"
                            st.session_state["gemma_sidebar_notice"] = dependency_message
                            st.session_state["gemma_sidebar_response"] = ""
                            _append_app_log(
                                log_type="error",
                                source="Sidebar LLM",
                                content="Sidebar LLM request failed because runtime dependencies are unavailable.",
                                request=prompt_text,
                                response=dependency_message,
                            )
                        elif not model_ready:
                            st.session_state["gemma_sidebar_status"] = "error"
                            st.session_state["gemma_sidebar_notice"] = "로컬 모델이 아직 준비되지 않았습니다. `model` 폴더에 모델을 먼저 배치해주세요."
                            st.session_state["gemma_sidebar_response"] = ""
                            _append_app_log(
                                log_type="error",
                                source="Sidebar LLM",
                                content=f"Sidebar LLM request failed because model `{_to_project_relative_path(selected_model_dir)}` is unavailable.",
                                request=prompt_text,
                                response=st.session_state["gemma_sidebar_notice"],
                            )
                        else:
                            tool_result = handle_user_command(prompt_text)
                            if tool_result.get("clear_dashboard_cache"):
                                load_dashboard_data.clear()
                                st.cache_data.clear()

                            answer = str(tool_result.get("message", ""))
                            status = tool_result.get("status")

                            if status == "needs_input":
                                st.session_state["gemma_sidebar_status"] = "completed"
                                st.session_state["gemma_sidebar_notice"] = "추가 정보가 필요합니다. 이어서 답변을 입력해주세요."
                                st.session_state["gemma_sidebar_response"] = answer
                                _append_app_log(
                                    log_type="start",
                                    source="Sidebar Agent",
                                    content="Agent asked a clarifying question.",
                                    request=prompt_text,
                                    response=answer,
                                )
                            elif status == "ok":
                                st.session_state["gemma_sidebar_status"] = "completed"
                                st.session_state["gemma_sidebar_notice"] = "요청하신 작업이 처리되었습니다."
                                st.session_state["gemma_sidebar_response"] = answer
                                _append_app_log(
                                    log_type="done",
                                    source="Sidebar Agent",
                                    content=(
                                        f"Tool `{tool_result.get('tool')}` completed with arguments "
                                        f"{tool_result.get('invoked_args', {})}."
                                    ),
                                    request=prompt_text,
                                    response=answer,
                                )
                                target_page = str(tool_result.get("target_page") or "").strip()
                                if target_page:
                                    st.switch_page(target_page)
                            elif status == "unsupported":
                                st.session_state["gemma_sidebar_status"] = "completed"
                                st.session_state["gemma_sidebar_notice"] = "지원하지 않는 요청입니다."
                                st.session_state["gemma_sidebar_response"] = answer
                                _append_app_log(
                                    log_type="done",
                                    source="Sidebar Agent",
                                    content="Agent found no matching tool for the request.",
                                    request=prompt_text,
                                    response=answer,
                                )
                            else:
                                st.session_state["gemma_sidebar_status"] = "error"
                                st.session_state["gemma_sidebar_notice"] = "요청 처리 중 오류가 발생했습니다."
                                st.session_state["gemma_sidebar_response"] = answer
                                _append_app_log(
                                    log_type="error",
                                    source="Sidebar Agent",
                                    content=(
                                        f"Tool `{tool_result.get('tool')}` failed with arguments "
                                        f"{tool_result.get('invoked_args', {})}."
                                    ),
                                    request=prompt_text,
                                    response=answer,
                                )
                except Exception as exc:
                    st.session_state["gemma_sidebar_status"] = "error"
                    st.session_state["gemma_sidebar_notice"] = "An error occurred while calling the LLM."
                    st.session_state["gemma_sidebar_response"] = str(exc)
                    _append_app_log(
                        log_type="error",
                        source="Sidebar LLM",
                        content=f"Sidebar LLM request failed with model `{_to_project_relative_path(selected_model_dir)}`.",
                        request=prompt_text,
                        response=str(exc),
                    )
            st.rerun()

        status = st.session_state["gemma_sidebar_status"]
        if status in {"running", "downloading"}:
            st.info(st.session_state["gemma_sidebar_notice"])
        elif status == "completed":
            st.success(st.session_state["gemma_sidebar_notice"])
        elif status == "error":
            st.error(st.session_state["gemma_sidebar_notice"])
        else:
            st.caption(st.session_state["gemma_sidebar_notice"])

        st.text_area(
            "Response box",
            value=st.session_state["gemma_sidebar_response"],
            height=220,
            disabled=True,
            placeholder="The model response will appear here.",
        )


def render_page_header(title: str, caption: str | None = None) -> None:
    st.session_state["current_page_title"] = title
    render_sidebar_llm_panel(title)
    st.title(title)
    if caption:
        st.caption(caption)


def build_aggregate_run(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not runs:
        return None

    latest_timestamp = max(run["timestamp"] for run in runs)

    total_count = sum(int(run.get("total_count", 0)) for run in runs)
    good_count = sum(int(run.get("good_count", 0)) for run in runs)
    bad_count = sum(int(run.get("bad_count", 0)) for run in runs)

    average_values = [float(run.get("average_inference_ms", 0.0)) for run in runs]
    total_process_values = [float(run.get("total_process_ms", 0.0)) for run in runs]

    label_counts: dict[str, int] = {}
    for run in runs:
        for label, count in run.get("label_counts", {}).items():
            label_counts[label] = label_counts.get(label, 0) + int(count)

    aggregate_run = {
        "name": "All Data",
        "timestamp": latest_timestamp,
        "path": "",
        "total_count": total_count,
        "label_counts": dict(sorted(label_counts.items(), key=lambda item: (-item[1], item[0]))),
        "good_count": good_count,
        "bad_count": bad_count,
        "model_dir": "",
        "average_inference_ms": sum(average_values) / len(average_values) if average_values else 0.0,
        "total_process_ms": sum(total_process_values),
    }
    return aggregate_run


CLASSIFICATION_INFERENCE_CACHE_PATH = BASE_DIR / "outputs" / "classification" / "inference_cache.csv"


def _read_classification_inference_cache() -> dict[str, str]:
    """Read the persisted {project-relative image path: predicted label} cache, if any."""
    if not CLASSIFICATION_INFERENCE_CACHE_PATH.exists():
        return {}
    try:
        frame = pd.read_csv(CLASSIFICATION_INFERENCE_CACHE_PATH, dtype=str)
    except Exception:
        return {}
    if "path" not in frame.columns or "label" not in frame.columns:
        return {}
    return {
        str(row["path"]): str(row["label"])
        for _, row in frame.iterrows()
        if pd.notna(row["path"]) and pd.notna(row["label"])
    }


def _write_classification_inference_cache(cache: dict[str, str]) -> None:
    CLASSIFICATION_INFERENCE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(sorted(cache.items()), columns=["path", "label"])
    frame.to_csv(CLASSIFICATION_INFERENCE_CACHE_PATH, index=False)


def _run_classification_inference_for_paths(
    image_paths: list[str],
) -> tuple[dict[str, str], int, int]:
    """Classify `image_paths` with the default GPU(TensorRT)/CPU(ONNX) runtime.

    Paths already present in CLASSIFICATION_INFERENCE_CACHE_PATH are reused as-is; only paths
    missing from the cache are actually run through the model, and the cache is updated with any
    newly computed labels so a later call over the same images does no model work at all.

    Returns (label_by_path for every resolvable path in image_paths, cached_count, inferred_count).
    """
    from scripts.classifier_runtime import get_default_classifier_runtime
    from PIL import Image

    cache = _read_classification_inference_cache()
    relative_paths = {path: _to_project_relative_path(path) for path in image_paths}
    missing_paths = [path for path in image_paths if relative_paths[path] not in cache]

    inferred_count = 0
    if missing_paths:
        runtime = get_default_classifier_runtime()
        for path in missing_paths:
            try:
                with Image.open(path) as image:
                    rgb_image = image.convert("RGB")
                pixel_values = runtime.preprocess(rgb_image)
                label = runtime.infer_label(pixel_values)
            except Exception:
                continue
            cache[relative_paths[path]] = label
            inferred_count += 1
        _write_classification_inference_cache(cache)

    label_by_path = {
        path: cache[relative_paths[path]]
        for path in image_paths
        if relative_paths[path] in cache
    }
    cached_count = len(label_by_path) - inferred_count
    return label_by_path, cached_count, inferred_count


def _apply_inference_labels_to_records(
    image_records: list[dict[str, Any]],
    label_by_path: dict[str, str],
) -> list[dict[str, Any]]:
    """Return a copy of image_records with `label`/`run_name` overridden from label_by_path."""
    updated_records: list[dict[str, Any]] = []
    for record in image_records:
        predicted_label = label_by_path.get(record["path"])
        if predicted_label is None:
            updated_records.append(record)
            continue
        updated_record = dict(record)
        updated_record["label"] = predicted_label
        updated_record["run_name"] = predicted_label
        updated_records.append(updated_record)
    return updated_records


def _build_runs_from_labeled_records(image_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Regroup image_records into the per-label "runs" structure load_dashboard_data produces,
    so build_aggregate_run/build_label_distribution_frame keep working unchanged after labels
    have been overridden (e.g. by a fresh classification inference pass)."""
    grouped_counts: Counter[str] = Counter()
    good_counts: Counter[str] = Counter()
    bad_counts: Counter[str] = Counter()
    latest_timestamps: dict[str, datetime] = {}

    for record in image_records:
        label = str(record["label"])
        grouped_counts[label] += 1
        timestamp = record.get("timestamp")
        if isinstance(timestamp, datetime):
            previous_timestamp = latest_timestamps.get(label)
            if previous_timestamp is None or timestamp > previous_timestamp:
                latest_timestamps[label] = timestamp
        if label == "Normal":
            good_counts[label] += 1
        else:
            bad_counts[label] += 1

    now = datetime.now()
    return [
        {
            "name": label,
            "timestamp": latest_timestamps.get(label, now),
            "path": "",
            "total_count": count,
            "label_counts": {label: count},
            "good_count": int(good_counts.get(label, 0)),
            "bad_count": int(bad_counts.get(label, 0)),
            "model_dir": "",
            "average_inference_ms": 0.0,
            "total_process_ms": 0.0,
        }
        for label, count in sorted(grouped_counts.items())
    ]


def _extract_features_from_images(
    image_paths: list[str],
    model_dir: Path,
) -> tuple[Any, list[str]]:
    """Extract feature vectors from images using the classifier model.
    
    Args:
        image_paths: List of image file paths
        model_dir: Path to the classifier model directory
        
    Returns:
        Tuple of (features array, processed image path list)
    """
    try:
        import numpy as np
        import torch
        from PIL import Image
        _suppress_transformers_path_alias_warning()
        from transformers import AutoImageProcessor, AutoModelForImageClassification
    except ImportError:
        st.error("Required libraries are missing. Please install torch, transformers, and Pillow.")
        return None, []
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        image_processor = AutoImageProcessor.from_pretrained(str(model_dir))
        model = AutoModelForImageClassification.from_pretrained(str(model_dir)).to(device)
        if hasattr(model.config, "output_hidden_states"):
            model.config.output_hidden_states = True

        def _to_feature_vector(output_tensor: Any) -> np.ndarray:
            array = output_tensor.detach().cpu().numpy()[0]
            if array.ndim == 1:
                return array.astype(np.float32)

            feature_axis = int(np.argmax(array.shape))
            reduce_axes = tuple(index for index in range(array.ndim) if index != feature_axis)
            reduced = array.mean(axis=reduce_axes)
            return np.asarray(reduced, dtype=np.float32).reshape(-1)

        features_list = []
        processed_paths = []
        
        model.eval()
        with torch.no_grad():
            for image_path in image_paths:
                try:
                    image = Image.open(image_path).convert("RGB")
                    inputs = image_processor(images=[image], return_tensors="pt").to(device)
                    
                    # Prefer richer hidden representations when available, then fall back to logits.
                    try:
                        outputs = model(**inputs, output_hidden_states=True)
                    except TypeError:
                        outputs = model(**inputs)
                    hidden_states = getattr(outputs, "hidden_states", None)
                    if hidden_states:
                        feature_vector = _to_feature_vector(hidden_states[-1])
                    else:
                        feature_vector = _to_feature_vector(outputs.logits)
                    features_list.append(feature_vector)
                    processed_paths.append(image_path)
                except Exception as e:
                    st.warning(f"Feature extraction failed for {Path(image_path).name}: {e}")
                    continue
        
        if not features_list:
            st.error("No images could be processed.")
            return None, []
        
        features_array = np.array(features_list, dtype=np.float32)
        
        # Ensure 2D shape: (num_samples, feature_dim)
        if features_array.ndim > 2:
            features_array = features_array.reshape(features_array.shape[0], -1)
        elif features_array.ndim == 1:
            features_array = features_array.reshape(-1, 1)
        
        return features_array, processed_paths
    except Exception as e:
        st.error(f"Failed to load the model: {e}")
        return None, []
