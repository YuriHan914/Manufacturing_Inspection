from __future__ import annotations

import csv
import logging
import math
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

try:
    from st_supabase_connection import SupabaseConnection
except ImportError:
    SupabaseConnection = None

from scripts.detail_finetune_mcp import resolve_base_model_dir


BASE_DIR = Path(__file__).resolve().parents[1]
SUPABASE_CONNECTION_NAME = "supabase"
SUPABASE_IMAGE_TABLE = "semiconductor"
SUPABASE_IMAGE_COLUMNS = "id,file_path,trained,create_date"
SUPABASE_QUERY_TTL = "0s"
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CSV_FALLBACK_DATA_PATH = BASE_DIR / "data" / "data.csv"


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


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    normalized = str(value).strip().lower()
    return normalized in {"1", "true", "t", "y", "yes"}


def _normalize_row_payload(rows: Any) -> list[dict[str, Any]]:
    normalized_rows: list[dict[str, Any]] = []
    for row in getattr(rows, "data", rows) or []:
        if isinstance(row, dict):
            normalized_rows.append(row)
            continue
        if hasattr(row, "items"):
            normalized_rows.append(dict(row.items()))
            continue
        normalized_rows.append(dict(row))
    return normalized_rows


def _normalize_image_path_key(value: str | Path | None) -> str:
    if value is None:
        return ""
    raw_value = str(value).strip()
    if not raw_value:
        return ""
    return str(Path(raw_value).expanduser().resolve())


def _is_supported_image_path(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES


def _fetch_supabase_semiconductor_rows() -> list[dict[str, Any]]:
    if SupabaseConnection is None:
        raise RuntimeError("The `st_supabase_connection` package could not be found.")

    connection = st.connection(SUPABASE_CONNECTION_NAME, type=SupabaseConnection)
    try:
        query_builder = connection.query(
            SUPABASE_IMAGE_COLUMNS,
            table=SUPABASE_IMAGE_TABLE,
            ttl=SUPABASE_QUERY_TTL,
        )
        if hasattr(query_builder, "order"):
            query_builder = query_builder.order("create_date", desc=False)
        result = query_builder.execute()
    except Exception as exc:
        client = getattr(connection, "client", None)
        if client is None:
            raise RuntimeError(f"Failed to query the Supabase semiconductor table: {exc}") from exc

        query_builder = client.table(SUPABASE_IMAGE_TABLE).select(SUPABASE_IMAGE_COLUMNS)
        if hasattr(query_builder, "order"):
            query_builder = query_builder.order("create_date", desc=False)
        result = query_builder.execute()

    rows = _normalize_row_payload(result)
    for row in rows:
        if "file_path" in row:
            row["image_path"] = row.pop("file_path")
        if "create_date" in row:
            row["created_at"] = row.pop("create_date")
    return rows


def _fetch_csv_status_rows() -> list[dict[str, Any]]:
    if not CSV_FALLBACK_DATA_PATH.exists():
        raise RuntimeError(f"The CSV fallback file does not exist: {CSV_FALLBACK_DATA_PATH}")

    rows: list[dict[str, Any]] = []
    with CSV_FALLBACK_DATA_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not isinstance(row, dict):
                continue
            rows.append(
                {
                    "id": row.get("id"),
                    "image_path": row.get("image_path"),
                    "trained": row.get("trained"),
                    "created_at": row.get("created_at"),
                }
            )
    return rows


def load_supabase_image_status_frame() -> pd.DataFrame:
    try:
        rows = _fetch_supabase_semiconductor_rows()
    except Exception as supabase_error:
        logger = logging.getLogger(__name__)
        logger.warning(
            "Margin sampling status source switched to CSV fallback: %s",
            supabase_error,
        )
        try:
            rows = _fetch_csv_status_rows()
        except Exception as csv_error:
            raise RuntimeError(
                "Failed to load margin sampling status from both Supabase and the CSV fallback. "
                f"supabase_error={supabase_error} | csv_error={csv_error}"
            ) from csv_error

    records: list[dict[str, Any]] = []
    for row in rows:
        records.append(
            {
                "record_id": row.get("id"),
                "image_paths": _normalize_image_path_key(row.get("image_path")),
                "trained": _coerce_bool(row.get("trained")),
                "created_at": row.get("created_at"),
            }
        )

    frame = pd.DataFrame.from_records(
        records,
        columns=["record_id", "image_paths", "trained", "created_at"],
    )
    if frame.empty:
        return frame

    frame = frame[frame["image_paths"].astype(str).str.strip().ne("")].copy()
    frame["trained"] = frame["trained"].map(_coerce_bool)
    return frame.reset_index(drop=True)


def _build_candidate_status_frame(
    image_paths: list[str] | tuple[str, ...] | None,
    supabase_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    status_frame = (
        supabase_frame.copy()
        if supabase_frame is not None
        else load_supabase_image_status_frame()
    )

    if status_frame.empty:
        status_frame = pd.DataFrame(columns=["image_paths", "trained", "created_at"])
    else:
        status_frame["image_paths"] = status_frame["image_paths"].map(_normalize_image_path_key)
        status_frame["trained"] = status_frame["trained"].map(_coerce_bool)
        if "created_at" in status_frame.columns:
            status_frame = status_frame.sort_values("created_at", kind="stable")
        status_frame = status_frame.drop_duplicates(subset="image_paths", keep="last")

    normalized_input_paths = [
        normalized_path
        for normalized_path in (_normalize_image_path_key(path) for path in image_paths or [])
        if normalized_path
    ]
    if normalized_input_paths:
        candidate_frame = pd.DataFrame({"image_paths": normalized_input_paths})
        if not status_frame.empty:
            candidate_frame = candidate_frame.merge(
                status_frame[["image_paths", "trained"]],
                on="image_paths",
                how="left",
            )
    else:
        candidate_frame = status_frame[["image_paths", "trained"]].copy()

    if candidate_frame.empty:
        return pd.DataFrame(columns=["image_paths", "trained"])

    candidate_frame["trained"] = candidate_frame["trained"].fillna(False).map(_coerce_bool)
    candidate_frame = candidate_frame.drop_duplicates(subset="image_paths", keep="first")
    return candidate_frame.reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def _load_margin_sampling_runtime(model_dir_value: str) -> tuple[Any, Any, str]:
    try:
        import torch

        _suppress_transformers_path_alias_warning()
        from transformers import AutoModelForImageClassification
        try:
            from transformers import AutoImageProcessor
        except ImportError:
            from transformers import AutoFeatureExtractor as AutoImageProcessor
    except ImportError as exc:
        raise RuntimeError("The torch/transformers packages required for margin sampling are not installed.") from exc

    resolved_model_dir = resolve_base_model_dir(model_dir_value)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_processor = AutoImageProcessor.from_pretrained(str(resolved_model_dir))
    model = AutoModelForImageClassification.from_pretrained(str(resolved_model_dir)).to(device)
    model.eval()
    return image_processor, model, device


def build_margin_sampling_frame(
    image_paths: list[str] | tuple[str, ...] | None,
    base_model_dir: str | Path,
    supabase_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    candidate_frame = _build_candidate_status_frame(image_paths, supabase_frame=supabase_frame)
    if candidate_frame.empty:
        return pd.DataFrame(
            columns=[
                "image_paths",
                "trained",
                "predicted_label",
                "top1_confidence",
                "top2_confidence",
                "margin_score",
            ]
        )

    try:
        import torch
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("The torch/Pillow packages required for margin sampling are not installed.") from exc

    resolved_model_dir = resolve_base_model_dir(base_model_dir)
    image_processor, model, device_name = _load_margin_sampling_runtime(str(resolved_model_dir))
    device = torch.device(device_name)
    rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for _, row in candidate_frame.iterrows():
            image_path = Path(str(row["image_paths"])).expanduser()
            if not image_path.exists() or not image_path.is_file() or not _is_supported_image_path(image_path):
                continue

            with Image.open(image_path) as image:
                rgb_image = image.convert("RGB")

            inputs = image_processor(images=rgb_image, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            logits = model(**inputs).logits[0]
            probabilities = torch.softmax(logits, dim=-1)

            top_k = min(2, int(probabilities.numel()))
            top_probabilities, top_indices = torch.topk(probabilities, k=top_k)
            top1_index = int(top_indices[0].item())
            top1_confidence = float(top_probabilities[0].item())
            top2_confidence = float(top_probabilities[1].item()) if top_k > 1 else 0.0
            margin_score = max(0.0, top1_confidence - top2_confidence)


            rows.append(
                {
                    "image_paths": str(image_path.resolve()),
                    "trained": _coerce_bool(row.get("trained")),
                    "predicted_label": str(model.config.id2label[top1_index]),
                    "top1_confidence": top1_confidence,
                    "top2_confidence": top2_confidence,
                    "margin_score": margin_score,
                }
            )

    frame = pd.DataFrame.from_records(
        rows,
        columns=[
            "image_paths",
            "trained",
            "predicted_label",
            "top1_confidence",
            "top2_confidence",
            "margin_score",
        ],
    )
    if frame.empty:
        return frame

    return frame.sort_values(
        by=["margin_score", "trained", "image_paths"],
        ascending=[True, True, True],
        kind="stable",
    ).reset_index(drop=True)


def select_margin_sampling_paths(
    image_paths: list[str] | tuple[str, ...] | None,
    base_model_dir: str | Path,
    selection_percentage: int,
) -> tuple[list[str], pd.DataFrame]:
    supabase_frame = load_supabase_image_status_frame()
    margin_frame = build_margin_sampling_frame(
        image_paths=image_paths,
        base_model_dir=base_model_dir,
        supabase_frame=supabase_frame,
    )
    if margin_frame.empty:
        return [], margin_frame

    candidate_frame = margin_frame.loc[~margin_frame["trained"]].copy()
    if candidate_frame.empty:
        candidate_frame = margin_frame.copy()

    sample_count = max(1, math.ceil(len(candidate_frame) * (selection_percentage / 100.0)))
    sample_count = min(sample_count, len(candidate_frame))
    selected_paths = candidate_frame.sort_values(
        by=["margin_score", "top1_confidence", "image_paths"],
        ascending=[True, True, True],
        kind="stable",
    ).head(sample_count)["image_paths"].tolist()
    return selected_paths, margin_frame
