from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

try:
    from st_supabase_connection import SupabaseConnection
except ImportError:
    SupabaseConnection = None

from scripts.detail_finetune_mcp import resolve_base_model_dir


SUPABASE_CONNECTION_NAME = "supabase"
SUPABASE_IMAGE_TABLE = "semiconductor"
SUPABASE_IMAGE_COLUMNS = "id,file_path,class,trained,create_date"
SUPABASE_QUERY_TTL = "0s"
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


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
    if pd.isna(value):
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


def _normalize_cluster_label(value: Any) -> str:
    if value is None or pd.isna(value):
        return "trained"
    label = str(value).strip() if value is not None else ""
    return label or "trained"


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

    return _normalize_row_payload(result)


def load_supabase_boundary_source_frame() -> pd.DataFrame:
    rows = _fetch_supabase_semiconductor_rows()
    records: list[dict[str, Any]] = []
    for row in rows:
        normalized_path = _normalize_image_path_key(row.get("file_path"))
        if not normalized_path:
            continue

        image_path = Path(normalized_path)
        if not image_path.exists() or not image_path.is_file() or not _is_supported_image_path(image_path):
            continue

        records.append(
            {
                "record_id": row.get("id"),
                "image_paths": normalized_path,
                "cluster_label": _normalize_cluster_label(row.get("class")),
                "trained": _coerce_bool(row.get("trained")),
                "created_at": row.get("create_date"),
            }
        )

    frame = pd.DataFrame.from_records(
        records,
        columns=["record_id", "image_paths", "cluster_label", "trained", "created_at"],
    )
    if frame.empty:
        return frame

    frame = frame.sort_values("created_at", kind="stable")
    frame = frame.drop_duplicates(subset="image_paths", keep="last")
    frame["trained"] = frame["trained"].map(_coerce_bool)
    frame["cluster_label"] = frame["cluster_label"].map(_normalize_cluster_label)
    return frame.reset_index(drop=True)


def _load_feature_extractor() -> Any:
    try:
        from utils import _extract_features_from_images
    except ImportError:
        from scripts.utils import _extract_features_from_images

    return _extract_features_from_images


def _extract_partition_features(
    frame: pd.DataFrame,
    model_dir: Path,
) -> tuple[np.ndarray, pd.DataFrame]:
    if frame.empty:
        return np.empty((0, 0), dtype=np.float32), frame.copy()

    extract_features = _load_feature_extractor()
    ordered_paths = frame["image_paths"].astype(str).tolist()
    features, processed_paths = extract_features(ordered_paths, model_dir)
    if features is None or len(processed_paths) == 0:
        return np.empty((0, 0), dtype=np.float32), frame.iloc[0:0].copy()

    processed_path_keys = [_normalize_image_path_key(path) for path in processed_paths]
    path_to_row = {
        _normalize_image_path_key(row["image_paths"]): row
        for row in frame.to_dict("records")
    }
    processed_rows = [
        path_to_row[path_key]
        for path_key in processed_path_keys
        if path_key in path_to_row
    ]
    processed_frame = pd.DataFrame.from_records(processed_rows, columns=frame.columns)
    if processed_frame.empty:
        return np.empty((0, 0), dtype=np.float32), processed_frame

    features_array = np.asarray(features, dtype=np.float32)
    if features_array.ndim > 2:
        features_array = features_array.reshape(features_array.shape[0], -1)
    elif features_array.ndim == 1:
        features_array = features_array.reshape(-1, 1)

    usable_count = min(len(processed_frame), len(features_array))
    processed_frame = processed_frame.iloc[:usable_count].reset_index(drop=True)
    features_array = features_array[:usable_count]
    return features_array, processed_frame


def _empty_boundary_result_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "record_id",
            "image_paths",
            "trained",
            "cluster_label",
            "created_at",
            "tsne_x",
            "tsne_y",
            "tsne_z",
            "boundary_distance",
            "nearest_boundary_cluster",
            "nearest_boundary_path",
        ]
    )


def _build_candidate_frame_from_visible_paths(
    image_paths: list[str] | tuple[str, ...] | None,
    source_frame: pd.DataFrame,
) -> pd.DataFrame:
    normalized_input_paths = [
        normalized_path
        for normalized_path in (_normalize_image_path_key(path) for path in image_paths or [])
        if normalized_path
    ]
    if not normalized_input_paths:
        return source_frame.loc[~source_frame["trained"]].copy()

    visible_candidate_frame = pd.DataFrame({"image_paths": normalized_input_paths})
    source_columns = ["image_paths", "record_id", "cluster_label", "trained", "created_at"]
    visible_candidate_frame = visible_candidate_frame.merge(
        source_frame[source_columns],
        on="image_paths",
        how="left",
    )
    visible_candidate_frame["trained"] = visible_candidate_frame["trained"].fillna(False).map(_coerce_bool)
    visible_candidate_frame["cluster_label"] = visible_candidate_frame["cluster_label"].map(_normalize_cluster_label)
    visible_candidate_frame = visible_candidate_frame.loc[~visible_candidate_frame["trained"]].copy()
    visible_candidate_frame = visible_candidate_frame.drop_duplicates(subset="image_paths", keep="first")
    return visible_candidate_frame.reset_index(drop=True)


def _choose_tsne_perplexity(sample_count: int) -> float:
    if sample_count <= 2:
        return 1.0
    return float(max(1, min(30, sample_count // 3, sample_count - 1)))


def _pad_to_three_dimensions(values: np.ndarray) -> np.ndarray:
    if values.ndim != 2:
        values = values.reshape(len(values), -1)
    if values.shape[1] >= 3:
        return values[:, :3].astype(np.float32, copy=False)

    padded = np.zeros((values.shape[0], 3), dtype=np.float32)
    padded[:, :values.shape[1]] = values.astype(np.float32, copy=False)
    return padded


def _build_shared_embedding(
    trained_features: np.ndarray,
    untrained_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    try:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError("The scikit-learn package required for boundary sampling is not installed.") from exc

    if trained_features.size == 0 or untrained_features.size == 0:
        return (
            np.empty((len(trained_features), 3), dtype=np.float32),
            np.empty((len(untrained_features), 3), dtype=np.float32),
            "unavailable",
        )

    combined_features = np.concatenate([trained_features, untrained_features], axis=0)
    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined_features)

    if len(combined_scaled) < 4:
        reducer = PCA(n_components=min(3, combined_scaled.shape[1], len(combined_scaled)), random_state=42)
        reduced = reducer.fit_transform(combined_scaled)
        reduced = _pad_to_three_dimensions(reduced)
        method = "PCA fallback"
    else:
        perplexity = _choose_tsne_perplexity(len(combined_scaled))
        reducer = TSNE(
            n_components=3,
            random_state=42,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
        )
        reduced = reducer.fit_transform(combined_scaled).astype(np.float32, copy=False)
        method = f"t-SNE (perplexity={perplexity:.0f})"

    trained_count = len(trained_features)
    trained_embedding = np.asarray(reduced[:trained_count], dtype=np.float32)
    untrained_embedding = np.asarray(reduced[trained_count:], dtype=np.float32)
    return trained_embedding, untrained_embedding, method


def _select_outer_points_by_radius(points_3d: np.ndarray) -> np.ndarray:
    if len(points_3d) <= 1:
        return np.arange(len(points_3d))

    centroid = points_3d.mean(axis=0, keepdims=True)
    radius = np.linalg.norm(points_3d - centroid, axis=1)
    outer_count = max(1, math.ceil(len(points_3d) * 0.15))
    return np.argsort(radius)[-outer_count:]


def _select_boundary_points_for_cluster(points_3d: np.ndarray) -> np.ndarray:
    if len(points_3d) <= 4:
        return np.arange(len(points_3d))

    try:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(points_3d)
        return np.unique(hull.vertices)
    except Exception:
        return _select_outer_points_by_radius(points_3d)


def _build_boundary_point_frame(trained_frame: pd.DataFrame) -> pd.DataFrame:
    if trained_frame.empty:
        return trained_frame.iloc[0:0].copy()

    boundary_chunks: list[pd.DataFrame] = []
    for cluster_label, cluster_frame in trained_frame.groupby("cluster_label", sort=True):
        cluster_points = cluster_frame[["tsne_x", "tsne_y", "tsne_z"]].to_numpy(dtype=np.float32)
        boundary_indices = _select_boundary_points_for_cluster(cluster_points)
        boundary_chunk = cluster_frame.iloc[np.unique(boundary_indices)].copy()
        boundary_chunk["boundary_cluster_label"] = cluster_label
        boundary_chunk["is_boundary_point"] = True
        boundary_chunks.append(boundary_chunk)

    if not boundary_chunks:
        return trained_frame.iloc[0:0].copy()

    boundary_frame = pd.concat(boundary_chunks, ignore_index=True)
    boundary_frame = boundary_frame.drop_duplicates(subset="image_paths", keep="first")
    return boundary_frame.reset_index(drop=True)


def _compute_min_boundary_distances(
    candidate_points: np.ndarray,
    boundary_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("The torch package required for boundary sampling is not installed.") from exc

    if len(candidate_points) == 0 or len(boundary_points) == 0:
        return (
            np.empty((len(candidate_points),), dtype=np.float32),
            np.empty((len(candidate_points),), dtype=np.int64),
            "cpu",
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidate_tensor = torch.as_tensor(candidate_points, dtype=torch.float32, device=device)
    boundary_tensor = torch.as_tensor(boundary_points, dtype=torch.float32, device=device)

    candidate_chunk_size = 1024
    boundary_chunk_size = 4096 if device.type == "cuda" else 2048
    mean_distance_chunks: list[Any] = []

    with torch.no_grad():
        for candidate_start in range(0, candidate_tensor.shape[0], candidate_chunk_size):
            candidate_batch = candidate_tensor[candidate_start:candidate_start + candidate_chunk_size]
            batch_distances = []

            for boundary_start in range(0, boundary_tensor.shape[0], boundary_chunk_size):
                boundary_batch = boundary_tensor[boundary_start:boundary_start + boundary_chunk_size]
                distance_batch = torch.cdist(candidate_batch, boundary_batch, p=2)
                batch_distances.append(distance_batch)

            # Concatenate distances from all boundary chunks and compute the mean distance.
            all_distances = torch.cat(batch_distances, dim=1)
            mean_distance = all_distances.mean(dim=1)
            
            mean_distance_chunks.append(mean_distance.detach().cpu())

    mean_distances = torch.cat(mean_distance_chunks).numpy().astype(np.float32, copy=False)
    # Return an empty array for nearest_indices to preserve function signature compatibility.
    nearest_indices = np.empty((len(candidate_points),), dtype=np.int64)
    return mean_distances, nearest_indices, device.type


def build_boundary_sampling_frame(
    image_paths: list[str] | tuple[str, ...] | None,
    base_model_dir: str | Path,
    supabase_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    source_frame = supabase_frame.copy() if supabase_frame is not None else load_supabase_boundary_source_frame()
    if source_frame.empty:
        return _empty_boundary_result_frame()

    source_frame["image_paths"] = source_frame["image_paths"].map(_normalize_image_path_key)
    source_frame["trained"] = source_frame["trained"].map(_coerce_bool)
    source_frame["cluster_label"] = source_frame["cluster_label"].map(_normalize_cluster_label)
    source_frame = source_frame.drop_duplicates(subset="image_paths", keep="last").reset_index(drop=True)

    trained_source_frame = source_frame.loc[source_frame["trained"]].copy()
    untrained_source_frame = _build_candidate_frame_from_visible_paths(image_paths, source_frame)

    if trained_source_frame.empty:
        raise RuntimeError("Boundary sampling cannot be computed because there is no `trained=True` data.")
    if untrained_source_frame.empty:
        return _empty_boundary_result_frame()

    resolved_model_dir = resolve_base_model_dir(base_model_dir)
    trained_features, trained_processed_frame = _extract_partition_features(trained_source_frame, resolved_model_dir)
    untrained_features, untrained_processed_frame = _extract_partition_features(untrained_source_frame, resolved_model_dir)

    if len(trained_features) == 0:
        raise RuntimeError("Failed to extract features from the `trained=True` data.")
    if len(untrained_features) == 0:
        return _empty_boundary_result_frame()

    # Distance between independently fitted t-SNE spaces is not meaningful,
    # so both partitions are embedded together and then split back out.
    trained_embedding, untrained_embedding, embedding_method = _build_shared_embedding(
        trained_features,
        untrained_features,
    )

    trained_embedding_frame = trained_processed_frame.copy()
    trained_embedding_frame["tsne_x"] = trained_embedding[:, 0]
    trained_embedding_frame["tsne_y"] = trained_embedding[:, 1]
    trained_embedding_frame["tsne_z"] = trained_embedding[:, 2]

    boundary_frame = _build_boundary_point_frame(trained_embedding_frame)
    if boundary_frame.empty:
        raise RuntimeError("Failed to compute boundary points from the `trained=True` data.")

    mean_distances, nearest_indices, distance_device = _compute_min_boundary_distances(
        candidate_points=untrained_embedding,
        boundary_points=boundary_frame[["tsne_x", "tsne_y", "tsne_z"]].to_numpy(dtype=np.float32),
    )

    nearest_boundary_rows = boundary_frame.iloc[nearest_indices].reset_index(drop=True)
    candidate_frame = untrained_processed_frame.copy()
    candidate_frame["tsne_x"] = untrained_embedding[:, 0]
    candidate_frame["tsne_y"] = untrained_embedding[:, 1]
    candidate_frame["tsne_z"] = untrained_embedding[:, 2]
    candidate_frame["boundary_distance"] = mean_distances
    candidate_frame["nearest_boundary_cluster"] = nearest_boundary_rows["boundary_cluster_label"].tolist()
    candidate_frame["nearest_boundary_path"] = nearest_boundary_rows["image_paths"].tolist()
    candidate_frame["trained"] = False

    candidate_frame = candidate_frame.sort_values(
        by=["boundary_distance", "created_at", "image_paths"],
        ascending=[False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    candidate_frame.attrs["embedding_method"] = embedding_method
    candidate_frame.attrs["boundary_point_count"] = int(len(boundary_frame))
    candidate_frame.attrs["trained_sample_count"] = int(len(trained_embedding_frame))
    candidate_frame.attrs["candidate_sample_count"] = int(len(candidate_frame))
    candidate_frame.attrs["distance_device"] = distance_device
    return candidate_frame


def select_boundary_sampling_paths(
    image_paths: list[str] | tuple[str, ...] | None,
    base_model_dir: str | Path,
    selection_percentage: int,
) -> tuple[list[str], pd.DataFrame]:
    supabase_frame = load_supabase_boundary_source_frame()
    boundary_frame = build_boundary_sampling_frame(
        image_paths=image_paths,
        base_model_dir=base_model_dir,
        supabase_frame=supabase_frame,
    )
    if boundary_frame.empty:
        return [], boundary_frame

    sample_count = max(1, math.ceil(len(boundary_frame) * (selection_percentage / 100.0)))
    sample_count = min(sample_count, len(boundary_frame))
    selected_paths = boundary_frame.head(sample_count)["image_paths"].tolist()
    return selected_paths, boundary_frame
