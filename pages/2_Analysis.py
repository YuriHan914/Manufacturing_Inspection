from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from huggingface_hub import InferenceClient

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "output"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
OPENXAI_DIR = ROOT_DIR / "third_party" / "OpenXAI"
if OPENXAI_DIR.exists() and str(OPENXAI_DIR) not in sys.path:
    sys.path.insert(0, str(OPENXAI_DIR))

AD_SCRIPTS_DIR = ROOT_DIR / "scripts" / "AD"
AD_PIPELINE_SCRIPT = AD_SCRIPTS_DIR / "run_feature_pipeline.sh"
AD_FEATURE_ROOT = ROOT_DIR / "data" / "features" / "app"
AD_RAW_DIR = AD_FEATURE_ROOT / "raw"
AD_AGG_DIR = AD_FEATURE_ROOT / "agg"
AD_MEMORY_BANK_PATH = ROOT_DIR / "data" / "memory_bank" / "MB.npy"
AD_HEATMAP_RANGE_PATH = ROOT_DIR / "data" / "memory_bank" / "heatmap_range.json"
AD_OUTPUT_DIR = ROOT_DIR / "outputs" / "AD"

from scripts.classification.classifier_runtime import get_default_classifier_runtime
from scripts.detail_finetune_mcp import CLASSIFIER_MODEL_DIR, resolve_base_model_dir
from scripts.utils import (
    CLASS_VISUALIZATION_ORDER,
    _append_app_log,
    _extract_features_from_images,
    _get_cached_detail_inference_result,
    _get_discrete_class_colors,
    _render_classifier_model_selector,
    _suppress_transformers_path_alias_warning,
    _to_project_relative_path,
    configure_page,
    load_dashboard_data,
    render_page_header,
)


def _summarize_selected_filenames(records: list[dict[str, Any]], limit: int = 10) -> str:
    filenames = ", ".join(record["filename"] for record in records[:limit])
    if len(records) > limit:
        filenames += f", ... (+{len(records) - limit} more)"
    return filenames


def _normalize_option_value(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _resolve_requested_option(requested: str, options: list[str]) -> str | None:
    requested_normalized = _normalize_option_value(str(requested))
    if not requested_normalized:
        return None
    for option in options:
        if _normalize_option_value(str(option)) == requested_normalized:
            return option
    return None


def _apply_requested_selectbox_value(request_key: str, widget_key: str, options: list[str], default: str) -> None:
    requested_value = st.session_state.pop(request_key, None)
    if requested_value is not None:
        resolved_value = _resolve_requested_option(str(requested_value), options)
        st.session_state[widget_key] = resolved_value or default
    elif st.session_state.get(widget_key) not in options:
        st.session_state[widget_key] = default


def _render_detail_inference_model_selector(selected_records: list[dict[str, Any]]) -> tuple[Path | None, bool]:
    selected_model_dir, model_changed = _render_classifier_model_selector(
        selected_records=selected_records,
        container=st.sidebar,
        selector_key="detail_inference_model_selector",
        active_key="detail_inference_model_active",
        section_title="Image Inference",
        helper_text="Choose a model, then click Run to run inference again on the currently selected images.",
        label="Inference model",
        add_divider=True,
    )
    if selected_records and model_changed:
        st.sidebar.caption("The model has changed. Click `Run` to rerun inference with the new model.")
    return selected_model_dir, model_changed


def _resolve_inference_output_targets(output_path: Path, inference_mode: str) -> tuple[Path, Path, Path]:
    base_output_dir = output_path.parent if output_path.suffix else output_path
    if base_output_dir.exists() and not base_output_dir.is_dir():
        raise NotADirectoryError(f"Output path exists but is not a directory: {base_output_dir}")

    base_output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S_") + f"{time.time_ns() % 1_000_000:06d}"
    timestamp_output_dir = base_output_dir / timestamp
    timestamp_output_dir.mkdir(parents=True, exist_ok=False)
    return (
        timestamp_output_dir / f"{inference_mode}_inference_results.json",
        timestamp_output_dir / f"{inference_mode}_inference_timing.txt",
        timestamp_output_dir,
    )


def _save_inference_results(results: dict[str, str], output_json_path: Path) -> None:
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_results = {
        _to_project_relative_path(path): label for path, label in results.items()
    }
    output_json_path.write_text(
        json.dumps(normalized_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _save_inference_timing(
    model_path: Path,
    initial_model_load_time_milliseconds: float,
    total_process_time_milliseconds: float,
    per_image_preprocess_times_milliseconds: dict[str, float],
    per_image_inference_times_milliseconds: dict[str, float],
    output_timing_path: Path,
) -> None:
    output_timing_path.parent.mkdir(parents=True, exist_ok=True)
    total_model_load_time_milliseconds = (
        initial_model_load_time_milliseconds + sum(per_image_preprocess_times_milliseconds.values())
    )
    total_inference_time_milliseconds = sum(per_image_inference_times_milliseconds.values())

    lines = [
        f"model_path: {_to_project_relative_path(model_path)}",
        f"initial_model_load_time_milliseconds: {initial_model_load_time_milliseconds:.3f}",
        f"model_load_time_milliseconds: {total_model_load_time_milliseconds:.3f}",
        f"inference_time_milliseconds: {total_inference_time_milliseconds:.3f}",
        f"total_process_time_milliseconds: {total_process_time_milliseconds:.3f}",
        f"image_count: {len(per_image_inference_times_milliseconds)}",
    ]

    if per_image_preprocess_times_milliseconds:
        average_preprocess_time = (
            sum(per_image_preprocess_times_milliseconds.values())
            / len(per_image_preprocess_times_milliseconds)
        )
        lines.append(f"average_per_image_preprocess_time_milliseconds: {average_preprocess_time:.3f}")

    if per_image_inference_times_milliseconds:
        average_inference_time = (
            sum(per_image_inference_times_milliseconds.values())
            / len(per_image_inference_times_milliseconds)
        )
        lines.append(f"average_per_image_inference_time_milliseconds: {average_inference_time:.3f}")

    if per_image_preprocess_times_milliseconds:
        lines.append("per_image_preprocess_time_milliseconds:")
        for path, value in per_image_preprocess_times_milliseconds.items():
            lines.append(f"{path}: {value:.3f}")

    if per_image_inference_times_milliseconds:
        lines.append("per_image_inference_time_milliseconds:")
        for path, value in per_image_inference_times_milliseconds.items():
            lines.append(f"{path}: {value:.3f}")

    output_timing_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class _TorchClassifierHandle:
    """Adapts an AutoModelForImageClassification model to the preprocess()/infer_label()
    interface shared with the default TensorRT/ONNX runtime (see _load_detail_classifier_runtime),
    so the reinference loop below doesn't need to special-case either backend."""

    def __init__(self, image_processor: Any, model: Any, device: str) -> None:
        import torch

        self._torch = torch
        self.processor = image_processor
        self.model = model
        self.device = device

    def preprocess(self, image: Any) -> Any:
        inputs = self.processor(images=image, return_tensors="pt")
        return {key: value.to(self.device) for key, value in inputs.items()}

    def infer_label(self, inputs: Any) -> str:
        with self._torch.no_grad():
            logits = self.model(**inputs).logits
        predicted_index = int(logits.argmax(dim=-1).item())
        return str(self.model.config.id2label[predicted_index])


def _load_detail_torch_classifier_runtime(model_dir: Path) -> tuple[Any, Any, str, float]:
    try:
        import torch

        _suppress_transformers_path_alias_warning()
        from transformers import AutoImageProcessor, AutoModelForImageClassification
    except ImportError as exc:
        raise RuntimeError("The torch/transformers packages required for image reinference are not installed.") from exc

    resolved_model_dir = resolve_base_model_dir(model_dir)
    cached_model_dir = st.session_state.get("detail_inference_runtime_model_dir")
    initial_model_load_time_milliseconds = 0.0
    if cached_model_dir != str(resolved_model_dir):
        cached_model = st.session_state.get("detail_inference_runtime_model")
        if cached_model is not None:
            try:
                cached_model.to("cpu")
            except Exception:
                pass

        st.session_state.pop("detail_inference_runtime_model", None)
        st.session_state.pop("detail_inference_runtime_processor", None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        load_start_ns = time.perf_counter_ns()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        image_processor = AutoImageProcessor.from_pretrained(str(resolved_model_dir))
        model = AutoModelForImageClassification.from_pretrained(str(resolved_model_dir)).to(device)
        model.eval()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        initial_model_load_time_milliseconds = (time.perf_counter_ns() - load_start_ns) / 1_000_000.0

        st.session_state["detail_inference_runtime_model_dir"] = str(resolved_model_dir)
        st.session_state["detail_inference_runtime_processor"] = image_processor
        st.session_state["detail_inference_runtime_model"] = model
        st.session_state["detail_inference_runtime_device"] = device
    st.session_state["detail_inference_runtime_initial_model_load_ms"] = initial_model_load_time_milliseconds

    return (
        st.session_state["detail_inference_runtime_processor"],
        st.session_state["detail_inference_runtime_model"],
        str(st.session_state["detail_inference_runtime_device"]),
        float(st.session_state.get("detail_inference_runtime_initial_model_load_ms", 0.0)),
    )


def _load_detail_classifier_runtime(model_dir: Path) -> tuple[Any, str, float]:
    """Prediction runtime for the Result/Run flow: the default classifier model_dir (the app's
    normal case) uses the GPU(TensorRT)/CPU(ONNX) engine; any other, explicitly-selected model_dir
    (e.g. a fine-tuned checkpoint from the Fine-tuning page) keeps using the original torch model,
    since only the default model has a compiled engine/ONNX export."""
    resolved_model_dir = resolve_base_model_dir(model_dir)

    if resolved_model_dir == resolve_base_model_dir(CLASSIFIER_MODEL_DIR):
        load_start_ns = time.perf_counter_ns()
        runtime = get_default_classifier_runtime()
        initial_model_load_time_milliseconds = (time.perf_counter_ns() - load_start_ns) / 1_000_000.0
        return runtime, runtime.device, initial_model_load_time_milliseconds

    image_processor, model, device_name, initial_model_load_time_milliseconds = _load_detail_torch_classifier_runtime(
        resolved_model_dir
    )
    return (
        _TorchClassifierHandle(image_processor, model, device_name),
        device_name,
        initial_model_load_time_milliseconds,
    )


def _predict_detail_records_with_model(
    selected_records: list[dict[str, Any]],
    model_dir: Path,
) -> tuple[list[dict[str, Any]], list[str], dict[str, str] | None]:
    resolved_model_dir = resolve_base_model_dir(model_dir)
    signature = (
        str(resolved_model_dir),
        tuple(record["path"] for record in selected_records),
    )

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("The Pillow package required for image reinference is not installed.") from exc

    handle, device_name, initial_model_load_time_milliseconds = _load_detail_classifier_runtime(resolved_model_dir)
    predicted_records: list[dict[str, Any]] = []
    prediction_errors: list[str] = []
    per_image_preprocess_times_milliseconds: dict[str, float] = {}
    per_image_inference_times_milliseconds: dict[str, float] = {}
    prediction_results: dict[str, str] = {}
    total_start_ns = time.perf_counter_ns()

    def _sync_if_needed() -> None:
        if device_name == "cuda":
            import torch

            torch.cuda.synchronize()

    with st.spinner(f"Running inference again on the selected images with {resolved_model_dir.name}..."):
        for record in selected_records:
            updated_record = dict(record)
            updated_record["model_dir"] = str(resolved_model_dir)
            updated_record["model_dir_display"] = _to_project_relative_path(resolved_model_dir)

            if not updated_record.get("exists"):
                predicted_records.append(updated_record)
                continue

            try:
                preprocess_start_ns = time.perf_counter_ns()
                with Image.open(updated_record["path"]) as image:
                    rgb_image = image.convert("RGB")
                inputs = handle.preprocess(rgb_image)
                preprocess_time_milliseconds = (time.perf_counter_ns() - preprocess_start_ns) / 1_000_000.0
                _sync_if_needed()
                inference_start_ns = time.perf_counter_ns()
                updated_record["label"] = handle.infer_label(inputs)
                _sync_if_needed()
                inference_time_milliseconds = (time.perf_counter_ns() - inference_start_ns) / 1_000_000.0
                relative_image_path = _to_project_relative_path(updated_record["path"])
                prediction_results[updated_record["path"]] = updated_record["label"]
                per_image_preprocess_times_milliseconds[relative_image_path] = preprocess_time_milliseconds
                per_image_inference_times_milliseconds[relative_image_path] = inference_time_milliseconds
            except Exception as exc:
                prediction_errors.append(f"{Path(updated_record['path']).name}: {exc}")
            predicted_records.append(updated_record)

    artifact_paths: dict[str, str] | None = None
    if prediction_results:
        total_process_time_milliseconds = (time.perf_counter_ns() - total_start_ns) / 1_000_000.0
        output_json_path, output_timing_path, output_dir = _resolve_inference_output_targets(OUTPUT_DIR, "batch")
        _save_inference_results(prediction_results, output_json_path)
        _save_inference_timing(
            model_path=resolved_model_dir,
            initial_model_load_time_milliseconds=initial_model_load_time_milliseconds,
            total_process_time_milliseconds=total_process_time_milliseconds,
            per_image_preprocess_times_milliseconds=per_image_preprocess_times_milliseconds,
            per_image_inference_times_milliseconds=per_image_inference_times_milliseconds,
            output_timing_path=output_timing_path,
        )
        artifact_paths = {
            "output_dir": str(output_dir),
            "results_path": str(output_json_path),
            "timing_path": str(output_timing_path),
        }

    st.session_state["detail_inference_prediction_signature"] = signature
    st.session_state["detail_inference_prediction_records"] = predicted_records
    st.session_state["detail_inference_prediction_errors"] = prediction_errors
    st.session_state["detail_inference_prediction_output_dir"] = (
        artifact_paths["output_dir"] if artifact_paths else ""
    )
    st.session_state["detail_inference_prediction_results_path"] = (
        artifact_paths["results_path"] if artifact_paths else ""
    )
    st.session_state["detail_inference_prediction_timing_path"] = (
        artifact_paths["timing_path"] if artifact_paths else ""
    )
    return predicted_records, prediction_errors, artifact_paths


def _log_detail_classification_run(
    selected_records: list[dict[str, Any]],
    model_dir: Path,
    prediction_errors: list[str],
    artifact_paths: dict[str, str] | None,
) -> None:
    content = (
        f"Classification re-inference run on {len(selected_records)} selected image(s) "
        f"with model `{_to_project_relative_path(resolve_base_model_dir(model_dir))}`. "
        f"Images: {_summarize_selected_filenames(selected_records)}."
    )
    if artifact_paths:
        content += f" Results saved to `{_to_project_relative_path(Path(artifact_paths['output_dir']))}`."
    if prediction_errors:
        content += f" {len(prediction_errors)} image(s) failed."

    _append_app_log(
        log_type="error" if prediction_errors and not artifact_paths else "done",
        source="Analysis Classification",
        content=content,
    )


def _resolve_selected_model_dirs(selected_records: list[dict[str, Any]]) -> list[Path]:
    resolved_dirs: list[Path] = []
    for record in selected_records:
        model_dir = resolve_base_model_dir(record.get("model_dir"))
        if model_dir not in resolved_dirs:
            resolved_dirs.append(model_dir)
    return resolved_dirs


def _get_detail_base_model_dir(selected_records: list[dict[str, Any]]) -> Path:
    resolved_dirs = _resolve_selected_model_dirs(selected_records)
    if resolved_dirs:
        return resolved_dirs[0]
    return resolve_base_model_dir("model")


def _get_detail_selected_records(image_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected_paths = st.session_state.get("detail_selected_image_paths", [])
    if not selected_paths:
        return []
    selected_records: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for record in image_records:
        path = record["path"]
        if path in selected_paths and path not in seen_paths:
            selected_records.append(record)
            seen_paths.add(path)
            if len(seen_paths) == len(selected_paths):
                break
    return selected_records


def _reset_detail_finetune_session(selected_paths: list[str]) -> None:
    prefix = "detail_finetune_"
    for key in list(st.session_state.keys()):
        if key.startswith(prefix):
            st.session_state.pop(key, None)
    st.session_state["detail_selected_image_paths"] = selected_paths
    st.session_state["detail_finetune_chat"] = []
    st.session_state["detail_finetune_plan"] = None
    st.session_state["detail_finetune_execution"] = None


def _resolve_default_hf_token() -> str:
    for env_name in ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HUGGINGFACE_API_TOKEN"):
        token = os.environ.get(env_name, "").strip()
        if token:
            return token
    try:
        from huggingface_hub import get_token

        cached_token = (get_token() or "").strip()
        if cached_token:
            return cached_token
    except Exception:
        pass
    return ""


def _extract_prediction_label(record: dict[str, Any]) -> str:
    """Use prediction-oriented fields only; never read ground-truth keys."""
    if "pred_label" in record and str(record.get("pred_label", "")).strip():
        return str(record["pred_label"])
    return str(record.get("label", "Unknown"))


def _sanitize_gemma_text(text: str) -> str:
    """Drop any line that may leak forbidden supervision metrics."""
    forbidden = (
        "ground_truth",
        "ground truth",
        "true_label",
        "true label",
        "accuracy",
        "acc=",
        "acc ",
    )
    safe_lines: list[str] = []
    for raw_line in str(text).splitlines():
        low = raw_line.lower()
        if any(token in low for token in forbidden):
            continue
        safe_lines.append(raw_line)
    return "\n".join(safe_lines)


def _build_three_reductions(features_scaled: np.ndarray) -> dict[str, np.ndarray]:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    reducers: dict[str, np.ndarray] = {}
    reducers["PCA"] = PCA(n_components=3, random_state=42).fit_transform(features_scaled)

    tsne = TSNE(n_components=3, random_state=42, perplexity=min(30, max(2, len(features_scaled) - 1)))
    reducers["t-SNE"] = tsne.fit_transform(features_scaled)

    try:
        import umap

        reducers["UMAP"] = umap.UMAP(n_components=3, random_state=42).fit_transform(features_scaled)
    except Exception:
        reducers["UMAP"] = reducers["PCA"].copy()

    return reducers


def _normalize_for_display(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    axis_std = points.std(axis=0)
    axis_std[axis_std == 0] = 1.0
    return (points - points.mean(axis=0)) / axis_std


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-9
    p_safe = np.clip(p.astype(np.float64), eps, None)
    q_safe = np.clip(q.astype(np.float64), eps, None)
    p_safe = p_safe / p_safe.sum()
    q_safe = q_safe / q_safe.sum()
    return float(np.sum(p_safe * np.log(p_safe / q_safe)))


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    return 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)


def _rbf_mmd(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return 0.0
    combined = np.vstack([x, y])
    sq_dists = np.sum((combined[:, None, :] - combined[None, :, :]) ** 2, axis=-1)
    median_sq = float(np.median(sq_dists[sq_dists > 0])) if np.any(sq_dists > 0) else 1.0
    gamma = 1.0 / max(median_sq, 1e-6)

    def _kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.exp(-gamma * np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1))

    k_xx = _kernel(x, x)
    k_yy = _kernel(y, y)
    k_xy = _kernel(x, y)
    return float(np.mean(k_xx) + np.mean(k_yy) - 2.0 * np.mean(k_xy))


def _hist_prob(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    hist, _ = np.histogram(values, bins=bins)
    hist = hist.astype(np.float64)
    if hist.sum() <= 0:
        return np.ones_like(hist) / len(hist)
    return hist / hist.sum()


def _build_embedding_metrics_snapshot(
    reductions: dict[str, np.ndarray],
    labels: list[str],
) -> str:
    from scipy.stats import ks_2samp, wasserstein_distance

    methods = ["PCA", "t-SNE", "UMAP"]
    unique_labels = sorted(set(labels))
    label_arr = np.asarray(labels)

    lines: list[str] = []
    lines.append("EMBEDDING_ANALYSIS_INPUT")
    lines.append(f"methods={', '.join(methods)}")
    lines.append(f"classes={', '.join(unique_labels)}")

    for method in methods:
        pts = np.asarray(reductions[method], dtype=np.float64)
        lines.append(f"[{method}] class_mean_var")
        for cls in unique_labels:
            cls_pts = pts[label_arr == cls]
            if len(cls_pts) == 0:
                continue
            mean_vec = np.mean(cls_pts, axis=0)
            var_vec = np.var(cls_pts, axis=0)
            lines.append(
                f"{cls}: mean=({mean_vec[0]:.4f},{mean_vec[1]:.4f},{mean_vec[2]:.4f}), "
                f"var=({var_vec[0]:.4f},{var_vec[1]:.4f},{var_vec[2]:.4f})"
            )

    pairs = [("PCA", "t-SNE"), ("PCA", "UMAP"), ("t-SNE", "UMAP")]
    lines.append("[PAIRWISE_METHOD_METRICS]")
    for a, b in pairs:
        a_pts = np.asarray(reductions[a], dtype=np.float64)
        b_pts = np.asarray(reductions[b], dtype=np.float64)
        a_norm = np.linalg.norm(a_pts, axis=1)
        b_norm = np.linalg.norm(b_pts, axis=1)

        bins = np.linspace(min(a_norm.min(), b_norm.min()), max(a_norm.max(), b_norm.max()), 31)
        if np.allclose(bins[0], bins[-1]):
            bins = np.linspace(float(bins[0]) - 1.0, float(bins[0]) + 1.0, 31)

        p = _hist_prob(a_norm, bins)
        q = _hist_prob(b_norm, bins)

        js = _js_divergence(p, q)
        kl = _kl_divergence(p, q)
        mmd = _rbf_mmd(a_pts, b_pts)
        wass = float(wasserstein_distance(a_norm, b_norm))
        ph_stat, ph_p = ks_2samp(a_norm, b_norm)

        lines.append(
            f"{a} vs {b}: JS={js:.6f}, MMD={mmd:.6f}, Wasserstein={wass:.6f}, "
            f"KL={kl:.6f}, PH_KS_stat={float(ph_stat):.6f}, PH_p={float(ph_p):.6f}"
        )

    return "\n".join(lines)


def _build_detail_gemma_prompt(class_names: list[str]) -> str:
    classes_text = ", ".join(sorted(set(class_names)))
    return (
        "You are analyzing a classification model from embedding statistics only. "
        "Do not use or infer any ground-truth labels or accuracy values. "
        f"Observed classes: {classes_text}. "
        "Use PCA/t-SNE/UMAP class mean/variance and JS/MMD/Wasserstein/KL/PH metrics to assess model health. "
        "Answer briefly with: 1) key risks, 2) update recommendation reason, 3) whether immediate update is needed. "
        "Final line must be: UPDATE_URGENCY_PERCENT=<0-100>."
    )


def _extract_update_urgency_percent(response_text: str) -> int:
    match = re.search(r"UPDATE_URGENCY_PERCENT\s*=\s*(\d{1,3})", response_text, flags=re.IGNORECASE)
    if match:
        return max(0, min(100, int(match.group(1))))

    fallback = re.findall(r"(\d{1,3})\s*%", response_text)
    if fallback:
        return max(0, min(100, int(fallback[-1])))
    return 50


def _call_hf_gemma4_analysis_text_only(
    prompt_text: str,
    snapshot_text: str,
    model_name: str,
    api_key: str,
) -> str:
    client = InferenceClient(api_key=api_key)
    safe_prompt = _sanitize_gemma_text(prompt_text)
    safe_snapshot = _sanitize_gemma_text(snapshot_text)
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": safe_prompt + "\n\n" + safe_snapshot,
            }
        ],
        max_tokens=700,
    )

    message = completion.choices[0].message
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                chunks.append(item["text"])
        return "\n".join(chunks).strip()
    return str(content)


def _render_detail_3d_visualization(selected_records: list[dict[str, Any]]) -> None:
    try:
        import plotly.graph_objects as go
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        st.error("Required libraries are missing. Please install scikit-learn and plotly.")
        return

    st.subheader("3D Model Prediction Visualization")
    st.caption("A TensorFlow Projector-style 3D feature space visualization.")

    if len(selected_records) < 3:
        st.info("Select at least 3 images to use 3D Visualization.")
        return

    image_paths = [record["path"] for record in selected_records if record["exists"]]
    if not image_paths:
        st.warning("There are no images to display.")
        return

    base_model_dir = _get_detail_base_model_dir(selected_records)
    selected_model_dirs = _resolve_selected_model_dirs(selected_records)
    if len(selected_model_dirs) > 1:
        st.warning("The selected images were generated with different inference models. The 3D features will be computed using the first model.")

    with st.spinner("Extracting features from the model..."):
        features, processed_paths = _extract_features_from_images(image_paths, base_model_dir)

    if features is None or len(features) == 0:
        st.error("Failed to extract features.")
        return

    record_by_path = {record["path"]: record for record in selected_records if record["exists"]}
    plotted_records = [record_by_path[path] for path in processed_paths if path in record_by_path]
    if not plotted_records:
        st.error("Could not find label information for visualization.")
        return

    labels = [_extract_prediction_label(record) for record in plotted_records]
    if features.ndim > 2:
        features = features.reshape(features.shape[0], -1)
    elif features.ndim == 1:
        features = features.reshape(-1, 1)

    reduction_options = ["PCA", "t-SNE", "UMAP"]
    _apply_requested_selectbox_value(
        "detail_3d_reduction_requested",
        "detail_3d_reduction_method",
        reduction_options,
        "PCA",
    )

    col1, col2 = st.columns(2)
    with col1:
        reduction_method = st.selectbox(
            "Dimensionality reduction",
            reduction_options,
            key="detail_3d_reduction_method",
            help="Choose how high-dimensional features should be reduced into 3D.",
        )
    with col2:
        show_labels = st.checkbox("Show labels", value=False)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    with st.spinner("Computing PCA/t-SNE/UMAP embeddings..."):
        reductions = _build_three_reductions(features_scaled)

    if reduction_method == "PCA":
        reduction_info = "PCA"
        features_3d = reductions["PCA"]
    elif reduction_method == "t-SNE":
        reduction_info = "t-SNE"
        features_3d = reductions["t-SNE"]
    else:
        reduction_info = "UMAP"
        features_3d = reductions["UMAP"]

    features_3d_display = _normalize_for_display(features_3d)

    unique_labels = [label for label in CLASS_VISUALIZATION_ORDER if label in set(labels)]
    unique_labels.extend(sorted(set(labels) - set(unique_labels)))
    color_map = _get_discrete_class_colors(labels)

    fig = go.Figure()
    for label in unique_labels:
        class_indices = [idx for idx, item in enumerate(labels) if item == label]
        if not class_indices:
            continue

        class_records = [plotted_records[idx] for idx in class_indices]
        hover_text = [
            f"{record['label']}<br>File: {record['filename']}"
            for record in class_records
        ]

        fig.add_trace(
            go.Scatter3d(
                x=features_3d_display[class_indices, 0],
                y=features_3d_display[class_indices, 1],
                z=features_3d_display[class_indices, 2],
                mode="markers+text" if show_labels else "markers",
                name=label,
                marker=dict(
                    size=8,
                    color=color_map[label],
                    line=dict(width=0.5, color="white"),
                ),
                text=[label] * len(class_indices) if show_labels else None,
                textposition="top center",
                hovertext=hover_text,
                hovertemplate="%{hovertext}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"3D Model Feature Space ({reduction_info})",
        scene=dict(
            xaxis=dict(title="Feature 1"),
            yaxis=dict(title="Feature 2"),
            zaxis=dict(title="Feature 3"),
            aspectmode="cube",
            camera=dict(
                eye=dict(x=1.7, y=1.5, z=1.4),
                projection=dict(type="perspective"),
            ),
        ),
        width=1000,
        height=700,
        hovermode="closest",
        dragmode="orbit",
        legend=dict(title="Class"),
    )

    st.plotly_chart(fig, width="stretch")

    st.subheader("Class Distribution")
    class_counts = pd.DataFrame(
        {"Class": unique_labels, "Count": [labels.count(label) for label in unique_labels]}
    )
    class_distribution_fig = go.Figure(
        data=[
            go.Bar(
                x=class_counts["Class"],
                y=class_counts["Count"],
                marker_color=[color_map[label] for label in class_counts["Class"]],
                text=class_counts["Count"],
                textposition="outside",
                hovertemplate="Class: %{x}<br>Count: %{y}<extra></extra>",
                showlegend=False,
            )
        ]
    )
    class_distribution_fig.update_layout(
        xaxis_title="Class",
        yaxis_title="Count",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(class_distribution_fig, width="stretch")

    with st.expander("Image details", expanded=False):
        details_df = pd.DataFrame(
            {
                "Filename": [record["filename"] for record in plotted_records],
                "Prediction": labels,
                "Path": [record["path"] for record in plotted_records],
            }
        )
        st.dataframe(details_df, width="stretch")

    st.subheader("Gemma4 Model Update Urgency (Hugging Face)")
    st.caption("Uses PCA/t-SNE/UMAP mean/variance statistics and JS, MMD, Wasserstein, KL, PH metrics.")

    hf_model_name = st.text_input(
        "Gemma4 model",
        value="google/gemma-4-31B-it:novita",
        key="detail_gemma_model_name",
    )
    hf_token_env = _resolve_default_hf_token()
    hf_token_input = st.text_input(
        "HF token",
        value=hf_token_env,
        type="password",
        key="detail_gemma_hf_token",
    )

    default_prompt = _build_detail_gemma_prompt(labels)
    gemma_prompt = st.text_area(
        "Analysis prompt",
        value=default_prompt,
        height=140,
        key="detail_gemma_prompt",
    )

    if st.button("Analyze Update Urgency with Gemma4", key="detail_gemma_analyze"):
        api_key = (hf_token_input or "").strip() or hf_token_env
        if not api_key:
            st.error("HF token is required. Set HF_TOKEN (or HUGGINGFACEHUB_API_TOKEN) or enter token above.")
        else:
            try:
                with st.spinner("Building metrics and calling Gemma4..."):
                    snapshot_text = _build_embedding_metrics_snapshot(reductions=reductions, labels=labels)
                    response_text = _call_hf_gemma4_analysis_text_only(
                        prompt_text=gemma_prompt,
                        snapshot_text=snapshot_text,
                        model_name=hf_model_name,
                        api_key=api_key,
                    )
                    urgency_percent = _extract_update_urgency_percent(response_text)

                st.session_state["detail_gemma_urgency_percent"] = urgency_percent
                st.session_state["detail_gemma_analysis_text"] = response_text
            except Exception as exc:
                st.error(f"Gemma4 analysis failed: {exc}")

    if "detail_gemma_urgency_percent" in st.session_state:
        urgency_percent = int(st.session_state["detail_gemma_urgency_percent"])
        st.metric("Update Urgency Percent", f"{urgency_percent}%")
        if urgency_percent >= 100:
            st.error("100% means immediate model update is required.")
        elif urgency_percent >= 70:
            st.warning("High urgency: model update is recommended soon.")
        else:
            st.success("Current model is relatively stable based on embedding metrics.")

    if st.session_state.get("detail_gemma_analysis_text"):
        st.markdown(st.session_state["detail_gemma_analysis_text"])


def _resolve_target_label_index(model: Any, label_name: str, input_tensor: Any) -> int:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("The torch package required for XAI is not installed.") from exc

    label2id = getattr(model.config, "label2id", {}) or {}
    if label_name in label2id:
        return int(label2id[label_name])

    with torch.no_grad():
        logits = model(pixel_values=input_tensor).logits
        return int(logits.argmax(dim=-1).item())


def _render_bottom_right_pagination_controls(
    *,
    total_items: int,
    page_key: str,
    page_size_key: str,
    default_page_size: int,
    show_total_pages_in_size_field: bool = False,
) -> tuple[int, int]:
    if page_size_key not in st.session_state:
        st.session_state[page_size_key] = int(default_page_size)
    if page_key not in st.session_state:
        st.session_state[page_key] = 1

    page_size = int(st.session_state[page_size_key])
    total_pages = max(1, (total_items + page_size - 1) // page_size)
    current_page = int(st.session_state[page_key])
    current_page = max(1, min(current_page, total_pages))
    st.session_state[page_key] = current_page
    page_display_key = f"{page_key}_display"
    page_size_display_key = f"{page_size_key}_display"
    st.session_state[page_display_key] = str(current_page)
    size_field_value = total_pages if show_total_pages_in_size_field else page_size
    st.session_state[page_size_display_key] = str(size_field_value)

    spacer, controls_col = st.columns([6, 4], gap="small")
    with spacer:
        st.empty()
    with controls_col:
        label_cols = st.columns([2, 2], gap="small")
        with label_cols[0]:
            st.caption("Page")
            page_cols = st.columns([3, 1, 1], gap="small")
            with page_cols[0]:
                st.text_input(
                    "Current page",
                    key=page_display_key,
                    label_visibility="collapsed",
                    disabled=True,
                )
            with page_cols[1]:
                if st.button("<", key=f"{page_key}_minus", width="stretch"):
                    st.session_state[page_key] = max(1, current_page - 1)
                    st.rerun()
            with page_cols[2]:
                if st.button(">", key=f"{page_key}_plus", width="stretch"):
                    st.session_state[page_key] = min(total_pages, current_page + 1)
                    st.rerun()

        with label_cols[1]:
            st.caption("Total Pages" if show_total_pages_in_size_field else "Page Size")
            page_size_cols = st.columns([3, 1, 1], gap="small")
            with page_size_cols[0]:
                st.text_input(
                    "Page size",
                    key=page_size_display_key,
                    label_visibility="collapsed",
                    disabled=True,
                )
            with page_size_cols[1]:
                st.empty()
            with page_size_cols[2]:
                st.empty()

    page_size = int(st.session_state[page_size_key])
    total_pages = max(1, (total_items + page_size - 1) // page_size)
    current_page = int(st.session_state[page_key])
    st.session_state[page_key] = max(1, min(current_page, total_pages))
    return st.session_state[page_key], page_size


def _render_detail_xai_visualization(
    selected_records: list[dict[str, Any]],
    selected_model_dir: Path | None,
) -> None:
    try:
        import numpy as np
        import torch
        import torch.nn.functional as F
        from PIL import Image
        from matplotlib import colormaps
        from openxai import Explainer
    except ImportError as exc:
        st.error(
            "The packages required for XAI visualization could not be loaded. "
            "Please make sure `captum` and the OpenXAI dependencies are installed."
        )
        st.caption(f"Import error: {exc}")
        return

    st.subheader("XAI Visualization (OpenXAI)")
    st.caption("Overlays a heatmap on the original image using OpenXAI attributions.")

    if not selected_records:
        st.info("There are no images to display for XAI.")
        return

    if selected_model_dir is None:
        st.info("Please select an inference model in the sidebar first.")
        return

    openxai_methods = ["grad", "sg", "itg", "ig", "lime", "shap", "control"]
    supported_methods = {"grad", "sg", "itg", "ig"}
    _apply_requested_selectbox_value(
        "detail_xai_method_requested",
        "detail_xai_method_selector",
        openxai_methods,
        "grad",
    )
    selected_method = st.selectbox(
        "OpenXAI method",
        options=openxai_methods,
        key="detail_xai_method_selector",
        help="For the current image classification model, grad, sg, itg, and ig are recommended.",
    )
    overlay_alpha = st.slider(
        "Heatmap overlay strength",
        min_value=0.1,
        max_value=0.9,
        value=0.45,
        step=0.05,
        key="detail_xai_overlay_alpha",
    )
    colormap_name = st.selectbox(
        "Heatmap colormap",
        options=["turbo", "jet", "magma", "viridis"],
        index=0,
        key="detail_xai_colormap",
    )

    if selected_method not in supported_methods:
        st.warning(
            "The selected method is not directly supported by the current image model pipeline. "
            "Please choose from `grad`, `sg`, `itg`, or `ig`."
        )
        return

    valid_records = [record for record in selected_records if record.get("exists")]
    if not valid_records:
        st.info("XAI cannot be computed because no existing images were found.")
        return

    total_images = len(valid_records)
    current_xai_page = int(st.session_state.get("detail_xai_page", 1))
    xai_page_size = int(st.session_state.get("detail_xai_page_size", 5))
    xai_total_pages = max(1, (total_images + xai_page_size - 1) // xai_page_size)
    current_xai_page = max(1, min(current_xai_page, xai_total_pages))
    st.session_state["detail_xai_page"] = current_xai_page
    st.session_state["detail_xai_page_size"] = xai_page_size
    page_start = (current_xai_page - 1) * xai_page_size
    page_end = page_start + xai_page_size
    page_records = valid_records[page_start:page_end]

    st.caption(f"{total_images} images total | Page {current_xai_page}/{xai_total_pages}")

    try:
        image_processor, base_model, device_name, _ = _load_detail_torch_classifier_runtime(selected_model_dir)
        device = torch.device(device_name)

        class _OpenXAILogitsModel(torch.nn.Module):
            def __init__(self, model: Any) -> None:
                super().__init__()
                self.model = model

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                output = self.model(pixel_values=x)
                return output.logits if hasattr(output, "logits") else output

        openxai_model = _OpenXAILogitsModel(base_model).to(device)
        explainer = Explainer(method=selected_method, model=openxai_model)
    except Exception as exc:
        st.error(f"Failed to initialize the OpenXAI explainer: {exc}")
        return

    cmap = colormaps.get_cmap(colormap_name)
    xai_errors: list[str] = []
    page_items: list[dict[str, Any]] = []

    for record in page_records:
        image_path = record["path"]
        filename = record.get("filename", Path(image_path).name)
        try:
            with Image.open(image_path) as image:
                rgb_image = image.convert("RGB")
            original_np = np.asarray(rgb_image).astype(np.float32) / 255.0
            height, width = original_np.shape[:2]

            model_inputs = image_processor(images=rgb_image, return_tensors="pt")
            pixel_values = model_inputs["pixel_values"].to(device)
            pixel_values = pixel_values.requires_grad_(True)

            target_idx = _resolve_target_label_index(base_model, str(record["label"]), pixel_values)
            target_tensor = torch.tensor([target_idx], dtype=torch.long, device=device)

            attribution = explainer.get_explanations(pixel_values, target_tensor)
            attribution = attribution.detach().to("cpu")

            if attribution.ndim == 4:
                attribution_map = attribution[0].abs().mean(dim=0)
            elif attribution.ndim == 3:
                attribution_map = attribution[0].abs()
            else:
                raise ValueError(f"Unexpected attribution shape: {tuple(attribution.shape)}")

            attribution_map = attribution_map.unsqueeze(0).unsqueeze(0)
            attribution_map = F.interpolate(
                attribution_map,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )[0, 0].numpy()

            attribution_min = float(attribution_map.min())
            attribution_max = float(attribution_map.max())
            normalized = (attribution_map - attribution_min) / (attribution_max - attribution_min + 1e-8)

            heatmap_rgb = cmap(normalized)[..., :3]
            overlay = np.clip((1.0 - overlay_alpha) * original_np + overlay_alpha * heatmap_rgb, 0.0, 1.0)

            page_items.append(
                {
                    "filename": filename,
                    "label": record["label"],
                    "target_idx": target_idx,
                    "original": (original_np * 255).astype(np.uint8),
                    "heatmap": (heatmap_rgb * 255).astype(np.uint8),
                    "overlay": (overlay * 255).astype(np.uint8),
                }
                )
        except Exception as exc:
            xai_errors.append(f"{filename}: {exc}")

    for item in page_items:
        st.markdown(f"**{item['filename']}**")
        _, c1, c2, c3, _ = st.columns([0.5, 1, 1, 1, 0.5])
        with c1:
            st.image(item["original"], caption="Original", width="stretch")
        with c2:
            st.image(item["heatmap"], caption=f"XAI ({selected_method})", width="stretch")
        with c3:
            st.image(item["overlay"], caption="Overlay", width="stretch")
        st.caption(f"Prediction: {item['label']} | Target ID: {item['target_idx']}")

    _render_bottom_right_pagination_controls(
        total_items=total_images,
        page_key="detail_xai_page",
        page_size_key="detail_xai_page_size",
        default_page_size=5,
        show_total_pages_in_size_field=True,
    )

    if xai_errors:
        st.warning("Some XAI computations failed: " + "; ".join(xai_errors[:3]))


def _load_ad_calibration() -> tuple[float, float, float]:
    """heatmap vmin/vmax와 Normal/Abnormal 판정 threshold를 heatmap_range.json에서 읽는다.

    둘 다 scripts/AD/find_heapmap_range.py가 정상 이미지 score 분포의 percentile로
    미리 계산해둔 값이다 (anomaly_score.py의 calibrate_score_range() /
    calibrate_image_score_threshold()와 동일한 방식). anomaly_threshold가 json에
    없으면(구버전 캘리브레이션 결과) vmax로 대체한다.
    """
    if AD_HEATMAP_RANGE_PATH.exists():
        try:
            data = json.loads(AD_HEATMAP_RANGE_PATH.read_text(encoding="utf-8"))
            vmin = float(data["vmin"])
            vmax = float(data["vmax"])
            anomaly_threshold = float(data.get("anomaly_threshold", vmax))
            return vmin, vmax, anomaly_threshold
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
    return 0.0, 1.0, 1.0


def _resolve_ad_agg_feature_path(image_path: str) -> Path:
    return AD_AGG_DIR / f"{Path(image_path).stem}_local.npy"


class _AdProgressSink:
    """File-like object that redirects tqdm's progress output into a Streamlit placeholder."""

    def __init__(self, placeholder: Any) -> None:
        self._placeholder = placeholder

    def write(self, text: str) -> int:
        cleaned = text.strip("\r\n")
        if cleaned:
            self._placeholder.text(cleaned)
        return len(text)

    def flush(self) -> None:
        pass


def _run_subprocess_with_live_progress(cmd: list[str], cwd: Path, placeholder: Any) -> None:
    """Run cmd, streaming its stdout/stderr (incl. tqdm's \\r-based updates) into placeholder."""
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    buffer = ""
    tail_lines: list[str] = []
    while True:
        char = process.stdout.read(1)
        if char == "" and process.poll() is not None:
            break
        if char in ("\r", "\n"):
            cleaned = buffer.strip()
            if cleaned:
                placeholder.text(cleaned)
                tail_lines.append(cleaned)
            buffer = ""
        else:
            buffer += char
    if buffer.strip():
        placeholder.text(buffer.strip())
        tail_lines.append(buffer.strip())

    if process.poll() != 0:
        raise RuntimeError("\n".join(tail_lines[-30:]) or f"Process exited with code {process.poll()}")


def _run_ad_feature_extraction_pipeline(image_paths: list[str], progress_placeholder: Any) -> None:
    """Run Step1 (feature extraction) + Step2 (PatchCore aggregation) for exactly image_paths.

    Mirrors: bash scripts/AD/run_feature_pipeline.sh
             RUN_EXTRACT=1 RUN_AGGREGATION=1 BUILD_MEMORY_BANK=0
    (BUILD_MEMORY_BANK has no CLI flag to force 0; it already defaults to 0.)

    run_feature_pipeline.sh's Step1 (wide_resnet_img_feature_extract.py) only accepts a whole
    --img_dir to scan or a --csv_path list; it never took the selected images before, so it
    silently fell back to its hardcoded default --img_dir (data/images) regardless of what was
    selected/queried from the DB. To target exactly image_paths (which can live anywhere, e.g.
    DB-queried paths outside data/images), write them to a throwaway CSV and use --input_mode csv.
    Streams the script's own tqdm progress (from wide_resnet_img_feature_extract.py) live.
    """
    import csv
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", prefix="ad_feature_extract_", delete=False, newline="", encoding="utf-8"
    ) as tmp_csv:
        writer = csv.writer(tmp_csv)
        writer.writerow(["data_path", "labels"])
        for path in image_paths:
            writer.writerow([path, "selected"])
        tmp_csv_path = Path(tmp_csv.name)

    try:
        cmd = [
            "bash",
            str(AD_PIPELINE_SCRIPT),
            "--input_mode", "csv",
            "--csv_path", str(tmp_csv_path),
            "--path_col", "data_path",
            "--label_col", "labels",
            "--label_filter", "selected",
            "--run_extract", "1",
            "--run_aggregation", "1",
            # run_feature_pipeline.sh hardcodes non-empty defaults for --raw_dir/--agg_dir/
            # --raw_input1/--agg_output, so --feature_root alone is silently ignored. All four
            # must be passed explicitly to redirect Step1+2 output into AD_FEATURE_ROOT.
            "--raw_dir", str(AD_RAW_DIR),
            "--agg_dir", str(AD_AGG_DIR),
            "--raw_input1", str(AD_RAW_DIR),
            "--agg_output", str(AD_AGG_DIR),
        ]
        progress_placeholder.text("Step1+2: starting feature extraction + PatchCore aggregation...")
        _run_subprocess_with_live_progress(cmd, ROOT_DIR, progress_placeholder)
    finally:
        tmp_csv_path.unlink(missing_ok=True)


def _run_ad_scoring_for_images(image_paths: list[str], progress_placeholder: Any) -> dict[str, Any]:
    """Score each image with anomaly_score.py's PatchCoreScorer against the memory bank."""
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from tqdm import tqdm

    if str(AD_SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(AD_SCRIPTS_DIR))
    from anomaly_score import (
        PatchCoreScorer,
        foreground_mask_from_image,
        load_memory_bank,
        load_test_feature,
        save_result,
    )

    vmin, vmax, _ = _load_ad_calibration()
    results: dict[str, Any] = {}
    errors: list[str] = []

    if not AD_MEMORY_BANK_PATH.exists():
        raise RuntimeError(f"Memory bank not found: {AD_MEMORY_BANK_PATH}")

    bank = load_memory_bank(AD_MEMORY_BANK_PATH)
    scorer = PatchCoreScorer(bank)

    sink = _AdProgressSink(progress_placeholder)
    for image_path in tqdm(image_paths, desc="Anomaly scoring", unit="img", file=sink, mininterval=0.0):
        feature_path = _resolve_ad_agg_feature_path(image_path)
        filename = Path(image_path).name
        if not feature_path.exists():
            errors.append(f"{filename}: feature file not found ({feature_path.name})")
            continue
        try:
            test_feat, grid = load_test_feature(feature_path)
            with Image.open(image_path) as image:
                out_size = (image.height, image.width)
                fg_mask_full = foreground_mask_from_image(image)

            fg_mask_grid = F.adaptive_max_pool2d(
                torch.from_numpy(fg_mask_full).float()[None, None], output_size=grid
            ).squeeze(1).bool()

            score_result = scorer.score(test_feat, grid=grid, out_size=out_size, fg_mask=fg_mask_grid)
            image_score = float(score_result["image_scores"][0])
            anomaly_map = score_result["anomaly_map"][0].numpy()
            patch_scores = score_result["patch_scores"][0].numpy()

            stem = feature_path.stem
            save_result(
                Path(image_path), anomaly_map, image_score, AD_OUTPUT_DIR / stem, stem,
                fg_mask=fg_mask_full, vmin=vmin, vmax=vmax,
            )

            results[image_path] = {
                "image_score": image_score,
                "anomaly_map": anomaly_map,
                "patch_scores": patch_scores,
                "grid": grid,
                "fg_mask": fg_mask_full,
                "feature_path": str(feature_path),
                "anomaly_map_png": str(AD_OUTPUT_DIR / stem / f"{stem}_anomaly_map.png"),
                "anomaly_map_npy": str(AD_OUTPUT_DIR / stem / f"{stem}_anomaly_map.npy"),
            }
        except Exception as exc:
            errors.append(f"{filename}: {exc}")

    return {"results": results, "errors": errors, "vmin": vmin, "vmax": vmax}


def _get_cached_ad_pipeline_data(selected_records: list[dict[str, Any]]) -> dict[str, Any] | None:
    image_paths = [record["path"] for record in selected_records if record.get("exists")]
    signature = tuple(sorted(image_paths))
    if st.session_state.get("ad_pipeline_signature") == signature:
        return st.session_state.get("ad_pipeline_data")
    return None


def _log_detail_ad_run(
    selected_records: list[dict[str, Any]],
    feature_extraction_performed: bool,
    missing_feature_count: int,
    pipeline_data: dict[str, Any] | None,
    error: str | None,
) -> None:
    content = (
        f"Anomaly detection run on {len(selected_records)} selected image(s). "
        f"Images: {_summarize_selected_filenames(selected_records)}. "
    )
    content += (
        f"Feature extraction performed for {missing_feature_count} image(s) missing from "
        f"`{_to_project_relative_path(AD_AGG_DIR)}`."
        if feature_extraction_performed
        else f"Feature extraction skipped (features already present in `{_to_project_relative_path(AD_AGG_DIR)}`)."
    )
    if error:
        content += f" Pipeline failed: {error}"
    elif pipeline_data is not None:
        errors = pipeline_data.get("errors", [])
        content += f" Scored {len(pipeline_data.get('results', {}))} image(s)."
        if errors:
            content += f" {len(errors)} image(s) failed scoring."

    _append_app_log(
        log_type="error" if error else "done",
        source="Analysis Anomaly Detection",
        content=content,
    )


def _render_ad_run_button(
    container: Any,
    selected_records: list[dict[str, Any]],
    progress_placeholder: Any,
) -> None:
    run_clicked = container.button(
        "Run",
        key="ad_run_button",
        width="stretch",
        disabled=not selected_records,
    )
    if not run_clicked:
        return

    image_paths = [record["path"] for record in selected_records if record.get("exists")]
    missing_features: list[str] = []
    try:
        missing_features = [
            path for path in image_paths if not _resolve_ad_agg_feature_path(path).exists()
        ]
        if missing_features:
            _run_ad_feature_extraction_pipeline(missing_features, progress_placeholder)
        else:
            progress_placeholder.text(
                f"[Step1+2 건너뜀] {len(image_paths)}개 이미지 feature가 이미 "
                f"{AD_AGG_DIR}에 있어 재추출을 생략합니다."
            )
        pipeline_data = _run_ad_scoring_for_images(image_paths, progress_placeholder)
    except Exception as exc:
        progress_placeholder.empty()
        st.error(f"Anomaly detection pipeline failed: {exc}")
        _log_detail_ad_run(selected_records, bool(missing_features), len(missing_features), None, str(exc))
        return

    progress_placeholder.empty()
    st.session_state["ad_pipeline_signature"] = tuple(sorted(image_paths))
    st.session_state["ad_pipeline_data"] = pipeline_data
    if pipeline_data["errors"]:
        st.warning("Some images could not be scored: " + "; ".join(pipeline_data["errors"][:3]))
    else:
        st.success(f"Anomaly detection completed for {len(pipeline_data['results'])} image(s).")
    _log_detail_ad_run(selected_records, bool(missing_features), len(missing_features), pipeline_data, None)
    st.rerun()


def _render_ad_result_tab(
    page_records: list[dict[str, Any]],
    pipeline_data: dict[str, Any] | None,
    threshold: float,
) -> None:
    if pipeline_data is None:
        st.info("Click **Run Anomaly Detection** in the sidebar to score the selected images.")
        return

    results = pipeline_data["results"]
    cols = st.columns(5, gap="large")
    for idx, record in enumerate(page_records):
        with cols[idx % 5]:
            if record["exists"]:
                st.image(record["path"], width="stretch")
            else:
                st.info("Image not found")
            st.caption(record["filename"])

            image_result = results.get(record["path"])
            if image_result is None:
                st.caption("Prediction: -")
            else:
                label = "Abnormal" if image_result["image_score"] > threshold else "Normal"
                st.caption(f"Prediction: {label}")


def _render_ad_3d_visualization(
    selected_records: list[dict[str, Any]],
    pipeline_data: dict[str, Any] | None,
    threshold: float,
) -> None:
    st.subheader("3D Feature Space (Per-Image)")
    st.caption(
        f"Each point is one image's feature from {AD_AGG_DIR} (patch-averaged, then PCA-compressed "
        "to a single 3D point). Red = abnormal, blue = normal."
    )

    if pipeline_data is None:
        st.info("Click **Run Anomaly Detection** in the sidebar first.")
        return

    results = pipeline_data["results"]
    valid_records = [record for record in selected_records if record["path"] in results]
    if len(valid_records) < 3:
        st.info("Select at least 3 images to use 3D Visualization.")
        return

    try:
        import plotly.graph_objects as go
        from sklearn.decomposition import PCA
    except ImportError:
        st.error("Required libraries are missing. Please install scikit-learn and plotly.")
        return

    image_vectors: list[np.ndarray] = []
    image_scores: list[float] = []
    filenames: list[str] = []
    for record in valid_records:
        feature_path = _resolve_ad_agg_feature_path(record["path"])
        feat = np.load(feature_path)                              # (1, C, Hf, Wf)
        channels = feat.shape[1]
        image_vector = feat[0].reshape(channels, -1).mean(axis=1)  # (C,) patch-averaged
        image_vectors.append(image_vector)
        image_scores.append(float(results[record["path"]]["image_score"]))
        filenames.append(record["filename"])

    vectors = np.stack(image_vectors, axis=0)                     # (N, C)
    scores = np.asarray(image_scores)                              # (N,)

    with st.spinner("Computing PCA embedding of per-image features..."):
        vectors_3d = PCA(n_components=3, random_state=42).fit_transform(vectors)  # (N, 3)

    is_abnormal = scores > threshold
    colors = np.where(is_abnormal, "red", "blue")

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=vectors_3d[:, 0],
                y=vectors_3d[:, 1],
                z=vectors_3d[:, 2],
                mode="markers",
                marker=dict(size=6, color=colors),
                text=[
                    f"{name}<br>score={score:.3f}<br>{'Abnormal' if abn else 'Normal'}"
                    for name, score, abn in zip(filenames, scores, is_abnormal)
                ],
                hovertemplate="%{text}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title="PCA of Per-Image Features (red=abnormal, blue=normal)",
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3", aspectmode="cube"),
        height=700,
        hovermode="closest",
    )
    st.plotly_chart(fig, width="stretch")


def _render_ad_xai_visualization(
    selected_records: list[dict[str, Any]],
    pipeline_data: dict[str, Any] | None,
    threshold: float,
) -> None:
    st.subheader("Anomaly Heatmap")
    st.caption("Replaces the OpenXAI panel for Anomaly Detection: shows the PatchCore anomaly map instead.")

    if pipeline_data is None:
        st.info("Click **Run Anomaly Detection** in the sidebar first.")
        return

    results = pipeline_data["results"]
    valid_records = [record for record in selected_records if record["path"] in results]
    if not valid_records:
        st.info("No scored images are available.")
        return

    from matplotlib import colormaps
    from PIL import Image

    overlay_alpha = st.slider(
        "Heatmap overlay strength", min_value=0.1, max_value=0.9, value=0.45, step=0.05, key="ad_xai_overlay_alpha"
    )
    # Matches anomaly_score.py's save_result(), which always renders with cmap="jet".
    cmap = colormaps.get_cmap("jet")
    vmin, vmax = float(pipeline_data["vmin"]), float(pipeline_data["vmax"])

    total_images = len(valid_records)
    current_page = int(st.session_state.get("ad_xai_page", 1))
    page_size = int(st.session_state.get("ad_xai_page_size", 5))
    total_pages = max(1, (total_images + page_size - 1) // page_size)
    current_page = max(1, min(current_page, total_pages))
    st.session_state["ad_xai_page"] = current_page
    st.session_state["ad_xai_page_size"] = page_size
    page_start = (current_page - 1) * page_size
    page_end = page_start + page_size
    page_records = valid_records[page_start:page_end]

    st.caption(f"{total_images} images total | Page {current_page}/{total_pages}")

    for record in page_records:
        result = results[record["path"]]
        image_score = float(result["image_score"])
        anomaly_map = result["anomaly_map"]
        fg_mask = result["fg_mask"]
        label = "Abnormal" if image_score > threshold else "Normal"

        with Image.open(record["path"]) as image:
            original_np = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0

        normalized = np.clip((anomaly_map - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
        heatmap_rgb = cmap(normalized)[..., :3]
        fg_mask_3d = fg_mask[..., None]
        heatmap_display = np.where(fg_mask_3d, heatmap_rgb, 0.5)
        overlay = np.where(
            fg_mask_3d,
            (1.0 - overlay_alpha) * original_np + overlay_alpha * heatmap_rgb,
            original_np,
        )

        st.markdown(f"**{record['filename']}** &nbsp;|&nbsp; Anomaly score: `{image_score:.4f}` &nbsp;|&nbsp; Prediction: **{label}**")
        _, c1, c2, c3, _ = st.columns([0.5, 1, 1, 1, 0.5])
        with c1:
            st.image((original_np * 255).astype(np.uint8), caption="Original", width="stretch")
        with c2:
            st.image((heatmap_display * 255).astype(np.uint8), caption="Anomaly Heatmap", width="stretch")
        with c3:
            st.image((overlay * 255).astype(np.uint8), caption="Overlay", width="stretch")

    _render_bottom_right_pagination_controls(
        total_items=total_images,
        page_key="ad_xai_page",
        page_size_key="ad_xai_page_size",
        default_page_size=5,
        show_total_pages_in_size_field=True,
    )


def render_detail_page(image_records) -> None:
    render_page_header("Analysis")
    all_dates = ["All dates"] + sorted({record["date"] for record in image_records}, reverse=True)
    all_classes = ["All classes"] + sorted({record["label"] for record in image_records})
    _apply_requested_selectbox_value(
        "detail_class_filter_requested",
        "detail_class_filter",
        all_classes,
        "All classes",
    )
    if st.session_state.get("detail_date_filter") not in all_dates:
        st.session_state["detail_date_filter"] = "All dates"

    method_options = ["Classification", "Anomaly Detection"]
    _apply_requested_selectbox_value(
        "detail_method_filter_requested",
        "detail_method_filter",
        method_options,
        "Classification",
    )
    method_col, method_run_button_col = st.columns([3, 1])
    with method_col:
        st.selectbox("Method", method_options, key="detail_method_filter")
    method = st.session_state["detail_method_filter"]
    method_run_button_placeholder = method_run_button_col.empty()
    st.markdown(
        "<style>.st-key-method_run_button_nudge { margin-top: 12px; }</style>",
        unsafe_allow_html=True,
    )

    filter_cols = st.columns([1, 1], gap="large")
    with filter_cols[0]:
        selected_date = st.selectbox("Date filter", all_dates, key="detail_date_filter")
    with filter_cols[1]:
        selected_class = st.selectbox("Class filter", all_classes, key="detail_class_filter")

    filtered = image_records
    if selected_date != "All dates":
        filtered = [record for record in filtered if record["date"] == selected_date]
    if selected_class != "All classes":
        filtered = [record for record in filtered if record["label"] == selected_class]

    if not filtered:
        st.info("No images matched the selected filter.")
        return

    select_options = []
    seen_paths = set()
    for record in filtered:
        if record["path"] not in seen_paths:
            select_options.append(record["path"])
            seen_paths.add(record["path"])

    previous_selected_paths = list(st.session_state.get("detail_selected_image_paths", []))
    selected_paths = [path for path in previous_selected_paths if path in select_options]
    selected_paths = st.multiselect(
        "Select images",
        options=select_options,
        default=selected_paths,
        format_func=lambda path: Path(path).name,
        key="detail_multi_select_paths",
    )
    if previous_selected_paths != selected_paths:
        _reset_detail_finetune_session(selected_paths)
    else:
        st.session_state["detail_selected_image_paths"] = selected_paths

    raw_selected_records = _get_detail_selected_records(image_records) if selected_paths else []
    if method == "Anomaly Detection":
        ad_progress_placeholder = st.empty()
        with method_run_button_placeholder.container():
            with st.container(key="method_run_button_nudge"):
                st.write("")
                _render_ad_run_button(st, raw_selected_records, ad_progress_placeholder)
    else:
        rerun_model_dir, _ = _render_detail_inference_model_selector(raw_selected_records)
        with method_run_button_placeholder.container():
            with st.container(key="method_run_button_nudge"):
                st.write("")
                classification_run_clicked = st.button(
                    "Run",
                    key="detail_classification_run_button",
                    width="stretch",
                    disabled=rerun_model_dir is None or not raw_selected_records,
                )
        if classification_run_clicked and rerun_model_dir is not None:
            _, prediction_errors, artifact_paths = _predict_detail_records_with_model(
                raw_selected_records, rerun_model_dir
            )
            _log_detail_classification_run(raw_selected_records, rerun_model_dir, prediction_errors, artifact_paths)
            st.rerun()
        elif rerun_model_dir is not None:
            cached_signature = (
                str(resolve_base_model_dir(rerun_model_dir)),
                tuple(record["path"] for record in raw_selected_records),
            )
            if st.session_state.get("detail_inference_prediction_signature") == cached_signature:
                raw_selected_records = st.session_state["detail_inference_prediction_records"]

    selected_records = []
    result_total = 0
    current_result_page = 1
    result_page_size = int(st.session_state.get("detail_result_page_size", 25))
    result_total_pages = 1
    result_start = 0
    result_end = result_page_size
    page_records: list[dict[str, Any]] = []
    selected_model_dir: Path | None = None
    ad_pipeline_data: dict[str, Any] | None = None
    ad_threshold: float = 0.0
    if selected_paths:
        if method == "Anomaly Detection":
            _, _, ad_threshold = _load_ad_calibration()
            ad_pipeline_data = _get_cached_ad_pipeline_data(raw_selected_records)
        else:
            selected_model_dir = _get_detail_base_model_dir(raw_selected_records) if raw_selected_records else None
        result_total = len(raw_selected_records)
        result_total_pages = max(1, (result_total + result_page_size - 1) // result_page_size)
        current_result_page = int(st.session_state.get("detail_result_page", 1))
        current_result_page = max(1, min(current_result_page, result_total_pages))
        st.session_state["detail_result_page"] = current_result_page
        st.session_state["detail_result_page_size"] = result_page_size

        result_start = (current_result_page - 1) * result_page_size
        result_end = result_start + result_page_size
        page_records = raw_selected_records[result_start:result_end]
        selected_records = raw_selected_records

        tab1, tab2, tab3 = st.tabs(["Result", "3D Visualization", "XAI"])

        result_total = len(selected_records)
        result_total_pages = max(1, (result_total + result_page_size - 1) // result_page_size)
        current_result_page = max(1, min(current_result_page, result_total_pages))
        st.session_state["detail_result_page"] = current_result_page
        st.session_state["detail_result_page_size"] = result_page_size
        result_start = (current_result_page - 1) * result_page_size
        result_end = result_start + result_page_size
        page_records = selected_records[result_start:result_end]

        with tab1:
            st.caption(f"{result_total} images total | Page {current_result_page}/{result_total_pages}")

            with st.expander(f"Selected images ({len(page_records)}/{result_total})", expanded=True):
                if method == "Anomaly Detection":
                    _render_ad_result_tab(page_records, ad_pipeline_data, ad_threshold)
                else:
                    cols = st.columns(5, gap="large")
                    for idx, record in enumerate(page_records):
                        with cols[idx % 5]:
                            if record["exists"]:
                                st.image(record["path"], width="stretch")
                            else:
                                st.info("Image not found")
                            st.caption(record["filename"])
                            st.caption(f"Prediction: {record['label']}")

            _render_bottom_right_pagination_controls(
                total_items=result_total,
                page_key="detail_result_page",
                page_size_key="detail_result_page_size",
                default_page_size=25,
                show_total_pages_in_size_field=True,
            )

        with tab2:
            if method == "Anomaly Detection":
                _render_ad_3d_visualization(selected_records, ad_pipeline_data, ad_threshold)
            elif len(selected_records) < 3:
                st.info("Select at least 3 images to use 3D Visualization.")
            else:
                _render_detail_3d_visualization(selected_records)

        with tab3:
            if method == "Anomaly Detection":
                _render_ad_xai_visualization(selected_records, ad_pipeline_data, ad_threshold)
            else:
                _render_detail_xai_visualization(selected_records, selected_model_dir)
    else:
        st.info("Select multiple items above to view multiple images.")

    if selected_records:
        st.caption("Interactive fine-tuning can be run on the `Fine-tuning` page.")


configure_page("Analysis")
if not bool(st.session_state.get("dashboard_data_loaded", False)):
    st.info("Please go to the Dashboard page and click **Load Data** first.")
    st.stop()
_query_date_start = str(st.session_state.get("dashboard_query_date_start", "")).strip() or None
_query_date_end = str(st.session_state.get("dashboard_query_date_end", "")).strip() or None
_config, _runs, image_records, _log_entries = load_dashboard_data(
    query_date_start=_query_date_start,
    query_date_end=_query_date_end,
)
render_detail_page(image_records)
