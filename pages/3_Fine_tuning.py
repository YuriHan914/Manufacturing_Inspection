from __future__ import annotations

import base64
import math
import random
import sys
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.detail_finetune_mcp import (
    DetailFineTunePlan,
    load_available_classes,
    resolve_base_model_dir,
    run_detail_finetune_plan,
)
from scripts.utils import (
    SUPABASE_CONNECTION_NAME,
    SUPABASE_IMAGE_TABLE,
    SupabaseConnection,
    _append_app_log,
    _format_display_path,
    _normalize_image_path_key,
    _predict_dashboard_labels,
    _render_classifier_model_selector,
    _to_project_relative_path,
    configure_page,
    load_dashboard_data,
    read_json_file,
    render_page_header,
)


@st.cache_data(show_spinner=False)
def _build_thumbnail_data_uri(image_path: str, size: tuple[int, int] = (96, 96)) -> str | None:
    try:
        from PIL import Image
    except ImportError:
        return None

    path = Path(image_path)
    if not path.exists():
        return None

    try:
        with Image.open(path) as image:
            rgb_image = image.convert("RGB")
            rgb_image.thumbnail(size)
            buffer = BytesIO()
            rgb_image.save(buffer, format="PNG")
    except Exception:
        return None

    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _build_unique_image_pool(image_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique_records: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for record in image_records:
        path = record["path"]
        if path in seen_paths:
            continue
        unique_records.append(dict(record))
        seen_paths.add(path)
    return unique_records


def _reset_fine_tuning_page_session(selected_paths: list[str]) -> None:
    prefix = "fine_tuning_page_"
    for key in list(st.session_state.keys()):
        if key.startswith(prefix):
            st.session_state.pop(key, None)
    st.session_state["fine_tuning_page_selected_image_paths"] = selected_paths
    st.session_state["fine_tuning_page_chat"] = []
    st.session_state["fine_tuning_page_plan"] = None
    st.session_state["fine_tuning_page_execution"] = None


def _build_path_signature(paths: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    normalized_paths = [
        normalized_path
        for normalized_path in (_normalize_image_path_key(path) for path in paths)
        if normalized_path
    ]
    return tuple(sorted(normalized_paths))


def _set_fine_tuning_selection_metadata(
    *,
    selected_paths: list[str],
    strategy: str,
    origin: str,
    notice: str = "",
    selection_percentage: int | None = None,
    model_dir: Path | str | None = None,
) -> None:
    st.session_state["fine_tuning_selection_strategy"] = strategy.strip() or "Manual Selection"
    st.session_state["fine_tuning_selection_origin"] = origin.strip() or "manual"
    st.session_state["fine_tuning_selection_notice"] = notice.strip()
    st.session_state["fine_tuning_selection_percentage_value"] = selection_percentage
    st.session_state["fine_tuning_selection_paths_signature"] = _build_path_signature(selected_paths)
    st.session_state["fine_tuning_selection_model_dir"] = (
        str(resolve_base_model_dir(model_dir)) if model_dir is not None else ""
    )


def _get_fine_tuning_selection_metadata(
    selected_records: list[dict[str, Any]],
    base_model_dir: Path | str,
) -> dict[str, Any]:
    selected_paths = [str(record["path"]) for record in selected_records]
    stored_signature = tuple(st.session_state.get("fine_tuning_selection_paths_signature", ()))
    current_signature = _build_path_signature(selected_paths)
    stored_model_dir = str(st.session_state.get("fine_tuning_selection_model_dir", "")).strip()
    current_model_dir = str(resolve_base_model_dir(base_model_dir))
    strategy = str(st.session_state.get("fine_tuning_selection_strategy", "")).strip() or "Manual Selection"
    origin = str(st.session_state.get("fine_tuning_selection_origin", "")).strip() or "manual"
    notice = str(st.session_state.get("fine_tuning_selection_notice", "")).strip()
    selection_percentage = st.session_state.get("fine_tuning_selection_percentage_value")

    manually_adjusted = bool(stored_signature) and stored_signature != current_signature
    model_changed_after_selection = bool(stored_model_dir) and stored_model_dir != current_model_dir
    if manually_adjusted or model_changed_after_selection:
        origin = "manual"
        strategy = "Manual Selection"
        adjustment_notes: list[str] = []
        if manually_adjusted:
            adjustment_notes.append("selected images were adjusted after auto sampling")
        if model_changed_after_selection:
            adjustment_notes.append("inference model changed after auto sampling")
        if adjustment_notes:
            extra_notice = "; ".join(adjustment_notes)
            notice = f"{notice} | {extra_notice}".strip(" |")

    return {
        "selection_strategy": strategy,
        "selection_origin": origin,
        "selection_notice": notice,
        "selection_percentage": selection_percentage,
        "selection_model_dir": _to_project_relative_path(current_model_dir),
        "selected_path_signature": list(current_signature),
    }


def _predict_image_pool_records_with_model(
    image_pool_records: list[dict[str, Any]],
    model_dir: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    resolved_model_dir = resolve_base_model_dir(model_dir)
    existing_image_paths = tuple(record["path"] for record in image_pool_records if record.get("exists"))
    predicted_labels_by_path: dict[str, str] = {}
    prediction_errors: list[str] = []

    if existing_image_paths:
        predicted_labels_by_path, _, _ = _predict_dashboard_labels(
            existing_image_paths,
            str(resolved_model_dir),
        )

    predicted_records: list[dict[str, Any]] = []
    for record in image_pool_records:
        updated_record = dict(record)
        updated_record["model_dir"] = str(resolved_model_dir)
        updated_record["model_dir_display"] = _to_project_relative_path(resolved_model_dir)

        predicted_label = predicted_labels_by_path.get(record["path"])
        if predicted_label:
            updated_record["label"] = predicted_label
        elif record.get("exists"):
            prediction_errors.append(f"{Path(record['path']).name}: prediction result missing")

        updated_record["predicted_label"] = str(updated_record.get("label", ""))
        predicted_records.append(updated_record)

    return predicted_records, prediction_errors


def _sync_records_with_model_dir(records: list[dict[str, Any]], model_dir: Path | str) -> list[dict[str, Any]]:
    resolved_model_dir = resolve_base_model_dir(model_dir)
    synchronized_records: list[dict[str, Any]] = []
    for record in records:
        updated_record = dict(record)
        updated_record["model_dir"] = str(resolved_model_dir)
        updated_record["model_dir_display"] = _to_project_relative_path(resolved_model_dir)
        synchronized_records.append(updated_record)
    return synchronized_records


def _select_margin_sampling_paths_for_fine_tuning(
    image_pool_records: list[dict[str, Any]],
    base_model_dir: Path | str,
    selection_percentage: int,
) -> tuple[list[str], pd.DataFrame]:
    scripts_dir = ROOT_DIR / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    from margin_sampling import select_margin_sampling_paths

    selectable_paths = [record["path"] for record in image_pool_records if record.get("exists")]
    sampled_paths, margin_frame = select_margin_sampling_paths(
        image_paths=selectable_paths,
        base_model_dir=resolve_base_model_dir(base_model_dir),
        selection_percentage=selection_percentage,
    )
    if not sampled_paths:
        return [], margin_frame

    sampled_path_keys = {_normalize_image_path_key(path) for path in sampled_paths}
    selected_paths = [
        record["path"]
        for record in image_pool_records
        if _normalize_image_path_key(record.get("path")) in sampled_path_keys
    ]
    return selected_paths, margin_frame


def _get_fine_tuning_label_overrides() -> dict[str, str]:
    stored_value = st.session_state.get("fine_tuning_label_overrides", {})
    if not isinstance(stored_value, dict):
        return {}
    return {
        str(path): str(label).strip()
        for path, label in stored_value.items()
        if str(path).strip() and str(label).strip()
    }


def _apply_fine_tuning_label_overrides(selected_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    label_overrides = _get_fine_tuning_label_overrides()
    labeled_records: list[dict[str, Any]] = []
    for record in selected_records:
        predicted_label = str(record.get("predicted_label") or record.get("label") or "").strip()
        assigned_label = label_overrides.get(record["path"], predicted_label)
        labeled_record = dict(record)
        labeled_record["predicted_label"] = predicted_label
        labeled_record["assigned_label"] = assigned_label
        labeled_record["label"] = assigned_label
        labeled_records.append(labeled_record)
    return labeled_records


def _save_fine_tuning_label_overrides(
    selected_records: list[dict[str, Any]],
    assigned_labels_by_path: dict[str, str],
) -> None:
    label_overrides = _get_fine_tuning_label_overrides()
    for record in selected_records:
        path = str(record["path"])
        predicted_label = str(record.get("predicted_label") or record.get("label") or "").strip()
        assigned_label = str(assigned_labels_by_path.get(path, predicted_label)).strip()
        if not assigned_label or assigned_label == predicted_label:
            label_overrides.pop(path, None)
        else:
            label_overrides[path] = assigned_label
    st.session_state["fine_tuning_label_overrides"] = label_overrides


def _build_fine_tuning_plan_from_labels(
    selected_records: list[dict[str, Any]],
    epochs: float,
    learning_rate: float,
    repeat_count: int,
) -> DetailFineTunePlan:
    label_counts = Counter(str(record.get("label") or "-") for record in selected_records)
    label_summary = ", ".join(f"{label}: {count}" for label, count in sorted(label_counts.items()))
    return DetailFineTunePlan(
        assistant_reply=f"Training will start immediately using the saved image labels. Label distribution: {label_summary}.",
        target_label=None,
        epochs=float(epochs),
        learning_rate=float(learning_rate),
        repeat_count=int(repeat_count),
        ready_to_train=bool(selected_records),
        create_new_class=False,
        new_class_name=None,
        notes="fine-tuning page saved labels",
    )


def _get_supabase_client() -> Any:
    if SupabaseConnection is None:
        raise RuntimeError("The `st_supabase_connection` package could not be found.")

    connection = st.connection(SUPABASE_CONNECTION_NAME, type=SupabaseConnection)
    client = getattr(connection, "client", None)
    if client is None:
        raise RuntimeError("The Supabase client is not available.")
    return client


def _is_missing_scalar(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _resolve_training_label_for_record(
    record: dict[str, Any],
    plan: DetailFineTunePlan | None,
    *,
    use_record_labels: bool,
) -> str:
    if use_record_labels:
        label = str(record.get("assigned_label") or record.get("label") or "").strip()
        if label:
            return label

    if plan is not None:
        plan_label = str(plan.new_class_name or plan.target_label or "").strip()
        if plan_label:
            return plan_label

    return str(record.get("assigned_label") or record.get("label") or "").strip()


def _sync_fine_tuning_records_to_supabase(
    selected_records: list[dict[str, Any]],
    plan: DetailFineTunePlan | None,
    *,
    use_record_labels: bool,
) -> dict[str, Any]:
    updates: list[dict[str, Any]] = []
    seen_targets: set[tuple[str, str]] = set()

    for record in selected_records:
        assigned_label = _resolve_training_label_for_record(record, plan, use_record_labels=use_record_labels)
        if not assigned_label:
            continue

        record_id = record.get("record_id")
        normalized_path = _normalize_image_path_key(record.get("path"))
        if _is_missing_scalar(record_id) and not normalized_path:
            continue

        if _is_missing_scalar(record_id):
            target_key = ("image_path", normalized_path)
            target_value = normalized_path
        else:
            target_key = ("id", str(record_id))
            target_value = record_id

        if target_key in seen_targets:
            continue
        seen_targets.add(target_key)
        updates.append(
            {
                "target_column": target_key[0],
                "target_value": target_value,
                "image_path": normalized_path,
                "filename": str(record.get("filename") or Path(normalized_path).name),
                "assigned_label": assigned_label,
            }
        )

    summary = {
        "attempted_count": len(updates),
        "updated_count": 0,
        "skipped_count": 0,
        "error_count": 0,
        "updated_records": [],
        "skipped_records": [],
        "errors": [],
    }
    if not updates:
        return summary

    client = _get_supabase_client()
    for update in updates:
        payload = {
            "trained": True,
            "class": update["assigned_label"],
        }
        try:
            query_builder = client.table(SUPABASE_IMAGE_TABLE).update(payload)
            query_builder = query_builder.eq(update["target_column"], update["target_value"])
            result = query_builder.execute()
            returned_rows = getattr(result, "data", None)
            if isinstance(returned_rows, list) and len(returned_rows) == 0:
                summary["skipped_count"] += 1
                summary["skipped_records"].append(
                    {
                        "filename": update["filename"],
                        "label": update["assigned_label"],
                        "reason": "no matching semiconductor row",
                    }
                )
                continue

            summary["updated_count"] += 1
            summary["updated_records"].append(
                {
                    "filename": update["filename"],
                    "label": update["assigned_label"],
                    "image_path": update["image_path"],
                }
            )
        except Exception as exc:
            summary["error_count"] += 1
            summary["errors"].append(f"{update['filename']}: {exc}")

    if summary["updated_count"] > 0:
        load_dashboard_data.clear()
    return summary


def _get_detail_manual_setting_defaults(base_model_dir: Path) -> dict[str, Any]:
    dataset_config = read_json_file(resolve_base_model_dir(base_model_dir) / "dataset_config.json", {})
    return {
        "epochs": max(1, min(5, int(round(float(dataset_config.get("num_epochs", 2.0)))))),
        "learning_rate": max(1e-6, min(1e-4, float(dataset_config.get("learning_rate", 1e-5)))),
        "repeat_count": max(4, min(64, int(dataset_config.get("interactive_repeat_count", 16)))),
    }


def _render_fine_tuning_training_panel(
    image_pool_records: list[dict[str, Any]],
    selected_records: list[dict[str, Any]],
    base_model_dir: Path,
    panel: Any | None = None,
    heading: str = "Interactive Fine-tuning",
    has_unsaved_label_changes: bool = False,
    available_classes: list[str] | None = None,
) -> None:
    panel = panel or st
    state_prefix = "fine_tuning_page"
    widget_token = f"{state_prefix}__{abs(hash(tuple(record['path'] for record in selected_records)))}"
    execution_state_key = f"{state_prefix}_execution"
    manual_epochs_key = f"{state_prefix}_manual_epochs_{widget_token}"
    manual_learning_rate_key = f"{state_prefix}_manual_learning_rate_{widget_token}"

    panel.subheader(heading)
    panel.caption(f"Base model: {_to_project_relative_path(base_model_dir)}")
    panel.caption(f"Labeled samples: {len(selected_records)}")

    if selected_records:
        label_counts = Counter(str(record.get("label") or "-") for record in selected_records)
        panel.caption("Saved labels: " + ", ".join(f"{label}: {count}" for label, count in sorted(label_counts.items())))

    resolved_available_classes = [
        str(label).strip()
        for label in (available_classes or [])
        if str(label).strip()
    ]
    class_reference_available = bool(resolved_available_classes)
    selected_label_set = {
        str(record.get("label") or "").strip()
        for record in selected_records
        if str(record.get("label") or "").strip()
    }
    has_new_class = class_reference_available and any(
        label not in resolved_available_classes for label in selected_label_set
    )
    if class_reference_available:
        incremental_selected_records = (
            list(selected_records)
            if has_new_class
            else [record for record in selected_records if not bool(record.get("trained", False))]
        )
    else:
        incremental_selected_records = list(selected_records)
        if selected_records:
            panel.info("The existing class list could not be loaded, so all selected data will be used for training.")

    if not has_new_class and class_reference_available:
        excluded_count = len(selected_records) - len(incremental_selected_records)
        panel.caption(
            "Training mode: existing-class incremental training (keep current weights and train only with additional data)"
        )
        if excluded_count > 0:
            panel.info(
                f"{excluded_count} images already marked as `trained=True` will be excluded, and only additional data will be used."
            )
    elif has_new_class:
        panel.caption("Training mode: includes new classes (train on all selected data)")

    if incremental_selected_records and len(incremental_selected_records) < 100:
        panel.warning(
            f"The current fine-tuning set contains {len(incremental_selected_records)} images. "
            "With fewer than 100 images, class bias may occur."
        )

    panel.divider()
    panel.write("Active Learning")
    panel.caption("Automatically selects images for training based on the chosen inference model using the available sampling strategy.")
    active_learning_notice = str(st.session_state.get(f"{state_prefix}_active_learning_notice", "")).strip()
    if active_learning_notice:
        panel.caption(active_learning_notice)

    strategy_col, slider_col = panel.columns([0.9, 1.1], gap="large")
    active_learning_strategy = strategy_col.selectbox(
        "Selection strategy",
        options=["Margin Sampling", "Random"],
        key=f"{state_prefix}_active_learning_strategy",
    )
    selection_percentage = slider_col.slider(
        "Selection rate (%)",
        min_value=1,
        max_value=100,
        value=int(st.session_state.get(f"{state_prefix}_active_learning_percentage", 10)),
        step=1,
        key=f"{state_prefix}_active_learning_percentage",
    )

    selectable_records = [record for record in image_pool_records if record.get("exists")]
    if panel.button(
        "Start active learning",
        key=f"{state_prefix}_active_learning_start",
        disabled=not selectable_records,
        width="stretch",
    ):
        try:
            if active_learning_strategy == "Margin Sampling":
                selected_paths, margin_frame = _select_margin_sampling_paths_for_fine_tuning(
                    image_pool_records=selectable_records,
                    base_model_dir=base_model_dir,
                    selection_percentage=selection_percentage,
                )
                if not selected_paths:
                    panel.warning("No images were selected by margin sampling.")
                else:
                    min_margin = float(margin_frame["margin_score"].min()) if not margin_frame.empty else 0.0
                    selection_notice = (
                        f"Margin Sampling selected {len(selected_paths)} images. "
                        f"Minimum margin={min_margin:.4f}"
                    )
                    _reset_fine_tuning_page_session(selected_paths)
                    st.session_state[f"{state_prefix}_active_learning_percentage"] = selection_percentage
                    st.session_state[f"{state_prefix}_active_learning_notice"] = selection_notice
                    _set_fine_tuning_selection_metadata(
                        selected_paths=selected_paths,
                        strategy="Margin Sampling",
                        origin="active_learning",
                        notice=selection_notice,
                        selection_percentage=selection_percentage,
                        model_dir=base_model_dir,
                    )
                    st.rerun()
            else:
                sample_count = max(1, math.ceil(len(selectable_records) * (selection_percentage / 100.0)))
                sample_count = min(sample_count, len(selectable_records))
                sampled_paths = set(random.sample([record["path"] for record in selectable_records], sample_count))
                selected_paths = [record["path"] for record in image_pool_records if record["path"] in sampled_paths]
                selection_notice = f"Random sampling selected {len(selected_paths)} images."
                _reset_fine_tuning_page_session(selected_paths)
                st.session_state[f"{state_prefix}_active_learning_percentage"] = selection_percentage
                st.session_state[f"{state_prefix}_active_learning_notice"] = selection_notice
                _set_fine_tuning_selection_metadata(
                    selected_paths=selected_paths,
                    strategy="Random",
                    origin="active_learning",
                    notice=selection_notice,
                    selection_percentage=selection_percentage,
                        model_dir=base_model_dir,
                    )
                st.rerun()
        except Exception as exc:
            panel.warning(f"An error occurred while running {active_learning_strategy}: {exc}")

    panel.divider()
    panel.write("Manual settings")
    manual_defaults = _get_detail_manual_setting_defaults(base_model_dir)
    default_epochs = int(manual_defaults["epochs"])
    if manual_epochs_key not in st.session_state:
        st.session_state[manual_epochs_key] = default_epochs
    else:
        st.session_state[manual_epochs_key] = max(
            1,
            min(5, int(round(float(st.session_state.get(manual_epochs_key, default_epochs))))),
        )
    if manual_learning_rate_key not in st.session_state:
        st.session_state[manual_learning_rate_key] = float(manual_defaults["learning_rate"])

    manual_epochs = panel.number_input(
        "Epochs",
        min_value=1,
        max_value=5,
        value=int(st.session_state.get(manual_epochs_key, default_epochs)),
        step=1,
        format="%d",
        key=manual_epochs_key,
    )
    manual_learning_rate = panel.number_input(
        "Learning rate",
        min_value=1e-6,
        max_value=1e-4,
        value=float(st.session_state.get(manual_learning_rate_key, manual_defaults["learning_rate"])),
        step=1e-6,
        format="%.6f",
        key=manual_learning_rate_key,
    )
    manual_repeat_count = int(manual_defaults["repeat_count"])

    if has_unsaved_label_changes:
        panel.info("Save the label changes in `Selected images` on the left before starting fine-tuning.")
    if selected_records and not incremental_selected_records:
        panel.warning("Without new classes, fine-tuning requires additional data with `trained=False`.")

    train_disabled = not incremental_selected_records or has_unsaved_label_changes
    if panel.button(
        "Start fine-tuning",
        key=f"{state_prefix}_start_{widget_token}",
        type="primary",
        disabled=train_disabled,
        width="stretch",
    ):
        live_log_placeholder = panel.empty()

        def _update_live_finetune_log(log_text: str) -> None:
            live_log_placeholder.text_area(
                "Training log",
                value=log_text[-12000:],
                height=220,
                disabled=True,
            )

        effective_plan = _build_fine_tuning_plan_from_labels(
            selected_records=incremental_selected_records,
            epochs=float(manual_epochs),
            learning_rate=float(manual_learning_rate),
            repeat_count=int(manual_repeat_count),
        )
        selection_metadata = _get_fine_tuning_selection_metadata(incremental_selected_records, base_model_dir)
        request_summary = (
            f"base_model={_to_project_relative_path(base_model_dir)}, "
            f"epochs={float(manual_epochs):.1f}, "
            f"learning_rate={float(manual_learning_rate):.6f}, "
            f"repeat_count={int(manual_repeat_count)}, "
            f"selection_strategy={selection_metadata['selection_strategy']}, "
            f"selected_images={len(selected_records)}, "
            f"train_images={len(incremental_selected_records)}, "
            f"incremental_only={str(not has_new_class).lower()}"
        )
        _append_app_log(
            log_type="start",
            source="Fine-tuning",
            content="Interactive fine-tuning started from saved labels.",
            request=request_summary,
            response="\n".join(
                (
                    f"{record.get('display_path', _to_project_relative_path(record['path']))} | "
                    f"predicted={record.get('predicted_label', '-')} | label={record['label']}"
                )
                for record in incremental_selected_records[:20]
            ),
        )
        with st.spinner("Fine-tuning the MobileViT model using the saved image labels..."):
            execution_result = run_detail_finetune_plan(
                effective_plan,
                incremental_selected_records,
                [],
                base_model_dir=base_model_dir,
                manual_target_class_input=None,
                selected_class_option=None,
                log_callback=_update_live_finetune_log,
                use_record_labels=True,
                selection_metadata=selection_metadata,
                incremental_only=not has_new_class,
            )
        supabase_sync_summary = None
        if execution_result.success:
            try:
                supabase_sync_summary = _sync_fine_tuning_records_to_supabase(
                    incremental_selected_records,
                    effective_plan,
                    use_record_labels=True,
                )
                _append_app_log(
                    log_type="done" if supabase_sync_summary["error_count"] == 0 else "Warning",
                    source="Supabase",
                    content=(
                        "The training image status was updated in the semiconductor table."
                        if supabase_sync_summary["error_count"] == 0
                        else "The training image status was only partially updated in the semiconductor table."
                    ),
                    request=(
                        f"trained=True, class=assigned_label, updated={supabase_sync_summary['updated_count']}, "
                        f"skipped={supabase_sync_summary['skipped_count']}, errors={supabase_sync_summary['error_count']}"
                    ),
                    response="\n".join(
                        [
                            *(
                                f"{item['filename']} -> {item['label']}"
                                for item in supabase_sync_summary["updated_records"][:20]
                            ),
                            *supabase_sync_summary["errors"][:10],
                        ]
                    ),
                )
            except Exception as exc:
                supabase_sync_summary = {
                    "attempted_count": 0,
                    "updated_count": 0,
                    "skipped_count": 0,
                    "error_count": 1,
                    "updated_records": [],
                    "skipped_records": [],
                    "errors": [str(exc)],
                }
                _append_app_log(
                    log_type="error",
                    source="Supabase",
                    content="Failed to update the semiconductor table after fine-tuning.",
                    request="trained=True, class=assigned_label",
                    response=str(exc),
                )
        st.session_state[execution_state_key] = {
            "success": execution_result.success,
            "command": execution_result.command,
            "output_dir": execution_result.output_dir,
            "llm_comment_path": execution_result.llm_comment_path,
            "selected_images_path": execution_result.selected_images_path,
            "context_json_path": execution_result.context_json_path,
            "audit_log_json_path": execution_result.audit_log_json_path,
            "audit_log_text_path": execution_result.audit_log_text_path,
            "supabase_sync_summary": supabase_sync_summary,
            "stdout": execution_result.stdout,
            "stderr": execution_result.stderr,
            "returncode": execution_result.returncode,
        }
        _append_app_log(
            log_type="done" if execution_result.success else "error",
            source="Fine-tuning",
            content=(
                "Interactive fine-tuning completed successfully."
                if execution_result.success
                else f"Interactive fine-tuning failed with return code {execution_result.returncode}."
            ),
            request=" ".join(execution_result.command),
            response="\n".join(
                filter(
                    None,
                    [
                        (execution_result.stdout or execution_result.stderr or "").strip(),
                        f"audit_json={execution_result.audit_log_json_path or '-'}",
                        f"audit_txt={execution_result.audit_log_text_path or '-'}",
                    ],
                )
            ),
        )

    execution_result = st.session_state.get(execution_state_key)
    if execution_result:
        with panel.expander("Fine-tuning result", expanded=not execution_result["success"]):
            if execution_result["success"]:
                panel.success("Fine-tuning completed successfully.")
                if execution_result.get("output_dir"):
                    panel.caption(f"Output directory: {_format_display_path(execution_result['output_dir'])}")
                if execution_result.get("llm_comment_path"):
                    panel.caption(f"LLM comment file: {_format_display_path(execution_result['llm_comment_path'])}")
                if execution_result.get("selected_images_path"):
                    panel.caption(f"Selected images file: {_format_display_path(execution_result['selected_images_path'])}")
                if execution_result.get("context_json_path"):
                    panel.caption(f"Context json file: {_format_display_path(execution_result['context_json_path'])}")
            else:
                panel.error(f"Fine-tuning failed. Return code: {execution_result['returncode']}")
            if execution_result.get("audit_log_json_path"):
                panel.caption(f"Audit json file: {_format_display_path(execution_result['audit_log_json_path'])}")
            if execution_result.get("audit_log_text_path"):
                panel.caption(f"Audit text file: {_format_display_path(execution_result['audit_log_text_path'])}")
            supabase_sync_summary = execution_result.get("supabase_sync_summary")
            if isinstance(supabase_sync_summary, dict):
                panel.caption(
                    "Supabase sync: "
                    f"updated={int(supabase_sync_summary.get('updated_count', 0))}, "
                    f"skipped={int(supabase_sync_summary.get('skipped_count', 0))}, "
                    f"errors={int(supabase_sync_summary.get('error_count', 0))}"
                )
                error_messages = supabase_sync_summary.get("errors") or []
                if error_messages:
                    panel.warning("Supabase update error: " + "; ".join(str(message) for message in error_messages[:3]))
            panel.code(" ".join(execution_result["command"]), language="bash")
            if execution_result["stdout"]:
                panel.text_area(
                    "stdout",
                    value=execution_result["stdout"],
                    height=180,
                    disabled=True,
                )
            if execution_result["stderr"]:
                panel.text_area(
                    "stderr",
                    value=execution_result["stderr"],
                    height=180,
                    disabled=True,
                )


def render_fine_tuning_page(image_records) -> None:
    render_page_header(
        "Fine-tuning",
        "Select training samples from the image pool, then run Interactive Fine-tuning. "
        "This is an additional training feature for the classification model.",
    )
    image_pool_records = _build_unique_image_pool(image_records)
    if not image_pool_records:
        st.info("There is no image pool available for fine-tuning.")
        return

    predicted_pool_records = image_pool_records
    selected_records = []
    has_unsaved_label_changes = False
    selected_model_dir = None
    available_classes: list[str] = []
    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        with st.container(border=True):
            selected_model_dir, model_changed = _render_classifier_model_selector(
                selected_records=image_pool_records,
                container=st,
                selector_key="fine_tuning_inference_model_selector",
                active_key="fine_tuning_inference_model_active",
                section_title="Image Pool",
                helper_text="Selecting a model reruns inference for the full image pool below with that model.",
                label="Inference model",
            )
            if selected_model_dir is None:
                return
            selected_model_dir = resolve_base_model_dir(selected_model_dir)

            previous_selected_paths = list(st.session_state.get("fine_tuning_page_selected_image_paths", []))
            available_paths = {record["path"] for record in image_pool_records}
            previous_selected_paths = [path for path in previous_selected_paths if path in available_paths]
            if model_changed:
                _reset_fine_tuning_page_session(previous_selected_paths)
                if previous_selected_paths:
                    _set_fine_tuning_selection_metadata(
                        selected_paths=previous_selected_paths,
                        strategy="Manual Selection",
                        origin="manual",
                        notice="inference model changed after selection",
                        selection_percentage=None,
                        model_dir=selected_model_dir,
                    )

            try:
                predicted_pool_records, prediction_errors = _predict_image_pool_records_with_model(
                    image_pool_records,
                    selected_model_dir,
                )
            except Exception as exc:
                st.warning(str(exc))
                predicted_pool_records = image_pool_records
                prediction_errors = []

            if prediction_errors:
                st.warning("Some images could not be reprocessed for inference: " + "; ".join(prediction_errors[:3]))

            available_classes = []
            try:
                available_classes = load_available_classes(selected_model_dir)
            except Exception as exc:
                st.warning(f"Could not load class information: {exc}")

            records_by_index = {
                index: record for index, record in enumerate(predicted_pool_records, start=1)
            }
            table_rows = [
                {
                    "Index": index,
                    "Raw Image": _build_thumbnail_data_uri(record["path"]) if record["exists"] else None,
                    "Trained": bool(record.get("trained", False)),
                    "Predicted Class": record.get("predicted_label", record["label"]),
                    "Select": record["path"] in previous_selected_paths,
                }
                for index, record in records_by_index.items()
            ]

            edited_table = st.data_editor(
                pd.DataFrame(table_rows),
                hide_index=True,
                use_container_width=True,
                height=760,
                disabled=["Index", "Raw Image", "Trained", "Predicted Class"],
                column_order=["Index", "Raw Image", "Trained", "Predicted Class", "Select"],
                column_config={
                    "Index": st.column_config.NumberColumn("Index", format="%d", width="small"),
                    "Raw Image": st.column_config.ImageColumn("Raw Image", width="medium"),
                    "Trained": st.column_config.CheckboxColumn(
                        "Trained",
                        help="Shows whether the data has already been reflected in training based on Supabase.",
                        width="small",
                    ),
                    "Predicted Class": st.column_config.TextColumn("Predicted Class", width="medium"),
                    "Select": st.column_config.CheckboxColumn("Select", help="Only selected images are used for fine-tuning."),
                },
                key="fine_tuning_image_pool_editor",
            )

            selected_indices = [
                int(row["Index"]) for _, row in edited_table.iterrows() if bool(row["Select"])
            ]
            raw_selected_records = [records_by_index[index] for index in selected_indices if index in records_by_index]
            selected_paths = [record["path"] for record in raw_selected_records]
            if previous_selected_paths != selected_paths:
                _reset_fine_tuning_page_session(selected_paths)
                _set_fine_tuning_selection_metadata(
                    selected_paths=selected_paths,
                    strategy="Manual Selection",
                    origin="manual",
                    notice="The user manually selected images from the fine-tuning image pool.",
                    selection_percentage=None,
                    model_dir=selected_model_dir,
                )
            else:
                st.session_state["fine_tuning_page_selected_image_paths"] = selected_paths

            selected_records = _sync_records_with_model_dir(
                _apply_fine_tuning_label_overrides(raw_selected_records),
                selected_model_dir,
            )
            st.caption(f"Selected {len(selected_records)} out of {len(predicted_pool_records)} total images.")
            if selected_records:
                with st.expander(f"Selected images ({len(selected_records)})", expanded=True):
                    selected_records_widget_token = str(abs(hash(tuple(record["path"] for record in selected_records))))
                    if available_classes:
                        st.caption(
                            "Available classes: "
                            + ", ".join(available_classes)
                            + " | Enter new classes directly in the `Label` column."
                        )
                    label_editor_rows = [
                        {
                            "Raw Image": _build_thumbnail_data_uri(record["path"]) if record["exists"] else None,
                            "Filename": record["filename"],
                            "Predicted Class": record.get("predicted_label", record["label"]),
                            "Label": record["label"],
                        }
                        for record in selected_records
                    ]

                    edited_labels = st.data_editor(
                        pd.DataFrame(label_editor_rows),
                        hide_index=True,
                        use_container_width=True,
                        height=min(420, 120 + len(label_editor_rows) * 70),
                        disabled=["Raw Image", "Filename", "Predicted Class"],
                        column_order=["Raw Image", "Filename", "Predicted Class", "Label"],
                        column_config={
                            "Raw Image": st.column_config.ImageColumn("Raw Image", width="medium"),
                            "Filename": st.column_config.TextColumn("Filename", width="medium"),
                            "Predicted Class": st.column_config.TextColumn("Predicted Class", width="medium"),
                            "Label": st.column_config.TextColumn(
                                "Label",
                                width="medium",
                                required=True,
                                help="Enter an existing class name or type a new class directly.",
                            ),
                        },
                        key=f"fine_tuning_selected_label_editor__{selected_records_widget_token}",
                    )
                    assigned_labels_by_path = {
                        record["path"]: str(edited_labels.iloc[index]["Label"]).strip()
                        for index, record in enumerate(selected_records)
                    }
                    saved_labels_by_path = {
                        record["path"]: str(record["label"]).strip()
                        for record in selected_records
                    }
                    has_unsaved_label_changes = assigned_labels_by_path != saved_labels_by_path

                    if st.button("Save labels", key="fine_tuning_save_labels", width="stretch"):
                        _save_fine_tuning_label_overrides(raw_selected_records, assigned_labels_by_path)
                        selected_records = _sync_records_with_model_dir(
                            _apply_fine_tuning_label_overrides(raw_selected_records),
                            selected_model_dir,
                        )
                        has_unsaved_label_changes = False
                        st.success("The selected image labels have been saved.")
                    elif has_unsaved_label_changes:
                        st.info("After editing labels, click `Save labels` to apply them to the fine-tuning panel on the right.")
            else:
                st.info("Use the `Select` checkbox in the Image Pool on the left to choose images for fine-tuning.")

    with right_col:
        with st.container(border=True):
            _render_fine_tuning_training_panel(
                image_pool_records=predicted_pool_records,
                selected_records=selected_records,
                base_model_dir=resolve_base_model_dir(selected_model_dir),
                panel=st,
                heading="Interactive Fine-tuning",
                has_unsaved_label_changes=has_unsaved_label_changes,
                available_classes=available_classes,
            )


configure_page("Fine-tuning")
if not bool(st.session_state.get("dashboard_data_loaded", False)):
    st.info("Please go to the Dashboard page and click **Load Data** first.")
    st.stop()
_query_date_start = str(st.session_state.get("dashboard_query_date_start", "")).strip() or None
_query_date_end = str(st.session_state.get("dashboard_query_date_end", "")).strip() or None
_config, _runs, image_records, _log_entries = load_dashboard_data(
    query_date_start=_query_date_start,
    query_date_end=_query_date_end,
)
render_fine_tuning_page(image_records)
