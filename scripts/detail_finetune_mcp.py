from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.local_gemma_model import are_runtime_dependencies_available, generate_response, is_model_downloaded
from scripts.local_gemma_model import unload_model


CLASSIFICATION_MODEL_ROOT = PROJECT_ROOT / "model" / "classification"
CLASSIFIER_MODEL_DIR = CLASSIFICATION_MODEL_ROOT / "mobilevit_small_9_classifier"
INTERACTIVE_FINETUNE_SCRIPT = PROJECT_ROOT / "scripts" / "interactive_finetune.py"
INTERACTIVE_OUTPUT_ROOT = CLASSIFICATION_MODEL_ROOT
DETAIL_FINETUNE_SYSTEM_PROMPT = """You are a fine-tuning coach for a semiconductor image classification model.
Respond only in English.
Your job is to organize how the user wants to retrain the selected images and propose a safe fine-tuning plan.
You may suggest creating a new class.
Output exactly one JSON object and nothing else.
JSON schema:
{
  "assistant_reply": "English reply shown to the user",
  "target_label": "Center|Donut|Edge-Loc|Edge-Ring|Local|Near-Full|Normal|Scratch or null",
  "create_new_class": false,
  "new_class_name": "new class name or null",
  "epochs": number between 1.0 and 5.0,
  "learning_rate": number between 0.000001 and 0.0001,
  "repeat_count": integer between 4 and 64,
  "ready_to_train": true or false,
  "notes": "short internal note"
}
Rules:
- If the user has not clearly specified the target class or goal yet, set ready_to_train=false and ask a short clarifying question.
- If an existing class is sufficient, choose target_label and keep create_new_class=false.
- If a new class is needed and does not already exist, set create_new_class=true and write the class name in new_class_name.
- When creating a new class, set target_label to null.
- new_class_name must use only letters, numbers, and underscores, with no spaces.
- Keep epochs, learning_rate, and repeat_count conservative.
- Never output explanatory text outside the JSON object."""


def _resolve_project_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    raw_value = str(value).strip()
    if not raw_value:
        return None
    path = Path(raw_value).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _to_project_relative_path(value: str | Path | None) -> str:
    if value is None:
        return "-"
    raw_value = str(value).strip()
    if not raw_value:
        return "-"
    path = Path(raw_value).expanduser()
    absolute_path = path if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    return os.path.relpath(str(absolute_path), str(PROJECT_ROOT))


def _normalize_record_for_artifact(record: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    if normalized.get("path"):
        normalized["path"] = _to_project_relative_path(normalized["path"])
    if normalized.get("display_path"):
        normalized["display_path"] = _to_project_relative_path(normalized["display_path"])
    if normalized.get("model_dir"):
        normalized["model_dir"] = _to_project_relative_path(normalized["model_dir"])
    if normalized.get("model_dir_display"):
        normalized["model_dir_display"] = _to_project_relative_path(normalized["model_dir_display"])
    return normalized


@dataclass
class DetailFineTunePlan:
    assistant_reply: str
    target_label: str | None
    epochs: float
    learning_rate: float
    repeat_count: int
    ready_to_train: bool
    create_new_class: bool = False
    new_class_name: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DetailFineTuneExecutionResult:
    success: bool
    command: list[str]
    output_dir: str | None
    llm_comment_path: str | None
    selected_images_path: str | None
    context_json_path: str | None
    audit_log_json_path: str | None
    audit_log_text_path: str | None
    stdout: str
    stderr: str
    returncode: int


def _is_valid_classifier_model_dir(candidate: Path) -> bool:
    required_files = (
        "model.safetensors",
        "config.json",
        "preprocessor_config.json",
        "label2id.json",
    )
    return candidate.is_dir() and all((candidate / filename).exists() for filename in required_files)


def resolve_base_model_dir(model_dir: str | Path | None = None) -> Path:
    candidate = _resolve_project_path(model_dir) if model_dir else CLASSIFIER_MODEL_DIR
    assert candidate is not None
    if candidate.is_file() and candidate.name == "model.safetensors":
        candidate = candidate.parent

    candidate_options: list[Path] = [candidate]
    if candidate.name:
        candidate_options.append(CLASSIFICATION_MODEL_ROOT / candidate.name)

    for current_candidate in candidate_options:
        if _is_valid_classifier_model_dir(current_candidate):
            return current_candidate

    if _is_valid_classifier_model_dir(CLASSIFIER_MODEL_DIR):
        return CLASSIFIER_MODEL_DIR

    if CLASSIFICATION_MODEL_ROOT.exists():
        available_model_dirs = sorted(
            path for path in CLASSIFICATION_MODEL_ROOT.iterdir() if _is_valid_classifier_model_dir(path)
        )
        if available_model_dirs:
            return available_model_dirs[0]

    return candidate


def load_available_classes(model_dir: Path | str | None = None) -> list[str]:
    resolved_model_dir = resolve_base_model_dir(model_dir)
    label2id_path = resolved_model_dir / "label2id.json"
    if not label2id_path.exists():
        raise FileNotFoundError(f"Label mapping file does not exist: {label2id_path}")

    label2id = json.loads(label2id_path.read_text(encoding="utf-8"))
    if not isinstance(label2id, dict) or not label2id:
        raise ValueError(f"Invalid label mapping in: {label2id_path}")
    return list(label2id.keys())


def build_detail_transcript(chat_history: list[dict[str, str]]) -> str:
    if not chat_history:
        return "No conversation history"
    return "\n".join(f"{message['role']}: {message['content']}" for message in chat_history)


def save_detail_comment_file(
    output_dir: Path,
    plan: DetailFineTunePlan,
    selected_records: list[dict[str, Any]],
    chat_history: list[dict[str, str]],
    base_model_dir: Path,
    manual_target_class_input: str | None = None,
    selected_class_option: str | None = None,
) -> Path:
    selected_lines: list[str] = []
    for record in selected_records:
        assigned_label = str(record.get("assigned_label") or record.get("label") or "-")
        predicted_label = str(record.get("predicted_label") or assigned_label or "-")
        line = (
            f"- {record['filename']} | predicted={predicted_label} | "
            f"assigned={assigned_label} | path={_to_project_relative_path(record['path'])}"
        )
        selected_lines.append(line)
    target_label = plan.new_class_name if plan.create_new_class else plan.target_label
    content = "\n".join(
        [
            f"saved_at: {datetime.now().isoformat(timespec='seconds')}",
            f"base_model_dir: {_to_project_relative_path(base_model_dir)}",
            f"target_label: {target_label or '-'}",
            f"manual_target_class_input: {manual_target_class_input or '-'}",
            f"selected_class_option: {selected_class_option or '-'}",
            f"create_new_class: {plan.create_new_class}",
            f"epochs: {plan.epochs}",
            f"learning_rate: {plan.learning_rate}",
            f"repeat_count: {plan.repeat_count}",
            f"notes: {plan.notes or '-'}",
            "",
            "[assistant_reply]",
            plan.assistant_reply,
            "",
            "[selected_records]",
            *selected_lines,
            "",
            "[chat_history]",
            build_detail_transcript(chat_history),
        ]
    )
    comment_path = output_dir / "llm_comment.txt"
    comment_path.write_text(content, encoding="utf-8")
    return comment_path


def save_selected_images_file(output_dir: Path, selected_records: list[dict[str, Any]]) -> Path:
    image_list_content = "\n".join(_to_project_relative_path(record["path"]) for record in selected_records)
    image_list_path = output_dir / "selected_images.txt"
    image_list_path.write_text(image_list_content + ("\n" if image_list_content else ""), encoding="utf-8")
    return image_list_path


def _make_json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_make_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_make_json_safe(item) for item in value]
    return value


def save_detail_context_json(
    output_dir: Path,
    plan: DetailFineTunePlan,
    selected_records: list[dict[str, Any]],
    chat_history: list[dict[str, str]],
    base_model_dir: Path,
    manual_target_class_input: str | None = None,
    selected_class_option: str | None = None,
) -> Path:
    context_payload = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "base_model_dir": _to_project_relative_path(base_model_dir),
        "plan": plan.to_dict(),
        "manual_target_class_input": manual_target_class_input,
        "selected_class_option": selected_class_option,
        "selected_records": [_normalize_record_for_artifact(record) for record in selected_records],
        "chat_history": chat_history,
    }
    context_path = output_dir / "detail_finetune_context.json"
    context_path.write_text(
        json.dumps(_make_json_safe(context_payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return context_path


def save_detail_audit_logs(
    output_dir: Path,
    plan: DetailFineTunePlan,
    selected_records: list[dict[str, Any]],
    base_model_dir: Path,
    saved_model_dir: Path,
    command: list[str],
    success: bool,
    returncode: int,
    manual_target_class_input: str | None = None,
    selected_class_option: str | None = None,
    selection_metadata: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    metadata = dict(selection_metadata or {})
    selection_strategy = str(metadata.get("selection_strategy") or "Manual Selection").strip() or "Manual Selection"
    selection_origin = str(metadata.get("selection_origin") or "manual").strip() or "manual"
    selection_notice = str(metadata.get("selection_notice") or "").strip() or None
    selection_percentage = metadata.get("selection_percentage")

    normalized_records: list[dict[str, Any]] = []
    for index, record in enumerate(selected_records, start=1):
        normalized_record = {
            "index": index,
            "filename": str(record.get("filename") or Path(str(record["path"])).name),
            "path": _to_project_relative_path(record["path"]),
            "predicted_label": str(record.get("predicted_label") or record.get("label") or "-"),
            "assigned_label": str(record.get("assigned_label") or record.get("label") or "-"),
            "trained_before_finetuning": bool(record.get("trained", False)),
            "record_id": record.get("record_id"),
            "source_label": str(record.get("source_label") or "").strip() or None,
        }
        normalized_records.append(normalized_record)

    audit_payload = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "success": bool(success),
        "returncode": int(returncode),
        "selection_strategy": selection_strategy,
        "selection_origin": selection_origin,
        "selection_notice": selection_notice,
        "selection_percentage": selection_percentage,
        "base_model_name": base_model_dir.name,
        "base_model_dir": _to_project_relative_path(base_model_dir),
        "trained_model_name": saved_model_dir.name,
        "trained_model_dir": _to_project_relative_path(saved_model_dir),
        "target_label": plan.new_class_name if plan.create_new_class else plan.target_label,
        "create_new_class": bool(plan.create_new_class),
        "manual_target_class_input": manual_target_class_input,
        "selected_class_option": selected_class_option,
        "epochs": float(plan.epochs),
        "learning_rate": float(plan.learning_rate),
        "repeat_count": int(plan.repeat_count),
        "command": list(command),
        "selected_record_count": len(normalized_records),
        "selected_records": normalized_records,
        "extra_metadata": _make_json_safe(metadata),
    }

    audit_json_path = output_dir / "fine_tuning_audit_log.json"
    audit_json_path.write_text(
        json.dumps(_make_json_safe(audit_payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    text_lines = [
        f"saved_at: {audit_payload['saved_at']}",
        f"success: {audit_payload['success']}",
        f"returncode: {audit_payload['returncode']}",
        f"selection_strategy: {selection_strategy}",
        f"selection_origin: {selection_origin}",
        f"selection_notice: {selection_notice or '-'}",
        f"selection_percentage: {selection_percentage if selection_percentage is not None else '-'}",
        f"base_model_name: {base_model_dir.name}",
        f"base_model_dir: {_to_project_relative_path(base_model_dir)}",
        f"trained_model_name: {saved_model_dir.name}",
        f"trained_model_dir: {_to_project_relative_path(saved_model_dir)}",
        f"target_label: {audit_payload['target_label'] or '-'}",
        f"manual_target_class_input: {manual_target_class_input or '-'}",
        f"selected_class_option: {selected_class_option or '-'}",
        f"epochs: {plan.epochs}",
        f"learning_rate: {plan.learning_rate}",
        f"repeat_count: {plan.repeat_count}",
        f"selected_record_count: {len(normalized_records)}",
        "",
        "[selected_records]",
    ]
    for record in normalized_records:
        text_lines.append(
            f"- {record['filename']} | predicted={record['predicted_label']} | "
            f"assigned={record['assigned_label']} | trained_before={record['trained_before_finetuning']} | "
            f"path={record['path']}"
        )
    text_lines.extend(
        [
            "",
            "[command]",
            " ".join(command),
        ]
    )

    audit_text_path = output_dir / "fine_tuning_audit_log.txt"
    audit_text_path.write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    return audit_json_path, audit_text_path


def _extract_json_payload(raw_text: str) -> dict[str, Any]:
    fenced_match = re.search(r"```json\s*(\{.*?\})\s*```", raw_text, flags=re.DOTALL)
    if fenced_match:
        return json.loads(fenced_match.group(1))

    start_index = raw_text.find("{")
    end_index = raw_text.rfind("}")
    if start_index == -1 or end_index == -1 or end_index <= start_index:
        raise ValueError("LLM response did not include a JSON object.")

    return json.loads(raw_text[start_index : end_index + 1])


def _clamp_float(value: Any, minimum: float, maximum: float, fallback: float) -> float:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return fallback
    return max(minimum, min(maximum, numeric_value))


def _clamp_int(value: Any, minimum: int, maximum: int, fallback: int) -> int:
    try:
        numeric_value = int(value)
    except (TypeError, ValueError):
        return fallback
    return max(minimum, min(maximum, numeric_value))


def _is_out_of_memory_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return exc.__class__.__name__ == "OutOfMemoryError" or "out of memory" in message


def parse_detail_finetune_plan(raw_text: str, available_classes: list[str]) -> DetailFineTunePlan:
    payload = _extract_json_payload(raw_text)
    target_label = payload.get("target_label")
    if target_label and target_label not in available_classes:
        target_label = None

    create_new_class = bool(payload.get("create_new_class", False))
    new_class_name = str(payload.get("new_class_name", "")).strip() if payload.get("new_class_name") else None
    if create_new_class and new_class_name:
        new_class_name = new_class_name.replace(" ", "_").replace("-", "_")
    else:
        new_class_name = None

    reply = str(payload.get("assistant_reply", "")).strip()
    if not reply:
        reply = "Please explain the retraining plan again."

    ready_to_train = bool(payload.get("ready_to_train", False)) and (target_label is not None or new_class_name is not None)
    if not ready_to_train and target_label is None and new_class_name is None and "class" not in reply.lower():
        reply = "Please tell me which class this image should be retrained as."

    return DetailFineTunePlan(
        assistant_reply=reply,
        target_label=target_label,
        epochs=_clamp_float(payload.get("epochs"), minimum=1.0, maximum=5.0, fallback=2.0),
        learning_rate=_clamp_float(payload.get("learning_rate"), minimum=1e-6, maximum=1e-4, fallback=1e-5),
        repeat_count=_clamp_int(payload.get("repeat_count"), minimum=4, maximum=64, fallback=16),
        ready_to_train=ready_to_train,
        create_new_class=create_new_class,
        new_class_name=new_class_name,
        notes=str(payload.get("notes", "")).strip(),
    )


def build_detail_plan_prompt(
    selected_records: list[dict[str, Any]],
    chat_history: list[dict[str, str]],
    user_message: str,
    available_classes: list[str],
    target_label_override: str | None = None,
) -> str:
    transcript = build_detail_transcript(chat_history)
    if len(selected_records) == 1:
        summary = (
            "Please create a retraining plan for the currently selected image.\n"
            f"Selected image filename: {selected_records[0]['filename']}\n"
            f"Current predicted label: {selected_records[0]['label']}\n"
            f"Image path: {_to_project_relative_path(selected_records[0]['path'])}\n"
        )
    else:
        summary_lines = [
            "Please create a retraining plan for the currently selected images."
        ]
        for record in selected_records:
            summary_lines.append(
                f"- {record['filename']} (current prediction: {record['label']}) path: {_to_project_relative_path(record['path'])}"
            )
        summary = "\n".join(summary_lines) + "\n"

    if target_label_override:
        summary += f"For this plan, keep the target class fixed as '{target_label_override}'.\n"

    return summary + (
        f"Available classes: {', '.join(available_classes)}\n"
        "You may also propose a new class if the image does not fit the existing classes.\n"
        "The model is a MobileViT 8-class classifier, and the user wants to retrain this image or these images toward the correct class.\n"
        "Conversation history:\n"
        f"{transcript}\n"
        f"Current user message: {user_message.strip()}\n"
        "If the class is clear, set target_label and propose conservative values for learning_rate, epochs, and repeat_count."
    )


def request_detail_finetune_plan(
    selected_records: list[dict[str, Any]],
    chat_history: list[dict[str, str]],
    user_message: str,
    model_dir: Path | str | None = None,
    target_label_override: str | None = None,
) -> DetailFineTunePlan:
    dependency_ready, dependency_message = are_runtime_dependencies_available()
    if not dependency_ready:
        return DetailFineTunePlan(
            assistant_reply=dependency_message,
            target_label=None,
            epochs=2.0,
            learning_rate=1e-5,
            repeat_count=16,
            ready_to_train=False,
            notes="runtime dependency missing",
        )
    if not is_model_downloaded():
        return DetailFineTunePlan(
            assistant_reply="The local Gemma model is not ready, so a retraining plan could not be created.",
            target_label=None,
            epochs=2.0,
            learning_rate=1e-5,
            repeat_count=16,
            ready_to_train=False,
            notes="gemma model missing",
        )

    available_classes = load_available_classes(model_dir)
    prompt = build_detail_plan_prompt(
        selected_records,
        chat_history,
        user_message,
        available_classes,
        target_label_override=target_label_override,
    )
    target_label = target_label_override if target_label_override in available_classes else None
    create_new_class = bool(target_label_override and target_label_override not in available_classes)
    new_class_name = None
    if create_new_class and target_label_override:
        new_class_name = target_label_override.replace(" ", "_").replace("-", "_")

    image_paths = [selected_records[0]["path"]] if len(selected_records) == 1 else []
    used_image_context = bool(image_paths)

    try:
        try:
            raw_response = generate_response(
                prompt=prompt,
                system_prompt=DETAIL_FINETUNE_SYSTEM_PROMPT,
                image_paths=image_paths,
                max_new_tokens=256,
            )
        except Exception as exc:
            if not image_paths or not _is_out_of_memory_error(exc):
                raise

            unload_model()
            used_image_context = False
            raw_response = generate_response(
                prompt=prompt,
                system_prompt=DETAIL_FINETUNE_SYSTEM_PROMPT,
                image_paths=[],
                max_new_tokens=256,
            )

        plan = parse_detail_finetune_plan(raw_response, available_classes)
        if not used_image_context:
            if plan.notes:
                plan.notes = f"{plan.notes} | plan generated without images to reduce GPU memory use"
            else:
                plan.notes = "plan generated without images to reduce GPU memory use"
        return plan
    except Exception as exc:
        return DetailFineTunePlan(
            assistant_reply=(
                "An error occurred while generating the retraining plan. "
                f"Error: {exc}"
            ),
            target_label=target_label,
            epochs=2.0,
            learning_rate=1e-5,
            repeat_count=16,
            ready_to_train=bool(target_label or new_class_name),
            create_new_class=create_new_class,
            new_class_name=new_class_name,
            notes="gemma inference error",
        )
    finally:
        unload_model()


def run_detail_finetune_plan(
    plan: DetailFineTunePlan,
    selected_records: list[dict[str, Any]],
    chat_history: list[dict[str, str]],
    base_model_dir: Path | str | None = None,
    manual_target_class_input: str | None = None,
    selected_class_option: str | None = None,
    log_callback: Callable[[str], None] | None = None,
    use_record_labels: bool = False,
    selection_metadata: dict[str, Any] | None = None,
    incremental_only: bool = False,
) -> DetailFineTuneExecutionResult:
    resolved_base_model_dir = resolve_base_model_dir(base_model_dir)
    INTERACTIVE_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-u",
        _to_project_relative_path(INTERACTIVE_FINETUNE_SCRIPT),
        "--base-model-dir",
        _to_project_relative_path(resolved_base_model_dir),
    ]
    selected_records_manifest_path: str | None = None
    if use_record_labels:
        labeled_records_payload: list[dict[str, str]] = []
        for record in selected_records:
            assigned_label = str(record.get("assigned_label") or record.get("label") or "").strip()
            if not assigned_label:
                raise ValueError("A selected image label is empty.")

            payload_record = {
                "path": _to_project_relative_path(record["path"]),
                "label": assigned_label,
            }
            predicted_label = str(record.get("predicted_label") or "").strip()
            if predicted_label:
                payload_record["predicted_label"] = predicted_label
            labeled_records_payload.append(payload_record)

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
            encoding="utf-8",
        ) as handle:
            json.dump(labeled_records_payload, handle, ensure_ascii=False, indent=2)
            selected_records_manifest_path = handle.name
        command.extend(["--selected-records-path", selected_records_manifest_path])
    elif len(selected_records) == 1:
        command.extend(["--selected-image", _to_project_relative_path(selected_records[0]["path"])])
    else:
        command.append("--selected-images")
        command.extend(_to_project_relative_path(record["path"]) for record in selected_records)

    if not use_record_labels:
        if plan.create_new_class and plan.new_class_name:
            command.extend(["--create-new-class", "--new-class-name", plan.new_class_name])
        else:
            command.extend(["--target-label", str(plan.target_label)])

    command.extend([
        "--predicted-label",
        str(selected_records[0].get("predicted_label") or selected_records[0]["label"]),
        "--epochs",
        str(plan.epochs),
        "--learning-rate",
        str(plan.learning_rate),
        "--repeat-count",
        str(plan.repeat_count),
        "--output-root",
        _to_project_relative_path(INTERACTIVE_OUTPUT_ROOT),
    ])
    if incremental_only:
        command.append("--incremental-only")
    if manual_target_class_input:
        command.extend(["--manual-target-class-input", manual_target_class_input])
    if selected_class_option:
        command.extend(["--selected-class-option", selected_class_option])
    try:
        if log_callback is None:
            completed = subprocess.run(
                command,
                cwd=str(PROJECT_ROOT),
                text=True,
                capture_output=True,
                check=False,
            )
        else:
            process = subprocess.Popen(
                command,
                cwd=str(PROJECT_ROOT),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
            )
            combined_stdout_lines: list[str] = []
            assert process.stdout is not None
            for line in process.stdout:
                combined_stdout_lines.append(line)
                log_callback("".join(combined_stdout_lines))
            process.wait()
            completed = subprocess.CompletedProcess(
                args=command,
                returncode=process.returncode,
                stdout="".join(combined_stdout_lines),
                stderr="",
            )
    finally:
        if selected_records_manifest_path:
            try:
                Path(selected_records_manifest_path).unlink(missing_ok=True)
            except OSError:
                pass

    output_dir = None
    for line in completed.stdout.splitlines():
        if line.startswith("OUTPUT_DIR="):
            output_dir = line.split("=", 1)[1].strip()
            break

    llm_comment_path = None
    selected_images_path = None
    context_json_path = None
    audit_log_json_path = None
    audit_log_text_path = None
    if output_dir:
        output_path = _resolve_project_path(output_dir)
        assert output_path is not None
        if output_path.exists():
            try:
                # These are pure side-car artifacts (comment/context/audit logs) never read back
                # by the app, so they go under metadata/ instead of the model directory root.
                metadata_path = output_path / "metadata"
                metadata_path.mkdir(parents=True, exist_ok=True)
                llm_comment_path = _to_project_relative_path(
                    save_detail_comment_file(
                        output_dir=metadata_path,
                        plan=plan,
                        selected_records=selected_records,
                        chat_history=chat_history,
                        base_model_dir=resolved_base_model_dir,
                        manual_target_class_input=manual_target_class_input,
                        selected_class_option=selected_class_option,
                    )
                )
                selected_images_path = _to_project_relative_path(save_selected_images_file(metadata_path, selected_records))
                context_json_path = _to_project_relative_path(
                    save_detail_context_json(
                        output_dir=metadata_path,
                        plan=plan,
                        selected_records=selected_records,
                        chat_history=chat_history,
                        base_model_dir=resolved_base_model_dir,
                        manual_target_class_input=manual_target_class_input,
                        selected_class_option=selected_class_option,
                    )
                )
                audit_json_path, audit_text_path = save_detail_audit_logs(
                    output_dir=metadata_path,
                    plan=plan,
                    selected_records=selected_records,
                    base_model_dir=resolved_base_model_dir,
                    saved_model_dir=output_path,
                    command=command,
                    success=completed.returncode == 0,
                    returncode=completed.returncode,
                    manual_target_class_input=manual_target_class_input,
                    selected_class_option=selected_class_option,
                    selection_metadata=selection_metadata,
                )
                audit_log_json_path = _to_project_relative_path(audit_json_path)
                audit_log_text_path = _to_project_relative_path(audit_text_path)
            except Exception as artifact_exc:
                completed.stderr = (
                    completed.stderr
                    + ("\n" if completed.stderr else "")
                    + f"[artifact-save-error] {artifact_exc}"
                )

    return DetailFineTuneExecutionResult(
        success=completed.returncode == 0,
        command=command,
        output_dir=_to_project_relative_path(output_dir) if output_dir else None,
        llm_comment_path=llm_comment_path,
        selected_images_path=selected_images_path,
        context_json_path=context_json_path,
        audit_log_json_path=audit_log_json_path,
        audit_log_text_path=audit_log_text_path,
        stdout=completed.stdout,
        stderr=completed.stderr,
        returncode=completed.returncode,
    )
