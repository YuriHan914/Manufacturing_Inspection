from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

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


_suppress_transformers_path_alias_warning()
from transformers import AutoModelForImageClassification, set_seed

try:
    from transformers import AutoImageProcessor
except ImportError:  # Compatibility with older Transformers releases.
    from transformers import AutoFeatureExtractor as AutoImageProcessor

from model_train import (
    FolderImageClassificationDataset,
    ImageClassificationCollator,
    build_trainer,
    build_training_arguments,
    build_training_args_payload,
    compute_class_weights,
    ensure_output_dir,
    evaluate_and_save_split,
    replace_training_args_bin_with_json,
    resolve_project_path,
    save_json,
    save_records_csv,
    to_project_relative_path,
)


BASE_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Incrementally fine-tune the MobileViT classifier with selected Detail image(s).",
    )
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--selected-image", type=Path, required=False)
    parser.add_argument("--selected-images", type=Path, nargs="+", action="append", required=False)
    parser.add_argument("--selected-records-path", type=Path, required=False)
    parser.add_argument("--predicted-label", type=str, default="")
    parser.add_argument("--target-label", type=str, required=False)
    parser.add_argument("--create-new-class", action="store_true", default=False)
    parser.add_argument("--new-class-name", type=str, required=False)
    parser.add_argument("--manual-target-class-input", type=str, default="", required=False)
    parser.add_argument("--selected-class-option", type=str, default="", required=False)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--repeat-count", type=int, default=16)
    parser.add_argument("--incremental-only", action="store_true", default=False)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=BASE_DIR / "model",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_records_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [{"path": row["path"], "label": row["label"]} for row in reader]


SUPABASE_IMAGE_TABLE = "semiconductor"
SUPABASE_PAGE_SIZE = 1000  # PostgREST's own default max-rows-per-request.


def _load_supabase_credentials() -> tuple[str, str]:
    import toml

    secrets_path = BASE_DIR / ".streamlit" / "secrets.toml"
    if not secrets_path.exists():
        raise RuntimeError(f"Supabase secrets file not found: {secrets_path}")
    connection_secrets = toml.load(secrets_path).get("connections", {}).get("supabase", {})
    url = str(connection_secrets.get("SUPABASE_URL") or "").strip()
    key = str(connection_secrets.get("SUPABASE_KEY") or "").strip()
    if not url or not key:
        raise RuntimeError("Supabase URL/key are not configured in .streamlit/secrets.toml.")
    return url, key


def load_records_from_supabase(dataset_type: str) -> list[dict[str, str]]:
    """Load {path, label} records straight from Supabase's `semiconductor` table, filtered by its
    `type` column (train/valid/test). Used for valid/test so evaluation doesn't depend on a local
    valid_split.csv/test_split.csv file, whose location can move between model directory layouts —
    Supabase's `type` column is the same fixed split for every fine-tuning round regardless."""
    from supabase import create_client

    url, key = _load_supabase_credentials()
    client = create_client(url, key)

    records: list[dict[str, str]] = []
    offset = 0
    while True:
        result = (
            client.table(SUPABASE_IMAGE_TABLE)
            .select("file_path,class")
            .eq("type", dataset_type)
            .range(offset, offset + SUPABASE_PAGE_SIZE - 1)
            .execute()
        )
        rows = result.data or []
        for row in rows:
            file_path = str(row.get("file_path") or "").strip()
            label = str(row.get("class") or "").strip()
            if file_path and label:
                records.append({"path": file_path, "label": label})
        if len(rows) < SUPABASE_PAGE_SIZE:
            break
        offset += SUPABASE_PAGE_SIZE

    if not records:
        raise RuntimeError(f"No Supabase rows found with type={dataset_type!r} in `{SUPABASE_IMAGE_TABLE}`.")
    return records


def load_selected_records_manifest(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    if not isinstance(payload, list):
        raise ValueError("The selected records manifest format is invalid.")

    selected_records: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("A selected records manifest entry has an invalid format.")

        resolved_image_path = resolve_project_path(item.get("path"))
        label = str(item.get("label") or "").strip()
        predicted_label = str(item.get("predicted_label") or "").strip() or None
        if resolved_image_path is None:
            raise ValueError("Could not resolve an image path from the selected records manifest.")
        if not label:
            raise ValueError("The selected records manifest contains an empty label.")

        selected_records.append(
            {
                "path": resolved_image_path,
                "label": label,
                "predicted_label": predicted_label,
            }
        )
    return selected_records


def build_interactive_config(base_model_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    dataset_config = load_json(base_model_dir / "dataset_config.json")
    dataset_config["model_name"] = to_project_relative_path(base_model_dir)
    dataset_config["num_epochs"] = float(args.epochs)
    dataset_config["learning_rate"] = float(args.learning_rate)
    dataset_config["train_batch_size"] = min(int(dataset_config.get("train_batch_size", 16)), 8)
    dataset_config["eval_batch_size"] = min(int(dataset_config.get("eval_batch_size", 16)), 8)
    dataset_config["num_workers"] = 0
    dataset_config["logging_steps"] = 1
    dataset_config["save_total_limit"] = 1
    dataset_config["gradient_accumulation_steps"] = 1
    return dataset_config


def build_label_mappings(base_model_dir: Path) -> tuple[dict[str, int], dict[int, str]]:
    label2id = load_json(base_model_dir / "label2id.json")
    id2label = {int(index): label for label, index in label2id.items()}
    return label2id, id2label


def ensure_selected_labels_in_mappings(
    selected_records: list[dict[str, Any]],
    label2id: dict[str, int],
    id2label: dict[int, str],
) -> list[str]:
    added_labels: list[str] = []
    next_id = (max((int(index) for index in label2id.values()), default=-1) + 1) if label2id else 0

    for selected_record in selected_records:
        selected_label = str(selected_record.get("label") or "").strip()
        if not selected_label:
            raise ValueError("A selected image label is empty.")

        selected_record["label"] = selected_label
        if selected_label in label2id:
            continue

        label2id[selected_label] = next_id
        id2label[next_id] = selected_label
        added_labels.append(selected_label)
        next_id += 1

    return added_labels


def build_augmented_train_records(
    base_train_records: list[dict[str, str]],
    selected_records: list[dict[str, Any]],
    repeat_count: int,
) -> list[dict[str, str]]:
    augmented_records = list(base_train_records)
    for selected_record in selected_records:
        selected_image = Path(selected_record["path"])
        target_label = str(selected_record["label"]).strip()
        for _ in range(max(1, int(repeat_count))):
            augmented_records.append({"path": str(selected_image), "label": target_label})
    return augmented_records


def main() -> None:
    args = parse_args()
    args.base_model_dir = resolve_project_path(args.base_model_dir)
    args.output_root = resolve_project_path(args.output_root)
    args.selected_records_path = resolve_project_path(args.selected_records_path)
    if args.base_model_dir is None or args.output_root is None:
        raise ValueError("Could not resolve the `base-model-dir` or `output-root` path.")
    if not args.base_model_dir.exists():
        raise FileNotFoundError(f"Base model directory does not exist: {args.base_model_dir}")
    label2id, id2label = build_label_mappings(args.base_model_dir)
    original_num_labels = len(label2id)

    target_label = None
    if args.create_new_class:
        if not args.new_class_name:
            raise ValueError("`--new-class-name` must be provided when using `--create-new-class`.")
        new_class_name = args.new_class_name.replace(" ", "_").replace("-", "_")
        if new_class_name in label2id:
            raise ValueError(f"The class '{new_class_name}' already exists.")
        next_id = max(int(id) for id in label2id.values()) + 1
        label2id[new_class_name] = next_id
        id2label[next_id] = new_class_name
        target_label = new_class_name
        print(f"New class created: {new_class_name} (ID: {next_id})")
    else:
        if not args.selected_records_path and not args.target_label:
            raise ValueError("`--target-label` must be provided.")
        target_label = args.target_label
        if target_label and target_label not in label2id:
            raise ValueError(
                f"Unknown target label '{target_label}'. Available labels: {sorted(label2id.keys())}"
            )

    selected_records: list[dict[str, Any]] = []
    if args.selected_records_path:
        if not args.selected_records_path.exists():
            raise FileNotFoundError(f"Selected records manifest does not exist: {args.selected_records_path}")
        selected_records = load_selected_records_manifest(args.selected_records_path)
    else:
        selected_images: list[Path | None] = []
        if args.selected_images:
            selected_images = [resolve_project_path(path) for group in args.selected_images for path in group]
        elif args.selected_image:
            resolved_selected_image = resolve_project_path(args.selected_image)
            selected_images = [resolved_selected_image]
        else:
            raise ValueError(
                "One of `--selected-records-path`, `--selected-image`, or `--selected-images` must be provided."
            )
        if any(path is None for path in selected_images):
            raise ValueError("Could not resolve a selected image path.")
        selected_records = [
            {
                "path": path,
                "label": target_label,
                "predicted_label": args.predicted_label.strip() or None,
            }
            for path in selected_images
            if path is not None
        ]

    if not selected_records:
        raise ValueError("There are no selected images available for training.")

    selected_images = [Path(record["path"]) for record in selected_records]
    for selected_image in selected_images:
        if not selected_image.exists():
            raise FileNotFoundError(f"Selected image does not exist: {selected_image}")

    added_selected_labels = ensure_selected_labels_in_mappings(selected_records, label2id, id2label)
    for added_label in added_selected_labels:
        print(f"New class created: {added_label} (selected record label)")

    config = build_interactive_config(args.base_model_dir, args)
    set_seed(int(config.get("seed", 42)))

    # Train only on the newly selected (active-learning or manually-picked) records — the
    # previous run's train_split.csv is never re-included, whether or not a new class is being
    # added. valid/test are queried straight from Supabase's `type` column (the same fixed
    # held-out split every round) instead of a local valid_split.csv/test_split.csv file, whose
    # location can move between model directory layouts.
    #
    # The Fine-tuning page's Active Learning "New sample" pool draws specifically from the
    # valid/test images, so a selected image can itself be a valid/test row — exclude every
    # selected path from valid/test so this round never evaluates on data it just trained on.
    base_train_records: list[dict[str, str]] = []
    selected_paths_for_exclusion = {str(Path(record["path"])) for record in selected_records}
    valid_records = [
        record for record in load_records_from_supabase("valid")
        if str(Path(record["path"])) not in selected_paths_for_exclusion
    ]
    test_records = [
        record for record in load_records_from_supabase("test")
        if str(Path(record["path"])) not in selected_paths_for_exclusion
    ]
    if not valid_records or not test_records:
        raise ValueError(
            "The valid/test split is empty after excluding this round's selected images — "
            "lower the Active Learning New sample rate so valid/test still has data left."
        )
    train_records = build_augmented_train_records(
        base_train_records=base_train_records,
        selected_records=selected_records,
        repeat_count=args.repeat_count,
    )

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = ensure_output_dir(args.output_root / run_name)
    print(f"OUTPUT_DIR={to_project_relative_path(output_dir)}")
    # Everything besides the loadable model itself (model.safetensors/config.json/
    # preprocessor_config.json) and the training-continuation inputs (label2id.json,
    # dataset_config.json) goes under metadata/ instead of cluttering the model directory:
    # checkpoints, trainer_state.json, metrics, the split CSV snapshots, and our own request/audit
    # json files.
    metadata_dir = output_dir / "metadata"

    image_processor = AutoImageProcessor.from_pretrained(args.base_model_dir)
    model = AutoModelForImageClassification.from_pretrained(
        args.base_model_dir,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=len(label2id) != original_num_labels,
        problem_type="single_label_classification",
    )

    train_dataset = FolderImageClassificationDataset(train_records, label2id)
    valid_dataset = FolderImageClassificationDataset(valid_records, label2id)
    test_dataset = FolderImageClassificationDataset(test_records, label2id)
    collator = ImageClassificationCollator(image_processor)
    class_weights = compute_class_weights(train_records, label2id)
    train_batch_size = max(1, int(config["train_batch_size"]))
    steps_per_epoch = math.ceil(len(train_dataset) / train_batch_size)
    device_name = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"TRAIN_DEVICE={device_name}")
    print("TRAIN_MODE=selected_only")
    print(f"TRAIN_DATASET_SIZE={len(train_dataset)}")
    print(f"VALID_DATASET_SIZE={len(valid_dataset)}")
    print(f"TEST_DATASET_SIZE={len(test_dataset)}")
    print(f"TRAIN_BATCH_SIZE={train_batch_size}")
    print(f"STEPS_PER_EPOCH={steps_per_epoch}")
    if device_name == "cpu":
        print("WARNING: CUDA is not available, so training will run on CPU. This may take a while.")

    training_args = build_training_arguments(config, metadata_dir, len(train_dataset))
    trainer = build_trainer(
        model=model,
        training_args=training_args,
        image_processor=image_processor,
        collator=collator,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        class_weights=class_weights,
    )

    print("TRAINING_STARTED")
    train_result = trainer.train()
    print("TRAINING_FINISHED")
    trainer.save_model(output_dir)
    image_processor.save_pretrained(output_dir)
    trainer.save_state()
    trainer.save_metrics("train", train_result.metrics)

    training_args_payload = build_training_args_payload(
        training_args=training_args,
        config=config,
        config_path=args.base_model_dir / "dataset_config.json",
    )
    replace_training_args_bin_with_json(metadata_dir, training_args_payload)

    # label2id.json/dataset_config.json are read back directly by the next incremental
    # fine-tuning round (build_label_mappings/build_interactive_config above), so they stay in
    # the model directory root; the split CSVs are now pure audit snapshots (train is always the
    # freshly selected data, valid/test come from Supabase), so they go under metadata/.
    save_records_csv(train_records, metadata_dir / "train_split.csv")
    save_records_csv(valid_records, metadata_dir / "valid_split.csv")
    save_records_csv(test_records, metadata_dir / "test_split.csv")
    save_json(label2id, output_dir / "label2id.json")
    save_json(config, output_dir / "dataset_config.json")

    save_json(train_result.metrics, metadata_dir / "train_summary.json")
    save_json(training_args_payload, metadata_dir / "training_args.json")
    save_json(
        {
            "selected_images": [to_project_relative_path(path) for path in selected_images],
            "selected_records": [
                {
                    "path": to_project_relative_path(record["path"]),
                    "label": str(record["label"]).strip(),
                    "predicted_label": str(record.get("predicted_label") or "").strip() or None,
                }
                for record in selected_records
            ],
            "predicted_label": args.predicted_label,
            "target_label": target_label,
            "create_new_class": args.create_new_class,
            "new_class_name": args.new_class_name,
            "added_selected_labels": added_selected_labels,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "repeat_count": args.repeat_count,
            "incremental_only": bool(args.incremental_only),
            "base_model_dir": to_project_relative_path(args.base_model_dir),
            "manual_target_class_input": args.manual_target_class_input or None,
            "selected_class_option": args.selected_class_option or None,
            "saved_model_dir": to_project_relative_path(output_dir),
        },
        metadata_dir / "interactive_request.json",
    )

    valid_metrics = trainer.evaluate(valid_dataset)
    trainer.save_metrics("valid", valid_metrics)
    evaluate_and_save_split(
        trainer=trainer,
        dataset=valid_dataset,
        id2label=id2label,
        output_dir=metadata_dir,
        split_name="valid",
    )

    test_metrics = trainer.evaluate(test_dataset)
    trainer.save_metrics("test", test_metrics)
    evaluate_and_save_split(
        trainer=trainer,
        dataset=test_dataset,
        id2label=id2label,
        output_dir=metadata_dir,
        split_name="test",
        save_report=True,
    )

    save_json(
        {
            "train_metrics": train_result.metrics,
            "valid_metrics": valid_metrics,
            "test_metrics": test_metrics,
        },
        metadata_dir / "all_results.json",
    )
    print("Interactive fine-tuning completed successfully.")


if __name__ == "__main__":
    main()
