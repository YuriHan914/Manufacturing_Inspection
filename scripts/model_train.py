import argparse
import csv
import inspect
import json
import logging
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFile
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import Dataset


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
from transformers import AutoModelForImageClassification, Trainer, TrainingArguments, set_seed

try:
    from transformers import AutoImageProcessor
except ImportError:  # Compatibility with older Transformers releases.
    from transformers import AutoFeatureExtractor as AutoImageProcessor


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING_CONFIG = {
    "model_name": "apple/mobilevit-small",
    "train_batch_size": 16,
    "eval_batch_size": 16,
    "num_epochs": 10,
    "learning_rate": 3e-5,
    "weight_decay": 1e-4,
    "warmup_ratio": 0.1,
    "gradient_accumulation_steps": 1,
    "num_workers": 4,
    "logging_steps": 10,
    "save_total_limit": 2,
    "seed": 42,
    "disable_class_weights": False,
    "no_fp16": False,
    "classes_to_train": None,
}
ImageFile.LOAD_TRUNCATED_IMAGES = True


def _resolve_record_image_path(record: dict[str, str]) -> Path:
    image_path = Path(record["path"])
    if not image_path.is_absolute():
        image_path = (BASE_DIR / image_path).resolve()
    return image_path


def _filter_missing_image_records(
    records: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[str]]:
    kept_records = []
    missing_paths = []
    for record in records:
        image_path = _resolve_record_image_path(record)
        if image_path.is_file():
            kept_records.append(record)
        else:
            missing_paths.append(str(image_path))
    return kept_records, missing_paths


class FolderImageClassificationDataset(Dataset):
    def __init__(self, records: list[dict[str, str]], label2id: dict[str, int]) -> None:
        self.records, missing_paths = _filter_missing_image_records(records)
        self.label2id = label2id
        if missing_paths:
            print(f"WARNING: skipping {len(missing_paths)} record(s) with missing image files:")
            for path in missing_paths[:10]:
                print(f"  - {path}")
            if len(missing_paths) > 10:
                print(f"  ... and {len(missing_paths) - 10} more")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_path = _resolve_record_image_path(record)
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")

        return {
            "image": rgb_image,
            "labels": self.label2id[record["label"]],
            "path": str(image_path),
        }


class ImageClassificationCollator:
    def __init__(self, image_processor: Any) -> None:
        self.image_processor = image_processor

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        images = [item["image"] for item in batch]
        labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
        model_inputs = self.image_processor(images=images, return_tensors="pt")
        model_inputs["labels"] = labels
        return model_inputs


class WeightedTrainer(Trainer):
    def __init__(self, *args: Any, class_weights: torch.Tensor | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        **_: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        labels = inputs["labels"]
        model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
        outputs = model(**model_inputs)
        logits = outputs.logits

        weight = None
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)

        loss = torch.nn.functional.cross_entropy(logits, labels, weight=weight)
        if return_outputs:
            return loss, outputs
        return loss


def parse_args() -> argparse.Namespace:
    default_output_dir = BASE_DIR / "output" / "mobilevit_small_classifier"

    parser = argparse.ArgumentParser(
        description="Train MobileViT using folder datasets and a JSON config file.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        required=True,
        help="Path to the dataset and training config JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Directory to save checkpoints, metrics, and the final model.",
    )
    return parser.parse_args()


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


def resolve_project_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    raw_value = str(value).strip()
    if not raw_value:
        return None
    path = Path(raw_value).expanduser()
    if path.is_absolute():
        return path
    return (BASE_DIR / path).resolve()


def to_project_relative_path(value: str | Path | None) -> str:
    if value is None:
        return "-"
    raw_value = str(value).strip()
    if not raw_value:
        return "-"
    path = Path(raw_value).expanduser()
    absolute_path = path if path.is_absolute() else (BASE_DIR / path).resolve()
    return os.path.relpath(str(absolute_path), str(BASE_DIR))


def _normalize_record_paths(records: list[dict[str, str]]) -> list[dict[str, str]]:
    normalized_records: list[dict[str, str]] = []
    for record in records:
        normalized_records.append(
            {
                "path": to_project_relative_path(record["path"]),
                "label": record["label"],
            }
        )
    return normalized_records


def _resolve_model_name_for_runtime(model_name: Any) -> str:
    raw_model_name = str(model_name).strip()
    if not raw_model_name:
        return raw_model_name
    if _looks_like_project_path(raw_model_name):
        resolved_path = resolve_project_path(raw_model_name)
        if resolved_path is not None and resolved_path.exists():
            return str(resolved_path)
    return raw_model_name


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        raw_config = json.load(file)

    merged_config = dict(DEFAULT_TRAINING_CONFIG)
    merged_config.update(raw_config)
    return merged_config


def discover_model_class_names(model_name: Any) -> list[str]:
    if not model_name:
        return []

    model_path = resolve_project_path(model_name)
    if model_path is None:
        return []
    if model_path.is_file() and model_path.name == "model.safetensors":
        model_path = model_path.parent
    if not model_path.exists():
        return []

    label2id_path = model_path / "label2id.json"
    if not label2id_path.exists():
        return []

    label2id = json.loads(label2id_path.read_text(encoding="utf-8"))
    if not isinstance(label2id, dict):
        return []

    ordered_items = sorted(label2id.items(), key=lambda item: int(item[1]))
    return [label for label, _ in ordered_items]


def synchronize_training_config(
    config_path: Path,
    base_model_dir: Path,
    additional_classes: list[str] | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    model_classes = discover_model_class_names(base_model_dir)
    configured_classes = config.get("classes_to_train") or []
    if not isinstance(configured_classes, list):
        configured_classes = []

    merged_classes: list[str] = []
    for class_name in [*model_classes, *configured_classes, *(additional_classes or [])]:
        normalized = str(class_name).strip()
        if normalized and normalized not in merged_classes:
            merged_classes.append(normalized)

    config["model_name"] = to_project_relative_path(base_model_dir)
    if merged_classes:
        config["classes_to_train"] = merged_classes

    save_json(config, config_path)
    return config


def ensure_output_dir(output_dir: Path) -> Path:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise NotADirectoryError(f"Output path exists but is not a directory: {output_dir}")

        candidate_index = 1
        while True:
            candidate_dir = output_dir.parent / f"{output_dir.name}_{candidate_index}"
            if not candidate_dir.exists():
                candidate_dir.mkdir(parents=True, exist_ok=False)
                return candidate_dir
            candidate_index += 1

    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def resolve_split_dir(config: dict[str, Any], split_key: str) -> Path:
    if "data_root_path" not in config:
        raise KeyError("Config must include 'data_root_path'.")
    if split_key not in config:
        raise KeyError(f"Config must include '{split_key}'.")

    data_root = resolve_project_path(config["data_root_path"])
    if data_root is None:
        raise ValueError("Configured 'data_root_path' is empty.")
    split_path = str(config[split_key]).lstrip("/\\")
    return data_root / split_path


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def discover_class_names(split_dir: Path) -> list[str]:
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory does not exist: {split_dir}")

    class_names = sorted(path.name for path in split_dir.iterdir() if path.is_dir())
    if not class_names:
        raise ValueError(f"No class directories were found in: {split_dir}")
    return class_names


def resolve_selected_classes(train_dir: Path, config: dict[str, Any]) -> tuple[list[str], list[str]]:
    available_classes = discover_class_names(train_dir)
    model_classes = discover_model_class_names(config.get("model_name"))
    requested_classes = config.get("classes_to_train")

    if requested_classes in (None, []):
        return available_classes, available_classes

    if not isinstance(requested_classes, list) or not all(
        isinstance(class_name, str) for class_name in requested_classes
    ):
        raise ValueError("'classes_to_train' must be a list of class name strings.")

    selected_classes: list[str] = []
    seen: set[str] = set()
    for raw_class_name in requested_classes:
        class_name = raw_class_name.strip()
        if not class_name:
            raise ValueError("'classes_to_train' cannot include empty class names.")
        if class_name in seen:
            raise ValueError(f"Duplicate class name in 'classes_to_train': {class_name}")
        seen.add(class_name)
        selected_classes.append(class_name)

    known_classes = set(available_classes) | set(model_classes)
    unknown_classes = sorted(set(selected_classes) - known_classes)
    if unknown_classes:
        raise ValueError(
            f"Unknown classes in 'classes_to_train': {unknown_classes}. "
            f"Available classes: {sorted(known_classes)}"
        )

    return selected_classes, available_classes


def collect_split_records(
    split_dir: Path,
    split_name: str,
    class_names: list[str],
    allow_missing_classes: bool = False,
) -> list[dict[str, str]]:
    available_classes = set(discover_class_names(split_dir))
    missing_classes = sorted(set(class_names) - available_classes)
    if missing_classes and not allow_missing_classes:
        raise ValueError(
            f"{split_name} split is missing selected classes: {missing_classes}"
        )

    records: list[dict[str, str]] = []
    for class_name in class_names:
        class_dir = split_dir / class_name
        if not class_dir.exists():
            continue
        image_paths = sorted(path for path in class_dir.rglob("*") if is_image_file(path))
        if not image_paths:
            if allow_missing_classes:
                continue
            raise ValueError(f"No images were found for class '{class_name}' in {split_dir}")

        for image_path in image_paths:
            records.append({"path": str(image_path), "label": class_name})

    if not records:
        raise ValueError(f"No image records were collected from: {split_dir}")

    return records


def build_label_mappings(class_names: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    label2id = {label: index for index, label in enumerate(class_names)}
    id2label = {index: label for label, index in label2id.items()}
    return label2id, id2label


def compute_class_weights(
    records: list[dict[str, str]],
    label2id: dict[str, int],
) -> torch.Tensor:
    label_counts = Counter(record["label"] for record in records)
    total_samples = sum(label_counts.values())
    num_classes = len(label2id)

    weights = torch.ones(num_classes, dtype=torch.float32)
    for label, label_id in label2id.items():
        class_count = label_counts.get(label, 0)
        if class_count == 0:
            weights[label_id] = 0.0
            continue
        weights[label_id] = total_samples / (num_classes * class_count)
    return weights


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "weighted_f1": f1_score(labels, predictions, average="weighted", zero_division=0),
    }


def build_training_arguments(config: dict[str, Any], output_dir: Path, train_dataset_size: int) -> TrainingArguments:
    training_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "remove_unused_columns": False,
        "learning_rate": float(config["learning_rate"]),
        "per_device_train_batch_size": int(config["train_batch_size"]),
        "per_device_eval_batch_size": int(config["eval_batch_size"]),
        "gradient_accumulation_steps": int(config["gradient_accumulation_steps"]),
        "num_train_epochs": float(config["num_epochs"]),
        "weight_decay": float(config["weight_decay"]),
        "logging_steps": int(config["logging_steps"]),
        "save_total_limit": int(config["save_total_limit"]),
        "seed": int(config["seed"]),
        "report_to": "none",
        "dataloader_num_workers": int(config["num_workers"]),
        "dataloader_pin_memory": torch.cuda.is_available(),
        "fp16": torch.cuda.is_available() and not bool(config["no_fp16"]),
        "push_to_hub": False,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "balanced_accuracy",
        "greater_is_better": True,
    }

    parameter_names = inspect.signature(TrainingArguments.__init__).parameters
    eval_key = "eval_strategy" if "eval_strategy" in parameter_names else "evaluation_strategy"
    training_kwargs[eval_key] = "epoch"

    if "warmup_ratio" in parameter_names:
        training_kwargs["warmup_ratio"] = float(config["warmup_ratio"])
    else:
        # transformers v5 dropped warmup_ratio from TrainingArguments; convert it to warmup_steps ourselves.
        optimizer_steps_per_epoch = max(
            1,
            math.ceil(train_dataset_size / (int(config["train_batch_size"]) * int(config["gradient_accumulation_steps"]))),
        )
        total_optimizer_steps = max(1, math.ceil(optimizer_steps_per_epoch * float(config["num_epochs"])))
        training_kwargs["warmup_steps"] = int(total_optimizer_steps * float(config["warmup_ratio"]))

    return TrainingArguments(**training_kwargs)


def build_trainer(
    model: torch.nn.Module,
    training_args: TrainingArguments,
    image_processor: Any,
    collator: ImageClassificationCollator,
    train_dataset: Dataset,
    valid_dataset: Dataset,
    class_weights: torch.Tensor | None,
) -> WeightedTrainer:
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "data_collator": collator,
        "train_dataset": train_dataset,
        "eval_dataset": valid_dataset,
        "compute_metrics": compute_metrics,
    }

    trainer_parameters = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_parameters:
        trainer_kwargs["processing_class"] = image_processor
    else:
        trainer_kwargs["tokenizer"] = image_processor

    return WeightedTrainer(class_weights=class_weights, **trainer_kwargs)


def save_records_csv(records: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["path", "label"])
        writer.writeheader()
        writer.writerows(_normalize_record_paths(records))


def save_json(data: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def build_training_args_payload(
    training_args: TrainingArguments,
    config: dict[str, Any],
    config_path: Path,
) -> dict[str, Any]:
    training_args_payload = training_args.to_dict()
    for key in ("output_dir", "logging_dir"):
        if training_args_payload.get(key):
            training_args_payload[key] = to_project_relative_path(training_args_payload[key])

    normalized_config = dict(config)
    if _looks_like_project_path(normalized_config.get("model_name")):
        normalized_config["model_name"] = to_project_relative_path(normalized_config["model_name"])
    if _looks_like_project_path(normalized_config.get("data_root_path")):
        normalized_config["data_root_path"] = to_project_relative_path(normalized_config["data_root_path"])

    return {
        "training_args": training_args_payload,
        "config_path": to_project_relative_path(config_path),
        "config_values": normalized_config,
    }


def replace_training_args_bin_with_json(
    output_dir: Path,
    training_args_payload: dict[str, Any],
) -> None:
    legacy_paths = sorted(output_dir.rglob("training_args.bin"))
    if not legacy_paths:
        save_json(training_args_payload, output_dir / "training_args.json")
        return

    for legacy_path in legacy_paths:
        save_json(training_args_payload, legacy_path.with_suffix(".json"))
        legacy_path.unlink()


def evaluate_and_save_split(
    trainer: Trainer,
    dataset: Dataset,
    id2label: dict[int, str],
    output_dir: Path,
    split_name: str,
    save_report: bool = False,
) -> dict[str, float]:
    predict_kwargs: dict[str, Any] = {}
    predict_parameters = inspect.signature(Trainer.predict).parameters
    if "metric_key_prefix" in predict_parameters:
        predict_kwargs["metric_key_prefix"] = split_name

    predictions_output = trainer.predict(dataset, **predict_kwargs)
    logits = predictions_output.predictions
    labels = predictions_output.label_ids
    predictions = np.argmax(logits, axis=-1)

    metrics = {
        f"{split_name}_{metric_name}": metric_value
        for metric_name, metric_value in compute_metrics((logits, labels)).items()
    }
    for key, value in predictions_output.metrics.items():
        if isinstance(value, (int, float)):
            metrics[key] = value

    save_json(metrics, output_dir / f"{split_name}_metrics.json")

    if save_report:
        label_names = [id2label[index] for index in range(len(id2label))]
        report = classification_report(
            labels,
            predictions,
            labels=list(range(len(label_names))),
            target_names=label_names,
            zero_division=0,
            output_dict=True,
        )
        confusion = confusion_matrix(labels, predictions, labels=list(range(len(label_names))))

        save_json(report, output_dir / "classification_report.json")
        save_json(
            {
                "labels": label_names,
                "matrix": confusion.tolist(),
            },
            output_dir / "confusion_matrix.json",
        )

    return metrics


def main() -> None:
    args = parse_args()
    args.config_path = resolve_project_path(args.config_path)
    args.output_dir = resolve_project_path(args.output_dir)
    if args.config_path is None or args.output_dir is None:
        raise ValueError("Could not resolve `config_path` or `output_dir`.")
    args.output_dir = ensure_output_dir(args.output_dir)

    config = load_config(args.config_path)
    set_seed(int(config["seed"]))

    train_dir = resolve_split_dir(config, "train_data_path")
    valid_dir = resolve_split_dir(config, "valid_data_path")
    test_dir = resolve_split_dir(config, "test_data_path")
    model_defined_classes = discover_model_class_names(config.get("model_name"))
    allow_missing_classes = bool(model_defined_classes)

    selected_classes, available_classes = resolve_selected_classes(train_dir, config)
    train_records = collect_split_records(
        train_dir,
        split_name="train",
        class_names=selected_classes,
        allow_missing_classes=allow_missing_classes,
    )
    valid_records = collect_split_records(
        valid_dir,
        split_name="valid",
        class_names=selected_classes,
        allow_missing_classes=allow_missing_classes,
    )
    test_records = collect_split_records(
        test_dir,
        split_name="test",
        class_names=selected_classes,
        allow_missing_classes=allow_missing_classes,
    )

    label2id, id2label = build_label_mappings(selected_classes)

    print(f"Train directory: {to_project_relative_path(train_dir)}")
    print(f"Valid directory: {to_project_relative_path(valid_dir)}")
    print(f"Test directory: {to_project_relative_path(test_dir)}")
    print(f"Available classes ({len(available_classes)}): {available_classes}")
    print(f"Selected classes ({len(selected_classes)}): {selected_classes}")
    print(f"Train samples: {len(train_records)}")
    print(f"Validation samples: {len(valid_records)}")
    print(f"Test samples: {len(test_records)}")

    model_name = _resolve_model_name_for_runtime(config["model_name"])
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        problem_type="single_label_classification",
    )

    train_dataset = FolderImageClassificationDataset(train_records, label2id)
    valid_dataset = FolderImageClassificationDataset(valid_records, label2id)
    test_dataset = FolderImageClassificationDataset(test_records, label2id)
    collator = ImageClassificationCollator(image_processor)

    class_weights = None
    if not bool(config["disable_class_weights"]):
        class_weights = compute_class_weights(train_records, label2id)
        print(f"Using class weights: {class_weights.tolist()}")

    training_args = build_training_arguments(config, args.output_dir, len(train_dataset))
    trainer = build_trainer(
        model=model,
        training_args=training_args,
        image_processor=image_processor,
        collator=collator,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        class_weights=class_weights,
    )

    train_result = trainer.train()
    trainer.save_model(args.output_dir)
    image_processor.save_pretrained(args.output_dir)
    trainer.save_state()
    trainer.save_metrics("train", train_result.metrics)

    training_args_payload = build_training_args_payload(
        training_args=training_args,
        config=config,
        config_path=args.config_path,
    )
    replace_training_args_bin_with_json(args.output_dir, training_args_payload)

    save_records_csv(train_records, args.output_dir / "train_split.csv")
    save_records_csv(valid_records, args.output_dir / "valid_split.csv")
    save_records_csv(test_records, args.output_dir / "test_split.csv")
    save_json(label2id, args.output_dir / "label2id.json")
    save_json(config, args.output_dir / "dataset_config.json")
    save_json(train_result.metrics, args.output_dir / "train_summary.json")
    save_json(training_args_payload, args.output_dir / "training_args.json")

    valid_metrics = trainer.evaluate(valid_dataset)
    trainer.save_metrics("valid", valid_metrics)
    evaluate_and_save_split(
        trainer=trainer,
        dataset=valid_dataset,
        id2label=id2label,
        output_dir=args.output_dir,
        split_name="valid",
        save_report=False,
    )
    evaluate_and_save_split(
        trainer=trainer,
        dataset=test_dataset,
        id2label=id2label,
        output_dir=args.output_dir,
        split_name="test",
        save_report=True,
    )

    print(f"Saved fine-tuned model and reports to: {args.output_dir}")


if __name__ == "__main__":
    main()
