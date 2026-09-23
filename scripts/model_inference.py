import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageFile


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
from transformers import AutoModelForImageClassification

try:
    from transformers import AutoImageProcessor
except ImportError:  # Compatibility with older Transformers releases.
    from transformers import AutoFeatureExtractor as AutoImageProcessor


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
BASE_DIR = Path(__file__).resolve().parents[1]
ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args() -> argparse.Namespace:
    default_model_path = BASE_DIR / "output" / "mobilevit_small_9_classifier" / "model.safetensors"
    default_data_path = Path("../../../Data/manufacture_semicond_seg/test")
    default_output_path = BASE_DIR / "output" / "inference_results"

    parser = argparse.ArgumentParser(
        description="Run inference with a fine-tuned MobileViT image classification model.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=default_model_path,
        help="Path to model.safetensors or to the saved model directory.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=default_data_path,
        help="Path to a single image file or to a folder containing images.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=default_output_path,
        help=(
            "Base output path. A timestamped folder is created under this path, "
            "and inference_results.json plus inference_timing.txt are saved inside it."
        ),
    )
    return parser.parse_args()


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


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def resolve_model_dir(model_path: Path) -> Path:
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    if model_path.is_dir():
        model_dir = model_path
    elif model_path.is_file() and model_path.name == "model.safetensors":
        model_dir = model_path.parent
    else:
        raise ValueError(
            "--model-path must point to a saved model directory or to a file named 'model.safetensors'."
        )

    required_files = [
        model_dir / "model.safetensors",
        model_dir / "config.json",
        model_dir / "preprocessor_config.json",
    ]
    missing_files = [str(path) for path in required_files if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            f"Model directory is missing required files: {missing_files}"
        )

    return model_dir


def collect_image_paths(data_path: Path) -> tuple[list[Path], str]:
    if not data_path.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")

    if data_path.is_file():
        if not is_image_file(data_path):
            raise ValueError(f"Data path is not a supported image file: {data_path}")
        return [data_path], "single"

    if not data_path.is_dir():
        raise ValueError(f"Data path must be an image file or a directory: {data_path}")

    image_paths = sorted(path for path in data_path.rglob("*") if is_image_file(path))
    if not image_paths:
        raise ValueError(f"No supported image files were found in: {data_path}")

    return image_paths, "batch"


def resolve_output_targets(output_path: Path, inference_mode: str) -> tuple[Path, Path, Path]:
    if output_path.suffix:
        base_output_dir = output_path.parent
    else:
        base_output_dir = output_path

    if base_output_dir.exists() and not base_output_dir.is_dir():
        raise NotADirectoryError(f"Output path exists but is not a directory: {base_output_dir}")

    base_output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    timestamp_output_dir = base_output_dir / timestamp
    timestamp_output_dir.mkdir(parents=True, exist_ok=False)

    return (
        timestamp_output_dir / f"{inference_mode}_inference_results.json",
        timestamp_output_dir / f"{inference_mode}_inference_timing.txt",
        timestamp_output_dir,
    )


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def predict_single_image(
    image_path: Path,
    image_processor: Any,
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[str, float, float]:
    preprocess_start_ns = time.perf_counter_ns()

    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")

    inputs = image_processor(images=rgb_image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    preprocess_time_milliseconds = (time.perf_counter_ns() - preprocess_start_ns) / 1_000_000.0

    synchronize_if_needed(device)
    inference_start_ns = time.perf_counter_ns()
    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_index = logits.argmax(dim=-1).item()
    predicted_label = model.config.id2label[predicted_index]
    synchronize_if_needed(device)
    inference_time_milliseconds = (time.perf_counter_ns() - inference_start_ns) / 1_000_000.0

    return predicted_label, preprocess_time_milliseconds, inference_time_milliseconds


def save_results(results: dict[str, str], output_json_path: Path) -> None:
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_results = {
        to_project_relative_path(path): label for path, label in results.items()
    }
    output_json_path.write_text(
        json.dumps(normalized_results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_timing(
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
        f"model_path: {to_project_relative_path(model_path)}",
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

    output_timing_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.model_path = resolve_project_path(args.model_path)
    args.data_path = resolve_project_path(args.data_path)
    args.output_path = resolve_project_path(args.output_path)
    if args.model_path is None or args.data_path is None or args.output_path is None:
        raise ValueError("Could not resolve the `model-path`, `data-path`, or `output-path`.")
    total_start_ns = time.perf_counter_ns()

    model_dir = resolve_model_dir(args.model_path)
    image_paths, inference_mode = collect_image_paths(args.data_path)
    output_json_path, output_timing_path, output_dir = resolve_output_targets(
        args.output_path,
        inference_mode,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    initial_model_load_start_ns = time.perf_counter_ns()
    image_processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModelForImageClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    synchronize_if_needed(device)
    initial_model_load_time_milliseconds = (
        time.perf_counter_ns() - initial_model_load_start_ns
    ) / 1_000_000.0

    results: dict[str, str] = {}
    per_image_preprocess_times_milliseconds: dict[str, float] = {}
    per_image_inference_times_milliseconds: dict[str, float] = {}

    for image_path in image_paths:
        predicted_label, preprocess_time_milliseconds, inference_time_milliseconds = predict_single_image(
            image_path=image_path,
            image_processor=image_processor,
            model=model,
            device=device,
        )
        relative_image_path = to_project_relative_path(image_path)
        results[relative_image_path] = predicted_label
        per_image_preprocess_times_milliseconds[relative_image_path] = preprocess_time_milliseconds
        per_image_inference_times_milliseconds[relative_image_path] = inference_time_milliseconds

    save_results(results, output_json_path)

    total_process_time_milliseconds = (time.perf_counter_ns() - total_start_ns) / 1_000_000.0
    save_timing(
        model_path=args.model_path,
        initial_model_load_time_milliseconds=initial_model_load_time_milliseconds,
        total_process_time_milliseconds=total_process_time_milliseconds,
        per_image_preprocess_times_milliseconds=per_image_preprocess_times_milliseconds,
        per_image_inference_times_milliseconds=per_image_inference_times_milliseconds,
        output_timing_path=output_timing_path,
    )

    print(f"Saved inference artifacts to: {to_project_relative_path(output_dir)}")
    print(f"Saved inference results to: {to_project_relative_path(output_json_path)}")
    print(f"Saved inference timing to: {to_project_relative_path(output_timing_path)}")


if __name__ == "__main__":
    main()
