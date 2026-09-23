import argparse
import csv
import math
import os
import random
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
BASE_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    default_output = BASE_DIR / "output" / "sampled_train_dataset.csv"

    parser = argparse.ArgumentParser(
        description="Sample train image paths from a class-folder directory and save them to a CSV file."
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        required=True,
        help="Path to the train split directory that contains class folders.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Total number of images to sample.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=default_output,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
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


def collect_class_images(train_dir: Path) -> dict[str, list[Path]]:
    class_to_images: dict[str, list[Path]] = {}

    for class_dir in sorted(train_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        image_paths = sorted(path for path in class_dir.iterdir() if is_image_file(path))
        if image_paths:
            class_to_images[class_dir.name] = image_paths

    if not class_to_images:
        raise ValueError(f"No class folders with images were found in: {train_dir}")

    return class_to_images


def find_normal_class_name(class_names: list[str]) -> str:
    for class_name in class_names:
        if class_name.lower() == "normal":
            return class_name
    raise ValueError("Could not find a 'Normal' class folder in the train directory.")


def proportional_allocate(
    total_count: int,
    class_to_size: dict[str, int],
) -> dict[str, int]:
    if total_count <= 0:
        return {class_name: 0 for class_name in class_to_size}

    total_size = sum(class_to_size.values())
    if total_size < total_count:
        raise ValueError(
            f"Requested {total_count} anomaly samples, but only {total_size} images are available."
        )

    raw_allocations = {
        class_name: (total_count * class_size) / total_size
        for class_name, class_size in class_to_size.items()
    }
    allocations = {
        class_name: min(class_to_size[class_name], math.floor(raw_value))
        for class_name, raw_value in raw_allocations.items()
    }

    remaining = total_count - sum(allocations.values())
    if remaining == 0:
        return allocations

    remainders = sorted(
        (
            raw_allocations[class_name] - allocations[class_name],
            class_name,
        )
        for class_name in class_to_size
    )

    for _, class_name in reversed(remainders):
        if remaining == 0:
            break
        if allocations[class_name] >= class_to_size[class_name]:
            continue
        allocations[class_name] += 1
        remaining -= 1

    if remaining != 0:
        raise ValueError("Failed to allocate the requested number of stratified samples.")

    return allocations


def build_sample_records(
    class_to_images: dict[str, list[Path]],
    num_samples: int,
    rng: random.Random,
) -> list[dict[str, str]]:
    if num_samples <= 0:
        raise ValueError("--num-samples must be a positive integer.")

    normal_class = find_normal_class_name(list(class_to_images.keys()))
    normal_images = class_to_images[normal_class]
    other_classes = {
        class_name: image_paths
        for class_name, image_paths in class_to_images.items()
        if class_name != normal_class
    }

    normal_target = min(num_samples, math.floor(num_samples * 0.8 + 0.5))
    other_target = num_samples - normal_target

    if len(normal_images) < normal_target:
        max_total = math.floor(len(normal_images) / 0.8)
        raise ValueError(
            f"Normal class has only {len(normal_images)} images, so it cannot satisfy "
            f"{normal_target} samples. Reduce --num-samples to {max_total} or less."
        )

    other_total_available = sum(len(image_paths) for image_paths in other_classes.values())
    if other_total_available < other_target:
        raise ValueError(
            f"Non-normal classes have only {other_total_available} images, so they cannot satisfy "
            f"{other_target} samples."
        )

    sampled_records: list[dict[str, str]] = []

    for image_path in rng.sample(normal_images, normal_target):
        sampled_records.append({"class": normal_class, "path": to_project_relative_path(image_path)})

    other_allocations = proportional_allocate(
        total_count=other_target,
        class_to_size={class_name: len(image_paths) for class_name, image_paths in other_classes.items()},
    )

    for class_name, sample_count in other_allocations.items():
        if sample_count == 0:
            continue
        for image_path in rng.sample(other_classes[class_name], sample_count):
            sampled_records.append({"class": class_name, "path": to_project_relative_path(image_path)})

    rng.shuffle(sampled_records)

    indexed_records = []
    for index, record in enumerate(sampled_records):
        indexed_records.append(
            {
                "index": index,
                "class": record["class"],
                "path": record["path"],
            }
        )

    return indexed_records


def write_csv(records: list[dict[str, str]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["index", "class", "path"])
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = parse_args()
    args.train_dir = resolve_project_path(args.train_dir)
    args.output_csv = resolve_project_path(args.output_csv)
    if args.train_dir is None or args.output_csv is None:
        raise ValueError("Could not resolve `train-dir` or `output-csv`.")
    train_dir = args.train_dir

    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory does not exist: {train_dir}")

    class_to_images = collect_class_images(train_dir)
    rng = random.Random(args.seed)
    records = build_sample_records(class_to_images, args.num_samples, rng)
    write_csv(records, args.output_csv)

    print(f"Saved {len(records)} sampled image paths to: {to_project_relative_path(args.output_csv)}")


if __name__ == "__main__":
    main()
