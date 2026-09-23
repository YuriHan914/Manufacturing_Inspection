from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.local_gemma_model import MODEL_DIR, MODEL_ID, MODEL_SIZE_NOTE


def main() -> None:
    print(f"Downloading {MODEL_ID} into {MODEL_DIR} ({MODEL_SIZE_NOTE})")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=MODEL_ID,
        local_dir=MODEL_DIR,
        token=os.getenv("HF_TOKEN"),
    )
    print(f"Download completed: {path}")


if __name__ == "__main__":
    main()
