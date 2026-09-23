from __future__ import annotations

from functools import lru_cache
import logging
from pathlib import Path
from typing import Any


MODEL_ID = "google/gemma-4-E2B-it"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = PROJECT_ROOT / "model"
MODEL_DIR = MODEL_ROOT / "google__gemma-4-E2B-it"
MODEL_SIZE_NOTE = "about 10.3 GB"
REQUIRED_MODEL_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
WEIGHT_FILE_GLOBS = ("*.safetensors", "*.safetensors.index.json")
PEFT_ADAPTER_CONFIG_FILE = "adapter_config.json"


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


def _import_runtime_dependencies() -> tuple[Any, Any, Any]:
    try:
        _suppress_transformers_path_alias_warning()
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "The packages required to run the local Gemma model are not installed. "
            "Please run `pip install -r requirements.txt` first."
        ) from exc

    return torch, AutoModelForImageTextToText, AutoProcessor


def are_runtime_dependencies_available() -> tuple[bool, str]:
    try:
        _import_runtime_dependencies()
    except RuntimeError as exc:
        return False, str(exc)
    return True, "Local runtime packages are ready."


def _resolve_model_dir(model_dir: str | Path | None = None) -> Path:
    candidate = Path(model_dir).expanduser() if model_dir else MODEL_DIR
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def _is_valid_model_dir(model_dir: Path) -> bool:
    # PEFT/LoRA outputs need adapter-aware loading; this runtime loads plain Gemma dirs.
    if (model_dir / PEFT_ADAPTER_CONFIG_FILE).exists():
        return False

    required_files_ready = all((model_dir / filename).exists() for filename in REQUIRED_MODEL_FILES)
    weight_files_ready = any(next(model_dir.glob(pattern), None) is not None for pattern in WEIGHT_FILE_GLOBS)
    return required_files_ready and weight_files_ready


def list_available_model_dirs() -> list[Path]:
    if not MODEL_ROOT.exists():
        return []
    return sorted(
        path for path in MODEL_ROOT.iterdir() if path.is_dir() and _is_valid_model_dir(path)
    )


def is_model_downloaded(model_dir: str | Path | None = None) -> bool:
    return _is_valid_model_dir(_resolve_model_dir(model_dir))


@lru_cache(maxsize=1)
def load_model(model_dir: str | Path | None = None) -> tuple[Any, Any, Any]:
    torch, AutoModelForImageTextToText, AutoProcessor = _import_runtime_dependencies()
    resolved_model_dir = _resolve_model_dir(model_dir)

    if not is_model_downloaded(resolved_model_dir):
        raise RuntimeError(
            "The local model files are missing. "
            f"Please prepare the model first at `{resolved_model_dir}`."
        )

    processor = AutoProcessor.from_pretrained(resolved_model_dir)
    model = AutoModelForImageTextToText.from_pretrained(
        resolved_model_dir,
        dtype="auto",
        device_map="auto",
    )
    model.eval()
    return torch, processor, model


def unload_model() -> None:
    try:
        torch, _auto_model, _auto_processor = _import_runtime_dependencies()
    except RuntimeError:
        load_model.cache_clear()
        return

    load_model.cache_clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def embed_text(text: str, model_dir: str | Path | None = None) -> list[float]:
    """Return a last-token hidden-state embedding for `text` from the local Gemma model.

    Used for cosine-similarity based tool/intent routing (see scripts.tool_router) — the same
    local model produces both generation and embeddings, no separate embedding model is loaded.

    Gemma is a causal (decoder-only) model, so only the LAST token's hidden state has
    attended to the full sequence via self-attention; every earlier position only saw a
    prefix. Mean-pooling across all positions would dilute the embedding with those
    partial-context representations, so last-token pooling is used instead — the standard
    approach for causal-LM embeddings (as in e.g. e5-mistral, LLM2Vec).
    """
    normalized = text.strip()
    if not normalized:
        return []

    resolved_model_dir = _resolve_model_dir(model_dir)
    torch, processor, model = load_model(str(resolved_model_dir))
    tokenizer = getattr(processor, "tokenizer", processor)

    inputs = tokenizer(normalized, return_tensors="pt", truncation=True, max_length=64).to(model.device)

    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)

    last_hidden_state = outputs.hidden_states[-1][0]  # (seq_len, hidden_dim)
    embedding = last_hidden_state[-1]  # hidden state of the final (fully-attended) token
    return embedding.float().cpu().tolist()


def _extract_response_text(processor: Any, output_tokens: Any, input_len: int) -> str:
    raw_text = processor.decode(output_tokens[0][input_len:], skip_special_tokens=False).strip()

    if hasattr(processor, "parse_response"):
        try:
            parsed = processor.parse_response(raw_text)
        except Exception:
            parsed = None
        if isinstance(parsed, str) and parsed.strip():
            return parsed.strip()
        if isinstance(parsed, dict):
            for key in ("text", "response", "content", "answer", "final"):
                value = parsed.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

    cleaned_text = processor.decode(output_tokens[0][input_len:], skip_special_tokens=True).strip()
    return cleaned_text or raw_text


def generate_response(
    prompt: str,
    system_prompt: str,
    image_paths: list[str] | None = None,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    model_dir: str | Path | None = None,
) -> str:
    if not prompt.strip():
        return ""

    resolved_model_dir = _resolve_model_dir(model_dir)
    torch, processor, model = load_model(str(resolved_model_dir))
    messages: list[dict[str, Any]] = []
    if system_prompt.strip():
        messages.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt.strip()}],
            }
        )

    user_content: list[dict[str, str]] = []
    for image_path in image_paths or []:
        resolved_path = Path(image_path)
        if not resolved_path.is_absolute():
            resolved_path = (PROJECT_ROOT / resolved_path).resolve()
        if resolved_path.exists():
            user_content.append({"type": "image", "path": str(resolved_path.resolve())})
    user_content.append({"type": "text", "text": prompt.strip()})
    messages.append({"role": "user", "content": user_content})

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    generation_temperature = max(0.0, float(temperature))
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": generation_temperature > 0.0,
    }
    if generation_temperature > 0.0:
        generation_kwargs["temperature"] = generation_temperature

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            **generation_kwargs,
        )

    decoded = _extract_response_text(processor, outputs, input_len).strip()
    return decoded or "The response was empty. Please check the prompt or the model state."
