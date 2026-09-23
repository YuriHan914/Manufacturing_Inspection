"""Default classification inference backend: TensorRT (GPU) with an ONNX Runtime (CPU) fallback.

Both `mobilevit_jax_engine_intro.engine` and `mobilevit_jax_model.onnx` were exported from the
same JAX-trained MobileViT-S weights (see jax_transform_onnx.py / convert_onnx_to_tensorrt.py),
so they share the same preprocessing and id2label mapping as
`model/classification/mobilevit_small_9_classifier`.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT_DIR = HERE.parent
MODEL_DIR = ROOT_DIR / "model" / "classification"
PROCESSOR_DIR = MODEL_DIR / "mobilevit_small_9_classifier"

ENGINE_PATH = PROCESSOR_DIR / "mobilevit_jax_engine_intro.engine"
ONNX_PATH = PROCESSOR_DIR / "mobilevit_jax_model.onnx"


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


def gpu_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def _load_id2label() -> dict[int, str]:
    config = json.loads((PROCESSOR_DIR / "config.json").read_text(encoding="utf-8"))
    return {int(index): str(label) for index, label in config["id2label"].items()}


class _TensorRTClassifier:
    """Runs mobilevit_jax_engine_intro.engine on GPU using torch CUDA tensors as device buffers
    (no pycuda needed since torch already owns a CUDA context)."""

    def __init__(self, engine_path: Path = ENGINE_PATH) -> None:
        import tensorrt as trt
        import torch

        self._torch = torch
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as engine_file:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(engine_file.read())
        self.context = self.engine.create_execution_context()

        tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input_name = next(
            name for name in tensor_names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        )
        self.output_name = next(
            name for name in tensor_names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
        )
        self.output_shape = tuple(self.engine.get_tensor_shape(self.output_name))

    def predict_logits(self, pixel_values: np.ndarray) -> np.ndarray:
        torch = self._torch
        x = torch.from_numpy(np.ascontiguousarray(pixel_values, dtype=np.float32)).to("cuda")
        out = torch.empty((x.shape[0], *self.output_shape[1:]), dtype=torch.float32, device="cuda")
        stream = torch.cuda.current_stream().cuda_stream
        # Engine input shape is fixed at batch=1, so images are fed through one at a time.
        for i in range(x.shape[0]):
            self.context.set_tensor_address(self.input_name, x[i].data_ptr())
            self.context.set_tensor_address(self.output_name, out[i].data_ptr())
            self.context.execute_async_v3(stream)
        torch.cuda.synchronize()
        return out.cpu().numpy()


class _OnnxClassifier:
    """Runs mobilevit_jax_model.onnx on CPU via onnxruntime."""

    def __init__(self, onnx_path: Path = ONNX_PATH) -> None:
        import onnxruntime as ort

        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3  # silence a harmless silu output-shape metadata warning
        self.session = ort.InferenceSession(
            str(onnx_path), sess_options=session_options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict_logits(self, pixel_values: np.ndarray) -> np.ndarray:
        pixel_values = np.ascontiguousarray(pixel_values, dtype=np.float32)
        # ONNX input shape is fixed at batch=1, so images are fed through one at a time.
        return np.concatenate(
            [
                self.session.run([self.output_name], {self.input_name: pixel_values[i : i + 1]})[0]
                for i in range(pixel_values.shape[0])
            ]
        )


class DefaultClassifierRuntime:
    """Default classifier: TensorRT engine on GPU, ONNX Runtime CPU fallback when no GPU is present
    (or if engine initialization fails for any reason).

    `backend`/`model_path` let a caller force a specific backend and export file instead of the
    auto-detect/fallback behavior above (used by the Classification Inference model picker in
    scripts/utils.py to run a user-selected TensorRT or ONNX export rather than the fixed default)."""

    def __init__(self, backend: str | None = None, model_path: Path | None = None) -> None:
        _suppress_transformers_path_alias_warning()
        from transformers import AutoImageProcessor

        self.processor = AutoImageProcessor.from_pretrained(str(PROCESSOR_DIR))
        self.id2label = _load_id2label()
        self._lock = threading.Lock()

        if backend in (None, "tensorrt") and (backend == "tensorrt" or gpu_available()):
            try:
                self._classifier = _TensorRTClassifier(model_path or ENGINE_PATH)
                self.backend = "tensorrt"
                self.device = "cuda"
                return
            except Exception:
                if backend == "tensorrt":
                    raise  # explicitly requested TensorRT, so don't silently swap backends
                pass  # auto mode: fall through to the CPU ONNX backend below

        self._classifier = _OnnxClassifier(model_path or ONNX_PATH)
        self.backend = "onnx"
        self.device = "cpu"

    def preprocess(self, image: Any) -> np.ndarray:
        pixel_values = self.processor(images=image, return_tensors="np")["pixel_values"]
        return np.asarray(pixel_values, dtype=np.float32)

    def infer_label(self, pixel_values: np.ndarray) -> str:
        with self._lock:
            logits = self._classifier.predict_logits(pixel_values)
        predicted_index = int(np.argmax(logits[0]))
        return self.id2label[predicted_index]


_runtime_singleton: DefaultClassifierRuntime | None = None
_runtime_lock = threading.Lock()


def get_default_classifier_runtime() -> DefaultClassifierRuntime:
    """Process-wide singleton so the engine/ONNX session is loaded only once."""
    global _runtime_singleton
    if _runtime_singleton is None:
        with _runtime_lock:
            if _runtime_singleton is None:
                _runtime_singleton = DefaultClassifierRuntime()
    return _runtime_singleton
