import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import onnxruntime as ort
import structlog

from pet_breed_mlops_project.config import config
from pet_breed_mlops_project.transform import imagenet_norm_trans

logger = structlog.getLogger()


class BaseModel(ABC):
    """Prediction interface that API, BentoML runner and ONNX model all sit behind."""

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def preprocess(self, inputs): ...

    @abstractmethod
    def predict_top3(self, inputs, k: int = 3): ...


class PetBreedModel(BaseModel):
    """ONNX-backed pet-breed classifier, loaded from a directory containing:
    model.onnx, class_names.json, transform_config.json
    """

    def __init__(self, model_path: str | Path, use_gpu: bool = False):
        self.model_path = Path(model_path)
        self.use_gpu = use_gpu
        self.class_names: list[str] | None = None
        self.session: ort.InferenceSession | None = None
        self._loaded = False

        logger.info("model.init", model_path=str(self.model_path), use_gpu=use_gpu)

    def load(self) -> None:
        logger.info("model.load.start", model_path=str(self.model_path))

        with open(self.model_path / config.class_name_filename) as f:
            self.class_names = json.load(f)

        with open(self.model_path / config.transform_config_filename) as f:
            transform_config = json.load(f)

        self._eval_transform = imagenet_norm_trans(
            transform_config["mean"], transform_config["std"]
        )["test"]

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.use_gpu
            else ["CPUExecutionProvider"]
        )
        onnx_path = self.model_path / config.onnx_filename
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self._input_name = self.session.get_inputs()[0].name

        self._loaded = True
        logger.info(
            "model.load.complete",
            num_classes=len(self.class_names),
            onnx_path=str(onnx_path),
            providers=providers,
        )

    def _ensure_loaded(self):
        if not self._loaded:
            logger.error("model.not_loaded", model_path=str(self.model_path))
            raise RuntimeError(
                f"{self.__class__.__name__} must be loaded before use. Call .load() first."
            )

    def preprocess(self, inputs) -> np.ndarray:
        if not isinstance(inputs, list):
            inputs = [inputs]
        tensors = [self._eval_transform(img) for img in inputs]
        return np.stack([t.numpy() for t in tensors]).astype(np.float32)

    def predict_top3(self, inputs, k: int = 3) -> list[list]:
        self._ensure_loaded()

        batch = self.preprocess(inputs)
        logits = self.session.run(None, {self._input_name: batch})[0]
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)

        top_idx = np.argsort(-probs, axis=1)[:, :k]
        results = []
        for i in range(probs.shape[0]):
            preds = [
                (self.class_names[idx], float(probs[i, idx])) for idx in top_idx[i]
            ]
            results.append(preds)

        logger.info("model.predict_top3.complete", num_images=probs.shape[0], k=k)
        return results
