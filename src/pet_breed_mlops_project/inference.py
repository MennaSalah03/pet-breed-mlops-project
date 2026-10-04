from abc import ABC, abstractmethod
from pathlib import Path

import structlog
import torch
import torch.nn.functional as F

logger = structlog.getLogger()


class BaseModel(ABC):
    """Base class for loadable, servable models. Subclasses define architecture-specific
    loading and preprocessing; prediction logic is shared here."""

    def __init__(self, model_path: str | Path, device: torch.device | None = None):
        self.model_path = Path(model_path)
        self.device = device or torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        self.model: torch.nn.Module | None = None
        self.class_names: list[str] | None = None
        self._loaded = False

        logger.info(
            "model.init", model_path=str(self.model_path), device=str(self.device)
        )

    @abstractmethod
    def load(self) -> None:
        """Load architecture, weights, class names, and preprocessing config.
        Must set self.model, self.class_names, and self._loaded = True."""
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, inputs):
        """Turn raw input (e.g. a PIL.Image or list of them) into a batched tensor
        ready to feed self.model."""
        raise NotImplementedError

    def _ensure_loaded(self):
        if not self._loaded:
            logger.error("model.not_loaded", model_path=str(self.model_path))
            raise RuntimeError(
                f"{self.__class__.__name__} must be loaded before use. Call .load() first."
            )

    def predict_top3(self, inputs, k: int = 3):
        self._ensure_loaded()
        self.model.eval()

        with torch.no_grad():
            batch = self.preprocess(inputs).to(self.device)
            outputs = self.model(batch)
            probs = F.softmax(outputs, dim=1)
            top_probs, top_idx = torch.topk(probs, k, dim=1)

        results = []
        for i in range(batch.size(0)):
            preds = [
                (self.class_names[top_idx[i][j].item()], top_probs[i][j].item())
                for j in range(k)
            ]
            results.append(preds)

        logger.info("model.predict_top3.complete", num_images=batch.size(0), k=k)
        return results
