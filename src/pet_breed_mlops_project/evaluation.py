import structlog
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm

from pet_breed_mlops_project.calibration import TemperatureScaler, compute_ece

logger = structlog.getLogger()


def collect_logits(model, dataloader, device):
    """Run model over dataloader, return (all_logits, all_labels) on CPU."""
    model.eval()
    all_logits, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="collect_logits"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_logits.append(outputs.cpu())
            all_labels.append(labels)

    return torch.cat(all_logits), torch.cat(all_labels)


def compute_metrics(
    logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0
) -> dict:
    """top1 accuracy, macro F1, ECE — optionally with temperature-scaled probabilities."""
    probs = torch.softmax(logits / temperature, dim=1)
    preds = torch.argmax(probs, dim=1)

    metrics = {
        "top1": (preds == labels).float().mean().item(),
        "f1_macro": f1_score(labels.numpy(), preds.numpy(), average="macro"),
        "ece": compute_ece(probs, labels),
    }
    logger.info(
        "evaluation.metrics",
        **{k: round(v, 4) for k, v in metrics.items()},
        temperature=temperature,
    )
    return metrics


def evaluate_with_calibration(model, dataloader, device) -> dict:
    """Collect logits, fit temperature on the same split, return calibrated metrics."""
    logger.info("evaluation.calibrated.start")

    logits, labels = collect_logits(model, dataloader, device)

    scaler = TemperatureScaler()
    temperature = scaler.fit(logits, labels)

    metrics = compute_metrics(logits, labels, temperature=temperature)
    metrics["temperature"] = temperature

    logger.info(
        "evaluation.calibrated.complete", **{k: round(v, 4) for k, v in metrics.items()}
    )
    return metrics
