import itertools

import structlog
import torch
import torch.nn as nn  # noqa: PLR0402

logger = structlog.getLogger()


def compute_ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """Expected Calibration Error."""
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(labels)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = torch.zeros(1)

    for lower, upper in itertools.pairwise(bin_boundaries):
        in_bin = (confidences > lower) & (confidences <= upper)
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            acc_in_bin = accuracies[in_bin].float().mean()
            conf_in_bin = confidences[in_bin].mean()
            ece += torch.abs(conf_in_bin - acc_in_bin) * prop_in_bin

    return ece.item()


class TemperatureScaler(nn.Module):
    """Learns a single scalar temperature to calibrate logits."""

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> float:
        logger.info("calibration.temperature_fit.start", max_iter=max_iter)

        nll_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        final_temp = self.temperature.item()
        logger.info(
            "calibration.temperature_fit.complete", temperature=round(final_temp, 4)
        )
        return final_temp
