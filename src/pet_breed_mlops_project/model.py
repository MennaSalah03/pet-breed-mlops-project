import structlog
import torch.nn as nn  # noqa: PLR0402
from torchvision import models

logger = structlog.getLogger()


def build_model(num_classes: int):
    """ResNet18 with frozen backbone and a fresh linear head for num_classes."""
    logger.info("model.build.start", architecture="resnet18", num_classes=num_classes)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    logger.info("model.build.complete", head_in_features=num_ftrs)
    return model
