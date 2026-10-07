import structlog
import torch.nn as nn  # noqa: PLR0402
from torchvision import models

logger = structlog.getLogger()

_BACKBONE_BUILDERS = {}


def register_backbone(name):
    """train model according to "name" backbone"""

    def decorator(fn):
        _BACKBONE_BUILDERS[name] = fn
        return fn

    return decorator


@register_backbone("resnet18")
def _build_resnet18(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model, model.fc.parameters()


@register_backbone("resnet50")
def _build_resnet50(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model, model.fc.parameters()


@register_backbone("mobilenet_v3_small")
def _build_mobilenet_v3_small(num_classes):
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    for p in model.parameters():
        p.requires_grad = False
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model, model.classifier[-1].parameters()


def build_model(backbone: str, num_classes: int):
    if backbone not in _BACKBONE_BUILDERS:
        logger.error(
            "model.build.unknown_backbone",
            backbone=backbone,
            available=list(_BACKBONE_BUILDERS),
        )
        raise ValueError(
            f"Unknown backbone '{backbone}'. Available: {list(_BACKBONE_BUILDERS)}"
        )

    logger.info("model.build.start", backbone=backbone, num_classes=num_classes)
    model, trainable_params = _BACKBONE_BUILDERS[backbone](num_classes)
    logger.info("model.build.complete", backbone=backbone)
    return model, trainable_params
