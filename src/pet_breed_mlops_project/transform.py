import structlog
from torchvision import transforms

logger = structlog.getLogger()
# Standard ImageNet normalization values


def imagenet_norm_trans(mean: list[float], std: list[float]):
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),  # Simple data augmentation
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        ),
    }

    logger.info("Data normalization done")

    return data_transforms
