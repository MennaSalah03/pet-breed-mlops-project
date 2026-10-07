import random

import numpy as np
import structlog
import torch
from torch.utils.data import DataLoader

from pet_breed_mlops_project.config import config
from pet_breed_mlops_project.pet_manifest_dataset import PetManifestDataset
from pet_breed_mlops_project.transform import imagenet_norm_trans

logger = structlog.getLogger()


def _get_data():
    logger.info("data.load.start", manifest_path=config.manifest_path)

    data_transforms = imagenet_norm_trans(config.mean, config.std)

    train_dataset = PetManifestDataset(
        manifest_path=config.manifest_path,
        split="train",
        corruption=None,  # clean images only. allow it after corruption suite is fully operational
        transform=data_transforms["train"],
    )

    test_dataset = PetManifestDataset(
        manifest_path=config.manifest_path,
        split="test",
        corruption=None,
        transform=data_transforms["test"],
    )

    logger.info(
        "data.load.complete",
        train_size=len(train_dataset),
        test_size=len(test_dataset),
        num_classes=len(test_dataset.classes),
    )

    return train_dataset, test_dataset


def seed_worker(worker_id):
    """Seed numpy/random per-worker so augmentation is reproducible across DataLoader workers."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    logger.debug(
        "dataloader.worker_seeded", worker_id=worker_id, worker_seed=worker_seed
    )


def load_data(batch_size: int | None = None):
    batch_size = batch_size or config.batch_size
    train_dataset, test_dataset = _get_data()

    generator = torch.Generator()
    generator.manual_seed(config.seed)

    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            worker_init_fn=seed_worker,
            generator=generator,
        ),
        "val": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        ),
    }
    logger.info(
        "dataloader.ready",
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        train_batches=len(dataloaders["train"]),
        val_batches=len(dataloaders["val"]),
    )

    return dataloaders


from pet_breed_mlops_project.train import get_device

device = get_device()
dataloaders = load_data()
class_names = dataloaders["train"].dataset.classes
print(class_names)
