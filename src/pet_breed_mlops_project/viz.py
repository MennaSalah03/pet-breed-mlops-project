"""for visualizing various things related to the data and model(s)"""

import numpy as np
import structlog
import torchvision
from matplotlib import pyplot as plt

from pet_breed_mlops_project.config import config

logger = structlog.getLogger()


def imshow(inp, title: str | None = None):
    """Display a single image tensor (already unnormalized to numpy HWC) via matplotlib."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array(config.mean)
    std = np.array(config.std)
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)

    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

    logger.debug("viz.imshow.rendered", title=title)


def show_batch(dataloaders: dict, class_names: list, split: str = "train", n: int = 5):
    """Pull one batch from dataloaders[split], grid the first n images, and display with labels."""
    logger.info("viz.show_batch.start", split=split, n=n)

    inputs, classes = next(iter(dataloaders[split]))

    if n > inputs.size(0):
        logger.warning(
            "viz.show_batch.n_exceeds_batch",
            requested_n=n,
            batch_size=inputs.size(0),
        )
        n = inputs.size(0)

    grid = torchvision.utils.make_grid(inputs[:n])
    titles = [class_names[x] for x in classes[:n]]
    imshow(grid, title=titles)

    logger.info("viz.show_batch.complete", split=split, n=n, titles=titles)
