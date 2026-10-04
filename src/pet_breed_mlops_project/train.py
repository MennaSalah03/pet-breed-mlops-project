import copy
import time

import structlog
import torch
from tqdm import tqdm

logger = structlog.getLogger()


def get_device() -> torch.device:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("device.selected", device=str(device))
    return device


def _run_phase(model, dataloader, criterion, optimizer, device, phase: str):
    """One epoch over one phase's dataloader. Returns (loss, acc)."""
    is_train = phase == "train"
    model.train() if is_train else model.eval()

    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in tqdm(dataloader, desc=phase):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    dataset_size = len(dataloader.dataset)
    epoch_loss = running_loss / dataset_size
    epoch_acc = (running_corrects.double() / dataset_size).item()
    return epoch_loss, epoch_acc


def train_model(
    model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=5
):
    """Train across train/val phases, keeping the weights with best val accuracy."""
    logger.info("training.start", num_epochs=num_epochs, device=str(device))
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        logger.info("training.epoch.start", epoch=epoch + 1, num_epochs=num_epochs)

        for phase in ["train", "val"]:
            epoch_loss, epoch_acc = _run_phase(
                model, dataloaders[phase], criterion, optimizer, device, phase
            )

            if phase == "train":
                scheduler.step()

            logger.info(
                "training.epoch.phase_complete",
                epoch=epoch + 1,
                phase=phase,
                loss=round(epoch_loss, 4),
                acc=round(epoch_acc, 4),
            )

            if phase == "val" and epoch_acc > best_acc:
                logger.info(
                    "training.epoch.new_best",
                    epoch=epoch + 1,
                    prev_best_acc=round(best_acc, 4),
                    new_best_acc=round(epoch_acc, 4),
                )
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    logger.info(
        "training.complete",
        time_elapsed_sec=round(time_elapsed, 1),
        best_val_acc=round(best_acc, 4),
    )

    model.load_state_dict(best_model_wts)
    return model
