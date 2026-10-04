import structlog
import torch
from tqdm import tqdm

logger = structlog.getLogger()


def evaluate_model(model, dataloader, criterion, device):
    """Full evaluation pass over dataloader. Returns (loss, acc)."""
    logger.info("evaluation.start", num_batches=len(dataloader))

    model.eval()
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="test"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    dataset_size = len(dataloader.dataset)
    test_loss = running_loss / dataset_size
    test_acc = (running_corrects.double() / dataset_size).item()

    logger.info("evaluation.complete", loss=round(test_loss, 4), acc=round(test_acc, 4))
    return test_loss, test_acc
