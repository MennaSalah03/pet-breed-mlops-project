import structlog
import torch.nn as nn  # noqa: PLR0402
import torch.optim as optim  # noqa: PLR0402
from torch.optim import lr_scheduler

from pet_breed_mlops_project.config import config
from pet_breed_mlops_project.data import load_data
from pet_breed_mlops_project.evaluation import evaluate_model
from pet_breed_mlops_project.inference import predict_top3
from pet_breed_mlops_project.model import build_model
from pet_breed_mlops_project.train import get_device, train_model

logger = structlog.getLogger()


def main():
    device = get_device()
    dataloaders = load_data()
    class_names = dataloaders["train"].dataset.classes

    model = build_model(num_classes=len(class_names)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=config.lr)
    scheduler = lr_scheduler.StepLR(
        optimizer, step_size=config.step_size, gamma=config.gamma
    )

    model_ft = train_model(
        model,
        dataloaders,
        criterion,
        optimizer,
        scheduler,
        device,
        num_epochs=config.num_epochs,
    )

    test_loss, test_acc = evaluate_model(
        model_ft, dataloaders["val"], criterion, device
    )
    logger.info("script.test_results", loss=round(test_loss, 4), acc=round(test_acc, 4))

    inputs, labels = next(iter(dataloaders["val"]))
    results = predict_top3(model_ft, inputs, class_names, device)

    for i, preds in enumerate(results[:10]):
        logger.info(
            "script.sample_prediction",
            image_index=i,
            top3=[{"class": name, "prob": round(prob, 4)} for name, prob in preds],
            true_label=class_names[labels[i]],
        )


if __name__ == "__main__":
    main()
