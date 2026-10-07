import argparse

import mlflow
import structlog
import torch.nn as nn  # noqa: PLR0402
from torch import optim
from torch.optim import lr_scheduler

from pet_breed_mlops_project.artifacts import save_run_artifacts
from pet_breed_mlops_project.config import config
from pet_breed_mlops_project.data import load_data
from pet_breed_mlops_project.evaluation import evaluate_with_calibration
from pet_breed_mlops_project.export import export_to_onnx
from pet_breed_mlops_project.mlflow_utils import init_mlflow, start_run
from pet_breed_mlops_project.model import build_model
from pet_breed_mlops_project.train import get_device, train_model

logger = structlog.getLogger()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backbone", default=config.backbone, choices=config.available_backbones
    )
    parser.add_argument("--lr", type=float, default=config.lr)
    parser.add_argument("--batch-size", type=int, default=config.batch_size)
    parser.add_argument("--num-epochs", type=int, default=config.num_epochs)
    return parser.parse_args()


def main():
    args = parse_args()
    init_mlflow()

    device = get_device()
    dataloaders = load_data(batch_size=args.batch_size)
    class_names = dataloaders["train"].dataset.classes

    model, trainable_params = build_model(
        backbone=args.backbone, num_classes=len(class_names)
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(trainable_params, lr=args.lr)
    scheduler = lr_scheduler.StepLR(
        optimizer, step_size=config.step_size, gamma=config.gamma
    )

    run_name = f"{args.backbone}_lr{args.lr}_bs{args.batch_size}"

    with start_run(run_name=run_name):
        mlflow.log_params(
            {
                "backbone": args.backbone,
                "lr": args.lr,
                "batch_size": args.batch_size,
            }
        )

        model_ft = train_model(
            model,
            dataloaders,
            criterion,
            optimizer,
            scheduler,
            device,
            num_epochs=args.num_epochs,
        )

        metrics = evaluate_with_calibration(model_ft, dataloaders["val"], device)
        mlflow.log_metrics(
            {
                "top1": metrics["top1"],
                "f1_macro": metrics["f1_macro"],
                "ece": metrics["ece"],
                "temperature": metrics["temperature"],
            }
        )

        output_dir = f"artifacts/{run_name}"
        onnx_path = export_to_onnx(model_ft, f"{output_dir}/model.onnx")
        save_run_artifacts(output_dir, class_names, config.mean, config.std)

        mlflow.log_artifact(str(onnx_path))
        mlflow.log_artifact(f"{output_dir}/class_names.json")
        mlflow.log_artifact(f"{output_dir}/transform_config.json")

        logger.info(
            "script.run_complete",
            run_name=run_name,
            backbone=args.backbone,
            metrics=metrics,
        )


if __name__ == "__main__":
    main()
