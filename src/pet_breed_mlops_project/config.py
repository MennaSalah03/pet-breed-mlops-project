"""Centralized configuration — paths, hyperparameters, service settings"""

# from pydantic import Field
import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """pydantic-settings configs"""

    # read from a .env file if it exists
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # File paths with sane defaults
    pickle_model_path: str = "models/baseline.pkl"
    onnx_model_path: str = "models/baseline.onnx"
    manifest_path: str = "data/oxford-iiit-pet/manifest.json"

    # Data normalization mean and std dev.
    mean: list[float] = [0.485, 0.456, 0.406]
    std: list[float] = [0.229, 0.224, 0.225]

    # Features & Hyperparameters
    seed: int = 42
    batch_size: int = 32
    num_workers: int = min(4, os.cpu_count() or 0)
    available_backbones: list[str] = ["resnet18", "resnet50", "mobilenet_v3_small"]

    backbone: str = "resnet18"  # baseline
    lr: float = 0.01
    step_size: int = 2
    gamma: float = 0.1
    num_epochs: int = 5

    # api_host: str
    # api_port: int

    mlflow_experiment_name: str = "pet-breed-classification"
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_model_uri: str

    minio_root_user: str
    minio_root_password: str

    postgres_user: str
    postgres_password: str
    postgres_db: str


# the single truth imported by the rest of the app.
config = Settings()
