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
    manifest: str = "data/manifest.json"

    # Features & Hyperparameters
    seed: int = 42
    batch_size = 32
    num_workers: int = min(4, os.cpu_count() or 0)


    # api_host: str
    # api_port: int




# the single truth imported by the rest of the app.
config = Settings()
