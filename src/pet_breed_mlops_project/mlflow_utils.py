from contextlib import contextmanager

import mlflow
import structlog

from pet_breed_mlops_project.config import config

logger = structlog.getLogger()


def init_mlflow():
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment_name)
    logger.info(
        "mlflow.init",
        tracking_uri=config.mlflow_tracking_uri,
        experiment_name=config.mlflow_experiment_name,
    )


@contextmanager
def start_run(run_name: str):
    with mlflow.start_run(run_name=run_name) as run:
        logger.info("mlflow.run.start", run_id=run.info.run_id, run_name=run_name)
        yield run
        logger.info("mlflow.run.end", run_id=run.info.run_id)
