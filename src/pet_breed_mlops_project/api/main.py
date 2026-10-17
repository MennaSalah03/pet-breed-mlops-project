import io
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
import structlog
from fastapi import FastAPI, UploadFile, status
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from pet_breed_mlops_project.api.schema import (
    ErrorResponse,
    HealthResponse,
    PredictionItem,
    PredictResponse,
)
from pet_breed_mlops_project.config import config
from pet_breed_mlops_project.inference import PetBreedModel

logger = structlog.getLogger()


model_state = {"model": None, "loaded": False}


def _load_production_model() -> PetBreedModel:
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()

    logger.info(
        "api.model_load.start", model_name=config.model_name, stage=config.model_alias
    )

    try:
        version = client.get_model_version_by_alias(
            config.model_name, config.model_alias
        )
    except Exception as e:
        logger.error(
            "api.model_load.alias_not_found",
            model_name=config.model_name,
            alias=config.model_alias,
        )
        raise RuntimeError(
            f"No version of '{config.model_name}' has alias '{config.model_alias}'"
        ) from e

    # 1. Download the entire model package in a single, cached operation
    model_uri = f"models:/{config.model_name}/{version.version}"
    downloaded_path = Path(mlflow.artifacts.download_artifacts(artifact_uri=model_uri))

    # 2. Locate the specific subfolder containing the model.
    # Because you logged the whole directory, the .onnx, .data, and JSONs are sitting side-by-side here.
    onnx_files = list(downloaded_path.rglob(config.onnx_filename))
    if not onnx_files:
        raise RuntimeError(
            f"{config.onnx_filename} not found in registered version {version.version}"
        )

    # Set the bundle directory to the folder natively containing the ONNX file
    bundle_dir = onnx_files[0].parent

    logger.info(
        "api.model_load.bundle_ready",
        version=version.version,
        run_id=version.run_id,
        files=sorted(p.name for p in bundle_dir.iterdir() if p.is_file()),
    )

    # 3. Pass the intact directory directly to your model inference class
    model = PetBreedModel(model_path=bundle_dir)
    model.load()

    logger.info(
        "api.model_load.complete", model_name=config.model_name, version=version.version
    )
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model_state["model"] = _load_production_model()
        model_state["loaded"] = True
    except Exception:
        logger.exception("api.startup.model_load_failed")
        model_state["model"] = None
        model_state["loaded"] = False
    yield
    logger.info("api.shutdown")


app = FastAPI(title="Pet Breed Classifier API", lifespan=lifespan)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    """Catch-all so nothing surfaces as a 500 to the client."""
    logger.exception("api.unhandled_exception", path=str(request.url))
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(detail="Request could not be processed.").model_dump(),
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    status_str = "ok" if model_state["loaded"] else "model_unavailable"
    logger.info("api.health_check", status=status_str)
    return HealthResponse(
        status=status_str,
        model_name=config.model_name,
        model_stage=config.model_alias,
        model_loaded=model_state["loaded"],
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={422: {"model": ErrorResponse}},
)
async def predict(file: UploadFile):
    if not model_state["loaded"] or model_state["model"] is None:
        logger.warning("api.predict.model_unavailable")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(detail="Model is not currently loaded.").model_dump(),
        )

    if file.content_type not in config.allowed_content_types:
        logger.warning(
            "api.predict.invalid_content_type",
            content_type=file.content_type,
            filename=file.filename,
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                detail=f"Unsupported content type '{file.content_type}'. Allowed: {sorted(config.allowed_content_types)}"
            ).model_dump(),
        )

    raw_bytes = await file.read()
    if not raw_bytes:
        logger.warning("api.predict.empty_file", filename=file.filename)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(detail="Uploaded file is empty.").model_dump(),
        )

    try:
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except UnidentifiedImageError:
        logger.warning("api.predict.unreadable_image", filename=file.filename)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                detail="File could not be read as an image."
            ).model_dump(),
        )

    try:
        results = model_state["model"].predict_top3(image)
    except Exception:
        logger.exception("api.predict.inference_failed", filename=file.filename)
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                detail="Inference failed for the given image."
            ).model_dump(),
        )

    predictions = [
        PredictionItem(class_name=name, probability=prob) for name, prob in results[0]
    ]
    logger.info(
        "api.predict.success", filename=file.filename, top1=predictions[0].class_name
    )

    return PredictResponse(filename=file.filename, predictions=predictions)
