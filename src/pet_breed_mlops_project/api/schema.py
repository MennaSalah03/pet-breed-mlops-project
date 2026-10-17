from pydantic import BaseModel, Field


class PredictionItem(BaseModel):
    class_name: str = Field(..., description="Predicted pet breed")
    probability: float = Field(..., ge=0.0, le=1.0, description="Confidence score")


class PredictResponse(BaseModel):
    filename: str
    predictions: list[PredictionItem]


class HealthResponse(BaseModel):
    status: str
    model_name: str
    model_stage: str
    model_loaded: bool


class ErrorResponse(BaseModel):
    detail: str
