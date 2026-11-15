from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str
    model_version: str
    categories: list[str]
    device: str


class ModelInfo(BaseModel):
    category: str
    threshold: float
    bank_size: int
    test_image_auroc: float
    test_pixel_auroc: float


class PredictionResponse(BaseModel):
    category: str
    anomaly_score: float
    threshold: float
    is_anomalous: bool
    latency_ms: float
    model_version: str
    heatmap_png_base64: str | None = None