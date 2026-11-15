import io
import time
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from anomaly.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from anomaly.models.patchcore import PatchCore

ALLOWED_FORMATS = {"PNG", "JPEG"}


class InvalidImageError(ValueError):
    """The uploaded bytes are not an acceptable image."""


@dataclass
class LoadedModel:
    category: str
    model: PatchCore
    threshold: float


@dataclass
class Prediction:
    score: float
    latency_ms: float
    overlay_png: bytes | None
    input_mean: float


def decode_image(data: bytes, max_pixels: int) -> Image.Image:
    """Validate and decode an upload. Checks size and format BEFORE loading pixels."""
    try:
        image = Image.open(io.BytesIO(data))
        width, height = image.size
        if width * height > max_pixels:
            raise InvalidImageError(f"Image too large: {width}x{height} pixels.")
        if image.format not in ALLOWED_FORMATS:
            raise InvalidImageError(f"Unsupported image format: {image.format}.")
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise InvalidImageError("Could not decode the uploaded file as an image.") from exc


def render_overlay(tensor: torch.Tensor, anomaly_map: np.ndarray, threshold: float) -> bytes:
    """Red overlay on the preprocessed image; strength grows with the anomaly score."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    base = (tensor.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

    strength = np.clip(anomaly_map / max(2.0 * threshold, 1e-8), 0.0, 1.0)[..., None]
    red = np.array([1.0, 0.0, 0.0])
    blended = base * (1.0 - 0.6 * strength) + red * 0.6 * strength

    buffer = io.BytesIO()
    Image.fromarray((blended * 255).astype(np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def predict(
    loaded: LoadedModel,
    image: Image.Image,
    transform,
    device: torch.device,
    include_heatmap: bool,
) -> Prediction:
    start = time.perf_counter()
    tensor = transform(image).unsqueeze(0).to(device)
    input_mean = float(tensor.mean())
    anomaly_map = loaded.model.anomaly_map(tensor)[0].cpu().numpy()
    score = float(anomaly_map.max())
    overlay = (
        render_overlay(tensor[0], anomaly_map, loaded.threshold) if include_heatmap else None
    )
    latency_ms = (time.perf_counter() - start) * 1000.0
    return Prediction(score=score, latency_ms=latency_ms, overlay_png=overlay, input_mean=input_mean)