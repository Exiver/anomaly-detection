import io

import numpy as np
import pytest
from PIL import Image

from anomaly.service.inferance import InvalidImageError, decode_image


def _encode(fmt: str, size: int = 32) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.full((size, size, 3), 128, dtype=np.uint8)).save(buffer, format=fmt)
    return buffer.getvalue()


def test_accepts_png_and_jpeg():
    assert decode_image(_encode("PNG"), 1_000_000).mode == "RGB"
    assert decode_image(_encode("JPEG"), 1_000_000).mode == "RGB"


def test_rejects_garbage():
    with pytest.raises(InvalidImageError):
        decode_image(b"definitely not an image", 1_000_000)


def test_rejects_oversized_images():
    with pytest.raises(InvalidImageError):
        decode_image(_encode("PNG", size=64), max_pixels=100)


def test_rejects_other_formats():
    with pytest.raises(InvalidImageError):
        decode_image(_encode("BMP"), 1_000_000)