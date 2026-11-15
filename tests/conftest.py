from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _save(path: Path, value: int, mode: str = "RGB") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    shape = (64, 64, 3) if mode == "RGB" else (64, 64)
    Image.fromarray(np.full(shape, value, dtype=np.uint8), mode=mode).save(path)


@pytest.fixture
def fake_dataset_root(tmp_path: Path) -> Path:
    cat = tmp_path / "bottle"
    for i in range(4):
        _save(cat / "train" / "good" / f"{i:03d}.png", 128)
    for i in range(2):
        _save(cat / "test" / "good" / f"{i:03d}.png", 128)
    for i in range(2):
        _save(cat / "test" / "broken" / f"{i:03d}.png", 255)
        _save(cat / "ground_truth" / "broken" / f"{i:03d}_mask.png", 255, mode="L")
    return tmp_path