from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from anomaly.data.mvtec import MVTecDataset


def _save(path: Path, mode: str = "RGB", value: int = 128) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    channels = 3 if mode == "RGB" else 1
    shape = (64, 64, 3) if channels == 3 else (64, 64)
    Image.fromarray(np.full(shape, value, dtype=np.uint8), mode=mode).save(path)

@pytest.fixture
def fake_root(tmp_path: Path) ->Path:
    cat = tmp_path / "bottle"
    for i in range(3):
        _save(cat / "train" / "good" / f"{i:03d}.png" )
    _save(cat / "test" / "good" / "000.png")

    for i in range(2):
        _save(cat / "test" / "broken" / f"{i:03d}.png")
        _save(cat / "ground_truth" / "broken" / f"{i:03d}_mask.png", mode="L", value=255)
    return tmp_path

def test_train_only_normal(fake_root):
    ds = MVTecDataset(fake_root, "bottle", "test", image_size=32, crop_size=28)
    assert len(ds) == 3

def test_test_split_has_anomalies_with_mask(fake_root):
    ds = MVTecDataset(fake_root, "bottle", "test", image_size=32, crop_size=28)
    assert len(ds) == 3
    labels = [s.label for s in ds.samples]
    assert labels.count(1) == 2
    item = next(ds[i] for i in range(len(ds)) if ds.samples[i].label == 1)
    assert item["mask"].sum() > 0


def test_output_shapes(fake_root):
    ds = MVTecDataset(fake_root, "bottle", "test", image_size=32, crop_size=28)
    item = ds[0]
    assert item["image"].shape == (3, 28, 28)
    assert item["mask"].shape == (1, 28, 28)

def test_grayscale_converted_to_rgb(tmp_path: Path):
    cat = tmp_path / "grid"
    _save(cat / "train" / "good" / "000.png", mode="L")
    ds = MVTecDataset(tmp_path, "grid", "train", image_size=32, crop_size=28)
    assert ds[0]["image"].shape[0] == 3


def test_missing_mask_raises(tmp_path: Path):
    cat = tmp_path / "bottle"
    _save(cat / "text" / "broken" / "000.png")
    with pytest.raises(FileNotFoundError):
        MVTecDataset(tmp_path, "bottle", "test")

def test_unknown_category_raises(tmp_path: Path):
    with pytest.raises(ValueError):
        MVTecDataset(tmp_path, "banana", "train")
