import torch
from PIL import Image

from anomaly.data.mvtec import MVTecDataset
from anomaly.data.transforms import build_image_transform


def test_dataset_and_serving_preprocessing_are_identical(fake_dataset_root):
    ds = MVTecDataset(fake_dataset_root, "bottle", "test", image_size=64, crop_size=64)
    expected = ds[0]["image"]
    image = Image.open(ds.samples[0].image_path).convert("RGB")
    assert torch.equal(build_image_transform(64, 64)(image), expected)