import pytest
import torch
from torch.utils.data import DataLoader

from anomaly.data.mvtec import MVTecDataset
from anomaly.evaluate import evaluate
from anomaly.models.patchcore import PatchCore
from anomaly.models.patchcore_features import FeatureExtractor

CPU = torch.device("cpu")


def _fitted_model(root) -> PatchCore:
    torch.manual_seed(0)
    extractor = FeatureExtractor(pretrained=False)
    model = PatchCore(extractor, CPU, embed_dim=64, coreset_ratio=0.5, blur_sigma=1.0, seed=0)
    train = MVTecDataset(root, "bottle", "train", image_size=64, crop_size=64)
    model.fit(DataLoader(train, batch_size=2))
    return model


def test_separates_normal_from_defective(fake_dataset_root):
    model = _fitted_model(fake_dataset_root)
    test = MVTecDataset(fake_dataset_root, "bottle", "test", image_size=64, crop_size=64)
    result = evaluate(model.anomaly_map, test, CPU, batch_size=2)
    assert result.anomaly_maps.shape == (4, 64, 64)
    assert result.image_auroc == 1.0
    assert result.pixel_auroc > 0.99


def test_scoring_before_fit_raises():
    model = PatchCore(FeatureExtractor(pretrained=False), CPU, embed_dim=64)
    with pytest.raises(RuntimeError):
        model.anomaly_map(torch.randn(1, 3, 64, 64))


def test_save_load_roundtrip(fake_dataset_root, tmp_path):
    model = _fitted_model(fake_dataset_root)
    path = tmp_path / "bank.pt"
    model.save(path)
    loaded = PatchCore.load(path, model.extractor, CPU)
    x = torch.randn(1, 3, 64, 64)
    assert torch.allclose(model.anomaly_map(x), loaded.anomaly_map(x), atol=1e-5)

def test_half_precision_bank_still_separates(fake_dataset_root):
    model = _fitted_model(fake_dataset_root)
    model.bank = model.bank.half()
    test = MVTecDataset(fake_dataset_root, "bottle", "test", image_size=64, crop_size=64)
    result = evaluate(model.anomaly_map, test, CPU, batch_size=2)
    assert result.image_auroc == 1.0