import torch

from anomaly.data.mvtec import MVTecDataset
from anomaly.evaluate import evaluate, per_defect_type_auroc


def test_perfect_scorer_gets_perfect_auroc(fake_dataset_root):
    ds = MVTecDataset(fake_dataset_root, "bottle", "test", image_size=64, crop_size=64)

    def brightness_scorer(images: torch.Tensor) -> torch.Tensor:
        return images.mean(dim=1)

    result = evaluate(brightness_scorer, ds, torch.device("cpu"), batch_size=2)
    assert result.image_auroc == 1.0
    assert result.pixel_auroc == 1.0
    assert result.anomaly_maps.shape == (4, 64, 64)

def test_result_carries_defect_types_and_per_defect_auroc(fake_dataset_root):
    ds = MVTecDataset(fake_dataset_root, "bottle", "test", image_size=64, crop_size=64)
    result = evaluate(lambda images: images.mean(dim=1), ds, torch.device("cpu"), batch_size=2)

    assert sorted(set(result.defect_types.tolist())) == ["broken", "good"]
    assert len(result.paths) == 4
    assert per_defect_type_auroc(result) == {"broken": 1.0}