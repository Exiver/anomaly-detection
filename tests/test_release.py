import numpy as np
import pytest

from anomaly.manifest import CategoryEntry, ReleaseManifest
from anomaly.release import choose_threshold, split_indices


def test_split_is_disjoint_complete_and_deterministic():
    fit, hold = split_indices(100, 0.1, seed=1)
    assert len(hold) == 10 and len(fit) == 90
    assert set(fit).isdisjoint(hold)
    assert sorted(fit + hold) == list(range(100))
    assert split_indices(100, 0.1, seed=1) == (fit, hold)


def test_split_enforces_minimum_holdout():
    _, hold = split_indices(30, 0.01, seed=0)
    assert len(hold) == 5


def test_split_rejects_tiny_datasets():
    with pytest.raises(ValueError):
        split_indices(5, 0.1, seed=0)


def test_threshold_rules():
    scores = np.array([1.0, 2.0, 3.0])
    assert choose_threshold(scores, "max", None) == 3.0
    assert choose_threshold(scores, "mean_k_std", 3.0) == pytest.approx(5.0)
    with pytest.raises(ValueError):
        choose_threshold(scores, "banana", None)


def test_manifest_roundtrip(tmp_path):
    manifest = ReleaseManifest(
        version="v1", created_at="2026-01-01T00:00:00+00:00", image_size=256, crop_size=224,
        embed_dim=1024, coreset_ratio=0.1, blur_sigma=4.0, seed=42, threshold_rule="max",
        threshold_k=None, holdout_fraction=0.1,
        categories={
            "bottle": CategoryEntry(
                threshold=1.5, holdout_images=20, bank_size=1000, test_image_auroc=0.99,
                test_pixel_auroc=0.98, test_recall_at_threshold=0.9, test_fpr_at_threshold=0.05,
            )
        },
        reference_score_ratios=[0.1, 0.2, 0.3],
        reference_input_mean=0.05,
        reference_input_mean_std=0.2,
    )
    path = tmp_path / "manifest.json"
    manifest.write(path)
    assert ReleaseManifest.read(path) == manifest