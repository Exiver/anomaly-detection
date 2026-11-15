import numpy as np
import pytest

from anomaly.metrics import bootstrap_auroc_ci, image_auroc, pixel_auroc, pro_score


def test_image_auroc_perfect():
    labels = np.array([0,0,1,1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert image_auroc(labels=labels, scores=scores)

def test_image_auroc_inverted():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    assert image_auroc(labels, scores) == 0.0

def test_pixel_auroc_perfect():
    masks = np.zeros((2,4,4))
    masks[:, :2, :2] = 1
    maps = masks * 0.9 + 0.05
    assert pixel_auroc(masks, maps) == 1.0
    
def test_pro_perfect_prediction():
    masks = np.zeros((3, 16, 16))
    masks[0, 2:6, 2:6] = 1
    masks[1, 8:12, 8:12] = 1
    assert pro_score(masks, masks.astype(float)) == pytest.approx(1.0)


def test_pro_inverted_prediction_is_near_zero():
    masks = np.zeros((3, 16, 16))
    masks[0, 2:6, 2:6] = 1
    assert pro_score(masks, 1.0 - masks) < 0.05


def test_pro_weights_regions_equally():
    # A large region found completely and a tiny region missed completely:
    # pixel recall is high, but PRO is 0.5.
    masks = np.zeros((2, 16, 16))
    masks[0, 0:8, 0:8] = 1   # 64 pixels
    masks[1, 0:2, 0:2] = 1   # 4 pixels
    maps = np.zeros_like(masks)
    maps[0, 0:8, 0:8] = 1.0
    assert pro_score(masks, maps) == pytest.approx(0.5)


def test_pro_requires_defects():
    with pytest.raises(ValueError):
        pro_score(np.zeros((2, 8, 8)), np.zeros((2, 8, 8)))


def test_bootstrap_ci_of_perfect_scores():
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    low, high = bootstrap_auroc_ci(labels, scores, n_boot=200, seed=0)
    assert low == 1.0 and high == 1.0