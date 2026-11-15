import numpy as np
import pytest

from anomaly.thresholds import (
    best_f1_threshold,
    binary_metrics,
    threshold_max,
    threshold_mean_k_std,
)


def test_binary_metrics_counts():
    labels = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.6, 0.7, 0.4])
    m = binary_metrics(labels, scores, threshold=0.5)  # flags indices 2 and 3
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)
    assert m["fpr"] == pytest.approx(1 / 3)


def test_threshold_rules():
    scores = np.array([1.0, 2.0, 3.0])
    assert threshold_max(scores) == 3.0
    assert threshold_mean_k_std(scores, k=3.0) == pytest.approx(5.0)


def test_best_f1_with_perfect_separation():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    _, f1 = best_f1_threshold(labels, scores)
    assert f1 == pytest.approx(1.0)