import numpy as np
from sklearn.metrics import precision_recall_curve


def threshold_max(normal_scores: np.ndarray) -> float:
    """Flag anything scoring at or above the worst normal image seen"""
    return float(np.max(normal_scores))


def threshold_mean_k_std(normal_scores: np.ndarray, k: float = 3.0) -> float:
    return float(np.mean(normal_scores) + k * np.std(normal_scores, ddof=1))


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Image is flagged anomalous whan score >= threshold."""

    predicted = scores >= threshold
    tp = int(((predicted == 1) & (labels == 1)).sum())
    fp = int(((predicted == 1) & (labels == 0)).sum())
    fn = int(((predicted == 0) & (labels == 1)).sum())
    tn = int(((predicted == 0) & (labels == 0)).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "fpr": fpr}


def best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """ORACLE: the F1-maximizing threshold, chosen with test labels. Upper bound only."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best = int(np.argmax(f1))
    return float(thresholds[best]), float(f1[best])
