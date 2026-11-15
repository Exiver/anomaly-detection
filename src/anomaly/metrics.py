import numpy as np
from scipy import ndimage
from sklearn.metrics import roc_auc_score


def image_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """labels: (N.) of 0/1. scores: (N.) higher = more anomalous."""
    return float(roc_auc_score(labels, scores))

def pixel_auroc(masks: np.ndarray, anomaly_maps:np.ndarray) ->float:
    """masks and anomaly_maps: (N, H, W). Flattened across all images."""
    return float(roc_auc_score(masks.astype(int).ravel(), anomaly_maps.ravel()))

def bootstrap_auroc_ci(
        labels: np.ndarray,
        scores: np.ndarray,
        n_boot: int = 1000,
        alpha: float = 0.005,
        seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for image-level AUROC."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(labels[idx])) < 2:
            continue
        values.append(roc_auc_score(labels[idx], scores[idx]))
    low, high = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)

def _trapezoid(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0))
def pro_score(
        masks: np.ndarray,
        anomaly_maps: np.ndarray,
        fpr_limit: float = 0.3,
        n_thresholds: int = 100,
) -> float:
    """Area under the per-region-overlap curve up to `fpr_limit`, normalized to [0, 1].

    masks, anomaly_maps: (N, H, W). For each threshold we compute the pixel
    false-positive rate over all normal pixels, and the mean over every
    connected defect region of the fraction of that region scored above the
    threshold.
    """
    masks  = masks > 0.5
    region_ids = np.zeros(masks.shape, dtype=np.int32)
    n_regions = 0
    for i in range(len(masks)):
        if masks[i].any():
            labeled, count = ndimage.label(masks[i])
            labeled[labeled > 0] += n_regions
            region_ids[i] = labeled
            n_regions += count
    if n_regions == 0:
        raise ValueError("No anomalies regions: pro is undefined")

    flat_ids = region_ids.ravel()
    flat_maps = anomaly_maps.ravel()
    flat_normal = flat_ids == 0
    region_sizes = np.bincount(flat_ids, minlength=n_regions + 1)[1:].astype(np.float64)
    n_normal = int(flat_normal.sum())


    normal_scores = flat_maps[flat_normal]
    targets = np.linspace(0.0, fpr_limit, n_thresholds)
    thresholds = np.quantile(normal_scores, 1.0 - targets)

    fprs, pros = [], []
    for t in thresholds:
        predicted = flat_maps > t
        false_positive = int((predicted & flat_normal).sum())
        hits = np.bincount(flat_ids[predicted & ~flat_normal], minlength=n_regions + 1)[1:]
        fprs.append(false_positive / n_normal)
        pros.append(float(np.mean(hits / region_sizes)))

    fprs_arr, pros_arr = np.array(fprs), np.array(pros)
    if fprs_arr[-1] < fpr_limit:
        fprs_arr = np.append(fprs_arr, fpr_limit)
        pros_arr = np.append(pros_arr, pros_arr[-1])
    return _trapezoid(fprs_arr, pros_arr) / fpr_limit
