from pathlib import Path

import numpy as np
import structlog
from torch import device as TorchDevice
from torch.utils.data import DataLoader, Subset

from anomaly.config import Settings
from anomaly.data.mvtec import MVTecDataset
from anomaly.evaluate import evaluate, image_scores
from anomaly.manifest import CategoryEntry
from anomaly.models.patchcore import PatchCore
from anomaly.models.patchcore_features import FeatureExtractor
from anomaly.thresholds import binary_metrics, threshold_max, threshold_mean_k_std

log = structlog.get_logger()


def split_indices(
    n: int, holdout_fraction: float, seed: int, min_holdout: int = 5
) -> tuple[list[int], list[int]]:
    """Deterministic (fit_indices, holdout_indices) split of n items."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n).tolist()
    n_hold = max(min_holdout, int(holdout_fraction * n))
    if n_hold >= n:
        raise ValueError("The holdout would consume the whole training set.")
    return order[n_hold:], order[:n_hold]


def choose_threshold(normal_scores: np.ndarray, rule: str, k: float | None) -> float:
    if rule == "max":
        return threshold_max(normal_scores)
    if rule == "mean_k_std":
        return threshold_mean_k_std(normal_scores, 3.0 if k is None else k)
    raise ValueError(f"Unknown threshold rule: {rule}")


def build_category(
    category: str,
    settings: Settings,
    device: TorchDevice,
    extractor: FeatureExtractor,
    release_dir: Path,
    rule: str,
    k: float | None,
    holdout_fraction: float,
) -> CategoryEntry:
    train_set = MVTecDataset(
        settings.data_root, category, "train", settings.image_size, settings.crop_size
    )
    fit_idx, hold_idx = split_indices(len(train_set), holdout_fraction, settings.seed)

    model = PatchCore(
        extractor,
        device,
        embed_dim=settings.embed_dim,
        coreset_ratio=settings.coreset_ratio,
        blur_sigma=settings.blur_sigma,
        nn_chunk=settings.nn_chunk,
        seed=settings.seed,
    )
    model.fit(
        DataLoader(
            Subset(train_set, fit_idx),
            batch_size=settings.batch_size,
            shuffle=False,
            num_workers=settings.num_workers,
        )
    )

    holdout_scores = image_scores(
        model.anomaly_map, Subset(train_set, hold_idx), device, settings.batch_size
    )
    threshold = choose_threshold(holdout_scores, rule, k)
    input_mean, input_std = input_mean_stats(train_set, fit_idx, settings.batch_size)
    reference_ratios = (holdout_scores / max(threshold, 1e-8)).tolist()
    test_set = MVTecDataset(
        settings.data_root, category, "test", settings.image_size, settings.crop_size
    )
    result = evaluate(model.anomaly_map, test_set, device, settings.batch_size)
    at_threshold = binary_metrics(result.labels, result.image_scores, threshold)

    model.save(release_dir / f"{category}.pt")
    log.info(
        "category_built",
        category=category,
        threshold=round(threshold, 4),
        holdout=len(hold_idx),
        image_auroc=round(result.image_auroc, 4),
    )
    return CategoryEntry(
        threshold=threshold,
        holdout_images=len(hold_idx),
        bank_size=int(model.bank.shape[0]),
        test_image_auroc=result.image_auroc,
        test_pixel_auroc=result.pixel_auroc,
        test_recall_at_threshold=at_threshold["recall"],
        test_fpr_at_threshold=at_threshold["fpr"],
        reference_score_ratios=reference_ratios,
        reference_input_mean=input_mean,
        reference_input_mean_std=input_std,
    )

def input_mean_stats(dataset, indices: list[int], batch_size: int) -> tuple[float, float]:
    """Mean and std (across images) of each preprocessed image's overall mean value."""
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False)
    means: list[float] = []
    for batch in loader:
        means.extend(batch["image"].flatten(1).mean(dim=1).tolist())
    values = np.array(means)
    return float(values.mean()), float(values.std(ddof=1))