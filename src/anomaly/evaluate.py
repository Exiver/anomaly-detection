import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from anomaly.metrics import image_auroc, pixel_auroc

Scorer = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class EvalResult:
    labels: np.ndarray        # (N,)
    image_scores: np.ndarray  # (N,)
    masks: np.ndarray         # (N, H, W)
    anomaly_maps: np.ndarray  # (N, H, W)
    image_auroc: float
    pixel_auroc: float
    defect_types: np.ndarray  # (N,) strings, "good" for normal images
    paths: list[str]


@torch.no_grad()
def evaluate(
    scorer: Scorer,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 16,
) -> EvalResult:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    label_batches, mask_batches, map_batches = [], [], []
    defect_types: list[str] = []
    paths: list[str] = []
    for batch in loader:
        images = batch["image"].to(device)
        map_batches.append(scorer(images).detach().cpu().numpy())
        label_batches.append(batch["label"].numpy())
        mask_batches.append(batch["mask"].squeeze(1).numpy())
        defect_types.extend(batch["defect_type"])
        paths.extend(batch["path"])

    labels = np.concatenate(label_batches)
    masks = np.concatenate(mask_batches)
    anomaly_maps = np.concatenate(map_batches)
    image_scores = anomaly_maps.reshape(len(anomaly_maps), -1).max(axis=1)

    return EvalResult(
        labels=labels,
        image_scores=image_scores,
        masks=masks,
        anomaly_maps=anomaly_maps,
        image_auroc=image_auroc(labels, image_scores),
        pixel_auroc=pixel_auroc(masks, anomaly_maps),
        defect_types=np.array(defect_types),
        paths=paths,
    )


@torch.no_grad()
def image_scores(
    scorer: Scorer, dataset: Dataset, device: torch.device, batch_size: int = 16
) -> np.ndarray:
    """Image-level scores only (no labels needed). Used for threshold calibration."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    out = []
    for batch in loader:
        maps = scorer(batch["image"].to(device))
        out.append(maps.flatten(1).max(dim=1).values.cpu().numpy())
    return np.concatenate(out)


def per_defect_type_auroc(result: EvalResult) -> dict[str, float]:
    """Image AUROC of 'all normal images' vs 'images with this one defect type'."""
    normal = result.labels == 0
    out: dict[str, float] = {}
    for defect in sorted(set(result.defect_types[result.labels == 1].tolist())):
        selected = normal | (result.defect_types == defect)
        out[defect] = image_auroc(result.labels[selected], result.image_scores[selected])
    return out


def benchmark_latency(
    scorer: Scorer,
    images: torch.Tensor,
    device: torch.device,
    repeats: int = 5,
) -> float:
    """Milliseconds per image for one batch, after a warm-up call."""
    images = images.to(device)
    with torch.no_grad():
        scorer(images)  # warm-up
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            scorer(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed / (repeats * images.shape[0]) * 1000.0