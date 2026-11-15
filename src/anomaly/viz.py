import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from anomaly.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from anomaly.evaluate import EvalResult


def denormalize(image: torch.Tensor) -> np.ndarray:
    """(3, H, W) normalized tensor -> (H, W, 3) array in [0, 1]."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (image * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

def save_heatmaps(dataset, result: EvalResult, out_path: Path, title: str,  n: int = 6) -> None:
    anomalous = np.flatnonzero(result.labels == 1)
    step = max(1, len(anomalous) // n)
    chosen = anomalous[::step][:n]
    vmax = float(result.anomaly_maps.max())

    fig, axes = plt.subplots(3, len(chosen), figsize=(3 * len(chosen), 9), squeeze=False)
    for col, idx in enumerate(chosen):
        item = dataset[int(idx)]
        image = denormalize(item["image"])

        axes[0, col].imshow(image)
        axes[0, col].set_title(item["defect_type"])
        axes[1, col].imshow(item["mask"].squeeze(0).numpy(), cmap="gray")
        axes[1, col].set_title("ground truth")
        axes[2, col].imshow(image)
        axes[2, col].imshow(result.anomaly_maps[idx], cmap="jet", alpha=0.5, vmin=0, vmax=vmax)
        axes[2, col].set_title("anomaly map")
        for row in range(3):
            axes[row, col].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def save_error_gallery(dataset, result: EvalResult, out_path: Path, title: str, n: int = 4) -> None:
    """Top two rows: the defects the model scored LOWEST (misses), with the ground truth
    outlined in red, then the anomaly map. Bottom two rows: the normal images it scored
    HIGHEST (false alarms). Ranking-based, so no threshold is needed."""
    anomalous = np.flatnonzero(result.labels == 1)
    normal = np.flatnonzero(result.labels == 0)
    missed = anomalous[np.argsort(result.image_scores[anomalous])[:n]]
    false_alarms = normal[np.argsort(result.image_scores[normal])[::-1][:n]]
    vmax = float(result.anomaly_maps.max())

    fig, axes = plt.subplots(4, n, figsize=(3 * n, 12), squeeze=False)
    for col in range(n):
        for row, indices, kind in ((0, missed, "missed"), (2, false_alarms, "false alarm")):
            if col >= len(indices):
                axes[row, col].axis("off")
                axes[row + 1, col].axis("off")
                continue
            idx = int(indices[col])
            item = dataset[idx]
            image = denormalize(item["image"])

            axes[row, col].imshow(image)
            if item["mask"].sum() > 0:
                axes[row, col].contour(
                    item["mask"].squeeze(0).numpy(), levels=[0.5], colors="red", linewidths=1
                )
            axes[row, col].set_title(
                f"{kind}: {item['defect_type']}  s={result.image_scores[idx]:.2f}", fontsize=8
            )
            axes[row + 1, col].imshow(image)
            axes[row + 1, col].imshow(
                result.anomaly_maps[idx], cmap="jet", alpha=0.5, vmin=0, vmax=vmax
            )
            axes[row, col].axis("off")
            axes[row + 1, col].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
