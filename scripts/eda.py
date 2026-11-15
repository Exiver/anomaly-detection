from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import structlog
from PIL import Image

from anomaly.config import get_settings
from anomaly.data.mvtec import CATEGORIES, MVTecDataset
from anomaly.logging_setup import configure_logging

log = structlog.get_logger()

def summarize(root: Path) -> pd.DataFrame:
    rows = []
    for category in CATEGORIES:
        train = MVTecDataset(root, category, "train")
        test = MVTecDataset(root, category, "test")
        test_labels = [s.label for s in test.samples]
        first = Image.open(train.samples[0].image_path)
        rows.append(
            {
                "category": category,
                "train_good": len(train),
                "test_total": len(test),
                "test_good": test_labels.count(0),
                "test_anomalous": test_labels.count(1),
                "defect_types": len({s.defect_type for s in test.samples} - {"good"}),
                "width": first.width,
                "height": first.height,
                "mode": first.mode, 
            }
        )
    return pd.DataFrame(rows)

def sample_grid(root: Path, category: str, out_path: Path, n: int = 4) -> None:
    train = MVTecDataset(root, category, "train")
    test = MVTecDataset(root, category, "test")
    anomalous = [s for s in test.samples if s.label == 1]

    train_step = max(1, len(train.samples) // n)
    anom_step = max(1, len(anomalous) // n)
    normals = train.samples[::train_step][:n]
    defects = anomalous[::anom_step][:n]

    fig, axes = plt.subplots(3, n, figsize=(3 *n, 9))
    for i in range(n):
        normal_img = Image.open(normals[1].image_path).convert("RGB")
        defect_img = Image.open(defects[1].image_path).convert("RGB")
        mask = np.array(Image.open(defects[i].mask_path).convert("L")) > 0

        overlay = np.array(defect_img).copy()
        overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)


        axes[0, i].imshow(normal_img)
        axes[0, i].set_title("normal (train)")
        axes[1, i].imshow(defect_img)
        axes[1, i].set_title(defects[i].defect_type)
        axes[2, i].imshow(overlay)
        axes[2, i].set_title("defect mask overlay")
        for row in range(3):
            axes[row, i].axis("off")

    fig.suptitle(category)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.json_logs)

    reports = Path("reports")
    reports.mkdir(exist_ok=True)

    df = summarize(settings.data_root)
    df.to_csv(reports / "dataset_summary.csv", index=False)
    log.info(
        "dataset_summary",
        train_total=int(df.train_good.sum()),
        test_total=int(df.test_total.sum()),
        anomalous_total=int(df.test_anomalous.sum()),
    )
    print(df.to_string(index=False))
    for category in ("bottle", "grid", "screw"):
        out = reports / f"samples_{category}.png"
        sample_grid(settings.data_root, category, out)
        log.info("saved_grid", path=str(out))

if __name__ == "__main__":
    main()