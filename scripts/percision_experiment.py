import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import structlog
from torch.utils.data import DataLoader, Subset

from anomaly.config import get_settings
from anomaly.data.mvtec import CATEGORIES, MVTecDataset
from anomaly.evaluate import evaluate, image_scores
from anomaly.logging_setup import configure_logging
from anomaly.models.patchcore import PatchCore
from anomaly.models.patchcore_features import FeatureExtractor
from anomaly.thresholds import (
    best_f1_threshold,
    binary_metrics,
    threshold_max,
    threshold_mean_k_std,
)
from anomaly.utils import get_device, seed_everything

log = structlog.get_logger()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", choices=CATEGORIES, help="Default: all 15")
    parser.add_argument("--k", type=float, default=3.0, help="k for the mean + k*std rule")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level, settings.json_logs)
    seed_everything(settings.seed)
    device = get_device()
    extractor = FeatureExtractor(pretrained=settings.pretrained_backbone)

    categories = [args.category] if args.category else CATEGORIES
    reports = Path("reports")
    reports.mkdir(exist_ok=True)

    rows = []
    for category in categories:
        train_set = MVTecDataset(
            settings.data_root, category, "train", settings.image_size, settings.crop_size
        )
        rng = np.random.default_rng(settings.seed)
        order = rng.permutation(len(train_set)).tolist()
        n_hold = max(5, int(0.1 * len(train_set)))
        hold_idx, fit_idx = order[:n_hold], order[n_hold:]

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
        hold_scores = image_scores(
            model.anomaly_map, Subset(train_set, hold_idx), device, settings.batch_size
        )

        test_set = MVTecDataset(
            settings.data_root, category, "test", settings.image_size, settings.crop_size
        )
        result = evaluate(model.anomaly_map, test_set, device, settings.batch_size)

        rules = {
            "max": threshold_max(hold_scores),
            f"mean+{args.k:g}std": threshold_mean_k_std(hold_scores, args.k),
        }
        for rule, thr in rules.items():
            m = binary_metrics(result.labels, result.image_scores, thr)
            rows.append(
                {"category": category, "rule": rule, "threshold": round(thr, 4),
                 **{name: round(value, 4) for name, value in m.items()}}
            )
        oracle_thr, _ = best_f1_threshold(result.labels, result.image_scores)
        m = binary_metrics(result.labels, result.image_scores, oracle_thr)
        rows.append(
            {"category": category, "rule": "ORACLE (uses test labels)",
             "threshold": round(oracle_thr, 4),
             **{name: round(value, 4) for name, value in m.items()}}
        )
        log.info("calibrated", category=category, holdout=len(hold_idx),
                 image_auroc=round(result.image_auroc, 4))

    df = pd.DataFrame(rows)
    name = "threshold_calibration.csv" if args.category is None else f"threshold_calibration_{args.category}.csv"
    df.to_csv(reports / name, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()