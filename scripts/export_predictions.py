import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import structlog
import torch

from anomaly.config import get_settings
from anomaly.data.mvtec import CATEGORIES, MVTecDataset
from anomaly.evaluate import evaluate, image_scores
from anomaly.factory import build_scorer
from anomaly.logging_setup import configure_logging
from anomaly.models.patchcore_features import FeatureExtractor
from anomaly.utils import get_device, seed_everything

log = structlog.get_logger()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["ae", "patchcore"], default="patchcore")
    parser.add_argument("--category", choices=CATEGORIES, help="Default: all 15")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level, settings.json_logs)
    seed_everything(settings.seed)
    device = get_device()

    extractor = (
        FeatureExtractor(pretrained=settings.pretrained_backbone)
        if args.method == "patchcore"
        else None
    )
    out_dir = Path("predictions") / args.method
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fit times were recorded in Phase 3.
    fit_seconds: dict[str, float] = {}
    phase3_csv = Path("reports") / "patchcore_results.csv"
    if args.method == "patchcore" and phase3_csv.exists():
        table = pd.read_csv(phase3_csv)
        fit_seconds = dict(zip(table.category, table.fit_seconds))

    categories = [args.category] if args.category else CATEGORIES
    for category in categories:
        scorer = build_scorer(args.method, category, settings, device, extractor)
        test_set = MVTecDataset(
            settings.data_root, category, "test", settings.image_size, settings.crop_size
        )

        # Timed pass: warm-up first, then end-to-end over the whole test set.
        sample = torch.stack([test_set[i]["image"] for i in range(min(8, len(test_set)))])
        scorer(sample.to(device))
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        image_scores(scorer, test_set, device, settings.batch_size)
        if device.type == "cuda":
            torch.cuda.synchronize()
        infer_ms = (time.perf_counter() - start) / len(test_set) * 1000.0
        peak_mb = torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else None

        result = evaluate(scorer, test_set, device, settings.batch_size)
        np.savez_compressed(
            out_dir / f"{category}.npz",
            labels=result.labels.astype(np.int64),
            image_scores=result.image_scores.astype(np.float32),
            masks=(result.masks > 0.5).astype(np.uint8),
            anomaly_maps=result.anomaly_maps.astype(np.float32),
            paths=np.array(result.paths),
            defect_types=result.defect_types,
        )
        meta = {
            "method": args.method,
            "fit_seconds": fit_seconds.get(category),
            "infer_ms_per_image": round(infer_ms, 2),
            "peak_gpu_mb_infer": round(peak_mb, 0) if peak_mb is not None else None,
            "n_images": len(result.labels),
        }
        (out_dir / f"{category}_meta.json").write_text(json.dumps(meta, indent=2))
        log.info("exported", category=category, **meta)


if __name__ == "__main__":
    main()