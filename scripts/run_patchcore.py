import argparse
import time
from pathlib import Path

import pandas as pd
import structlog
import torch
from torch.utils.data import DataLoader

from anomaly.config import get_settings
from anomaly.data.mvtec import CATEGORIES, MVTecDataset
from anomaly.evaluate import benchmark_latency, evaluate
from anomaly.logging_setup import configure_logging
from anomaly.models.patchcore import PatchCore
from anomaly.models.patchcore_features import FeatureExtractor
from anomaly.utils import get_device, seed_everything
from anomaly.viz import save_heatmaps

log = structlog.get_logger()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", choices=CATEGORIES, help="Default: all 15")
    parser.add_argument("--skip-fit", action="store_true", help="Load saved memory banks")
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
        bank_path = settings.artifacts_dir / "patchcore" / f"{category}.pt"
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        if args.skip_fit:
            model = PatchCore.load(bank_path, extractor, device, nn_chunk=settings.nn_chunk)
            fit_seconds = float("nan")

        else:
            train_set = MVTecDataset(
                settings.data_root, category, "train", settings.image_size, settings.crop_size
            )
            loader = DataLoader(
                train_set,
                batch_size= settings.batch_size,
                shuffle=True,
                num_workers= settings.num_workers
            )
            model = PatchCore(
                extractor,
                device,
                embed_dim= settings.embed_dim,
                coreset_ratio= settings.coreset_ratio,
                blur_sigma=settings.blur_sigma,
                nn_chunk=settings.nn_chunk,
                seed = settings.seed
            )

            start = time.perf_counter()
            model.fit(loader)
            fit_seconds = time.perf_counter() - start
            model.save(bank_path)

        test_set = MVTecDataset(
            settings.data_root, category, "test", settings.batch_size, settings.crop_size
        )
        result = evaluate(model.anomaly_map, test_set, device, settings.batch_size)
        sample = torch.stack([test_set[i]["image"] for i in range(min(8, len(test_set)))])
        latency_ms = benchmark_latency(model.anomaly_map, sample, device)
        peak_mb = (
            torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else float("nan")
        )
        log.info(
            "evaluated",
            category = category,
            image_auroc = round(result.image_auroc, 4),
            pixel_auroc = round(result.pixel_auroc, 4),
            bank_size = (model.bank.shape[0]),
            ms_per_image = round(latency_ms, 1),
        )
        rows.append(
            {
                "category": category,
                "image_auroc": round(result.image_auroc, 4),
                "pixel_auroc": round(result.pixel_auroc, 4),
                "bank_size": int(model.bank.shape[0]),
                "fit_seconds": round(fit_seconds, 1),
                "ms_per_image": round(latency_ms, 1),
                "peak_gpu_mb": round(peak_mb, 0),
            }
        )
        save_heatmaps(test_set, result, reports / f"pc_heatmaps_{category}.png", category)
        df = pd.DataFrame(rows)
        name = "patchcore_results.csv" if args.category is None else f"patchcore_results_{args.category}.csv"
        df.to_csv(reports / name, index=False)
        print(df.to_string(index=False))
        if len(df) > 1:
            print(f"\nmean image_auroc={df.image_auroc.mean():.4f}"
                  f"mean pixel_auroc{df.pixel_auroc.mean():.4f}")

        ae_path = reports / "ae_results.csv"
        if args.category is None and ae_path.exists():
            ae = pd.read_csv(ae_path).rename(
                columns={"image_auroc": "ae_image", "pixel_auroc": "ae_pixel"}
            )

            pc = df[["category", "image_auroc", "pixel_auroc"]].rename(
            columns={"image_auroc": "pc_image", "pixel_auroc": "pc_pixel"}
        )
            merged = pc.merge(ae, on="category")
            merged.to_csv(reports / "ae_vs_patchcore.csv", index=False)
            print("\n", merged.to_string(index=False))


if __name__ == "__main__":
    main()
