import argparse
from pathlib import Path

import pandas as pd
import structlog
import torch

from anomaly.config import get_settings
from anomaly.data.mvtec import CATEGORIES, MVTecDataset
from anomaly.evaluate import benchmark_latency, evaluate
from anomaly.logging_setup import configure_logging
from anomaly.metrics import pro_score
from anomaly.models.patchcore import PatchCore
from anomaly.models.patchcore_features import FeatureExtractor
from anomaly.utils import get_device, seed_everything

log = structlog.get_logger()

# (name, fp16 backbone, fp16 stored bank)
VARIANTS = [
    ("fp32", False, False),
    ("fp16_backbone", True, False),
    ("fp16_backbone_fp16_bank", True, True),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", nargs="+", default=["bottle", "carpet"], choices=CATEGORIES)
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level, settings.json_logs)
    seed_everything(settings.seed)
    device = get_device()
    extractor = FeatureExtractor(pretrained=settings.pretrained_backbone)

    rows = []
    for category in args.categories:
        bank_path = settings.artifacts_dir / "patchcore" / f"{category}.pt"
        test_set = MVTecDataset(
            settings.data_root, category, "test", settings.image_size, settings.crop_size
        )
        sample = torch.stack([test_set[i]["image"] for i in range(min(8, len(test_set)))])

        for name, use_fp16, bank_fp16 in VARIANTS:
            model = PatchCore.load(bank_path, extractor, device, nn_chunk=settings.nn_chunk)
            model.use_fp16 = use_fp16
            if bank_fp16:
                model.bank = model.bank.half()

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            result = evaluate(model.anomaly_map, test_set, device, settings.batch_size)
            latency = benchmark_latency(model.anomaly_map, sample, device)
            peak_mb = torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else float("nan")
            bank_mb = model.bank.numel() * model.bank.element_size() / 1e6

            rows.append(
                {
                    "category": category,
                    "variant": name,
                    "image_auroc": round(result.image_auroc, 4),
                    "pixel_auroc": round(result.pixel_auroc, 4),
                    "pro": round(pro_score(result.masks, result.anomaly_maps), 4),
                    "ms_per_image": round(latency, 1),
                    "peak_gpu_mb": round(peak_mb, 0),
                    "bank_mb": round(bank_mb, 1),
                }
            )
            log.info("variant_done", category=category, variant=name)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    reports = Path("reports")
    reports.mkdir(exist_ok=True)
    df.to_csv(reports / "precision_experiment.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()