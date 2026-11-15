import argparse
from pathlib import Path

import pandas as pd
import structlog

from anomaly.config import get_settings
from anomaly.data.mvtec import CATEGORIES, MVTecDataset
from anomaly.evaluate import evaluate, per_defect_type_auroc
from anomaly.logging_setup import configure_logging
from anomaly.metrics import bootstrap_auroc_ci, pro_score
from anomaly.models.patchcore import PatchCore
from anomaly.models.patchcore_features import FeatureExtractor
from anomaly.scoring import load_autoencoder, make_ae_scorer
from anomaly.utils import get_device, seed_everything
from anomaly.viz import save_error_gallery

log = structlog.get_logger()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["ae", "patchcore"], required=True)
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
    categories = [args.category] if args.category else CATEGORIES
    reports = Path("reports")
    reports.mkdir(exist_ok=True)

    rows, defect_rows = [], []
    for category in categories:
        if args.method == "patchcore":
            bank = settings.artifacts_dir / "patchcore" / f"{category}.pt"
            model = PatchCore.load(bank, extractor, device, nn_chunk=settings.nn_chunk)
            scorer = model.anomaly_map
        else:
            ckpt = settings.artifacts_dir / "ae" / f"{category}.pt"
            net = load_autoencoder(ckpt, settings.latent_channels, device)
            scorer = make_ae_scorer(net, settings.blur_sigma)

        test_set = MVTecDataset(
            settings.data_root, category, "test", settings.image_size, settings.crop_size
        )
        result = evaluate(scorer, test_set, device, settings.batch_size)
        pro = pro_score(result.masks, result.anomaly_maps)
        low, high = bootstrap_auroc_ci(result.labels, result.image_scores, seed=settings.seed)

        log.info(
            "evaluated",
            method=args.method,
            category=category,
            image_auroc=round(result.image_auroc, 4),
            ci=(round(low, 3), round(high, 3)),
            pixel_auroc=round(result.pixel_auroc, 4),
            pro=round(pro, 4),
        )
        rows.append(
            {
                "category": category,
                "image_auroc": round(result.image_auroc, 4),
                "ci_low": round(low, 4),
                "ci_high": round(high, 4),
                "pixel_auroc": round(result.pixel_auroc, 4),
                "pro": round(pro, 4),
                "n_test": len(result.labels),
                "n_anomalous": int(result.labels.sum()),
            }
        )
        for defect, auroc in per_defect_type_auroc(result).items():
            defect_rows.append(
                {"category": category, "defect_type": defect, "image_auroc": round(auroc, 4)}
            )
        save_error_gallery(
            test_set, result, reports / f"errors_{args.method}_{category}.png",
            f"{args.method} - {category}",
        )

    suffix = args.method if args.category is None else f"{args.method}_{args.category}"
    summary = pd.DataFrame(rows)
    per_defect = pd.DataFrame(defect_rows)
    summary.to_csv(reports / f"evaluation_{suffix}.csv", index=False)
    per_defect.to_csv(reports / f"per_defect_{suffix}.csv", index=False)

    print(summary.to_string(index=False))
    if len(summary) > 1:
        print(
            f"\nmean image_auroc={summary.image_auroc.mean():.4f}  "
            f"pixel_auroc={summary.pixel_auroc.mean():.4f}  pro={summary.pro.mean():.4f}"
        )
    print("\nHardest defect types:")
    print(per_defect.sort_values("image_auroc").head(10).to_string(index=False))

    other = "patchcore" if args.method == "ae" else "ae"
    other_path = reports / f"evaluation_{other}.csv"
    if args.category is None and other_path.exists():
        a = summary if args.method == "ae" else pd.read_csv(other_path)
        p = summary if args.method == "patchcore" else pd.read_csv(other_path)
        both = a.merge(p, on="category", suffixes=("_ae", "_pc"))
        cols = ["category", "image_auroc_ae", "image_auroc_pc", "pixel_auroc_ae",
                "pixel_auroc_pc", "pro_ae", "pro_pc"]
        both[cols].to_csv(reports / "method_comparison.csv", index=False)
        print("\n", both[cols].to_string(index=False))


if __name__ == "__main__":
    main()