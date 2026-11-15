import argparse
from pathlib import Path

import pandas as pd
import structlog

from anomaly.config import get_settings
from anomaly.data.mvtec import CATEGORIES, MVTecDataset
from anomaly.evaluate import evaluate
from anomaly.logging_setup import configure_logging
from anomaly.scoring import load_autoencoder, make_ae_scorer
from anomaly.training.autoencoder import train_autoencoder
from anomaly.utils import get_device
from anomaly.viz import save_heatmaps

log = structlog.get_logger()


def main() -> None :
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", choices=CATEGORIES, help="Default: all 15")
    parser.add_argument("--skip-train", action="store_true", help="Use existing checkpoints")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level, settings.json_logs)
    device = get_device()

    categories = [args.category] if args.category else CATEGORIES

    reports = Path("reports")
    reports.mkdir(exist_ok=True)

    rows = []

    for category in categories:
        checkpoint = settings.artifacts_dir / "ae" / f"{category}.pt"
        if not args.skip_train:
            checkpoint = train_autoencoder(category=category, settings=settings, device=device)

        model = load_autoencoder(checkpoint=checkpoint, latent_channels=settings.latent_channels, device=device)
        test_set = MVTecDataset(
            settings.data_root, category, "test", settings.image_size, settings.crop_size
        )
        result = evaluate(
            make_ae_scorer(model, settings.blur_sigma), test_set, device, settings.batch_size
        )
        log.info(
            "evaluated",
            category=category,
            image_auroc= round(result.image_auroc, 4),
            pixel_auroc = round(result.pixel_auroc, 4),
        )

        rows.append({
            "category": category,
            "image_auroc": round(result.image_auroc, 4),
            "pixel_auroc": round(result.pixel_auroc, 4),
        })
        save_heatmaps(test_set, result, reports /f"ae_heatmaps_{category}.png", category)

    df = pd.DataFrame(rows)
    name = "ae_results.csv" if args.category is None else f"ae_results_{args.category}.csv"
    df.to_csv(reports / name, index=False)
    print(df.to_string(index=False))
    if len(df) > 1:
        print(f"\nmean image_auroc={df.image_auroc.mean():.4f}  "
              f"mean pixel_auroc={df.pixel_auroc.mean():.4f}")


if __name__ == "__main__":
    main()