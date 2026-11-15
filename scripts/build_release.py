import argparse
from datetime import datetime, timezone
from pathlib import Path

import structlog

from anomaly.config import get_settings
from anomaly.data.mvtec import CATEGORIES
from anomaly.logging_setup import configure_logging
from anomaly.manifest import ReleaseManifest
from anomaly.models.patchcore_features import FeatureExtractor
from anomaly.release import build_category
from anomaly.utils import get_device, seed_everything

log = structlog.get_logger()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an immutable model release bundle.")
    parser.add_argument("--version", required=True, help="e.g. v1")
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES, default=CATEGORIES)
    parser.add_argument("--rule", choices=["max", "mean_k_std"], default="max")
    parser.add_argument("--k", type=float, default=3.0, help="Used by the mean_k_std rule")
    parser.add_argument("--holdout-fraction", type=float, default=0.1)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing release")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level, settings.json_logs)
    seed_everything(settings.seed)
    device = get_device()

    release_dir = Path("releases") / args.version
    if (release_dir / "manifest.json").exists() and not args.force:
        raise SystemExit(f"{release_dir} already exists. Releases are immutable; pick a new version.")
    release_dir.mkdir(parents=True, exist_ok=True)

    extractor = FeatureExtractor(pretrained=settings.pretrained_backbone)
    entries = {}
    for category in args.categories:
        entries[category] = build_category(
            category, settings, device, extractor, release_dir,
            rule=args.rule,
            k=args.k if args.rule == "mean_k_std" else None,
            holdout_fraction=args.holdout_fraction,
        )

    manifest = ReleaseManifest(
        version=args.version,
        created_at=datetime.now(timezone.utc).isoformat(),
        image_size=settings.image_size,
        crop_size=settings.crop_size,
        embed_dim=settings.embed_dim,
        coreset_ratio=settings.coreset_ratio,
        blur_sigma=settings.blur_sigma,
        seed=settings.seed,
        threshold_rule=args.rule,
        threshold_k=args.k if args.rule == "mean_k_std" else None,
        holdout_fraction=args.holdout_fraction,
        categories=entries,
    )
    manifest.write(release_dir / "manifest.json")
    log.info("release_written", path=str(release_dir), categories=list(entries))


if __name__ == "__main__":
    main()