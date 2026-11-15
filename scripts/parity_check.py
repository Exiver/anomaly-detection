import argparse
import time
from pathlib import Path

import httpx

from anomaly.config import get_settings
from anomaly.data.mvtec import CATEGORIES, MVTecDataset
from anomaly.manifest import ReleaseManifest
from anomaly.models.patchcore import PatchCore
from anomaly.models.patchcore_features import FeatureExtractor
from anomaly.service.config import ServiceSettings
from anomaly.utils import get_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--category", required=True, choices=CATEGORIES)
    parser.add_argument("--release-dir", default="releases/v1")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key; defaults to AD_API_KEY loaded from .env/environment",
    )
    args = parser.parse_args()

    settings = get_settings()
    device = get_device()
    release_dir = Path(args.release_dir)
    manifest = ReleaseManifest.read(release_dir / "manifest.json")

    extractor = FeatureExtractor(pretrained=settings.pretrained_backbone)
    model = PatchCore.load(
        release_dir / f"{args.category}.pt", extractor, device, nn_chunk=settings.nn_chunk
    )
    test_set = MVTecDataset(
        settings.data_root, args.category, "test", manifest.image_size, manifest.crop_size
    )

    step = max(1, len(test_set) // args.n)
    indices = list(range(0, len(test_set), step))[: args.n]
    api_key = args.api_key
    if api_key is not None and api_key.startswith("sha256:"):
        # Accept the common notation used when documenting a SHA-256 key value.
        api_key = api_key.removeprefix("sha256:")
    if api_key is None:
        configured_key = ServiceSettings().api_key
        api_key = configured_key.get_secret_value() if configured_key is not None else None
    headers = {"X-API-Key": api_key} if api_key else {}

    diffs, latencies = [], []
    with httpx.Client(base_url=args.url, headers=headers, timeout=60) as client:
        for i in indices:
            sample = test_set.samples[i]
            image = test_set[i]["image"].unsqueeze(0).to(device)
            offline = float(model.anomaly_map(image).max())

            start = time.perf_counter()
            with open(sample.image_path, "rb") as handle:
                response = client.post(
                    "/v1/predict",
                    params={"category": args.category},
                    files={"file": (sample.image_path.name, handle, "image/png")},
                )
            latencies.append((time.perf_counter() - start) * 1000.0)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                if response.status_code == 401:
                    raise SystemExit(
                        "API returned 401 Unauthorized. The parity check loads AD_API_KEY "
                        "from the current environment or .env; restart the API after changing "
                        "its key, and run this script from the project directory."
                    ) from exc
                raise
            online = response.json()["anomaly_score"]
            diffs.append(abs(offline - online))
            print(f"{sample.defect_type:>18}  offline={offline:.5f}  online={online:.5f}  diff={diffs[-1]:.2e}")

    latencies.sort()
    print(f"\nmax |offline - online| = {max(diffs):.2e}")
    print(f"HTTP latency  p50={latencies[len(latencies) // 2]:.0f} ms   "
          f"p95={latencies[int(0.95 * (len(latencies) - 1))]:.0f} ms   (n={len(latencies)})")


if __name__ == "__main__":
    main()
