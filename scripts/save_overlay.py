import argparse
import base64
from pathlib import Path

import httpx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--category", required=True)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--out", default="reports/demo_samples")
    parser.add_argument("images", nargs="+", help="PNG files, e.g. from data/mvtec_ad/bottle/test/...")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    headers = {"X-API-Key": args.api_key} if args.api_key else {}

    with httpx.Client(base_url=args.url, headers=headers, timeout=120) as client:
        for image in args.images:
            path = Path(image)
            with open(path, "rb") as handle:
                response = client.post(
                    "/v1/predict",
                    params={"category": args.category, "include_heatmap": True},
                    files={"file": (path.name, handle, "image/png")},
                )
            response.raise_for_status()
            body = response.json()
            verdict = "anomaly" if body["is_anomalous"] else "normal"
            name = f"{args.category}_{path.parent.name}_{path.stem}_{verdict}.png"
            (out / name).write_bytes(base64.b64decode(body["heatmap_png_base64"]))
            print(f"{name}  score={body['anomaly_score']:.3f}  threshold={body['threshold']:.3f}")


if __name__ == "__main__":
    main()