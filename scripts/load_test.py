import argparse
import io
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from PIL import Image, ImageEnhance


def load_bytes(path: Path, brightness: float) -> bytes:
    if brightness == 1.0:
        return path.read_bytes()
    image = Image.open(path).convert("RGB")
    image = ImageEnhance.Brightness(image).enhance(brightness)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--category", required=True)
    parser.add_argument("--image-dir", required=True, help="e.g. data/mvtec_ad/bottle/test")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--brightness", type=float, default=1.0, help="FAULT: scale brightness (e.g. 0.5)")
    parser.add_argument("--only-defects", action="store_true", help="FAULT: send only defective images")
    args = parser.parse_args()

    paths = sorted(Path(args.image_dir).rglob("*.png"))
    if args.only_defects:
        paths = [p for p in paths if p.parent.name != "good"]
    if not paths:
        raise SystemExit("No images found.")
    random.Random(0).shuffle(paths)
    chosen = [paths[i % len(paths)] for i in range(args.requests)]
    headers = {"X-API-Key": args.api_key} if args.api_key else {}

    with httpx.Client(base_url=args.url, headers=headers, timeout=60) as client:

        def send(path: Path) -> tuple[int, float]:
            payload = load_bytes(path, args.brightness)
            start = time.perf_counter()
            response = client.post(
                "/v1/predict",
                params={"category": args.category},
                files={"file": (path.name, payload, "image/png")},
            )
            return response.status_code, (time.perf_counter() - start) * 1000.0

        wall = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            results = list(pool.map(send, chosen))
        wall = time.perf_counter() - wall

    statuses: dict[int, int] = {}
    for status, _ in results:
        statuses[status] = statuses.get(status, 0) + 1
    latencies = sorted(ms for status, ms in results if status == 200)
    print("status counts:", statuses)
    if latencies:
        pick = lambda q: latencies[int(q * (len(latencies) - 1))]
        print(f"latency ms  p50={pick(0.5):.0f}  p95={pick(0.95):.0f}  p99={pick(0.99):.0f}")
    print(f"throughput: {len(results) / wall:.1f} req/s over {wall:.1f}s")


if __name__ == "__main__":
    main()