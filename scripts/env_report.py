import json
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch

PACKAGES = [
    "numpy", "pillow", "scikit-learn", "scipy", "pandas", "matplotlib", "pydantic",
    "pydantic-settings", "structlog", "torch", "torchvision", "fastapi", "uvicorn",
    "prometheus-client", "httpx", "onnx", "onnxruntime",
]


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def anomalib_version() -> str:
    """Read anomalib's version from metadata emitted by its separate export environment."""
    for path in sorted(Path("predictions").glob("*/*_meta.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if "anomalib_version" in data:
            return str(data["anomalib_version"])
    return "unknown (no export metadata found)"


def main() -> None:
    lines = [
        f"python: {sys.version.split()[0]}",
        f"platform: {platform.platform()}",
        f"git commit: {git_commit()}",
        f"cuda available: {torch.cuda.is_available()}",
        f"cuda (torch build): {torch.version.cuda}",
        f"gpu: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}",
        f"anomalib (separate env): {anomalib_version()}",
        "",
    ]
    lines.extend(f"{name}=={package_version(name)}" for name in PACKAGES)
    output = Path("reports") / "environment.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
