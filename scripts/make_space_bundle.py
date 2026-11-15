import argparse
import re
import shutil
from pathlib import Path

README = """---
title: Visual Anomaly Detection
sdk: docker
app_port: 7860
pinned: false
---

# Visual anomaly detection

A PatchCore anomaly detector implemented from scratch and served with FastAPI.
Open `/demo` for the web UI and `/docs` for the API.

Trained on the public MVTec AD benchmark (non-commercial license). See the project
repository for method, evaluation, and limitations.
"""


def set_arg_default(text: str, name: str, value: str) -> str:
    updated, count = re.subn(rf"^ARG {name}=.*$", f"ARG {name}={value}", text, flags=re.MULTILINE)
    if count != 1:
        raise SystemExit(f"Expected exactly one 'ARG {name}=' line in the Dockerfile, found {count}.")
    return updated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", required=True, help="e.g. v2")
    parser.add_argument("--out", default="space_bundle")
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} already exists; remove it first.")
    release_dir = Path("releases") / args.release
    if not (release_dir / "manifest.json").exists():
        raise SystemExit(f"No release at {release_dir}")

    out.mkdir(parents=True)
    shutil.copytree("src", out / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"))
    shutil.copy("pyproject.toml", out / "pyproject.toml")
    shutil.copytree(release_dir, out / "releases" / args.release)

    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    dockerfile = set_arg_default(dockerfile, "APP_UID", "1000")
    dockerfile = set_arg_default(dockerfile, "PORT", "7860")
    dockerfile = set_arg_default(dockerfile, "RELEASE", args.release)
    (out / "Dockerfile").write_text(dockerfile, encoding="utf-8")

    (out / "README.md").write_text(README, encoding="utf-8")
    (out / ".gitattributes").write_text("*.pt filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8")
    print(f"Space bundle ready in {out}/")


if __name__ == "__main__":
    main()