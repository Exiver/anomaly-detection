import argparse
import re
import subprocess
import sys
from pathlib import Path

WARN_BYTES = 5 * 1024 * 1024
GITHUB_WARN_BYTES = 50 * 1024 * 1024
GITHUB_BLOCK_BYTES = 100 * 1024 * 1024
SCAN_LIMIT_BYTES = 2 * 1024 * 1024

SECRET_PATTERNS = {
    "generic key assignment": re.compile(
        r"(?i)(api[_-]?key|secret|token|passwd|password)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}"
    ),
    "AWS access key id": re.compile(r"AKIA[0-9A-Z]{16}"),
    "GitHub token": re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"),
    "Google API key": re.compile(r"AIza[0-9A-Za-z_\-]{35}"),
    "Hugging Face token": re.compile(r"hf_[A-Za-z0-9]{30,}"),
    "Groq key": re.compile(r"gsk_[A-Za-z0-9]{20,}"),
    "private key block": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
}

FORBIDDEN_PREFIXES = (
    "data/", "artifacts/", "predictions/", "anomalib_runs/", "results/",
    "space_bundle/", "lightning_logs/", "static_demo/",
)
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".onnx", ".npz", ".npy", ".safetensors"}

REQUIRED_PATHS = [
    "README.md", "LICENSE", "MODEL_CARD.md", ".gitignore", "pyproject.toml", "Dockerfile",
    ".github/workflows/ci.yml", "docs/REPRODUCE.md",
    "src/anomaly/data/mvtec.py", "src/anomaly/data/transforms.py",
    "src/anomaly/models/patchcore.py", "src/anomaly/service/app.py", "tests/conftest.py",
]


def find_secrets(text: str) -> list[tuple[str, int]]:
    """Return (pattern name, line number). Never returns the matched text."""
    hits = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(line):
                hits.append((name, lineno))
    return hits


def forbidden(path: str) -> str | None:
    p = path.replace("\\", "/")
    name = p.rsplit("/", 1)[-1]
    if name == ".env" or (name.startswith(".env.") and name != ".env.example"):
        return "environment file that may hold secrets"
    if p.startswith(FORBIDDEN_PREFIXES):
        return "dataset or generated artifact folder"
    if p.startswith("releases/") and name != ".gitkeep":
        return "model release (publish separately, if at all)"
    suffix = Path(p).suffix.lower()
    if suffix in FORBIDDEN_SUFFIXES:
        return f"model or data file ({suffix})"
    if "__pycache__" in p or ".ipynb_checkpoints" in p or "egg-info" in p:
        return "generated cache"
    return None


def candidate_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        capture_output=True,
        check=True,
    )
    return [p for p in out.stdout.decode("utf-8").split("\0") if p]


def looks_binary(path: Path) -> bool:
    with open(path, "rb") as handle:
        return b"\0" in handle.read(1024)


def env_in_history() -> bool:
    result = subprocess.run(
        ["git", "log", "--all", "--oneline", "--", ".env"], capture_output=True, text=True
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def main() -> int:
    argparse.ArgumentParser(description="Check what would be published to GitHub.").parse_args()
    errors: list[str] = []
    warnings: list[str] = []

    files = candidate_files()
    present = set(files)
    for required in REQUIRED_PATHS:
        if required not in present:
            errors.append(f"missing from what would be committed: {required}")

    for path in files:
        reason = forbidden(path)
        if reason:
            errors.append(f"{path}: {reason}")
        file = Path(path)
        if not file.is_file():
            continue
        size = file.stat().st_size
        if size > GITHUB_BLOCK_BYTES:
            errors.append(f"{path}: {size / 1e6:.0f} MB exceeds GitHub's 100 MB file limit")
        elif size > GITHUB_WARN_BYTES:
            warnings.append(f"{path}: {size / 1e6:.0f} MB (GitHub warns above 50 MB)")
        elif size > WARN_BYTES:
            warnings.append(f"{path}: {size / 1e6:.1f} MB; is this meant to be in the repository?")
        if size <= SCAN_LIMIT_BYTES and not looks_binary(file):
            text = file.read_text(encoding="utf-8", errors="ignore")
            for name, lineno in find_secrets(text):
                errors.append(f"{path}:{lineno}: possible {name} (review it; the value is not printed)")

    readme = Path("README.md")
    if readme.exists() and "<!-- FILL" in readme.read_text(encoding="utf-8"):
        errors.append("README.md still contains <!-- FILL --> placeholders")
    card = Path("MODEL_CARD.md")
    if card.exists() and "TODO" in card.read_text(encoding="utf-8"):
        errors.append("MODEL_CARD.md still contains TODO sections")

    if env_in_history():
        errors.append(".env appears in git history: rotate every key it ever held")

    for message in warnings:
        print("WARN ", message)
    for message in errors:
        print("ERROR", message)
    print(f"\n{len(files)} files checked: {len(errors)} errors, {len(warnings)} warnings")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())