import argparse
from pathlib import Path

from anomaly.reporting import (
    README_BEGIN,
    README_END,
    build_results,
    readme_results,
    replace_between_markers,
    update_model_card,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reproducible project reports and model-card metrics.")
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--version", default=None, help="Release version (default: newest manifest by mtime)")
    parser.add_argument("--model-card", default="MODEL_CARD.md")
    args = parser.parse_args()

    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)
    results_path = reports / "RESULTS.md"
    results_path.write_text(build_results(reports), encoding="utf-8")
    print("wrote", results_path)

    readme_path = Path("README.md")
    if readme_path.exists():
        readme = readme_path.read_text(encoding="utf-8")
        if README_BEGIN in readme or README_END in readme:
            readme_path.write_text(
                replace_between_markers(readme, readme_results(reports), README_BEGIN, README_END),
                encoding="utf-8",
            )
            print("updated", readme_path, "results block")

    manifests = list(Path("releases").glob("*/manifest.json"))
    if args.version:
        manifests = [Path("releases") / args.version / "manifest.json"]
    manifests = [path for path in manifests if path.exists()]
    if not manifests:
        print("No release manifest found; skipped the model card.")
        return

    manifest = max(manifests, key=lambda path: path.stat().st_mtime)
    card_path = Path(args.model_card)
    update_model_card(card_path, manifest)
    print("updated", card_path, "from", manifest)


if __name__ == "__main__":
    main()
