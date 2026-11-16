"""Generate project reports and the machine-generated part of the model card."""

import json
from pathlib import Path

import pandas as pd

BEGIN = "<!-- BEGIN AUTO -->"
END = "<!-- END AUTO -->"
README_BEGIN = "<!-- BEGIN AUTO RESULTS -->"
README_END = "<!-- END AUTO RESULTS -->"


def fmt(value, digits: int = 3) -> str:
    if pd.isna(value):
        return "n/a"
    if isinstance(value, (float,)):
        return f"{value:.{digits}f}"
    return str(value)


def to_markdown(df: pd.DataFrame, digits: int = 3) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(str(header) for header in headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[column], digits) for column in headers) + " |")
    return "\n".join(lines)


def replace_between_markers(text: str, new: str, begin: str = BEGIN, end: str = END) -> str:
    """Replace only the generated block, keeping the hand-written text intact."""
    if text.count(begin) != 1 or text.count(end) != 1:
        raise ValueError(f"Expected exactly one {begin} and one {end} marker.")
    head, rest = text.split(begin, 1)
    if end not in rest:
        raise ValueError(f"Expected {end} after {begin}.")
    _, tail = rest.split(end, 1)
    return f"{head}{begin}\n{new}\n{end}{tail}"


def main_results(reports: Path) -> str:
    parts: list[str] = []
    frames: dict[str, pd.DataFrame] = {}
    for method, label in (("ae", "Autoencoder baseline"), ("patchcore", "PatchCore (from scratch)")):
        path = reports / f"evaluation_{method}.csv"
        if not path.exists():
            parts.append(f"### {label}\n\n_Not run: `{path.name}` is missing._")
            continue

        df = pd.read_csv(path)
        frames[method] = df
        rows = [
            {
                "category": row.category,
                "image AUROC (99.5% CI)": (
                    f"{row.image_auroc:.3f} ({row.ci_low:.3f}-{row.ci_high:.3f})"
                ),
                "pixel AUROC": row.pixel_auroc,
                "PRO": row.pro,
                "test images": row.n_test,
                "defective": row.n_anomalous,
            }
            for row in df.itertuples(index=False)
        ]
        rows.append(
            {
                "category": "**macro mean**",
                "image AUROC (99.5% CI)": f"{df.image_auroc.mean():.3f} (n/a)",
                "pixel AUROC": df.pixel_auroc.mean(),
                "PRO": df.pro.mean(),
                "test images": int(df.n_test.sum()),
                "defective": int(df.n_anomalous.sum()),
            }
        )
        parts.append(f"### {label}\n\n" + to_markdown(pd.DataFrame(rows)))

    if len(frames) == 2:
        merged = frames["ae"].merge(frames["patchcore"], on="category", suffixes=("_ae", "_pc"))
        overlap = ~(
            (merged.ci_high_ae < merged.ci_low_pc) | (merged.ci_high_pc < merged.ci_low_ae)
        )
        parts.append(
            f"The saved category-level intervals overlap in {int(overlap.sum())} of {len(merged)} "
            "categories. This is a descriptive overlap check, not a paired statistical test."
        )
    return "\n\n".join(parts)


def framework_results(reports: Path) -> str:
    matches = sorted(reports.glob("framework_comparison_*.csv"))
    if not matches:
        return "_Not run._"
    columns = [
        "category", "img_a", "img_b", "ci_overlap", "pix_a", "pix_b", "pro_a", "pro_b",
        "score_spearman", "map_px_a", "map_px_b", "fit_s_a", "fit_s_b", "ms_img_a",
        "ms_img_b", "peak_mb_a", "peak_mb_b",
    ]
    out = []
    for path in matches:
        df = pd.read_csv(path)
        present = [column for column in columns if column in df.columns]
        out.append(
            f"`{path.stem}` (A and B follow the order in the file name)\n\n"
            + to_markdown(df[present])
        )
    return "\n\n".join(out)


def threshold_results(reports: Path) -> str:
    summary_path = reports / "threshold_calibration.csv"
    paths = [summary_path] if summary_path.exists() else sorted(reports.glob("threshold_calibration_*.csv"))
    if not paths:
        return "_Not run._"
    df = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    summary = df.groupby("rule")[["precision", "recall", "f1", "fpr"]].mean().reset_index()
    files = ", ".join(f"`{path.name}`" for path in paths)
    return (
        f"Macro-average over the available calibration files ({files}). The ORACLE row uses test "
        "labels to pick its threshold, so it is an upper bound, not a deployable result. "
        "Per-category rows are available in those CSV files.\n\n" + to_markdown(summary)
    )


def csv_table(reports: Path, name: str) -> str:
    path = reports / name
    if not path.exists():
        return "_Not run._"
    return to_markdown(pd.read_csv(path))


def weakest_defects(reports: Path, method: str, count: int = 10) -> str:
    path = reports / f"per_defect_{method}.csv"
    if not path.exists():
        return "_Not run._"
    df = pd.read_csv(path).sort_values("image_auroc").head(count)
    return to_markdown(df[["category", "defect_type", "image_auroc"]])


def environment_block(reports: Path) -> str:
    path = reports / "environment.txt"
    if not path.exists():
        return "_Run `python scripts/env_report.py` to generate this._"
    return "```text\n" + path.read_text(encoding="utf-8").strip() + "\n```"


def build_results(reports: Path) -> str:
    sections = [
        ("Main results", main_results(reports)),
        ("Framework comparison", framework_results(reports)),
        ("Operating thresholds", threshold_results(reports)),
        ("Weakest PatchCore defect AUROCs", weakest_defects(reports, "patchcore")),
        ("Weakest autoencoder defect AUROCs", weakest_defects(reports, "ae")),
        ("Precision experiment (GPU)", csv_table(reports, "precision_experiment.csv")),
        ("ONNX / int8 experiment (CPU)", csv_table(reports, "onnx_experiment.csv")),
        ("Environment", environment_block(reports)),
    ]
    text = (
        "# Results\n\n"
        "_Generated by `scripts/make_docs.py` from the CSV files in `reports/`. "
        "Do not edit by hand; re-run the script._\n\n"
        "The image-AUROC intervals below are copied from the evaluation CSVs. The evaluator "
        "currently uses `alpha=0.005` (99.5% percentile bootstrap intervals), not 95% intervals; "
        "see `src/anomaly/metrics.py`. Macro means are unweighted means across categories and "
        "do not have a pooled confidence interval. The saved Phase 4 notes describe an earlier "
        "evaluation transform without ImageNet normalization, while the current data pipeline "
        "does normalize. The CSVs do not record their preprocessing revision, so rerun evaluation "
        "before presenting these values as measurements of the current code.\n\n"
    )
    for title, body in sections:
        text += f"## {title}\n\n{body}\n\n"
    return text.rstrip() + "\n"


def readme_results(reports: Path) -> str:
    """Return the main tables for an auto-updated README section."""
    return (
        "_Generated from the evaluation CSVs by `python scripts/make_docs.py`. "
        "See [the full results report](reports/RESULTS.md) for the interval method, "
        "framework comparison, calibration, per-defect evidence, and environment._\n\n"
        + main_results(reports)
    )


def model_card_auto(manifest_path: Path) -> str:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    rule = data["threshold_rule"]
    if data.get("threshold_k") is not None:
        rule += f" (k={data['threshold_k']})"
    lines = [
        f"- **Release:** {data['version']} (built {data['created_at']})",
        f"- **Preprocessing:** resize to {data['image_size']}x{data['image_size']}, "
        f"center crop {data['crop_size']}, ImageNet normalization",
        f"- **Embedding:** WideResNet50-2 layer2 + layer3, {data['embed_dim']}-dimensional patch "
        f"embeddings, coreset ratio {data['coreset_ratio']}",
        f"- **Map smoothing:** Gaussian blur, sigma {data['blur_sigma']}",
        f"- **Threshold rule:** {rule}, calibrated on a {data['holdout_fraction']:.0%} holdout of "
        "normal training images",
        f"- **Seed:** {data['seed']}",
    ]
    rows = [
        {
            "category": name,
            "threshold": entry["threshold"],
            "holdout images": entry["holdout_images"],
            "bank size": entry["bank_size"],
            "test image AUROC": entry["test_image_auroc"],
            "test pixel AUROC": entry["test_pixel_auroc"],
            "recall @ threshold": entry["test_recall_at_threshold"],
            "FPR @ threshold": entry["test_fpr_at_threshold"],
        }
        for name, entry in data["categories"].items()
    ]
    return (
        "\n".join(lines)
        + "\n\n"
        + to_markdown(pd.DataFrame(rows))
        + "\n\n_Test metrics are informational; the thresholds were selected without test labels._"
    )


MODEL_CARD_TEMPLATE = f"""# Model card: PatchCore anomaly detector

## Intended use
TODO

## Out of scope
TODO

## Data and license
TODO

## Configuration and thresholds
{BEGIN}
{END}

## Known failure modes
TODO

## Monitoring
TODO

## Limitations
TODO
"""


def update_model_card(card_path: Path, manifest_path: Path) -> None:
    text = card_path.read_text(encoding="utf-8") if card_path.exists() else MODEL_CARD_TEMPLATE
    text = replace_between_markers(text, model_card_auto(manifest_path))
    card_path.write_text(text, encoding="utf-8")
