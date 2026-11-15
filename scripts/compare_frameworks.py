import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from anomaly.data.mvtec import CATEGORIES
from anomaly.metrics import bootstrap_auroc_ci, image_auroc, pixel_auroc, pro_score


def load(root: Path, category: str) -> tuple[dict, dict]:
    data = np.load(root / f"{category}.npz", allow_pickle=False)
    meta = json.loads((root / f"{category}_meta.json").read_text())
    return {key: data[key] for key in data.files}, meta


def metrics(arrays: dict) -> dict:
    labels, scores = arrays["labels"], arrays["image_scores"]
    masks, maps = arrays["masks"], arrays["anomaly_maps"]
    low, high = bootstrap_auroc_ci(labels, scores)
    return {
        "image_auroc": image_auroc(labels, scores),
        "ci_low": low,
        "ci_high": high,
        "pixel_auroc": pixel_auroc(masks, maps),
        "pro": pro_score(masks, maps),
        "map_size": maps.shape[-1],
    }


def key(path: str) -> str:
    p = Path(path)
    return f"{p.parent.name}/{p.name}"  # e.g. "broken_large/000.png"


def percentile_rank(scores: np.ndarray) -> np.ndarray:
    return (rankdata(scores) - 1) / max(1, len(scores) - 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", default="patchcore")
    parser.add_argument("--b", default="anomalib_patchcore")
    parser.add_argument("--root", default="predictions")
    args = parser.parse_args()

    root_a, root_b = Path(args.root) / args.a, Path(args.root) / args.b
    rows, disagreements = [], []

    for category in CATEGORIES:
        if not (root_a / f"{category}.npz").exists() or not (root_b / f"{category}.npz").exists():
            continue
        arr_a, meta_a = load(root_a, category)
        arr_b, meta_b = load(root_b, category)
        m_a, m_b = metrics(arr_a), metrics(arr_b)

        # Align the two prediction sets by image identity.
        index_a = {key(p): i for i, p in enumerate(arr_a["paths"])}
        index_b = {key(p): i for i, p in enumerate(arr_b["paths"])}
        common = sorted(set(index_a) & set(index_b))
        if len(common) != len(index_a) or len(common) != len(index_b):
            print(f"WARNING {category}: only {len(common)} of "
                  f"{len(index_a)}/{len(index_b)} images match between the two runs")
        sa = arr_a["image_scores"][[index_a[k] for k in common]]
        sb = arr_b["image_scores"][[index_b[k] for k in common]]
        rho = float(spearmanr(sa, sb).statistic)
        diff = np.abs(percentile_rank(sa) - percentile_rank(sb))
        labels_common = arr_a["labels"][[index_a[k] for k in common]]
        for k, d, lab in zip(common, diff, labels_common):
            disagreements.append({"category": category, "image": k, "label": int(lab),
                                  "rank_gap": round(float(d), 3)})

        overlap = not (m_a["ci_high"] < m_b["ci_low"] or m_b["ci_high"] < m_a["ci_low"])
        rows.append(
            {
                "category": category,
                "img_a": round(m_a["image_auroc"], 4),
                "img_b": round(m_b["image_auroc"], 4),
                "ci_overlap": overlap,
                "pix_a": round(m_a["pixel_auroc"], 4),
                "pix_b": round(m_b["pixel_auroc"], 4),
                "pro_a": round(m_a["pro"], 4),
                "pro_b": round(m_b["pro"], 4),
                "score_spearman": round(rho, 3),
                "map_px_a": m_a["map_size"],
                "map_px_b": m_b["map_size"],
                "fit_s_a": meta_a.get("fit_seconds"),
                "fit_s_b": meta_b.get("fit_seconds"),
                "ms_img_a": meta_a.get("infer_ms_per_image"),
                "ms_img_b": meta_b.get("infer_ms_per_image"),
                "peak_mb_a": meta_a.get("peak_gpu_mb_infer"),
                "peak_mb_b": meta_b.get("peak_gpu_mb_infer"),
            }
        )

    if not rows:
        raise SystemExit("No categories found in both prediction folders.")

    df = pd.DataFrame(rows)
    reports = Path("reports")
    reports.mkdir(exist_ok=True)
    df.to_csv(reports / f"framework_comparison_{args.a}_vs_{args.b}.csv", index=False)
    pd.DataFrame(disagreements).sort_values("rank_gap", ascending=False).head(30).to_csv(
        reports / f"framework_disagreements_{args.a}_vs_{args.b}.csv", index=False
    )

    print(f"A = {args.a}    B = {args.b}\n")
    print(df.to_string(index=False))
    numeric = df[["img_a", "img_b", "pix_a", "pix_b", "pro_a", "pro_b", "score_spearman"]].mean()
    print("\nmeans:\n", numeric.round(4).to_string())
    print(f"\ncategories with overlapping image-AUROC confidence intervals: "
          f"{int(df.ci_overlap.sum())} of {len(df)}")


if __name__ == "__main__":
    main()