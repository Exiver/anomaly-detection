"""Run an anomalib model on MVTec AD and export predictions in a neutral format.

Run in the SEPARATE anomalib environment. Imports nothing from the `anomaly` package.
"""
import argparse
import json
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from anomalib import models as anomalib_models
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from lightning.pytorch import seed_everything

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]


def to_hw(tensor: torch.Tensor) -> np.ndarray:
    """(B, 1, H, W) or (B, H, W) -> (B, H, W) float32."""
    t = tensor.detach().cpu().float()
    if t.ndim == 4 and t.shape[1] == 1:
        t = t[:, 0]
    return t.numpy()


def describe(batch) -> None:
    print(f"batch type: {type(batch).__name__}")
    for name in ("image", "gt_label", "gt_mask", "pred_score", "anomaly_map"):
        value = getattr(batch, name, None)
        shape = tuple(value.shape) if hasattr(value, "shape") else value
        print(f"  {name}: {shape}")


def build_model(name: str, coreset_ratio: float, num_neighbors: int):
    if name == "Patchcore":
        return anomalib_models.Patchcore(
            backbone="wide_resnet50_2",
            layers=("layer2", "layer3"),
            pre_trained=True,
            coreset_sampling_ratio=coreset_ratio,
            num_neighbors=num_neighbors,
        )
    return getattr(anomalib_models, name)()


def collect(predictions) -> dict:
    labels, scores, masks, maps, paths = [], [], [], [], []
    for batch in predictions:
        if batch.gt_mask is None or batch.anomaly_map is None:
            raise RuntimeError(f"Missing fields on {type(batch).__name__}; print(batch) to inspect.")
        labels.append(batch.gt_label.detach().cpu().numpy().astype(np.int64).reshape(-1))
        scores.append(batch.pred_score.detach().cpu().float().numpy().reshape(-1))
        masks.append(to_hw(batch.gt_mask))
        maps.append(to_hw(batch.anomaly_map))
        paths.extend(str(p) for p in batch.image_path)

    out = {
        "labels": np.concatenate(labels),
        "image_scores": np.concatenate(scores).astype(np.float32),
        "masks": (np.concatenate(masks) > 0.5).astype(np.uint8),
        "anomaly_maps": np.concatenate(maps).astype(np.float32),
        "paths": np.array(paths),
    }
    if out["masks"].shape != out["anomaly_maps"].shape:
        raise RuntimeError(
            f"Mask {out['masks'].shape} and map {out['anomaly_maps'].shape} shapes differ."
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/mvtec_ad")
    parser.add_argument("--category", choices=CATEGORIES, help="Default: all 15")
    parser.add_argument("--model", default="Patchcore", help="Any class name in anomalib.models")
    parser.add_argument("--run-name", default="anomalib_patchcore")
    parser.add_argument("--coreset-ratio", type=float, default=0.1)
    parser.add_argument("--num-neighbors", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path("predictions") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    categories = [args.category] if args.category else CATEGORIES
    on_gpu = torch.cuda.is_available()

    for index, category in enumerate(categories):
        seed_everything(args.seed)
        datamodule = MVTecAD(
            root=args.data_root,
            category=category,
            train_batch_size=args.batch_size,
            eval_batch_size=args.batch_size,
            num_workers=0,
        )
        model = build_model(args.model, args.coreset_ratio, args.num_neighbors)
        engine = Engine(default_root_dir=str(Path("anomalib_runs") / args.run_name / category))

        start = time.perf_counter()
        engine.fit(model=model, datamodule=datamodule)
        fit_seconds = time.perf_counter() - start

        predictions = engine.predict(model=model, datamodule=datamodule)  # kept + warm-up
        if index == 0:
            describe(predictions[0])
        arrays = collect(predictions)

        if on_gpu:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        engine.predict(model=model, datamodule=datamodule)  # timing pass, result discarded
        if on_gpu:
            torch.cuda.synchronize()
        n_images = len(arrays["labels"])
        infer_ms = (time.perf_counter() - start) / n_images * 1000.0
        peak_mb = torch.cuda.max_memory_allocated() / 1e6 if on_gpu else None

        defect_types = np.array([Path(p).parent.name for p in arrays["paths"]])
        np.savez_compressed(out_dir / f"{category}.npz", defect_types=defect_types, **arrays)
        meta = {
            "method": f"anomalib:{args.model}",
            "anomalib_version": version("anomalib"),
            "coreset_ratio": args.coreset_ratio,
            "num_neighbors": args.num_neighbors,
            "fit_seconds": round(fit_seconds, 1),
            "infer_ms_per_image": round(infer_ms, 2),
            "peak_gpu_mb_infer": round(peak_mb, 0) if peak_mb is not None else None,
            "n_images": n_images,
        }
        (out_dir / f"{category}_meta.json").write_text(json.dumps(meta, indent=2))
        print(category, meta)


if __name__ == "__main__":
    main()