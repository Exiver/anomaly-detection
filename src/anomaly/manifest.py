from pathlib import Path

from pydantic import BaseModel


class CategoryEntry(BaseModel):
    threshold: float
    holdout_images: int
    bank_size: int
    # Informational "model card" numbers. They were NOT used to choose the threshold.
    test_image_auroc: float
    test_pixel_auroc: float
    test_recall_at_threshold: float
    test_fpr_at_threshold: float
    reference_score_ratios: list[float] = []
    reference_input_mean: float | None = None
    reference_input_mean_std: float | None = None

class ReleaseManifest(BaseModel):
    version: str
    created_at: str
    image_size: int
    crop_size: int
    embed_dim: int
    coreset_ratio: float
    blur_sigma: float
    seed: int
    threshold_rule: str
    threshold_k: float | None
    holdout_fraction: float
    categories: dict[str, CategoryEntry]

    def write(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> "ReleaseManifest":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))