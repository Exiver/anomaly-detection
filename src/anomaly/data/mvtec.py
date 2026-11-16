from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from anomaly.data.transforms import build_image_transform

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]



split = Literal["train", "test"]

@dataclass(frozen=True)
class Sample:
    image_path : Path
    mask_path : Path | None
    label : int
    defect_type: str


class MVTecDataset(Dataset):
    def __init__(self, root: Path, category : str, split: split, image_size: int = 256, crop_size: int = 224)-> None:
        if category not in CATEGORIES:
            raise ValueError(f"Unknown category: {category}")
        self.root = Path(root)
        self.category = category
        self.split = split
        self.samples = self._collect()

        self.image_transform = build_image_transform(image_size, crop_size)
        self.mask_transform = T.Compose(
            [
                T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST),
                T.CenterCrop(crop_size),
                T.ToTensor(),
            ]
        )
        self.crop_size = crop_size

    def _collect(self) -> list[Sample]:
        category_dir = self.root / self.category
        split_dir = category_dir / self.split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing directory: {split_dir}")

        samples: list[Sample] = []
        for defect_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            defect_type = defect_dir.name
            for image_path in sorted(defect_dir.glob("*.png")):
                if defect_type == "good":
                    samples.append(Sample(image_path, None, 0, "good"))
                else:
                    mask_path = (
                        category_dir
                        / "ground_truth"
                        / defect_type
                        / f"{image_path.stem}_mask.png"
                    )
                    if not mask_path.is_file():
                        raise FileNotFoundError(f"Missing mask: {mask_path}")
                    samples.append(Sample(image_path, mask_path, 1, defect_type))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        image_tensor = self.image_transform(image)


        if sample.mask_path is None:
            mask_tensor = torch.zeros(1, self.crop_size, self.crop_size)
        else:
            mask = Image.open(sample.mask_path).convert("L")
            mask_tensor = (self.mask_transform(mask) > 0.5).float()

        return{
            "image": image_tensor,
            "mask": mask_tensor,
            "label": sample.label,
            "defect_type": sample.defect_type,
            "path": str(sample.image_path)
        }
