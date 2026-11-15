from typing import Callable

import torch

from anomaly.config import Settings
from anomaly.models.patchcore import PatchCore
from anomaly.models.patchcore_features import FeatureExtractor
from anomaly.scoring import load_autoencoder, make_ae_scorer

Scorer = Callable[[torch.Tensor], torch.Tensor]


def build_scorer(
    method: str,
    category: str,
    settings: Settings,
    device: torch.device,
    extractor: FeatureExtractor | None = None,
) -> Scorer:
    if method == "patchcore":
        if extractor is None:
            raise ValueError("PatchCore needs a FeatureExtractor.")
        path = settings.artifacts_dir / "patchcore" / f"{category}.pt"
        return PatchCore.load(path, extractor, device, nn_chunk=settings.nn_chunk).anomaly_map
    if method == "ae":
        path = settings.artifacts_dir / "ae" / f"{category}.pt"
        net = load_autoencoder(path, settings.latent_channels, device)
        return make_ae_scorer(net, settings.blur_sigma)
    raise ValueError(f"Unknown method: {method}")