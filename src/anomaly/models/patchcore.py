import time
from pathlib import Path

import structlog
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from anomaly.models.coreset import greedy_coreset
from anomaly.models.patchcore_features import FeatureExtractor, build_patch_embedding
from anomaly.scoring import gaussian_blur

log = structlog.get_logger()

@torch.no_grad()
def nearest_distance(queries: torch.Tensor, bank: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
    """Euclidean distance from each quary row ro its nearest row in 'bank'."""

    # torch.cdist requires matching dtypes; embeddings are float32 even when a
    # saved or explicitly compressed memory bank uses float16.
    bank = bank.to(device=queries.device, dtype=queries.dtype)
    out = []
    for part in queries.split(chunk):
        out.append(torch.cdist(part, bank).min(dim=1).values)
    return torch.cat(out)


class PatchCore:
    def __init__(
            self, 
            extractor: FeatureExtractor,
            device: torch.device,
            embed_dim: int = 1024,
            coreset_ratio: float = 0.1,
            blur_sigma: float = 4.0,
            nn_chunk: int = 2048,
            seed: int = 42,
            use_fp16: bool = False,
    ) -> None:
        self.extractor = extractor.to(device)
        self.device = device
        self.embed_dim = embed_dim
        self.coreset_ratio = coreset_ratio
        self.blur_sigma = blur_sigma
        self.nn_chunk = nn_chunk
        self.seed = seed
        self.use_fp16 = use_fp16
        self.bank: torch.Tensor | None = None 

    @torch.no_grad()
    def embed(self, images: torch.Tensor) -> torch.Tensor:
        """(N, 3, H, W) -> (N, embed_dim, h, w) patch embeddings, always returned as float32."""
        enabled = self.use_fp16 and self.device.type == "cuda"
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=enabled):
            f2, f3 = self.extractor(images)
            embedding = build_patch_embedding(f2, f3, self.embed_dim)
        return embedding.float()


    @torch.no_grad()
    def fit(self, loader: DataLoader)-> None:
        start = time.perf_counter()
        chunks, n_images = [], 0
        for batch in loader:
            feats = self.embed(batch["image"].to(self.device))
            n_images += feats.shape[0]

            chunks.append(feats.permute(0, 2, 3, 1).reshape(-1, feats.shape[1]).cpu())
        all_feats = torch.cat(chunks)
        k = max(1, int(len(all_feats) * self.coreset_ratio))
        indices = greedy_coreset(all_feats, k, self.device, seed=self.seed)
        self.bank = all_feats[indices].to(self.device)

        log.info(
            "patchcore_fit",
            images=len(all_feats),
            bank_size=len(self.bank),
            seconds=round(time.perf_counter() - start, 1),
        )

    @torch.no_grad()
    def anomaly_map(self, images: torch.Tensor) -> torch.Tensor:
        """(N, 3, H, W) -> (N, H, W). higher = more anomalous. matches the scorer interface."""
        if self.bank is None:
            raise RuntimeError("call fit() or load() before scoring")
        feats = self.embed(images)
        n, d, h, w = feats.shape
        patches = feats.permute(0, 2, 3, 1).reshape(-1, d)
        dist = nearest_distance(patches, self.bank, self.nn_chunk).reshape(n, 1, h, w)
        dist = F.interpolate(dist, size=images.shape[-2:], mode="bilinear", align_corners=False)
        return gaussian_blur(dist.squeeze(1), self.blur_sigma)


    def save(self, path: Path) -> None:
        if self.bank is None:
            raise RuntimeError("Nothing to save : model is not fitted")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "bank": self.bank.cpu(),
                "embed_dim": self.embed_dim,
                "coreset_ratio": self.coreset_ratio,
                "blur_sigma": self.blur_sigma,
            },
            path,
        )

    @classmethod
    def load(
        cls,
        path: Path,
        extractor: FeatureExtractor,
        device: torch.device,
        nn_chunk: int = 2048,
    ) -> "PatchCore":
        state = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(
            extractor,
            device,
            embed_dim = state["embed_dim"],
            coreset_ratio= state["coreset_ratio"],
            blur_sigma= state["blur_sigma"],
            nn_chunk=nn_chunk,

        )
        model.bank = state["bank"].to(device)
        return model
