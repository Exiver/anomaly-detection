import math

import torch


def greedy_coreset(
        features: torch.Tensor,
        k: int ,
        device: torch.device,
        proj_dim: int = 128,
        seed: int = 0,
        chunk : int = 65536,
) -> torch.Tensor:
    "Return the indices (cpu, long) of k well-spread rows of features (n, d)."
    n, d = features.shape
    k = min(k, n)
    generator = torch.Generator().manual_seed(seed)
    projection = torch.randn(d, proj_dim, generator=generator) / math.sqrt(proj_dim)
    projection = projection.to(device)
    reduced = torch.cat([part.to(device) @ projection for part in features.split(chunk)])
    sq_norm = reduced.pow(2).sum(dim=1)

    selected = torch.empty(k, dtype=torch.long)
    min_dist = torch.full((n,), float("inf"), device=device)
    idx = int(torch.randint(n, (1,), generator=generator))

    for i in range(k):
        selected[i] = idx
        #squrared distance from every point to the newly selected one 
        dist = (sq_norm - 2.0 * (reduced @ reduced[idx]) + sq_norm[idx]).clamp_min(0)
        min_dist = torch.minimum(min_dist, dist)
        idx = int(torch.argmax(min_dist))

    return selected
