from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from anomaly.models.autoencoder import ConvAutoencoder


def gaussian_kernel_1d(sigma: float) -> torch.Tensor:
    radius = max(1, int(4 * sigma + 0.5))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kerenel = torch.exp(-(x **2) / (2 * sigma*2))
    return kerenel / kerenel.sum()

def gaussian_blur(maps: torch.Tensor, sigma: float) -> torch.Tensor:
    """maps: (N, H, W) -> (N, H, W)."""
    if sigma <= 0:
        return maps
    kernel = gaussian_kernel_1d(sigma).to(device=maps.device, dtype=maps.dtype)
    radius = kernel.numel() // 2
    x = maps.unsqueeze(1)
    x = F.pad(x, (radius, radius, 0, 0), mode="reflect")
    x = F.conv2d(x, kernel.view(1, 1, 1, -1))
    x = F.pad(x, (0, 0, radius, radius), mode="reflect")
    x = F.conv2d(x, kernel.view(1, 1, -1, 1))
    return x.squeeze(1)


def reconstruction_error_map(
        images: torch.Tensor, reconstructions: torch.Tensor, sigma: float = 4.0) -> torch.tensor:
    """per- pixel squared error averaged over channels, than blured. -> (N, H, W)."""
    error = (images - reconstructions).pow(2).mean(dim=1)
    return gaussian_blur(error, sigma)

def load_autoencoder(checkpoint: Path, latent_channels: int, device: torch.device) -> ConvAutoencoder:
    model = ConvAutoencoder(latent_channels).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.eval()

def make_ae_scorer(model: ConvAutoencoder, sigma: float) -> Callable[[torch.Tensor], torch.Tensor]:
    model.eval()
    @torch.no_grad()
    def scorer(images: torch.Tensor) -> torch.Tensor:
        return reconstruction_error_map(images, model(images), sigma)

    return scorer
