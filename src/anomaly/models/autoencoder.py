import torch
from torch import nn


def _down(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )

def _up(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True)
    )

class ConvAutoencoder(nn.Module):
    """Import height and width must be divisible by 32."""

    def __init__(self, latent_channels: int = 64) ->None:
        super().__init__()
        self.encoder = nn.Sequential(
            _down(3, 32),
            _down(32, 64),
            _down(64, 128),
            _down(128, 256),
            _down(256, 256),
            nn.Conv2d(256, latent_channels, kernel_size=1)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 256, kernel_size=1),
            _up(256, 256),
            _up(256, 128),
            _up(128, 64),
            _up(64, 32),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    
