import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2


class FeatureExtractor(nn.Module):
    """Frozen wideResNet50-2 that return the layer2 aand layer3 feature maps."""

    def __init__(self, pretrained: bool = True) ->None:
        super().__init__()
        weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1 if pretrained else None
        net = wide_resnet50_2(weights=weights)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3

        for param in self.parameters():
            param.requires_grad_(False)
        self.eval()


    def train(self, mode: bool = True) -> "FeatureExtractor":
            # this network is never trained, so BatchNorm statistics must stay frozen.
        return super().train(False)

    @torch.no_grad()
    def forward(self, x:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.layer1(x)
        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        return f2, f3

def _local_aggregate(feature: torch.Tensor) -> torch.Tensor:
    """Average each position with its 3x3 neighborhood."""
    return F.avg_pool2d(feature, kernel_size=3, stride=1, padding=1)

def _reduce_channels(features: torch.Tensor, out_dim: int) -> torch.Tensor:
    """ (N C, H, W) -> (N, OUT_DIM, H, W) by addaptive avarage pooling over channels"""
    n, c, h, w = features.shape
    flat = features.permute(0, 2, 3, 1).reshape(-1, 1, c)
    flat = F.adaptive_avg_pool1d(flat, out_dim)
    return flat.reshape(n, h, w, out_dim).permute(0, 3, 1,2)

def build_patch_embedding(
        f2: torch.Tensor, f3: torch.Tensor, embed_dim: int = 1024
) -> torch.Tensor:
    """ combine layer2 and layer3 into one (N, embed_dim, H, W) patch embedding."""
    f2 = _reduce_channels(_local_aggregate(f2), embed_dim)
    f3 = _reduce_channels(_local_aggregate(f3), embed_dim)

    f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
    merged = torch.cat([f2, f3], dim=1)
    return _reduce_channels(merged, embed_dim)
