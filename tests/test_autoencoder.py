import torch

from anomaly.models.autoencoder import ConvAutoencoder


def test_output_shape_matches_input():
    model = ConvAutoencoder(latent_channels=8)
    x = torch.randn(2, 3, 64, 64)
    assert model(x).shape == x.shape


def test_bottleneck_shape():
    model = ConvAutoencoder(latent_channels=8)
    x = torch.randn(2, 3, 64, 64)
    assert model.encoder(x).shape == (2, 8, 2, 2)