import torch

from anomaly.config import Settings
from anomaly.scoring import load_autoencoder
from anomaly.training.autoencoder import train_autoencoder


def test_training_writes_loadable_checkpoint(fake_dataset_root, tmp_path):
    settings = Settings(
        data_root=fake_dataset_root,
        artifacts_dir=tmp_path / "artifacts_dir",
        image_size=64,
        crop_size=64,
        epochs=2,
        batch_size=2,
        latent_channels=8,
        val_fraction=0.25,
        num_workers=0,
    )
    device = torch.device("cpu")
    checkpoint = train_autoencoder("bottle", settings, device)

    assert checkpoint.is_file()
    model = load_autoencoder(checkpoint, settings.latent_channels, device)
    x = torch.randn(1, 3, 64, 64)
    assert model(x).shape == x.shape