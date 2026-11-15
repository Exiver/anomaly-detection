from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config= SettingsConfigDict(
        env_file=".env",
        env_prefix="AD_",
        extra="ignore"
    )
    data_root : Path = Path("data/mvtec_ad")
    artifacts_dir : Path = Path("artifacts")
    image_size : int = 256
    crop_size : int = 224
    seed : int = 42
    log_level : str = "INFO"
    json_logs : bool = False

    batch_size : int = 16
    epochs : int = 100
    lr : float = 1e-3
    latent_channels : int = 64
    val_fraction : float = 0.1
    patience : int = 10
    num_workers : int = 0

    #scoring
    blur_sigma : float = 4.0
    embed_dim : int = 1024
    coreset_ratio : float = 0.1
    nn_chunk: int = 2048
    pretrained_backbone : bool = True

@lru_cache
def get_settings() -> Settings:
    return Settings()
