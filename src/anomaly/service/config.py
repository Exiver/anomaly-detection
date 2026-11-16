from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServiceSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="AD_", extra="ignore")

    release_dir: Path = Path("releases/v1")
    preload_categories: str = ""  # comma-separated; empty means every category in the release
    device: Literal["auto", "cpu", "cuda"] = "auto"
    nn_chunk: int = 2048
    pretrained_backbone: bool = True

    max_upload_bytes: int = 5_000_000
    max_image_pixels: int = 25_000_000
    max_concurrency: int = Field(default=1, ge=1)
    queue_timeout_seconds: float = 5.0

    api_key: SecretStr | None = None

    metrics_enabled: bool = True
    demo_enabled: bool = True
    drift_window: int = Field(default=200, ge=5)
    drift_min_samples: int = Field(default=30, ge=2)

    log_level: str = "INFO"
    json_logs: bool = False

    def categories(self) -> list[str]:
        return [c.strip() for c in self.preload_categories.split(",") if c.strip()]

    cors_origins: str = ""  # comma-separated exact origins; empty disables CORS

    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]