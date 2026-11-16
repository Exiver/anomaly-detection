import threading

import structlog
import torch

from anomaly.data.transforms import build_image_transform
from anomaly.manifest import ReleaseManifest
from anomaly.models.patchcore import PatchCore
from anomaly.models.patchcore_features import FeatureExtractor
from anomaly.service.config import ServiceSettings
from anomaly.service.drifts import DriftMonitor
from anomaly.service.inferance import LoadedModel

log = structlog.get_logger()


class UnknownCategoryError(KeyError):
    """The requested category is not part of the loaded release."""


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("AD_DEVICE=cuda was requested but CUDA is not available.")
    return torch.device(choice)


class ModelRegistry:
    def __init__(self, settings: ServiceSettings, extractor: FeatureExtractor | None = None) -> None:
        self._settings = settings
        self._extractor = extractor
        self._models: dict[str, LoadedModel] = {}
        self.manifest: ReleaseManifest | None = None
        self.transform = None
        self.device = resolve_device(settings.device)
        self.semaphore = threading.BoundedSemaphore(settings.max_concurrency)
        self.ready = False
        self._monitors: dict[str, DriftMonitor] = {}

    @property
    def categories(self) -> list[str]:
        return sorted(self._models)

    def load(self) -> None:
        if self.ready:
            return
        release_dir = self._settings.release_dir
        manifest_path = release_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"No release manifest at {manifest_path}")

        manifest = ReleaseManifest.read(manifest_path)
        wanted = self._settings.categories() or list(manifest.categories)
        unknown = [c for c in wanted if c not in manifest.categories]
        if unknown:
            raise ValueError(f"Categories not in release {manifest.version}: {unknown}")

        extractor = self._extractor or FeatureExtractor(pretrained=self._settings.pretrained_backbone)
        for category in wanted:
            model = PatchCore.load(
                release_dir / f"{category}.pt", extractor, self.device, nn_chunk=self._settings.nn_chunk
            )
            self._models[category] = LoadedModel(
                category=category, model=model, threshold=manifest.categories[category].threshold
            )
            entry = manifest.categories[category]
            self._monitors[category] = DriftMonitor(
                entry.reference_score_ratios,
                entry.reference_input_mean,
                entry.reference_input_mean_std,
                window=self._settings.drift_window,
                min_samples=self._settings.drift_min_samples,
            )

        self.transform = build_image_transform(manifest.image_size, manifest.crop_size)
        self.manifest = manifest
        self.ready = True
        log.info(
            "models_loaded",
            version=manifest.version,
            categories=self.categories,
            device=str(self.device),
        )

    def get(self, category: str) -> LoadedModel:
        try:
            return self._models[category]
        except KeyError:
            raise UnknownCategoryError(category) from None

    def monitor(self, category: str) -> DriftMonitor:
        return self._monitors[category]
