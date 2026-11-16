import base64
import io
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image
from torch.utils.data import DataLoader

from anomaly.data.mvtec import MVTecDataset
from anomaly.data.transforms import build_image_transform
from anomaly.manifest import CategoryEntry, ReleaseManifest
from anomaly.models.patchcore import PatchCore
from anomaly.models.patchcore_features import FeatureExtractor
from anomaly.service.app import create_app
from anomaly.service.config import ServiceSettings
from anomaly.service.registry import ModelRegistry

CPU = torch.device("cpu")


def png_bytes(value: int, size: int = 64) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.full((size, size, 3), value, dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def release(fake_dataset_root, tmp_path):
    torch.manual_seed(0)
    extractor = FeatureExtractor(pretrained=False)
    model = PatchCore(extractor, CPU, embed_dim=64, coreset_ratio=0.5, blur_sigma=1.0, seed=0)
    train = MVTecDataset(fake_dataset_root, "bottle", "train", image_size=64, crop_size=64)
    model.fit(DataLoader(train, batch_size=2))

    white = build_image_transform(64, 64)(Image.open(io.BytesIO(png_bytes(255))).convert("RGB"))
    defect_score = float(model.anomaly_map(white.unsqueeze(0)).max())

    release_dir = tmp_path / "release"
    release_dir.mkdir()
    model.save(release_dir / "bottle.pt")
    ReleaseManifest(
        version="v-test",
        created_at=datetime.now(timezone.utc).isoformat(),
        image_size=64, crop_size=64, embed_dim=64, coreset_ratio=0.5, blur_sigma=1.0, seed=0,
        threshold_rule="max", threshold_k=None, holdout_fraction=0.1,
        categories={
            "bottle": CategoryEntry(
                threshold=defect_score / 2, holdout_images=1, bank_size=int(model.bank.shape[0]),
                test_image_auroc=1.0, test_pixel_auroc=1.0,
                test_recall_at_threshold=1.0, test_fpr_at_threshold=0.0,
                reference_score_ratios=[0.05] * 10,
                reference_input_mean=0.0,
                reference_input_mean_std=0.1,
            )
        },
    ).write(release_dir / "manifest.json")
    return release_dir, extractor


def make_client(release, **overrides) -> TestClient:
    release_dir, extractor = release
    settings_kwargs = {
        "release_dir": release_dir,
        "pretrained_backbone": False,
        "device": "cpu",
        "api_key": None,
        **overrides,
    }
    settings = ServiceSettings(**settings_kwargs)
    return TestClient(create_app(settings, ModelRegistry(settings, extractor=extractor)))


def post(client, data: bytes, **params):
    params.setdefault("category", "bottle")
    return client.post("/v1/predict", params=params, files={"file": ("x.png", data, "image/png")})


def test_health_and_readiness(release):
    with make_client(release) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        ready = client.get("/readyz").json()
        assert ready["status"] == "ready"
        assert ready["categories"] == ["bottle"]
        assert ready["model_version"] == "v-test"


def test_normal_and_defective_images(release):
    with make_client(release) as client:
        normal = post(client, png_bytes(128))
        defect = post(client, png_bytes(255))
        assert normal.status_code == 200 and defect.status_code == 200
        assert normal.json()["is_anomalous"] is False
        assert defect.json()["is_anomalous"] is True
        assert defect.json()["anomaly_score"] > normal.json()["anomaly_score"]


def test_heatmap_is_a_valid_png(release):
    with make_client(release) as client:
        body = post(client, png_bytes(255), include_heatmap=True).json()
        png = base64.b64decode(body["heatmap_png_base64"])
        assert Image.open(io.BytesIO(png)).size == (64, 64)


def test_unknown_category_is_404(release):
    with make_client(release) as client:
        assert post(client, png_bytes(128), category="banana").status_code == 404


def test_garbage_upload_is_400(release):
    with make_client(release) as client:
        assert post(client, b"not an image").status_code == 400


def test_oversized_upload_is_413(release):
    with make_client(release, max_upload_bytes=10) as client:
        assert post(client, png_bytes(128)).status_code == 413


def test_api_key_is_enforced_when_configured(release):
    with make_client(release, api_key="s3cret") as client:
        assert post(client, png_bytes(128)).status_code == 401
        ok = client.post(
            "/v1/predict",
            params={"category": "bottle"},
            files={"file": ("x.png", png_bytes(128), "image/png")},
            headers={"X-API-Key": "s3cret"},
        )
        assert ok.status_code == 200
        assert client.get("/healthz").status_code == 200  # probes never need a key


def test_request_id_is_echoed(release):
    with make_client(release) as client:
        response = client.get("/healthz", headers={"X-Request-ID": "abc123"})
        assert response.headers["x-request-id"] == "abc123"


def test_models_endpoint(release):
    with make_client(release) as client:
        models = client.get("/v1/models").json()
        assert [m["category"] for m in models] == ["bottle"]


def test_service_does_not_import_training_dependencies():
    code = (
        "import sys, anomaly.service.app;"
        "bad = [m for m in ('sklearn', 'pandas', 'matplotlib') if m in sys.modules];"
        "assert not bad, bad"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

def test_metrics_count_predictions_and_requests(release):
    with make_client(release) as client:
        post(client, png_bytes(128))
        post(client, png_bytes(255))
        text = client.get("/metrics").text
        assert 'ad_predictions_total{category="bottle",outcome="normal"} 1.0' in text
        assert 'ad_predictions_total{category="bottle",outcome="anomalous"} 1.0' in text
        assert 'ad_http_requests_total{method="POST",path="/v1/predict",status="200"} 2.0' in text
        assert "ad_score_ratio_bucket" in text


def test_rejections_are_counted_by_reason(release):
    with make_client(release) as client:
        post(client, png_bytes(128), category="banana")
        post(client, b"not an image")
        text = client.get("/metrics").text
        assert 'ad_rejected_requests_total{reason="unknown_category"} 1.0' in text
        assert 'ad_rejected_requests_total{reason="invalid_image"} 1.0' in text


def test_metrics_need_the_key_and_accept_bearer(release):
    with make_client(release, api_key="s3cret") as client:
        assert client.get("/metrics").status_code == 401
        ok = client.get("/metrics", headers={"Authorization": "Bearer s3cret"})
        assert ok.status_code == 200


def test_metrics_can_be_disabled(release):
    with make_client(release, metrics_enabled=False) as client:
        assert client.get("/metrics").status_code == 404


def test_drift_gauges_appear_after_enough_samples(release):
    with make_client(release, drift_min_samples=3) as client:
        for _ in range(4):
            post(client, png_bytes(255))
        text = client.get("/metrics").text
        assert 'ad_window_anomaly_rate{category="bottle"} 1.0' in text
        assert 'ad_drift_ks_statistic{category="bottle"}' in text
        assert 'ad_input_mean_zscore{category="bottle"}' in text


def test_demo_page_is_served(release):
    with make_client(release) as client:
        response = client.get("/demo")
        assert response.status_code == 200
        assert "Visual anomaly detection" in response.text

def test_cors_preflight_allowed_for_configured_origin(release):
    with make_client(release, cors_origins="https://demo.example") as client:
        response = client.options(
            "/v1/predict",
            headers={
                "Origin": "https://demo.example",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "x-api-key",
            },
        )
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "https://demo.example"


def test_no_cors_headers_by_default(release):
    with make_client(release) as client:
        response = client.get("/healthz", headers={"Origin": "https://other.example"})
        assert "access-control-allow-origin" not in response.headers