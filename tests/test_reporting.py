from datetime import datetime, timezone

import pandas as pd
import pytest

from anomaly.manifest import CategoryEntry, ReleaseManifest
from anomaly.reporting import (
    BEGIN,
    END,
    build_results,
    model_card_auto,
    replace_between_markers,
    to_markdown,
)


def test_to_markdown_shape():
    markdown = to_markdown(pd.DataFrame({"a": [1.23456, 2.0], "b": ["x", "y"]}), digits=2)
    lines = markdown.splitlines()
    assert lines[0] == "| a | b |"
    assert lines[1] == "|---|---|"
    assert lines[2] == "| 1.23 | x |"


def test_replace_keeps_hand_written_text():
    text = f"intro\n{BEGIN}\nold\n{END}\noutro"
    updated = replace_between_markers(text, "new")
    assert "old" not in updated and "new" in updated
    assert updated.startswith("intro") and updated.endswith("outro")


def test_replace_requires_markers():
    with pytest.raises(ValueError):
        replace_between_markers("no markers here", "x")


def test_results_report_missing_files(tmp_path):
    assert "Not run" in build_results(tmp_path)


def _evaluation_csv(path, auroc):
    pd.DataFrame(
        [
            {
                "category": "bottle",
                "image_auroc": auroc,
                "ci_low": auroc - 0.05,
                "ci_high": min(1.0, auroc + 0.05),
                "pixel_auroc": 0.9,
                "pro": 0.8,
                "n_test": 83,
                "n_anomalous": 63,
            }
        ]
    ).to_csv(path, index=False)


def test_results_with_both_methods(tmp_path):
    _evaluation_csv(tmp_path / "evaluation_ae.csv", 0.60)
    _evaluation_csv(tmp_path / "evaluation_patchcore.csv", 0.99)
    text = build_results(tmp_path)
    assert "0.990 (0.940-1.000)" in text
    assert "intervals overlap in 0 of 1" in text


def test_model_card_block_contains_manifest_values(tmp_path):
    path = tmp_path / "manifest.json"
    ReleaseManifest(
        version="v9",
        created_at=datetime.now(timezone.utc).isoformat(),
        image_size=256,
        crop_size=224,
        embed_dim=1024,
        coreset_ratio=0.1,
        blur_sigma=4.0,
        seed=42,
        threshold_rule="max",
        threshold_k=None,
        holdout_fraction=0.1,
        categories={
            "bottle": CategoryEntry(
                threshold=1.5,
                holdout_images=20,
                bank_size=1000,
                test_image_auroc=0.99,
                test_pixel_auroc=0.98,
                test_recall_at_threshold=0.9,
                test_fpr_at_threshold=0.05,
            )
        },
    ).write(path)
    block = model_card_auto(path)
    assert "v9" in block and "bottle" in block and "max" in block
