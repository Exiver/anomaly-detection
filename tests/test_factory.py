import pytest
import torch

from anomaly.config import Settings
from anomaly.factory import build_scorer


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        build_scorer("banana", "bottle", Settings(), torch.device("cpu"))


def test_patchcore_requires_extractor():
    with pytest.raises(ValueError):
        build_scorer("patchcore", "bottle", Settings(), torch.device("cpu"), extractor=None)