import torch

from anomaly.models.coreset import greedy_coreset
from anomaly.models.patchcore import nearest_distance

CPU = torch.device("cpu")


def test_coreset_size_and_uniqueness():
    feats = torch.randn(500, 32)
    idx = greedy_coreset(feats, 50, CPU, proj_dim=8, seed=0)
    assert idx.shape == (50,)
    assert len(set(idx.tolist())) == 50


def test_coreset_covers_both_clusters():
    a = torch.randn(100, 16) * 0.01
    b = torch.randn(100, 16) * 0.01 + 10.0
    feats = torch.cat([a, b])
    idx = greedy_coreset(feats, 2, CPU, proj_dim=8, seed=0)
    assert {int(i) // 100 for i in idx} == {0, 1}


def test_coreset_k_larger_than_n_is_clamped():
    feats = torch.randn(10, 8)
    assert greedy_coreset(feats, 50, CPU, proj_dim=4).shape == (10,)


def test_nearest_distance_zero_for_members():
    bank = torch.randn(20, 8)
    d = nearest_distance(bank[:5], bank, chunk=2)
    assert torch.allclose(d, torch.zeros(5), atol=1e-3)