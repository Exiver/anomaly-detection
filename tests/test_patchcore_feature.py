import torch

from anomaly.models. patchcore_features import FeatureExtractor, build_patch_embedding


def test_feature_map_shapes():
    extractor = FeatureExtractor(pretrained=False)
    f2, f3 = extractor(torch.randn(2, 3, 64, 64))
    assert f2.shape == (2, 512, 8, 8)
    assert f3.shape == (2, 1024, 4, 4)

def test_patch_embedding_shape():
    extractor = FeatureExtractor(pretrained=False)
    f2, f3 = extractor(torch.randn(2, 3, 64, 64))
    emb = build_patch_embedding(f2, f3, embed_dim=64)
    assert emb.shape == (2, 64, 8, 8)

def test_extractor_is_frozen_and_stays_in_eval():
    extractor = FeatureExtractor(pretrained=False)
    assert all(not p.requires_grad for p in extractor.parameters())
    extractor.train()
    assert not extractor.training