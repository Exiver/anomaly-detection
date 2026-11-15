import torch

from anomaly.scoring import gaussian_blur, gaussian_kernel_1d, reconstruction_error_map


def test_kernel_sums_to_one():
    assert abs(gaussian_kernel_1d(2.0).sum().item() - 1.0) < 1e-6


def test_blur_keeps_shape():
    maps = torch.rand(2, 28, 28)
    assert gaussian_blur(maps, sigma=1.0).shape == maps.shape


def test_blur_of_constant_map_is_constant():
    maps = torch.full((1, 28, 28), 3.0)
    out = gaussian_blur(maps, sigma=1.5)
    assert torch.allclose(out, maps, atol=1e-5)


def test_error_map_zero_for_identical_images():
    x = torch.rand(2, 3, 28, 28)
    out = reconstruction_error_map(x, x.clone(), sigma=1.0)
    assert out.shape == (2, 28, 28)
    assert out.abs().max().item() == 0.0