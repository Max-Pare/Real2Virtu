import pytest
import torch

from r2v.stages.splat import visible_mask


def test_visible_mask_needs_both_radii_and_any_camera():
    # [C=2, N=3, 2]: Gaussian 0 drawn by camera 0, 1 by none (one radius 0), 2 by camera 1.
    radii = torch.tensor([[[3, 2], [4, 0], [0, 0]],
                          [[0, 0], [0, 0], [1, 1]]])
    assert visible_mask(radii).tolist() == [True, False, True]


def test_visible_mask_accepts_one_radius_per_gaussian():
    assert visible_mask(torch.tensor([[0, 5, 0]])).tolist() == [False, True, False]


def test_exposure_gain_is_relative_to_the_average_frame():
    from r2v.stages.splat import exposure_gain

    table = torch.tensor([[0.2, 0.0, 0.0], [0.0, 0.0, 0.0], [-0.2, 0.0, 0.3]], requires_grad=True)
    # Adding the same amount to every frame changes nothing.
    shifted = table + 5.0
    assert torch.allclose(exposure_gain(table, 0), exposure_gain(shifted, 0), atol=1e-5)
    assert torch.allclose(exposure_gain(table, 1), torch.tensor([1.0, 1.0, 0.9048]), atol=1e-4)
    # Only the frame in use receives a gradient.
    exposure_gain(table, 2).sum().backward()
    assert table.grad[:2].abs().sum() == 0 and table.grad[2].abs().sum() > 0


def test_loss_without_convolutions_matches_the_convolution_one():
    from r2v.stages.splat import _photo_loss_conv, _photo_loss_shifted

    g = torch.Generator().manual_seed(0)
    gt = torch.rand(37, 52, 3, generator=g)
    img = (gt + 0.2 * torch.randn(37, 52, 3, generator=g)).requires_grad_()
    want = _photo_loss_conv(img, gt, 0.2)
    got = _photo_loss_shifted(img, gt, 0.2)
    assert abs(float(want) - float(got)) < 1e-6
    g_want, = torch.autograd.grad(want, img)
    g_got, = torch.autograd.grad(got, img)
    assert (g_want - g_got).abs().max() < 1e-7 + 1e-4 * g_want.abs().max()


def test_rocm_loss_on_the_gpu_matches_the_cpu_reference():
    """The compiled loss the AMD path trains with, against the plain one on the CPU."""
    from r2v import device
    from r2v.stages.splat import _photo_loss_conv, make_photo_loss

    if device.gpu_backend() != "rocm" or not torch.cuda.is_available():
        pytest.skip("AMD ROCm only")
    g = torch.Generator().manual_seed(1)
    gt = torch.rand(120, 160, 3, generator=g)
    img = (gt + 0.2 * torch.randn(120, 160, 3, generator=g)).requires_grad_()
    want = _photo_loss_conv(img, gt, 0.2)
    g_want, = torch.autograd.grad(want, img)
    loss = make_photo_loss("rocm")
    img_gpu = img.detach().cuda().requires_grad_()
    for _ in range(2):   # second call: the compiled kernels, not the compile itself
        got = loss(img_gpu, gt.cuda(), 0.2)
        g_got, = torch.autograd.grad(got, img_gpu)
    assert abs(float(want) - float(got)) < 1e-5
    assert (g_want - g_got.cpu()).abs().max() < 1e-7 + 1e-3 * g_want.abs().max()
