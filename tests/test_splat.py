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
