import torch

from r2v.stages.splat import visible_mask


def test_visible_mask_needs_both_radii_and_any_camera():
    # [C=2, N=3, 2]: Gaussian 0 drawn by camera 0, 1 by none (one radius 0), 2 by camera 1.
    radii = torch.tensor([[[3, 2], [4, 0], [0, 0]],
                          [[0, 0], [0, 0], [1, 1]]])
    assert visible_mask(radii).tolist() == [True, False, True]


def test_visible_mask_accepts_one_radius_per_gaussian():
    assert visible_mask(torch.tensor([[0, 5, 0]])).tolist() == [False, True, False]
