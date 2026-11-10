"""Splat cleanup filters and the gap backdrop written to scene.json."""
import numpy as np
import pytest

from r2v import manifest
from r2v.config import CleanCfg
from r2v.stages import clean


def test_haze_filters_keep_the_room_and_far_scenery():
    rng = np.random.default_rng(0)
    n = 4000
    means = rng.uniform([-2, 0, -2], [2, 0.02, 2], size=(n, 3))  # a densely covered floor
    sigma = np.full((n, 3), 0.02)
    opacity = np.full(n, 0.8)
    cams = np.array([[0.0, 1.5, 0.0], [0.5, 1.5, 0.5]])

    opacity[0] = 0.005                      # haze
    sigma[1] = [0.02, 0.02, 0.6]            # blob inside the room
    # Large, but far away: the view through a window, made of several Gaussians.
    means[2], sigma[2] = [0, 5, 80], 6.0
    means[6:26] = [0, 5, 80] + rng.uniform(-8, 8, size=(20, 3))
    sigma[6:26] = 6.0
    sigma[3] = [0.005, 0.005, 0.4]          # needle
    means[4] = [0, 9, 0]                    # alone, far from everything
    sigma[5] = [0.01, 0.01, 0.15]           # thin but short: a cable, keep

    r = clean.haze_mask(means, sigma, opacity, cams, CleanCfg())
    assert list(np.flatnonzero(r["transparent"])) == [0]
    assert r["oversized"][1] and not r["oversized"][2]
    assert r["needle"][3] and not r["needle"][5]
    assert r["isolated"][4] and not r["isolated"][2]
    removed = r["transparent"] | r["oversized"] | r["needle"] | r["isolated"]
    assert removed[[0, 1, 3, 4]].all() and not removed[6:26].any()
    assert removed.sum() <= 6  # the room itself stays


def test_free_space_needs_agreeing_views_and_spares_edges():
    torch = pytest.importorskip("torch")
    K = torch.tensor([[40.0, 0, 16], [0, 40.0, 12], [0, 0, 1]])
    depth = torch.full((24, 32), 2.0)       # a wall 2 m ahead
    depth[:, 20:] = 0.6                     # a near object on the right
    views = []
    for x in (-0.05, 0.0, 0.05, 0.1):
        c2w = torch.eye(4)
        c2w[0, 3] = x
        views.append(c2w)
    c2ws = torch.stack(views)
    pts = torch.tensor([
        [0.0, 0.0, 1.0],    # floating in front of the wall
        [0.0, 0.0, 2.0],    # on the wall
        [0.0, 0.0, 3.0],    # behind the wall: hidden, not judged
        [0.16, 0.0, 1.0],   # at the object's edge: its window holds the near depth, so it is spared
    ])
    args = dict(margin=0.12, margin_rel=0.06, min_share=0.8)
    mask = clean.free_space_mask(pts, [depth] * 4, K.expand(4, 3, 3), c2ws, min_views=3, **args)
    assert mask.tolist() == [True, False, False, False]
    # Too few views to be sure.
    mask = clean.free_space_mask(pts, [depth] * 2, K.expand(2, 3, 3), c2ws[:2], min_views=3, **args)
    assert not mask.any()


def test_hole_must_be_reported_from_several_views():
    pts = np.array([[0.0, 0, 0], [0.005, 0, 0], [0.004, 0.001, 0], [1.0, 0, 0], [1.0, 0, 0.001]])
    view = np.array([0, 1, 2, 5, 5])
    assert clean.seen_by(pts, view, 0.02, 3).tolist() == [True, True, True, False, False]


def test_backdrop_colours_and_floor_extent():
    rgba = np.array([[200, 100, 0, 255], [0, 0, 0, 0]], dtype=np.uint8)  # second one is invisible
    verts = np.concatenate([np.random.default_rng(0).uniform([-3, 0, -2], [3, 0.05, 2], (100, 3)),
                            [[0, 2.0, 0]]])
    cols = np.tile(np.array([120, 60, 30, 255], dtype=np.uint8), (101, 1))
    cols[-1] = [255, 255, 255, 255]
    b = manifest.backdrop(rgba, verts, cols)
    assert np.allclose(b["background"], [200 / 255, 100 / 255, 0], atol=1e-6)
    assert np.allclose(b["floor"]["color"], [120 / 255, 60 / 255, 30 / 255])
    assert b["mesh"] and b["floor"]["min"][0] < -4.9 and b["floor"]["max"][1] > 3.9
    # A mesh without colours: no mesh backdrop, floor falls back to the background colour.
    b = manifest.backdrop(rgba, verts, None)
    assert not b["mesh"] and b["floor"]["color"] == b["background"]
