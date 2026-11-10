import numpy as np

from r2v.config import AlignCfg
from r2v.stages import align


def _room(rng, w=4.0, h=2.7, d=5.0, n=4000):
    """Points on the six faces of a box room; floor at y=0, Y up, metres."""
    pts = []
    for _ in range(n):
        face = rng.integers(6)
        p = rng.uniform([0, 0, 0], [w, h, d])
        axis, side = divmod(face, 2)
        p[axis] = [w, h, d][axis] * side
        pts.append(p)
    return np.array(pts) - [w / 2, 0, d / 2]


def _cameras(rng, n=40):
    """Upright cameras at about 1.5 m, OpenCV axes (x right, y down, z forward)."""
    out = []
    for _ in range(n):
        yaw = rng.uniform(0, 2 * np.pi)
        fwd = np.array([np.sin(yaw), 0, np.cos(yaw)])
        down = np.array([0.0, -1.0, 0.0])
        right = np.cross(down, fwd)
        c2w = np.eye(4)
        c2w[:3, :3] = np.stack([right, down, fwd], axis=1)
        c2w[:3, 3] = [rng.uniform(-1, 1), 1.5 + rng.uniform(-0.05, 0.05), rng.uniform(-1, 1)]
        out.append(c2w)
    return np.array(out)


def _random_rotation(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q if np.linalg.det(q) > 0 else -q


def _to_colmap_frame(rng, pts, cams, s0=0.37):
    Q, t0 = _random_rotation(rng), rng.normal(size=3) * 3
    T = {"scale": s0, "R": Q, "t": t0}
    return align.apply_points(T, pts), align.apply_cameras(T, cams)


def test_rotation_to_y():
    rng = np.random.default_rng(0)
    for _ in range(20):
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        R = align.rotation_to_y(u)
        assert np.allclose(R @ u, [0, 1, 0], atol=1e-9)
        assert np.isclose(np.linalg.det(R), 1.0)
    assert np.allclose(align.rotation_to_y(np.array([0, -1.0, 0])) @ [0, -1, 0], [0, 1, 0])


def test_recovers_metric_upright_room():
    rng = np.random.default_rng(1)
    pts, cams = _room(rng), _cameras(rng)
    pts_c, cams_c = _to_colmap_frame(rng, pts, cams)

    T = align.solve(cams_c, pts_c, AlignCfg(ceiling_height=2.7))
    out = align.apply_points(T, pts_c)
    cam_out = align.apply_cameras(T, cams_c)

    y = out[:, 1]
    assert abs(np.percentile(y, 2)) < 0.03          # floor at zero
    assert abs(np.percentile(y, 98) - 2.7) < 0.03   # ceiling at 2.7 m
    assert np.allclose(cam_out[:, 1, 3], 1.5, atol=0.1)  # cameras at eye height
    # Walls line up with the axes: footprint is 4 x 5 in some order.
    extent = np.sort(np.ptp(out[:, [0, 2]], axis=0))
    assert np.allclose(extent, [4.0, 5.0], atol=0.1)
    # Cameras stay upright: camera-down points along -Y.
    assert np.allclose(cam_out[:, :3, 1], [0, -1, 0], atol=0.02)
    assert not T["scale_assumed"]


def test_default_scale_is_flagged_and_overrides_win():
    rng = np.random.default_rng(2)
    pts_c, cams_c = _to_colmap_frame(rng, _room(rng), _cameras(rng))
    T = align.solve(cams_c, pts_c, AlignCfg())  # default: camera held 1.5 m up
    assert T["scale_assumed"]
    assert abs(T["room_height_m"] - 2.7) < 0.1  # test cameras really are at 1.5 m
    assert align.solve(cams_c, pts_c, AlignCfg(scale=2.0))["scale"] == 2.0
    T = align.solve(cams_c, pts_c, AlignCfg(camera_height=3.0))
    assert abs(T["camera_height_m"] - 3.0) < 1e-6 and not T["scale_assumed"]


def _pitched_cameras(rng, n=40, pitch_deg=35.0, yaw_range_deg=20.0):
    """Cameras pointing down, turning only a little: the case that fooled mean camera-up."""
    out = []
    p = np.radians(pitch_deg)
    for _ in range(n):
        yaw = np.radians(rng.uniform(-yaw_range_deg, yaw_range_deg))
        level_fwd = np.array([np.sin(yaw), 0, np.cos(yaw)])
        right = np.cross([0, -1.0, 0], level_fwd)  # level: no roll
        fwd = np.cos(p) * level_fwd - np.sin(p) * np.array([0, 1.0, 0])
        down = np.cross(fwd, right)
        c2w = np.eye(4)
        c2w[:3, :3] = np.stack([right, down, fwd], axis=1)
        c2w[:3, 3] = [rng.uniform(-0.5, 0.5), 1.5, rng.uniform(-1.5, -1.0)]
        out.append(c2w)
    return np.array(out)


def test_up_survives_camera_pointing_down():
    rng = np.random.default_rng(3)
    cams = _pitched_cameras(rng)
    assert np.allclose(np.linalg.det(cams[:, :3, :3]), 1.0)
    y = np.array([0, 1.0, 0])
    naive = -cams[:, :3, 1].mean(axis=0)
    naive /= np.linalg.norm(naive)
    assert np.degrees(np.arccos(naive @ y)) > 25          # the old estimate is far off
    assert np.degrees(np.arccos(align.estimate_up(cams) @ y)) < 5

    pts_c, cams_c = _to_colmap_frame(rng, _room(rng), cams)
    T = align.solve(cams_c, pts_c, AlignCfg())
    out = align.apply_points(T, pts_c)
    assert abs(np.percentile(out[:, 1], 2)) < 0.05
    assert abs(np.percentile(out[:, 1], 98) - 2.7) < 0.08   # level and correctly scaled


def test_floor_ignores_points_below_it():
    rng = np.random.default_rng(4)
    pts, cams = _room(rng), _cameras(rng)
    junk = rng.uniform([-2, -1.5, -2.5], [2, -0.3, 2.5], size=(150, 3))  # reflections under the floor
    pts_c, cams_c = _to_colmap_frame(rng, np.vstack([pts, junk]), cams)
    T = align.solve(cams_c, pts_c, AlignCfg(camera_height=1.5))
    out = align.apply_points(T, pts_c)[: len(pts)]
    assert abs(np.median(out[np.abs(out[:, 1]) < 0.2, 1])) < 0.03
