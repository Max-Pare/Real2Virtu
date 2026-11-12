"""LiDAR path: Stray Scanner reader, known-up alignment, depth loss, stage selection."""
import cv2
import numpy as np
import pytest

from r2v import cli
from r2v.config import AlignCfg, Config
from r2v.stages import align, ingest
from r2v.workspace import Workspace

ROOM_LO = np.array([-2.0, -1.2, -2.5])  # floor 1.2 m below the ARKit origin
ROOM_HI = np.array([2.0, 1.4, 2.5])
VIDEO = (64, 48)   # w, h
DEPTH = (32, 24)
K_VIDEO = np.array([[50.0, 0, 32], [0, 50.0, 24], [0, 0, 1]])


def _pose(yaw_deg):
    """Level camera at the origin, OpenCV axes, world +Y up."""
    s, c = np.sin(np.radians(yaw_deg)), np.cos(np.radians(yaw_deg))
    c2w = np.eye(4)
    c2w[:3, :3] = np.array([[-c, 0, s], [0, -1, 0], [s, 0, c]]).T  # rows: camera x, y, z
    return c2w


def _room_depth(c2w, K, size):
    """z-depth of the box room's inside, seen through a pinhole camera."""
    w, h = size
    v, u = np.mgrid[0:h, 0:w]
    rays = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0], (v + 0.5 - K[1, 2]) / K[1, 1], np.ones((h, w))], -1)
    d = rays @ c2w[:3, :3].T
    o = c2w[:3, 3]
    with np.errstate(divide="ignore"):
        t = np.where(d > 0, (ROOM_HI - o) / d, (ROOM_LO - o) / d)
    return t.min(axis=-1)  # rays have z = 1 in the camera, so t is the z-depth


def _write_scan(folder, n=90, per_frame_k=True, yaw_step=4.0):
    folder.mkdir()
    (folder / "depth").mkdir()
    (folder / "confidence").mkdir()
    rng = np.random.default_rng(0)
    video = cv2.VideoWriter(str(folder / "rgb.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, VIDEO)
    assert video.isOpened()
    head = "timestamp, frame, x, y, z, qx, qy, qz, qw"
    lines = [head + (", fx, fy, cx, cy" if per_frame_k else "")]
    for i in range(n):
        video.write(rng.integers(0, 255, (VIDEO[1], VIDEO[0], 3), dtype=np.uint8))
        c2w = _pose(i * yaw_step)
        k = K_VIDEO.copy()
        if per_frame_k:
            k[0, 0] = k[1, 1] = 50 + i * 0.01  # focus breathing
        k_depth = ingest.scale_K(k, DEPTH[0] / VIDEO[0], DEPTH[1] / VIDEO[1])
        mm = (_room_depth(c2w, k_depth, DEPTH) * 1000).round().astype(np.uint16)
        cv2.imwrite(str(folder / "depth" / f"{i:06d}.png"), mm)
        conf = np.full(DEPTH[::-1], 2, dtype=np.uint8)
        conf[:, :4] = 0  # an untrusted strip on the left
        cv2.imwrite(str(folder / "confidence" / f"{i:06d}.png"), conf)
        from scipy.spatial.transform import Rotation
        q = Rotation.from_matrix(c2w[:3, :3]).as_quat()
        row = f"{i / 30:.4f}, {i:06d}, 0, 0, 0, {q[0]}, {q[1]}, {q[2]}, {q[3]}"
        lines.append(row + (f", {50 + i * 0.01}, {50 + i * 0.01}, 32, 24" if per_frame_k else ""))
    video.release()
    (folder / "odometry.csv").write_text("\n".join(lines) + "\n")
    np.savetxt(folder / "camera_matrix.csv", K_VIDEO, delimiter=", ")
    return folder


def _ingest(tmp_path, **kw):
    scan = _write_scan(tmp_path / "scan", **kw)
    ws = Workspace(tmp_path / "ws")
    cfg = Config(source="lidar")
    cfg.ingest.scan = str(scan)
    cfg.ingest.fps = 10.0  # 30 fps video, 3 candidates per kept frame
    ingest.run(ws, cfg)
    return ws, cfg, ingest.load_model(ws)


def _wall_distance(p):
    return np.minimum(np.abs(p - ROOM_LO), np.abs(p - ROOM_HI)).min(axis=1)


def test_ingest_reads_scan(tmp_path):
    ws, _, m = _ingest(tmp_path)
    assert len(m["names"]) == 30 and len(m["depth_names"]) == 30
    assert tuple(m["sizes"][0]) == VIDEO
    # Per-frame intrinsics from odometry.csv win over camera_matrix.csv.
    frame = int(str(m["names"][3])[:6])
    assert np.isclose(m["K"][3][0, 0], 50 + frame * 0.01)
    # Start points sit on the room's surfaces, in ARKit's metric frame.
    assert len(m["points"]) > 1000
    assert _wall_distance(m["points"]).max() < 0.01
    assert m["colors"].dtype == np.uint8 and m["colors"].shape == m["points"].shape
    # Low-confidence pixels are blanked in the stored depth maps.
    d = cv2.imread(str(ingest.depth_maps(ws) / str(m["depth_names"][0])), cv2.IMREAD_UNCHANGED)
    assert d.dtype == np.uint16 and (d[:, :4] == 0).all() and (d[:, 4:] > 0).all()
    assert (ingest.images(ws) / str(m["names"][0])).exists()


def test_ingest_falls_back_to_camera_matrix(tmp_path):
    _, _, m = _ingest(tmp_path, per_frame_k=False)
    assert np.allclose(m["K"], K_VIDEO)


def test_still_camera_frames_are_dropped(tmp_path):
    with pytest.raises(SystemExit, match="only 1 frames"):
        _ingest(tmp_path, yaw_step=0.0)
    a, b = _pose(0), _pose(1.0)
    assert not ingest.moved(a, b, 0.02, 2.0)
    assert ingest.moved(a, _pose(3.0), 0.02, 2.0)
    b[:3, 3] = [0.03, 0, 0]
    assert ingest.moved(a, b, 0.02, 2.0)


def test_align_keeps_metric_scale_and_known_up(tmp_path):
    ws, cfg, m = _ingest(tmp_path)
    T = align.solve(m["c2w"], m["points"], AlignCfg(scale=1.0), up=ingest.WORLD_UP)
    assert T["scale"] == 1.0 and not T["scale_assumed"]
    assert np.allclose(T["R"] @ [0, 1, 0], [0, 1, 0])
    assert abs(T["camera_height_m"] - 1.2) < 0.05

    cfg.align.scale = 1.0
    align.run(ws, cfg)
    scene = dict(np.load(ws.dir("align") / "scene.npz", allow_pickle=False))
    assert str(scene["source"]) == "lidar" and float(scene["depth_scale"]) == 1.0
    assert list(scene["depth_names"]) == list(m["depth_names"])
    assert abs(scene["points"][:, 1].min()) < 0.05  # floor at y = 0
    assert abs(np.median(scene["c2w"][:, 1, 3]) - 1.2) < 0.05


def test_depth_loss_and_pose_delta():
    torch = pytest.importorskip("torch")
    from r2v.stages import splat

    target = torch.full((24, 32), 2.0)
    target[:, :8] = 0.0  # no measurement there
    rendered = torch.full((48, 64), 2.5)
    rendered[:, :16] = 9.0  # wrong only where the sensor has no data
    assert torch.isclose(splat.depth_loss(rendered, target), torch.tensor(0.5))
    assert splat.depth_loss(rendered, torch.zeros(24, 32)) == 0

    assert torch.allclose(splat.pose_delta_matrix(torch.zeros(6)), torch.eye(4))
    m = splat.pose_delta_matrix(torch.tensor([0.0, 0.0, torch.pi / 2, 1.0, 0.0, 0.0]))
    assert torch.allclose(m[:3, :3], torch.tensor([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]]), atol=1e-5)
    assert torch.allclose(m[:3, :3].T @ m[:3, :3], torch.eye(3), atol=1e-5)


def test_auto_size_shrinks_budget_for_small_scans():
    from r2v.stages import splat

    assert splat.auto_size(20_000, 1_500_000, 66, 196_944) == (9_900, 590_832)
    assert splat.auto_size(20_000, 1_500_000, 10, 1_000) == (6_000, 300_000)      # floors
    assert splat.auto_size(20_000, 1_500_000, 900, 400_000) == (20_000, 1_200_000)  # limits hold
    assert splat.auto_size(20_000, 1_500_000, 900, 900_000) == (20_000, 1_500_000)


def test_train_scale_is_a_percentage_of_the_source():
    from r2v.stages import splat

    assert Config().splat.train_scale == 67.0
    assert splat.scaled_size(1600, 1200, 67) == (1072, 804)
    assert splat.scaled_size(1600, 1200, 50) == (800, 600)
    assert splat.scaled_size(1600, 1200, 100) == (1600, 1200)
    assert splat.scaled_size(1600, 1200, 150) == (1600, 1200)  # never upscales


def test_stage_list_follows_source():
    assert cli.stages_for("lidar") == ["ingest", "align", "splat", "clean", "collision", "export"]
    assert cli.stages_for("video")[3:] == ["splat", "clean", "collision", "export"]
    assert cli.stages_for("video")[:2] == ["frames", "poses"]


def test_slot_length_covers_every_video_frame_between_kept_frames():
    from r2v.stages.ingest import slot_length

    assert slot_length(60.0, 2.0) == 30
    assert slot_length(60.0, 3.0) == 20
    assert slot_length(30.0, 60.0) == 1
