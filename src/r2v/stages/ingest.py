"""LiDAR path, stage 1: read a Stray Scanner scan (iPhone / iPad Pro).

Replaces the frames and poses stages. The scan already holds what COLMAP has to
guess: camera poses in metres (ARKit), a gravity-aligned world, and depth maps.

Scan folder:
  rgb.mp4               video, one frame per odometry row
  depth/NNNNNN.png      uint16 millimetres, 256x192
  confidence/NNNNNN.png 0 (low) to 2 (high)
  odometry.csv          timestamp, frame, x, y, z, qx, qy, qz, qw[, fx, fy, cx, cy]
  camera_matrix.csv     3x3 intrinsics of the video; used when odometry has none

Poses are camera-to-world with OpenCV axes (x right, y down, z forward); the
world has +Y up. Pixel centres sit at half-integers, as in gsplat.
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from ..config import Config
from ..workspace import Workspace
from .frames import sharpness_of

WORLD_UP = np.array([0.0, 1.0, 0.0])
REQUIRED = ["rgb.mp4", "odometry.csv", "depth"]


def check_scan(scan: Path) -> None:
    missing = [n for n in REQUIRED if not (Path(scan) / n).exists()]
    if missing:
        raise SystemExit(f"ingest: {scan} is not a Stray Scanner scan (missing {', '.join(missing)})")


def read_odometry(scan: Path) -> dict:
    """Return {frame number: (c2w [4,4], K [3,3])} for the full-size video."""
    scan = Path(scan)
    with open(scan / "odometry.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    col = {name.strip(): i for i, name in enumerate(rows[0])}
    per_frame_k = all(k in col for k in ("fx", "fy", "cx", "cy"))
    fixed_k = None
    if (scan / "camera_matrix.csv").exists():
        fixed_k = np.loadtxt(scan / "camera_matrix.csv", delimiter=",").reshape(3, 3)

    out = {}
    for r in rows[1:]:
        if len(r) < 9:
            continue

        def val(name):
            return float(r[col[name]])

        c2w = np.eye(4)
        c2w[:3, :3] = Rotation.from_quat([val("qx"), val("qy"), val("qz"), val("qw")]).as_matrix()
        c2w[:3, 3] = [val("x"), val("y"), val("z")]
        if per_frame_k and r[col["fx"]].strip():
            K = np.array([[val("fx"), 0, val("cx")], [0, val("fy"), val("cy")], [0, 0, 1.0]])
        elif fixed_k is not None:
            K = fixed_k.copy()
        else:
            raise SystemExit("ingest: no intrinsics in odometry.csv and no camera_matrix.csv")
        out[int(r[col["frame"]])] = (c2w, K)
    return out


def scale_K(K: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Intrinsics of the same view resized by (sx, sy)."""
    out = K.copy()
    out[0] *= sx
    out[1] *= sy
    return out


def moved(a: np.ndarray, b: np.ndarray, min_move: float, min_turn_deg: float) -> bool:
    """True when camera pose `b` differs enough from `a` to be worth keeping."""
    if np.linalg.norm(a[:3, 3] - b[:3, 3]) >= min_move:
        return True
    cos = (np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2
    return np.degrees(np.arccos(np.clip(cos, -1, 1))) >= min_turn_deg


def backproject(depth_m: np.ndarray, K: np.ndarray, c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World points for every pixel with depth > 0. Returns (points [M,3], pixel index [M,2] as v,u)."""
    v, u = np.nonzero(depth_m > 0)
    z = depth_m[v, u].astype(np.float64)
    cam = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0] * z, (v + 0.5 - K[1, 2]) / K[1, 1] * z, z], axis=1)
    return cam @ c2w[:3, :3].T + c2w[:3, 3], np.stack([v, u], axis=1)


def thin(points: np.ndarray, colors: np.ndarray, voxel: float) -> tuple[np.ndarray, np.ndarray]:
    """Keep one point per voxel."""
    if len(points) == 0:
        return points, colors
    _, idx = np.unique(np.floor(points / voxel).astype(np.int64), axis=0, return_index=True)
    return points[idx], colors[idx]


def read_depth(scan: Path, frame: int, min_conf: int, lo: float, hi: float) -> np.ndarray | None:
    """Depth in millimetres (uint16); 0 where the sensor was unsure or out of range."""
    d = cv2.imread(str(Path(scan) / "depth" / f"{frame:06d}.png"), cv2.IMREAD_UNCHANGED)
    if d is None:
        return None
    bad = (d < lo * 1000) | (d > hi * 1000)
    conf = cv2.imread(str(Path(scan) / "confidence" / f"{frame:06d}.png"), cv2.IMREAD_UNCHANGED)
    if conf is not None and min_conf > 0:
        bad |= conf < min_conf
    d = d.copy()
    d[bad] = 0
    return d


def run(ws: Workspace, cfg: Config) -> None:
    c = cfg.ingest
    if not c.scan:
        raise SystemExit("ingest: no scan folder given (--lidar)")
    scan = Path(c.scan)
    check_scan(scan)
    odo = read_odometry(scan)

    cap = cv2.VideoCapture(str(scan / "rgb.mp4"))
    if not cap.isOpened():
        raise SystemExit(f"ingest: cannot open {scan / 'rgb.mp4'}")
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)  # poses and intrinsics refer to the stored frame
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    stride = max(1, int(round(video_fps / (c.fps * c.keep_one_in))))

    d = ws.reset("ingest")
    (d / "images").mkdir()
    (d / "depth").mkdir()
    names, depth_names, Ks, sizes, c2ws = [], [], [], [], []
    pts, cols = np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)
    last_pose = None
    seen = still = no_depth = 0

    def flush(group):
        nonlocal last_pose, pts, cols, still, no_depth
        if not group:
            return
        _, frame_no, bgr = max(group, key=lambda g: g[0])
        c2w, K = odo[frame_no]
        if last_pose is not None and not moved(last_pose, c2w, c.min_move, c.min_turn_deg):
            still += 1
            return
        depth = read_depth(scan, frame_no, c.depth_confidence, c.min_depth, c.max_depth)
        if depth is None:
            no_depth += 1
            return
        last_pose = c2w

        h0, w0 = bgr.shape[:2]
        f = min(1.0, c.long_side / max(h0, w0))
        if f < 1.0:
            bgr = cv2.resize(bgr, (int(round(w0 * f)), int(round(h0 * f))), interpolation=cv2.INTER_AREA)
        h, w = bgr.shape[:2]
        name = f"{frame_no:06d}"
        cv2.imwrite(str(d / "images" / f"{name}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(d / "depth" / f"{name}.png"), depth)
        names.append(f"{name}.jpg")
        depth_names.append(f"{name}.png")
        Ks.append(scale_K(K, w / w0, h / h0))
        sizes.append((w, h))
        c2ws.append(c2w)

        # Start points for training: the most trusted depth pixels, coloured from the frame.
        sure = read_depth(scan, frame_no, c.points_confidence, c.min_depth, c.max_depth)
        dh, dw = sure.shape
        p, vu = backproject(sure / 1000.0, scale_K(K, dw / w0, dh / h0), c2w)
        px = np.clip(((vu + 0.5) * [h / dh, w / dw]).astype(int), 0, [h - 1, w - 1])
        rgb = bgr[px[:, 0], px[:, 1], ::-1]
        p, rgb = thin(p, rgb, c.voxel)
        pts, cols = np.concatenate([pts, p]), np.concatenate([cols, rgb])
        if len(pts) > 4 * c.max_points:
            pts, cols = thin(pts, cols, c.voxel)

    group, frame_no = [], 0
    while True:
        if frame_no % stride:
            if not cap.grab():
                break
        else:
            ok, bgr = cap.read()
            if not ok:
                break
            if frame_no in odo:
                seen += 1
                group.append((sharpness_of(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)), frame_no, bgr))
            if len(group) == c.keep_one_in:
                flush(group)
                group = []
        frame_no += 1
    flush(group)
    cap.release()

    pts, cols = thin(pts, cols, c.voxel)
    if len(pts) > c.max_points:
        idx = np.random.default_rng(0).choice(len(pts), c.max_points, replace=False)
        pts, cols = pts[idx], cols[idx]
    print(f"ingest: {scan} -> {len(names)} frames kept of {frame_no} "
          f"({still} dropped: camera not moving; {no_depth} without depth), {len(pts)} start points")
    if len(names) < 20:
        raise SystemExit(f"ingest: only {len(names)} frames; scan too short, or raise --fps")

    np.savez(d / "model.npz",
             names=np.array(names), depth_names=np.array(depth_names),
             K=np.array(Ks), sizes=np.array(sizes), c2w=np.array(c2ws),
             points=pts, colors=cols.astype(np.uint8))


def load_model(ws: Workspace) -> dict:
    """Same keys as colmap_io.read_model, plus depth_names."""
    return dict(np.load(ws.dir("ingest") / "model.npz", allow_pickle=False))


def images(ws: Workspace) -> Path:
    return ws.dir("ingest") / "images"


def depth_maps(ws: Workspace) -> Path:
    return ws.dir("ingest") / "depth"
