"""Stage 3: move the COLMAP model into a Godot-style frame.

Result: +Y up, floor at y = 0, walls roughly along X and Z, units in metres,
origin under the centre of the camera path. Cameras keep OpenCV axes
(x right, y down, z forward).
"""
from __future__ import annotations

import json

import numpy as np

from .. import colmap_io
from ..config import Config
from ..workspace import Workspace
from . import ingest, poses


def estimate_up(c2w: np.ndarray) -> np.ndarray:
    """World up from camera orientations.

    Mean camera-up alone is biased when the phone points down. Handheld video
    has little roll, so every camera's right-axis is close to horizontal: up is
    the direction near mean camera-up that is most perpendicular to all of them.
    """
    mean_up = -c2w[:, :3, 1].mean(axis=0)
    mean_up /= np.linalg.norm(mean_up)
    right = c2w[:, :3, 0]
    spread = right.T @ right / len(right)
    up = np.linalg.solve(spread + 0.002 * np.eye(3), mean_up)
    return up / np.linalg.norm(up)


def _peakiness(points: np.ndarray, normal: np.ndarray, bin_width: float) -> float:
    """High when many points share a few heights along `normal` (floor, tables, shelves, ceiling)."""
    h = points @ normal
    counts = np.bincount(((h - h.min()) / bin_width).astype(int))
    return float((counts.astype(np.float64) ** 2).sum()) / len(points) ** 2


def sharpen_up(points: np.ndarray, up: np.ndarray, bin_width: float, max_deg: float = 24.0) -> np.ndarray:
    """Tilt `up` by at most `max_deg` to the direction that best stacks the points
    into horizontal layers. Keeps `up` when no direction is clearly better."""
    if len(points) > 20000:
        points = points[np.random.default_rng(0).choice(len(points), 20000, replace=False)]
    a = np.cross(up, [1.0, 0, 0] if abs(up[0]) < 0.9 else [0, 1.0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(up, a)

    def search(centre, half_range, step):
        best, best_score = centre, _peakiness(points, centre, bin_width)
        ca = np.cross(centre, b)
        ca /= np.linalg.norm(ca)
        cb = np.cross(centre, ca)
        for da in np.arange(-half_range, half_range + 1e-9, step):
            for db in np.arange(-half_range, half_range + 1e-9, step):
                n = centre + np.tan(np.radians(da)) * ca + np.tan(np.radians(db)) * cb
                n /= np.linalg.norm(n)
                score = _peakiness(points, n, bin_width)
                if score > best_score:
                    best, best_score = n, score
        return best, best_score

    base = _peakiness(points, up, bin_width)
    coarse, _ = search(up, max_deg, 2.0)
    fine, score = search(coarse, 2.0, 0.25)
    return fine if score > 1.15 * base else up


def rotation_to_y(u: np.ndarray) -> np.ndarray:
    """Rotation R with R @ u = +Y."""
    y = np.array([0.0, 1.0, 0.0])
    v = np.cross(u, y)
    s, c = np.linalg.norm(v), float(u @ y)
    if s < 1e-8:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k * ((1 - c) / (s * s))


def yaw_matrix(deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def best_yaw(pts: np.ndarray) -> float:
    """Yaw (degrees, 0-90) that makes the horizontal bounding box smallest,
    which lines the walls up with the X and Z axes."""
    best, best_area = 0.0, np.inf
    for deg in np.arange(0.0, 90.0, 0.5):
        p = pts @ yaw_matrix(deg).T
        lo, hi = np.percentile(p[:, [0, 2]], [2, 98], axis=0)
        area = float(np.prod(hi - lo))
        if area < best_area:
            best, best_area = float(deg), area
    return best


def floor_and_ceiling(heights: np.ndarray, cam_heights: np.ndarray, bin_width: float) -> tuple[float, float]:
    """Floor = lowest well-populated height layer below the cameras; ceiling = highest
    one above them. Falls back to the 2nd / 98th percentile when no layer stands out."""
    lo, hi = np.percentile(heights, [2, 98])
    edges = np.arange(heights.min(), heights.max() + bin_width, bin_width)
    counts, _ = np.histogram(heights, bins=edges)
    counts = np.convolve(counts, np.ones(3), mode="same")
    centres = (edges[:-1] + edges[1:]) / 2
    strong = counts >= max(0.1 * counts.max(), 20)

    def layer(mask, pick):
        idx = np.flatnonzero(strong & mask)
        if len(idx) == 0:
            return None
        c = centres[pick(idx)]
        return float(np.median(heights[np.abs(heights - c) <= bin_width]))

    floor = layer(centres < cam_heights.min(), np.min)
    ceiling = layer(centres > cam_heights.max(), np.max)
    return (lo if floor is None else floor), (hi if ceiling is None else ceiling)


def solve(c2w: np.ndarray, points: np.ndarray, cfg, up: np.ndarray | None = None) -> dict:
    """Similarity x' = s * R @ x + t. Pass `up` when the input frame is already
    gravity-aligned (ARKit); it is then used as is."""
    lo3, hi3 = np.percentile(points, [1, 99], axis=0)
    bin_width = 0.01 * float(np.linalg.norm(hi3 - lo3))
    if up is None:
        up = sharpen_up(points, estimate_up(c2w), bin_width)
    up = np.asarray(up, dtype=np.float64)
    cam_h = c2w[:, :3, 3] @ up
    floor, ceiling = floor_and_ceiling(points @ up, cam_h, bin_width)
    cam_above_floor = float(np.median(cam_h)) - floor
    if cam_above_floor <= 0 or ceiling <= floor:
        raise SystemExit("align: could not find a floor below the cameras; pass --scale")

    assumed = False
    if cfg.scale is not None:
        s = float(cfg.scale)
    elif cfg.ceiling_height is not None:
        s = cfg.ceiling_height / (ceiling - floor)
    else:
        target = cfg.camera_height
        if target is None:
            target, assumed = cfg.default_camera_height, True
        s = target / cam_above_floor
    A = rotation_to_y(up)
    R = yaw_matrix(best_yaw(points @ A.T)) @ A
    centre = R @ c2w[:, :3, 3].mean(axis=0)
    t = np.array([-s * centre[0], -s * floor, -s * centre[2]])
    return {"scale": s, "R": R, "t": t, "scale_assumed": assumed,
            "room_height_m": s * (ceiling - floor), "camera_height_m": s * cam_above_floor}


def apply_points(T: dict, pts: np.ndarray) -> np.ndarray:
    return T["scale"] * pts @ T["R"].T + T["t"]


def apply_cameras(T: dict, c2w: np.ndarray) -> np.ndarray:
    out = np.tile(np.eye(4), (len(c2w), 1, 1))
    out[:, :3, :3] = T["R"] @ c2w[:, :3, :3]
    out[:, :3, 3] = apply_points(T, c2w[:, :3, 3])
    return out


def run(ws: Workspace, cfg: Config) -> None:
    lidar = cfg.source == "lidar"
    if lidar:
        model = ingest.load_model(ws)
    else:
        model = colmap_io.read_model(poses.sparse_txt(ws))
    if len(model["points"]) < 500:
        raise SystemExit(f"align: only {len(model['points'])} points; reconstruction too weak")
    # ARKit's world is gravity-aligned with +Y up, so up is known on the LiDAR path.
    T = solve(model["c2w"], model["points"], cfg.align, up=ingest.WORLD_UP if lidar else None)
    extra = {}
    if lidar:
        # Depth maps are in metres; `depth_scale` turns them into scene units.
        extra = {"depth_names": model["depth_names"], "depth_scale": np.float64(T["scale"])}
    if T["scale_assumed"]:
        print(f"align: WARNING no --camera-height, --ceiling-height or --scale given; assuming "
              f"the camera was held {cfg.align.default_camera_height} m above the floor. Sizes may be off.")
    d = ws.reset("align")
    np.savez(
        d / "scene.npz",
        names=model["names"], K=model["K"], sizes=model["sizes"],
        c2w=apply_cameras(T, model["c2w"]),
        points=apply_points(T, model["points"]), colors=model["colors"],
        source=np.array(cfg.source), **extra,
    )
    (d / "align.json").write_text(json.dumps(
        {"scale": T["scale"], "R": T["R"].tolist(), "t": T["t"].tolist(),
         "scale_assumed": T["scale_assumed"], "room_height_m": T["room_height_m"],
         "camera_height_m": T["camera_height_m"]},
        indent=2))
    print(f"align: {len(model['names'])} cameras, {len(model['points'])} points, "
          f"scale {T['scale']:.4f}, camera height {T['camera_height_m']:.2f} m, "
          f"floor to top layer {T['room_height_m']:.2f} m")
