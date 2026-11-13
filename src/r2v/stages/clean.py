"""Stage 5: tidy the trained splat.

1. Drop Gaussians that only add haze: near-transparent, oversized, needle-shaped, isolated.
2. LiDAR only: drop Gaussians floating in space that the sensor measured as empty.
3. LiDAR only, off by default: put new Gaussians on measured surfaces that the splat
   leaves see-through.

Later stages read the result through `output()`.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .. import device as device_mod
from .. import splat_io
from ..config import Config
from ..workspace import Workspace
from . import ingest, splat


def output(ws: Workspace, name: str) -> Path:
    """Path of a splat file for later stages: the cleaned one when it exists."""
    cleaned = ws.root / "clean" / name
    return cleaned if cleaned.exists() else ws.dir("splat") / name


def load_params(ws: Workspace, device):
    import torch
    ckpt = torch.load(output(ws, "ckpt.pt"), map_location=device)
    return {k: v.to(device) for k, v in ckpt.items()}


def haze_mask(means: np.ndarray, sigma: np.ndarray, opacity: np.ndarray, cams: np.ndarray, c) -> dict:
    """Boolean masks (True = remove) per reason. `sigma` [N,3] are linear sizes.

    Size limits grow with the distance to the nearest camera: a 5 m Gaussian is a
    blob inside the room and a normal piece of the view through a window.
    """
    from scipy.spatial import cKDTree

    s = np.sort(sigma, axis=1)
    big, mid = s[:, 2], s[:, 1]
    dist = cKDTree(cams).query(means)[0]
    reasons = {
        "transparent": opacity < c.min_opacity,
        "oversized": big > np.maximum(c.max_size, c.max_size_per_m * dist),
        "needle": (big > c.needle_ratio * np.maximum(mid, 1e-9))
                  & (big > np.maximum(c.needle_length, c.needle_length_per_m * dist)),
    }
    gone = reasons["transparent"] | reasons["oversized"] | reasons["needle"]
    # Isolation is judged among the Gaussians that survive the tests above.
    lonely = np.zeros(len(means), dtype=bool)
    keep = np.flatnonzero(~gone)
    if len(keep) > 4:
        gap = cKDTree(means[keep]).query(means[keep], k=4)[0][:, 1:].mean(axis=1)
        limit = np.maximum.reduce([np.full(len(keep), c.isolated_gap), 3 * big[keep],
                                   c.isolated_gap_per_m * dist[keep]])
        lonely[keep] = gap > limit
    reasons["isolated"] = lonely
    return reasons


def _neighbourhood(depth, size: int = 3):
    """Per pixel: nearest and farthest valid depth in a size x size window (0 = no data)."""
    import torch
    import torch.nn.functional as F

    d = depth[None, None]
    pad = size // 2
    far = F.max_pool2d(d, size, stride=1, padding=pad)[0, 0]
    inf = torch.where(d > 0, d, torch.full_like(d, float("inf")))
    near = -F.max_pool2d(-inf, size, stride=1, padding=pad)[0, 0]
    return near, far


def free_space_mask(means, depths, Ks, c2ws, margin: float, margin_rel: float,
                    min_views: int, min_share: float):
    """True for points that lie well in front of the measured surface in most of the
    views that see them. `depths` [V,h,w] in scene units (0 = no data), `Ks` for the
    depth maps, cameras with OpenCV axes. Torch tensors on one device.

    The nearest depth in a 3x3 window is used, so points on the edge of a foreground
    object are not judged against the background behind it.
    """
    import torch

    n = len(means)
    free = torch.zeros(n, device=means.device)
    seen = torch.zeros(n, device=means.device)
    for depth, K, c2w in zip(depths, Ks, c2ws):
        h, w = depth.shape
        cam = (means - c2w[:3, 3]) @ c2w[:3, :3]
        z = cam[:, 2]
        u = (cam[:, 0] / z.clamp_min(1e-6) * K[0, 0] + K[0, 2]).floor().long()
        v = (cam[:, 1] / z.clamp_min(1e-6) * K[1, 1] + K[1, 2]).floor().long()
        ok = (z > 0.05) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        near, _ = _neighbourhood(depth)
        idx = (v.clamp(0, h - 1), u.clamp(0, w - 1))
        d_here, d_near = depth[idx], near[idx]
        ok &= (d_here > 0) & torch.isfinite(d_near)
        tol = torch.maximum(torch.full_like(z, margin), margin_rel * d_near)
        seen += (ok & (z < d_here + tol)).float()   # not hidden behind the surface
        free += (ok & (z < d_near - tol)).float()
    return (free >= min_views) & (free >= min_share * seen.clamp_min(1))


def hole_points(alpha, depth, photo, K, c2w, smooth_rel: float):
    """World points and colours for sensor pixels where the splat is see-through.

    Pixels at depth jumps are skipped: the sensor blends foreground and background
    there. All inputs are at the depth map's size.

    A splat surface lying behind the measured one is not treated as a hole. That
    happens where the sensor is wrong (glass, screens, glossy surfaces), and filling
    there put opaque blobs in front of real content (-3.4 dB PSNR on the test scan).
    """
    import torch

    near, far = _neighbourhood(depth)
    smooth = torch.isfinite(near) & (far - near < smooth_rel * depth)
    hole = (depth > 0) & smooth & (alpha < 0.5)
    v, u = torch.nonzero(hole, as_tuple=True)
    z = depth[v, u]
    cam = torch.stack([(u + 0.5 - K[0, 2]) / K[0, 0] * z, (v + 0.5 - K[1, 2]) / K[1, 1] * z, z], dim=1)
    return cam @ c2w[:3, :3].T + c2w[:3, 3], photo[v, u]


def seen_by(points: np.ndarray, view: np.ndarray, voxel: float, min_views: int) -> np.ndarray:
    """True for points whose voxel was proposed from at least `min_views` different views."""
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    cells = np.floor(points / voxel).astype(np.int64)
    _, cell_id = np.unique(cells, axis=0, return_inverse=True)
    cell_id = cell_id.reshape(-1)
    pairs = np.unique(np.stack([cell_id, view], axis=1), axis=0)
    return np.bincount(pairs[:, 0], minlength=cell_id.max() + 1)[cell_id] >= min_views


def new_gaussians(points: np.ndarray, colors: np.ndarray, size: float, sh_coeffs: int) -> dict:
    """Round, mostly opaque Gaussians of the given colour (0-255) in training-parameter form."""
    n = len(points)
    quats = np.zeros((n, 4), dtype=np.float32)
    quats[:, 0] = 1.0
    return {
        "means": points.astype(np.float32),
        "scales": np.full((n, 3), math.log(size), dtype=np.float32),
        "quats": quats,
        "opacities": np.full(n, math.log(0.9 / 0.1), dtype=np.float32),
        "sh0": ((colors / 255.0 - 0.5) / splat_io.SH_C0)[:, None, :].astype(np.float32),
        "shN": np.zeros((n, sh_coeffs - 1, 3), dtype=np.float32),
    }


def _lidar_views(ws: Workspace, scene: dict, device):
    """Depth maps (scene units), their intrinsics, poses and photo paths."""
    import cv2
    import torch

    ref = ws.dir("splat") / "refined_c2w.npy"
    c2ws = np.load(ref) if ref.exists() else scene["c2w"]
    unit = float(scene["depth_scale"]) / 1000.0
    depths, Ks = [], []
    for i, name in enumerate(scene["depth_names"]):
        mm = cv2.imread(str(ingest.depth_maps(ws) / str(name)), cv2.IMREAD_UNCHANGED)
        if mm is None:
            raise SystemExit(f"clean: missing depth map {name}")
        w, h = scene["sizes"][i]
        depths.append(torch.from_numpy(mm.astype(np.float32) * unit).to(device))
        Ks.append(ingest.scale_K(scene["K"][i], mm.shape[1] / w, mm.shape[0] / h))
    return (depths, torch.tensor(np.array(Ks), dtype=torch.float32, device=device),
            torch.tensor(c2ws, dtype=torch.float32, device=device))


def run(ws: Workspace, cfg: Config) -> None:
    import cv2
    import torch

    c = cfg.clean
    scene = splat.load_scene(ws)
    ckpt = torch.load(ws.dir("splat") / "ckpt.pt", map_location="cpu")
    arr = {k: v.numpy() for k, v in ckpt.items()}
    n0 = len(arr["means"])
    stats = {"before": n0}

    reasons = haze_mask(arr["means"], np.exp(arr["scales"]), splat_io._sigmoid(arr["opacities"]),
                        scene["c2w"][:, :3, 3], c)
    gone = np.zeros(n0, dtype=bool)
    for name, mask in reasons.items():
        stats[name] = int((mask & ~gone).sum())
        gone |= mask

    lidar = "depth_names" in scene
    if lidar and (c.carve or c.fill):
        device = device_mod.resolve(cfg.device)
        depths, Ks, c2ws = _lidar_views(ws, scene, device)

    if lidar and c.carve:
        with torch.no_grad():
            floating = free_space_mask(
                torch.tensor(arr["means"], dtype=torch.float32, device=device), depths, Ks, c2ws,
                c.margin, c.margin_rel, c.carve_min_views, c.carve_min_share).cpu().numpy()
        stats["in_free_space"] = int((floating & ~gone).sum())
        gone |= floating
    arr = {k: v[~gone] for k, v in arr.items()}

    stats["filled"] = 0
    if lidar and c.fill:
        import torch.nn.functional as F
        params = {k: torch.tensor(v, device=device) for k, v in arr.items()}
        K_full = torch.tensor(scene["K"], dtype=torch.float32, device=device)
        pts, cols, views = [], [], []
        with torch.no_grad():
            for i, depth in enumerate(depths):
                h, w = depth.shape
                # Coverage is rendered at photo size and averaged down. Rendered straight
                # at the depth map's size, fine detail falls below one pixel and reads
                # as see-through.
                wf, hf = (int(v) for v in scene["sizes"][i])
                _, alpha, _ = splat.render(params, c2ws[i], K_full[i], wf, hf, sh_degree=0, mode="ED")
                alpha = F.interpolate(alpha.permute(2, 0, 1)[None], size=(h, w), mode="area")[0].permute(1, 2, 0)
                bgr = cv2.imread(str(ingest.images(ws) / str(scene["names"][i])))
                photo = torch.from_numpy(cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)[..., ::-1].copy())
                p, col = hole_points(alpha[..., 0], depth, photo.to(device), Ks[i], c2ws[i],
                                     c.fill_smooth_rel)
                pts.append(p)
                cols.append(col)
                views.append(np.full(len(p), i))
            pts, cols, views = torch.cat(pts), torch.cat(cols), np.concatenate(views)
            # A new point must not sit in space that any other view measured as empty.
            ok = ~free_space_mask(pts, depths, Ks, c2ws, c.margin, c.margin_rel, 1, 0.0)
        ok = ok.cpu().numpy()
        pts, cols = pts.cpu().numpy().astype(np.float64), cols.cpu().numpy()
        # One view alone is not trusted: thin objects and glass give the sensor wrong depth.
        ok &= seen_by(pts, views, c.fill_voxel, c.fill_min_views)
        stats["fill_rejected"] = int((~ok).sum())
        pts, cols = ingest.thin(pts[ok], cols[ok], c.fill_voxel)
        if len(pts) > c.fill_max:
            idx = np.random.default_rng(0).choice(len(pts), c.fill_max, replace=False)
            pts, cols = pts[idx], cols[idx]
        stats["filled"] = len(pts)
        if len(pts):
            add = new_gaussians(pts, cols, 0.6 * c.fill_voxel, arr["shN"].shape[1] + 1)
            arr = {k: np.concatenate([arr[k], add[k]]) for k in arr}
        del params

    stats["after"] = len(arr["means"])
    d = ws.reset("clean")
    torch.save({k: torch.from_numpy(v) for k, v in arr.items()}, d / "ckpt.pt")
    splat_io.write_ply(d / "scene.ply", **{k: arr[k] for k in splat.PARAM_NAMES})
    splat_io.write_splat(d / "scene.splat", arr["means"], arr["scales"], arr["quats"],
                         arr["opacities"], arr["sh0"])
    (d / "clean.json").write_text(json.dumps(stats, indent=2))
    removed = ", ".join(f"{stats[k]} {k.replace('_', ' ')}" for k in
                        ("transparent", "oversized", "needle", "isolated", "in_free_space") if k in stats)
    added = f"; added {stats['filled']} on uncovered surfaces" if c.fill else ""
    print(f"clean: {n0} Gaussians -> {stats['after']} (removed {removed}{added})")
