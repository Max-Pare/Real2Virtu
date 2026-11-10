"""Stage 6: collision mesh from depth rendered by the trained splat (TSDF fusion).

The mesh carries vertex colours taken from the splat, so the game can also show it
behind the splat: gaps then show a plain surface instead of the background.
"""
from __future__ import annotations

import json
import math

import numpy as np

from ..config import Config
from ..workspace import Workspace
from . import clean, splat


FLOOR_BIN = 0.02              # metres
FLOOR_RANGE = (-4.0, 4.0)     # heights searched around the align-stage floor


def pick_floor(area: np.ndarray, centres: np.ndarray, cam_min_y: float) -> float | None:
    """Floor height from a histogram of upward-facing surface area by height.

    The floor is the lowest layer below the cameras that holds a real share of the
    area. Sparse feature points cannot tell this (a cluttered desk has far more
    features than a plain floor), dense depth can. None if no layer stands out.

    Depth on a plain floor is noisy, so the floor is a broad hump rather than a
    spike: compare area summed over a 0.3 m window, enter the lowest hump that
    reaches 10% of the biggest layer, and climb to its peak.
    """
    half = int(round(0.15 / FLOOR_BIN))
    mass = np.convolve(area, np.ones(2 * half + 1), mode="same")
    below = centres < cam_min_y - 0.3
    if not below.any() or mass[below].max() <= 0:
        return None
    i = np.flatnonzero(below & (mass >= 0.10 * mass[below].max()))[0]
    while i + 1 < len(mass) and below[i + 1] and mass[i + 1] >= mass[i]:
        i += 1
    near = slice(max(i - half, 0), i + half + 1)
    if area[near].sum() <= 0:
        return None
    return float(np.average(centres[near], weights=area[near]))


def _upward_area(depth, valid, K, c2w, n_bins: int):
    """Histogram (by height) of the area of surfaces facing up, seen in one depth map."""
    import torch

    h, w = depth.shape
    v, u = torch.meshgrid(torch.arange(h, device=depth.device, dtype=torch.float32),
                          torch.arange(w, device=depth.device, dtype=torch.float32), indexing="ij")
    cam = torch.stack([(u - K[0, 2]) / K[0, 0] * depth, (v - K[1, 2]) / K[1, 1] * depth, depth], dim=-1)
    du = cam[1:-1, 2:] - cam[1:-1, :-2]
    dv = cam[2:, 1:-1] - cam[:-2, 1:-1]
    normal = torch.cross(du, dv, dim=-1)
    normal = -normal / normal.norm(dim=-1, keepdim=True).clamp_min(1e-9)  # face the camera
    R, t = c2w[:3, :3], c2w[:3, 3]
    normal_y = normal @ R[1]
    height = cam[1:-1, 1:-1] @ R[1] + t[1]
    ok = (valid[1:-1, 1:-1] & valid[1:-1, 2:] & valid[1:-1, :-2] & valid[2:, 1:-1] & valid[:-2, 1:-1]
          & (normal_y > 0.9) & (height > FLOOR_RANGE[0]) & (height < FLOOR_RANGE[1]))
    pixel_area = depth[1:-1, 1:-1] ** 2 / (K[0, 0] * K[1, 1])
    idx = ((height[ok] - FLOOR_RANGE[0]) / FLOOR_BIN).long().clamp(0, n_bins - 1)
    return torch.bincount(idx, weights=pixel_area[ok], minlength=n_bins)


def refine_transform(floor: float | None, cam_median_y: float, align_cfg) -> dict:
    """Correction applied after training: x' = scale * (x - [0, floor_shift, 0])."""
    if floor is None:
        return {"floor_shift": 0.0, "scale": 1.0, "floor_found": False}
    scale = 1.0
    if align_cfg.scale is None and align_cfg.ceiling_height is None:
        target = align_cfg.camera_height or align_cfg.default_camera_height
        scale = target / (cam_median_y - floor)
    return {"floor_shift": floor, "scale": scale, "floor_found": True}


def run(ws: Workspace, cfg: Config) -> None:
    import open3d as o3d
    import torch
    import trimesh

    c = cfg.collision
    device = splat._require_cuda()
    scene = splat.load_scene(ws)
    params = clean.load_params(ws, device)

    # A cube around the scene. (Open3D 0.20's ScalableTSDFVolume returned empty
    # meshes on this setup, so the uniform volume is used.)
    lo, hi = np.percentile(scene["points"], [1, 99], axis=0)
    cams = scene["c2w"][:, :3, 3]
    lo = np.minimum(lo, cams.min(axis=0)) - 0.5
    hi = np.maximum(hi, cams.max(axis=0)) + 0.5
    length = float((hi - lo).max())
    resolution = min(int(np.ceil(length / c.voxel)), c.max_resolution)
    voxel = length / resolution
    if voxel > c.voxel * 1.01:
        print(f"collision: scene is {length:.1f} m wide; voxel size raised to {voxel * 100:.1f} cm")
    volume = o3d.pipelines.integration.UniformTSDFVolume(
        length=length, resolution=resolution, sdf_trunc=voxel * 4,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        origin=(lo + hi) / 2 - length / 2)

    n_bins = int(round((FLOOR_RANGE[1] - FLOOR_RANGE[0]) / FLOOR_BIN))
    up_area = torch.zeros(n_bins, device=device, dtype=torch.float32)
    used = 0
    with torch.no_grad():
        for i in range(0, len(scene["names"]), c.use_one_in):
            w, h = (int(v) // c.downscale for v in scene["sizes"][i])
            K = scene["K"][i].copy()
            K[:2] /= c.downscale
            c2w = torch.tensor(scene["c2w"][i], dtype=torch.float32, device=device)
            sh_degree = int(round(math.sqrt(params["shN"].shape[1] + 1))) - 1
            out, alpha, _ = splat.render(
                params, c2w, torch.tensor(K, dtype=torch.float32, device=device),
                w, h, sh_degree=sh_degree, mode="RGB+ED")
            depth = out[..., 3]
            color = (out[..., :3].clamp(0, 1) * 255).byte().cpu().numpy()
            valid = alpha[..., 0] >= 0.5
            depth[~valid] = 0.0  # see-through pixels carry no surface
            up_area += _upward_area(depth, valid, torch.tensor(K, dtype=torch.float32, device=device),
                                    c2w, n_bins)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.ascontiguousarray(color)),
                o3d.geometry.Image(np.ascontiguousarray(depth.cpu().numpy(), dtype=np.float32)),
                depth_scale=1.0, depth_trunc=c.depth_trunc, convert_rgb_to_intensity=False)
            intr = o3d.camera.PinholeCameraIntrinsic(w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
            volume.integrate(rgbd, intr, np.linalg.inv(scene["c2w"][i]))
            used += 1

    mesh = volume.extract_triangle_mesh()
    if len(mesh.triangles) == 0:
        raise SystemExit("collision: TSDF produced an empty mesh")

    # Drop floating fragments, then simplify.
    labels, counts, _ = mesh.cluster_connected_triangles()
    small = np.asarray(counts)[np.asarray(labels)] < c.min_component_tris
    mesh.remove_triangles_by_mask(small)
    mesh.remove_unreferenced_vertices()
    if len(mesh.triangles) > c.target_tris:
        mesh = mesh.simplify_quadric_decimation(c.target_tris)
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    # Dense floor check: correct the floor height (and the scale, when it came from
    # camera height) that the align stage could only guess from sparse points.
    centres = FLOOR_RANGE[0] + (np.arange(n_bins) + 0.5) * FLOOR_BIN
    cam_y = scene["c2w"][:, 1, 3]
    floor = pick_floor(up_area.cpu().numpy(), centres, float(cam_y.min()))
    T = refine_transform(floor, float(np.median(cam_y)), cfg.align)
    if T["floor_found"]:
        print(f"collision: dense floor at {T['floor_shift']:+.2f} m relative to the sparse estimate; "
              f"scale correction x{T['scale']:.3f}")
    else:
        print("collision: no clear floor in the rendered depth; keeping the sparse estimate")
    verts = T["scale"] * (np.asarray(mesh.vertices) - [0.0, T["floor_shift"], 0.0])

    d = ws.reset("collision")
    (d / "refine.json").write_text(json.dumps(T, indent=2))
    colors = None
    if mesh.has_vertex_colors():
        colors = (np.asarray(mesh.vertex_colors) * 255).round().astype(np.uint8)
    tm = trimesh.Trimesh(verts, np.asarray(mesh.triangles), vertex_colors=colors, process=False)
    tm.export(d / "collision.glb")
    print(f"collision: {used} depth views -> {len(tm.faces)} triangles")


def load_refine(ws: Workspace) -> dict:
    """The correction that export must apply to the splat and the cameras."""
    p = ws.dir("collision") / "refine.json"
    return json.loads(p.read_text()) if p.exists() else {"floor_shift": 0.0, "scale": 1.0}
