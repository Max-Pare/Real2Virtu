"""Stage 6: assemble the Godot project from the template and pipeline outputs."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np
import trimesh

from .. import manifest
from ..config import Config
from ..workspace import Workspace
from . import clean, collision

TEMPLATE = Path(__file__).resolve().parents[3] / "godot_template"


def run(ws: Workspace, cfg: Config) -> None:
    c = cfg.export
    out = Path(c.out)
    if not TEMPLATE.exists():
        raise SystemExit(f"export: template not found at {TEMPLATE}")
    shutil.copytree(TEMPLATE, out, dirs_exist_ok=True)

    assets = out / "assets"
    assets.mkdir(exist_ok=True)
    shutil.copy2(ws.dir("collision") / "collision.glb", assets / "collision.glb")

    # The collision stage may have corrected floor height and scale after training
    # (the mesh already has it). Apply the same to the splat and the cameras:
    # x' = k * (x - [0, floor, 0]). No rotation, so colours and orientations stay valid.
    T = collision.load_refine(ws)
    k, shift = T["scale"], np.array([0.0, T["floor_shift"], 0.0])

    # .splat records are 32 bytes: position f32x3, linear scale f32x3, then 8 bytes of colour/rotation.
    raw = np.fromfile(clean.output(ws, "scene.splat"), dtype="<f4").reshape(-1, 8).copy()
    raw[:, :3] = k * (raw[:, :3] - shift)
    raw[:, 3:6] *= k
    raw.tofile(assets / "scene.splat")
    center = raw[:, :3].astype(np.float64).mean(axis=0) if len(raw) else np.zeros(3)

    c2w = np.load(ws.dir("align") / "scene.npz", allow_pickle=False)["c2w"].copy()
    c2w[:, :3, 3] = k * (c2w[:, :3, 3] - shift)
    mesh_colors = None
    try:
        mesh = trimesh.load(assets / "collision.glb", force="mesh")
        obstacles = np.asarray(mesh.vertices)
        if mesh.visual.kind == "vertex":
            mesh_colors = np.asarray(mesh.visual.vertex_colors)
    except Exception:
        obstacles = np.zeros((0, 3))
    rgba = raw[:, 6:7].copy().view(np.uint8)  # bytes 24-27 of each record
    spawn_cam = manifest.pick_spawn_camera(c2w, obstacles)
    eye = c.eye_height if c.eye_height is not None else manifest.eye_height_from_cameras(c2w)
    m = manifest.build(
        splat_center=[float(v) for v in center],
        spawn=manifest.spawn_from_camera(c2w[spawn_cam]),
        eye_height=eye,
        splat_path="res://assets/scene.splat",
        collision_path="res://assets/collision.glb",
        backdrop=manifest.backdrop(rgba, obstacles, mesh_colors),
    )
    manifest.write(out / "scene.json", m)

    proj = out / "project.godot"
    name = c.project_name.replace('"', "'")
    proj.write_text(re.sub(r'config/name=".*"', f'config/name="{name}"', proj.read_text()))
    print(f"export: Godot project written to {out.resolve()}")
