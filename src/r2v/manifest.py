"""scene.json: the contract between the pipeline and the Godot project.

The Godot side builds its scene from this file, so later versions add
interactable objects by adding entries to `objects`.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

VERSION = 1


def pick_spawn_camera(c2w: np.ndarray, obstacles: np.ndarray, radius: float = 0.35) -> int:
    """Index of the filmed camera position with the least collision geometry in the
    column under it, so the player does not start inside a chair or table.
    `obstacles` are collision-mesh vertices [M,3]."""
    best, best_count = 0, None
    for i, cam in enumerate(c2w[:, :3, 3]):
        near = np.linalg.norm(obstacles[:, [0, 2]] - cam[[0, 2]], axis=1) < radius
        count = int((near & (obstacles[:, 1] > 0.15) & (obstacles[:, 1] < cam[1])).sum())
        if best_count is None or count < best_count:
            best, best_count = i, count
    return best


def eye_height_from_cameras(c2w: np.ndarray) -> float:
    """Splats look right only near filmed viewpoints, so the player's eyes go where
    the camera was held (kept within a human range)."""
    return float(np.clip(np.median(c2w[:, 1, 3]), 1.0, 1.9))


def spawn_from_camera(c2w: np.ndarray) -> dict:
    """Player spawn under a camera, facing the way the camera looked."""
    pos = c2w[:3, 3]
    fwd = c2w[:3, 2]  # OpenCV camera looks along +Z
    # A Godot body with rotation.y = yaw faces (-sin yaw, 0, -cos yaw).
    yaw = math.degrees(math.atan2(-fwd[0], -fwd[2]))
    return {"position": [float(pos[0]), 0.05, float(pos[2])], "yaw_deg": yaw}


def build(spawn: dict, eye_height: float, splat_path: str, collision_path: str,
          splat_center: list[float]) -> dict:
    return {
        "version": VERSION,
        "units": "metres",
        "up": "+Y",
        # `center` is the mean Gaussian position. The Godot splat importer recentres
        # the data on it, so the splat node must be placed there.
        "splat": {"path": splat_path, "format": Path(splat_path).suffix.lstrip("."),
                  "center": splat_center},
        # floor_y: height of the aligned floor. Godot adds a flat safety floor there,
        # because the scanned mesh has holes where the floor was never filmed.
        "collision": {"path": collision_path, "floor_y": 0.0},
        "spawn": spawn,
        "eye_height": eye_height,
        "objects": [],  # reserved: {id, splat, collision, body: "rigid"|"static", transform}
    }


def write(path: Path, manifest: dict) -> None:
    Path(path).write_text(json.dumps(manifest, indent=2))


def read(path: Path) -> dict:
    m = json.loads(Path(path).read_text())
    if m.get("version") != VERSION:
        raise ValueError(f"unsupported scene.json version {m.get('version')}")
    return m
