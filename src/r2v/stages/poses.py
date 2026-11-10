"""Stage 2: camera poses and sparse points with COLMAP, then undistort to PINHOLE."""
from __future__ import annotations

import struct
import subprocess
from pathlib import Path

from ..config import Config
from ..workspace import Workspace


def _colmap(exe: str, *args: object) -> None:
    cmd = [exe, *[str(a) for a in args]]
    print("poses:", " ".join(cmd[:2]))
    subprocess.run(cmd, check=True)


def _registered(model: Path) -> int:
    """Number of registered images: the leading uint64 of COLMAP's images.bin."""
    with open(model / "images.bin", "rb") as f:
        return struct.unpack("<Q", f.read(8))[0]


def run(ws: Workspace, cfg: Config) -> None:
    c = cfg.poses
    images = ws.dir("frames") / "images"
    n = sum(1 for _ in images.rglob("*.jpg"))
    d = ws.reset("poses")
    db = d / "database.db"
    sparse = d / "sparse"
    sparse.mkdir()

    _colmap(c.colmap, "feature_extractor",
            "--database_path", db, "--image_path", images,
            "--ImageReader.camera_model", "OPENCV",
            "--ImageReader.single_camera_per_folder", 1)

    if n <= c.exhaustive_max:
        _colmap(c.colmap, "exhaustive_matcher", "--database_path", db)
    else:
        args = ["sequential_matcher", "--database_path", db]
        if c.vocab_tree:
            args += ["--SequentialMatching.loop_detection", 1,
                     "--SequentialMatching.vocab_tree_path", c.vocab_tree]
        else:
            print(f"poses: {n} frames > {c.exhaustive_max} and no vocab tree; "
                  "no loop closure, separate videos may not link")
        _colmap(c.colmap, *args)

    _colmap(c.colmap, "mapper",
            "--database_path", db, "--image_path", images, "--output_path", sparse)

    models = [m for m in sparse.iterdir() if (m / "images.bin").exists()]
    if not models:
        raise SystemExit("poses: COLMAP produced no model; capture needs more overlap or texture")
    best = max(models, key=_registered)
    reg = _registered(best)
    print(f"poses: {reg} of {n} frames registered")
    if len(models) > 1 or reg < 0.8 * n:
        print(f"poses: WARNING reconstruction split into {len(models)} unconnected parts; only the "
              f"largest is used, so {n - reg} frames ({100 * (n - reg) // n}%) are dropped. Usual cause: "
              "a fast pan or a blank wall where no features can be tracked. Re-film that stretch slowly, "
              "keeping textured objects in view.")

    undist = d / "undistorted"
    _colmap(c.colmap, "image_undistorter",
            "--image_path", images, "--input_path", best,
            "--output_path", undist, "--output_type", "COLMAP")
    txt = undist / "sparse_txt"
    txt.mkdir()
    _colmap(c.colmap, "model_converter",
            "--input_path", undist / "sparse", "--output_path", txt, "--output_type", "TXT")


def undistorted_images(ws: Workspace) -> Path:
    return ws.dir("poses") / "undistorted" / "images"


def sparse_txt(ws: Workspace) -> Path:
    return ws.dir("poses") / "undistorted" / "sparse_txt"
