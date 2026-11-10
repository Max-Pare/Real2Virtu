"""Read a COLMAP text model with PINHOLE cameras (image_undistorter output)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _data_lines(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if not line.startswith("#")]


def qvec_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def read_model(sparse_txt: Path) -> dict:
    """Return names, K [N,3,3], sizes [N,2] (w,h), c2w [N,4,4] (OpenCV axes),
    points [M,3], colors [M,3] uint8."""
    cams = {}
    for line in _data_lines(sparse_txt / "cameras.txt"):
        if not line.strip():
            continue
        p = line.split()
        model, w, h = p[1], int(p[2]), int(p[3])
        vals = [float(v) for v in p[4:]]
        if model == "PINHOLE":
            fx, fy, cx, cy = vals[:4]
        elif model == "SIMPLE_PINHOLE":
            fx = fy = vals[0]
            cx, cy = vals[1:3]
        else:
            raise ValueError(f"camera model {model} is not undistorted; run image_undistorter")
        cams[int(p[0])] = (np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]), (w, h))

    names, Ks, sizes, c2ws = [], [], [], []
    lines = _data_lines(sparse_txt / "images.txt")
    for line in lines[0::2]:  # every second line holds 2D points
        p = line.split()
        if len(p) < 10:
            continue
        q = np.array([float(v) for v in p[1:5]])
        t = np.array([float(v) for v in p[5:8]])
        R = qvec_to_rot(q)  # world to camera
        c2w = np.eye(4)
        c2w[:3, :3] = R.T
        c2w[:3, 3] = -R.T @ t
        K, size = cams[int(p[8])]
        names.append(" ".join(p[9:]))
        Ks.append(K)
        sizes.append(size)
        c2ws.append(c2w)

    order = np.argsort(names)
    pts, cols = [], []
    for line in _data_lines(sparse_txt / "points3D.txt"):
        p = line.split()
        if len(p) < 7:
            continue
        pts.append([float(v) for v in p[1:4]])
        cols.append([int(v) for v in p[4:7]])

    return {
        "names": np.array(names)[order],
        "K": np.array(Ks)[order],
        "sizes": np.array(sizes)[order],
        "c2w": np.array(c2ws)[order],
        "points": np.array(pts, dtype=np.float64).reshape(-1, 3),
        "colors": np.array(cols, dtype=np.uint8).reshape(-1, 3),
    }
