"""Write trained Gaussians to disk. Numpy only, so it is testable without a GPU.

Inputs are raw training parameters:
  means [N,3], scales [N,3] (log), quats [N,4] (wxyz, any norm),
  opacities [N] (logit), sh0 [N,1,3], shN [N,K,3].
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SH_C0 = 0.28209479177387814


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def write_ply(path: Path, means, scales, quats, opacities, sh0, shN) -> None:
    """Standard 3DGS PLY layout (INRIA), readable by most splat tools."""
    n = len(means)
    f_dc = sh0.reshape(n, 3)
    f_rest = shN.transpose(0, 2, 1).reshape(n, -1)  # channel-major, as INRIA stores it
    names = (["x", "y", "z", "nx", "ny", "nz"]
             + [f"f_dc_{i}" for i in range(3)]
             + [f"f_rest_{i}" for i in range(f_rest.shape[1])]
             + ["opacity"] + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)])
    data = np.concatenate(
        [means, np.zeros((n, 3)), f_dc, f_rest, opacities.reshape(n, 1), scales, quats],
        axis=1).astype("<f4")
    header = "ply\nformat binary_little_endian 1.0\n" + f"element vertex {n}\n"
    header += "".join(f"property float {name}\n" for name in names) + "end_header\n"
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())


def read_ply(path: Path) -> tuple[list[str], np.ndarray]:
    """Property names and an [n, properties] float32 table of a PLY written by write_ply."""
    with open(path, "rb") as f:
        names, n = [], 0
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property"):
                kind, name = line.split()[1:]
                if kind != "float":
                    raise ValueError(f"{path}: property {name} is {kind}, expected float")
                names.append(name)
            elif line == "end_header":
                break
        data = np.frombuffer(f.read(), dtype="<f4").reshape(n, len(names)).copy()
    return names, data


def transform_ply(src: Path, dst: Path, scale: float, shift: np.ndarray) -> np.ndarray:
    """Writes `src` to `dst` with positions x' = scale * (x - shift) and sizes scaled.
    No rotation, so orientations and view-dependent colour stay valid. Returns positions."""
    names, data = read_ply(src)
    xyz = [names.index(k) for k in ("x", "y", "z")]
    data[:, xyz] = scale * (data[:, xyz] - np.asarray(shift, dtype=np.float32))
    data[:, [names.index(f"scale_{i}") for i in range(3)]] += np.log(scale)  # stored as logarithms
    header = "ply\nformat binary_little_endian 1.0\n" + f"element vertex {len(data)}\n"
    header += "".join(f"property float {name}\n" for name in names) + "end_header\n"
    with open(dst, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.astype("<f4").tobytes())
    return data[:, xyz]


def write_splat(path: Path, means, scales, quats, opacities, sh0) -> None:
    """Compact .splat layout: 32 bytes per Gaussian, base colour only.
    position f32x3, scale f32x3, rgba u8x4, rotation u8x4 (wxyz)."""
    n = len(means)
    lin_scale = np.exp(scales)
    alpha = _sigmoid(opacities.reshape(n))
    order = np.argsort(-(lin_scale.prod(axis=1) * alpha))  # most visible first
    rgb = np.clip(0.5 + SH_C0 * sh0.reshape(n, 3), 0, 1)
    q = quats / np.linalg.norm(quats, axis=1, keepdims=True)

    rec = np.zeros(n, dtype=[("pos", "<f4", 3), ("scale", "<f4", 3),
                             ("rgba", "u1", 4), ("rot", "u1", 4)])
    rec["pos"] = means[order]
    rec["scale"] = lin_scale[order]
    rec["rgba"][:, :3] = (rgb[order] * 255).round()
    rec["rgba"][:, 3] = (alpha[order] * 255).round()
    rec["rot"] = np.clip(q[order] * 128 + 128, 0, 255).round()
    with open(path, "wb") as f:
        f.write(rec.tobytes())
