"""Stage 1: pull frames from each video and keep the sharpest of each group."""
from __future__ import annotations

import subprocess
from pathlib import Path

import cv2

from ..config import Config
from ..workspace import Workspace


def sharpness(path: Path) -> float:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return -1.0 if img is None else sharpness_of(img)


def sharpness_of(img) -> float:
    """Variance of the Laplacian of a greyscale image; higher is sharper."""
    img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def select_sharpest(frames: list[Path], group: int) -> list[Path]:
    keep = []
    for i in range(0, len(frames), group):
        chunk = frames[i:i + group]
        keep.append(max(chunk, key=sharpness))
    return keep


def run(ws: Workspace, cfg: Config) -> None:
    c = cfg.frames
    if not c.videos:
        raise SystemExit("frames: no input videos given")
    out = ws.reset("frames") / "images"
    s = c.long_side
    vf = (f"fps={c.extract_fps},"
          f"scale='if(gt(iw,ih),min({s},iw),-2)':'if(gt(iw,ih),-2,min({s},ih))'")
    total = 0
    for i, video in enumerate(c.videos):
        # One folder per video: COLMAP then assigns one camera per folder.
        d = out / f"v{i:02d}"
        d.mkdir(parents=True)
        subprocess.run(
            [c.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
             "-vf", vf, "-q:v", "2", str(d / "%06d.jpg")],
            check=True,
        )
        frames = sorted(d.glob("*.jpg"))
        keep = set(select_sharpest(frames, c.keep_one_in))
        for f in frames:
            if f not in keep:
                f.unlink()
        total += len(keep)
        print(f"frames: {video} -> {len(keep)} of {len(frames)} frames kept")
    if total < 20:
        raise SystemExit(f"frames: only {total} frames; video too short for reconstruction")
