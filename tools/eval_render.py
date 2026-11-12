"""Compares a trained scan with its photos, to tell a training problem from a viewer problem.

Run: .venv\\Scripts\\python.exe tools\\eval_render.py <workspace>

Renders the cleaned splat from the training cameras (rendering only, no training) and
prints PSNR against the photos twice:
  full colour  - as training sees it (view-dependent colour)
  base colour  - what the exported .splat holds and Godot shows
Writes side-by-side pictures (photo | full colour | base colour) to <workspace>\\eval.

Reading the result:
  full colour looks good, base colour bad -> the export loses the quality
  both look good, Godot looks bad         -> the Godot renderer loses it
  both look bad                           -> training or the capture is the limit
These are views the splat was trained on. Away from the walked path it looks worse.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from r2v.stages import ingest, poses, splat
from r2v.workspace import Workspace

N_PICTURES = 8


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    ws = Workspace(Path(sys.argv[1]))
    dev = torch.device("cuda")
    scene = splat.load_scene(ws)
    lidar = "source" in scene and str(scene["source"]) == "lidar"
    img_dir = ingest.images(ws) if lidar else poses.undistorted_images(ws)

    stage = "clean" if (ws.dir("clean") / "ckpt.pt").exists() else "splat"
    params = {k: v.to(dev) for k, v in
              torch.load(ws.dir(stage) / "ckpt.pt", map_location=dev).items()}
    refined = ws.dir("splat") / "refined_c2w.npy"
    c2w = torch.tensor(np.load(refined) if refined.exists() else scene["c2w"],
                       dtype=torch.float32, device=dev)
    if lidar and not refined.exists():
        print("note: no refined camera poses in this workspace (older run); "
              "pictures will be slightly shifted and PSNR too low")
    names = [str(n) for n in scene["names"]]
    picked = set(np.linspace(0, len(names) - 1, N_PICTURES).astype(int).tolist())
    out = ws.root / "eval"
    out.mkdir(exist_ok=True)

    def to_bgr(img):
        return cv2.cvtColor((img.cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    def psnr(a, b):
        return -10 * torch.log10(((a - b) ** 2).mean()).item()

    full, base = [], []
    with torch.no_grad():
        for i in range(0, len(names), 4):
            bgr = cv2.imread(str(img_dir / names[i]))
            gt = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).to(dev).float() / 255
            h, w = gt.shape[:2]
            K = torch.tensor(scene["K"][i], dtype=torch.float32, device=dev)
            a = splat.render(params, c2w[i], K, w, h, 3)[0][..., :3].clamp(0, 1)
            b = splat.render(params, c2w[i], K, w, h, 0)[0][..., :3].clamp(0, 1)
            full.append(psnr(a, gt))
            base.append(psnr(b, gt))
            if i in picked or not picked.isdisjoint(range(i, i + 4)):
                picked -= set(range(i, i + 4))
                cv2.imwrite(str(out / f"{Path(names[i]).stem}.jpg"),
                            np.hstack([bgr, to_bgr(a), to_bgr(b)]),
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"{stage} checkpoint: {len(params['means'])} Gaussians, {len(full)} training views")
    print(f"PSNR full colour: {np.mean(full):.2f} dB (worst view {np.min(full):.2f})")
    print(f"PSNR base colour: {np.mean(base):.2f} dB (worst view {np.min(base):.2f})  <- what Godot shows")
    print(f"pictures (photo | full colour | base colour): {out}")


if __name__ == "__main__":
    main()
