"""Stage 4: train 3D Gaussians with gsplat (MCMC strategy, capped count)."""
from __future__ import annotations

import math
import time

import numpy as np

from .. import gsplat_compat, splat_io
from ..config import Config
from ..workspace import Workspace
from . import ingest, poses

PARAM_NAMES = ["means", "scales", "quats", "opacities", "sh0", "shN"]


def _require_cuda():
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("splat: CUDA not available; install a CUDA build of torch")
    return torch.device("cuda")


def load_scene(ws: Workspace) -> dict:
    return dict(np.load(ws.dir("align") / "scene.npz", allow_pickle=False))


def render(params, c2w, K, width: int, height: int, sh_degree: int, mode: str = "RGB"):
    """Render one camera. Returns (image [H,W,C], alpha [H,W,1], info)."""
    import torch
    gsplat_compat.prepare()
    from gsplat import rasterization

    colors = torch.cat([params["sh0"], params["shN"]], dim=1)
    out, alpha, info = rasterization(
        means=params["means"],
        quats=params["quats"],
        scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]),
        colors=colors,
        viewmats=torch.linalg.inv(c2w)[None],
        Ks=K[None],
        width=width, height=height,
        sh_degree=sh_degree,
        render_mode=mode,
        packed=False,
    )
    return out[0], alpha[0], info


def depth_loss(rendered, target):
    """Mean absolute depth error. `rendered` [H,W] is resized to the sensor map
    `target` [h,w]; target pixels equal to 0 carry no measurement."""
    import torch.nn.functional as F

    r = F.interpolate(rendered[None, None], size=tuple(target.shape), mode="bilinear",
                      align_corners=False)[0, 0]
    valid = target > 0
    if not bool(valid.any()):
        return rendered.sum() * 0.0
    return (r[valid] - target[valid]).abs().mean()


def pose_delta_matrix(delta):
    """[6] (rotation vector, translation) -> [4,4] rigid transform; zeros give identity."""
    import torch

    w, t = delta[:3], delta[3:]
    z = delta.new_zeros(())
    gen = torch.stack([
        torch.stack([z, -w[2], w[1], t[0]]),
        torch.stack([w[2], z, -w[0], t[1]]),
        torch.stack([-w[1], w[0], z, t[2]]),
        delta.new_zeros(4),
    ])
    return torch.linalg.matrix_exp(gen)


def _ssim(a, b):
    """Mean SSIM of two [3,H,W] images, 11 px Gaussian window."""
    import torch
    import torch.nn.functional as F

    g = torch.exp(-(torch.arange(11, device=a.device) - 5.0) ** 2 / (2 * 1.5 ** 2))
    g = g / g.sum()
    win = (g[:, None] * g[None, :]).expand(3, 1, 11, 11).contiguous()

    def blur(x):
        return F.conv2d(x[None], win, padding=5, groups=3)[0]

    mu_a, mu_b = blur(a), blur(b)
    var_a = blur(a * a) - mu_a ** 2
    var_b = blur(b * b) - mu_b ** 2
    cov = blur(a * b) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))
    return s.mean()


def auto_size(steps: int, cap: int, n_frames: int, n_points: int) -> tuple[int, int]:
    """Smaller budget for small captures; `steps` and `cap` are upper limits.

    Time per step grows with the number of Gaussians, and dense LiDAR start points
    reach the cap within a few thousand steps. A short scan trained with the full
    budget spends almost every step at 1.5M Gaussians and gains nothing from it.
    Steps: about 150 passes over the frames. Cap: 3 Gaussians per start point
    (start points are one per 2 cm of scanned surface).
    """
    return (min(steps, max(6_000, 150 * n_frames)),
            min(cap, max(300_000, 3 * n_points)))


def _init_params(scene: dict, cfg, cap: int, device):
    import torch
    from scipy.spatial import cKDTree

    pts, cols = scene["points"], scene["colors"]
    if len(pts) > cap:
        idx = np.random.default_rng(cfg.seed).choice(len(pts), cap, replace=False)
        pts, cols = pts[idx], cols[idx]
    # Size each Gaussian from the distance to its 3 nearest neighbours.
    dist = cKDTree(pts).query(pts, k=4)[0][:, 1:].mean(axis=1)
    dist = np.clip(dist, 1e-4, None)
    n = len(pts)
    k = (cfg.sh_degree + 1) ** 2

    def t(x):
        return torch.nn.Parameter(torch.tensor(x, dtype=torch.float32, device=device))

    quats = np.zeros((n, 4))
    quats[:, 0] = 1.0
    return torch.nn.ParameterDict({
        "means": t(pts),
        "scales": t(np.log(dist * 0.1)[:, None].repeat(3, axis=1)),  # small start, as MCMC expects
        "quats": t(quats),
        "opacities": t(np.full(n, math.log(0.5 / 0.5))),
        "sh0": t(((cols / 255.0 - 0.5) / splat_io.SH_C0)[:, None, :]),
        "shN": t(np.zeros((n, k - 1, 3))),
    })


def run(ws: Workspace, cfg: Config) -> None:
    import cv2
    import torch
    gsplat_compat.prepare()
    from gsplat.strategy import MCMCStrategy

    c = cfg.splat
    device = _require_cuda()
    torch.manual_seed(c.seed)
    scene = load_scene(ws)
    lidar = "source" in scene and str(scene["source"]) == "lidar"
    img_dir = ingest.images(ws) if lidar else poses.undistorted_images(ws)

    # Images stay on the CPU as uint8; one moves to the GPU per step.
    images = []
    for name in scene["names"]:
        bgr = cv2.imread(str(img_dir / str(name)))
        if bgr is None:
            raise SystemExit(f"splat: missing image {name}")
        images.append(torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    # Sensor depth (LiDAR path): small uint16 maps in millimetres, 0 = no measurement.
    depths = None
    if "depth_names" in scene and c.depth_weight > 0:
        depths = []
        for name in scene["depth_names"]:
            mm = cv2.imread(str(ingest.depth_maps(ws) / str(name)), cv2.IMREAD_UNCHANGED)
            if mm is None:
                raise SystemExit(f"splat: missing depth map {name}")
            depths.append(torch.from_numpy(mm.astype(np.int32)))
        depth_unit = float(scene["depth_scale"]) / 1000.0  # millimetres -> scene units
        print(f"splat: depth supervision from {len(depths)} sensor maps")
    c2w = torch.tensor(scene["c2w"], dtype=torch.float32, device=device)
    Ks = torch.tensor(scene["K"], dtype=torch.float32, device=device)

    steps, cap = c.steps, c.cap
    if c.auto_size:
        steps, cap = auto_size(steps, cap, len(images), len(scene["points"]))
        print(f"splat: {len(images)} frames, {len(scene['points'])} start points -> "
              f"{steps} steps, at most {cap} Gaussians", flush=True)
    params = _init_params(scene, c, cap, device)
    centres = scene["c2w"][:, :3, 3]
    scene_scale = float(np.linalg.norm(centres - centres.mean(axis=0), axis=1).max()) * 1.1

    lrs = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3,
           "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    optimizers = {
        name: torch.optim.Adam([{"params": params[name], "lr": lrs[name], "name": name}], eps=1e-15)
        for name in PARAM_NAMES
    }
    sched = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=0.01 ** (1.0 / steps))

    # ARKit poses are a few centimetres off; let each camera move a little.
    pose_delta = pose_opt = pose_sched = None
    if c.pose_opt:
        pose_delta = torch.nn.Parameter(torch.zeros(len(images), 6, device=device))
        pose_opt = torch.optim.Adam([pose_delta], lr=c.pose_lr, weight_decay=1e-6)
        pose_sched = torch.optim.lr_scheduler.ExponentialLR(pose_opt, gamma=0.01 ** (1.0 / steps))

    strategy = MCMCStrategy(cap_max=cap, refine_start_iter=500,
                            refine_stop_iter=int(steps * 0.83), refine_every=100)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state()

    n = len(images)
    order = torch.randperm(n)
    started = lap = time.perf_counter()
    for step in range(steps):
        if step % n == 0:
            order = torch.randperm(n)
        i = int(order[step % n])
        gt = images[i].to(device).float() / 255.0
        h, w = gt.shape[:2]
        degree = min(step // 1000, c.sh_degree)
        cam = c2w[i] if pose_delta is None else c2w[i] @ pose_delta_matrix(pose_delta[i])
        img, _, info = render(params, cam, Ks[i], w, h, degree,
                              mode="RGB" if depths is None else "RGB+ED")
        d_err = None
        if depths is not None:
            img, d_render = img[..., :3], img[..., 3]
            d_err = depth_loss(d_render, depths[i].to(device).float() * depth_unit)
        strategy.step_pre_backward(params, optimizers, state, step, info)

        l1 = (img - gt).abs().mean()
        ssim = _ssim(img.permute(2, 0, 1), gt.permute(2, 0, 1))
        loss = (1 - c.ssim_weight) * l1 + c.ssim_weight * (1 - ssim)
        loss = loss + c.opacity_reg * torch.sigmoid(params["opacities"]).mean()
        loss = loss + c.scale_reg * torch.exp(params["scales"]).mean()
        if d_err is not None:
            loss = loss + c.depth_weight * d_err
        loss.backward()

        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        sched.step()
        if pose_opt is not None:
            pose_opt.step()
            pose_opt.zero_grad(set_to_none=True)
            pose_sched.step()
        strategy.step_post_backward(params, optimizers, state, step, info,
                                    lr=sched.get_last_lr()[0])

        if step % 500 == 0 or step == steps - 1:
            vram = torch.cuda.max_memory_allocated() / 2 ** 30
            extra = "" if d_err is None else f"depth err {d_err.item() * 100:.1f} cm "
            if pose_delta is not None:
                extra += (f"pose shift {pose_delta[:, 3:].norm(dim=1).mean().item() * 100:.2f} cm "
                          f"turn {math.degrees(pose_delta[:, :3].norm(dim=1).mean().item()):.2f} deg ")
            now = time.perf_counter()
            rate = (500 if step else 1) / max(now - lap, 1e-9)
            lap = now
            # flush: progress must show up while training runs, also when piped to a file.
            print(f"splat: step {step}/{steps} loss {loss.item():.4f} {extra}"
                  f"gaussians {len(params['means'])} peak VRAM {vram:.1f} GB "
                  f"{rate:.1f} steps/s, {(now - started) / 60:.1f} min", flush=True)

    d = ws.reset("splat")
    out = {k: v.detach().cpu() for k, v in params.items()}
    torch.save(out, d / "ckpt.pt")
    arr = {k: v.numpy() for k, v in out.items()}
    splat_io.write_ply(d / "scene.ply", **arr)
    splat_io.write_splat(d / "scene.splat", arr["means"], arr["scales"], arr["quats"],
                         arr["opacities"], arr["sh0"])


def load_params(ws: Workspace, device):
    import torch
    ckpt = torch.load(ws.dir("splat") / "ckpt.pt", map_location=device)
    return {k: v.to(device) for k, v in ckpt.items()}
