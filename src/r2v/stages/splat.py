"""Stage 4: train 3D Gaussians with gsplat (MCMC strategy, capped count)."""
from __future__ import annotations

import math
import time

import numpy as np

from .. import device as device_mod
from .. import gsplat_compat, splat_io
from ..config import Config
from ..workspace import Workspace
from . import ingest, poses

PARAM_NAMES = ["means", "scales", "quats", "opacities", "sh0", "shN"]


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


def visible_mask(radii):
    """[N] bool: Gaussians drawn in the rendered frame. `radii` is gsplat's
    screen radius per camera and Gaussian, [C,N] or [C,N,2]; 0 means not drawn."""
    drawn = radii > 0
    if drawn.dim() == 3:
        drawn = drawn.all(dim=-1)
    return drawn.any(dim=0)


def exposure_gain(log_gain, i: int):
    """[3] colour gain of frame `i` from the learned table `log_gain` [n,3].

    Gains are taken relative to the average frame, so the table cannot drift as a
    whole (a darker scene with every gain raised would fit the photos equally well).
    The average is held constant in the gradient: only frame `i` is updated.
    """
    return (log_gain[i] - log_gain.mean(dim=0).detach()).exp()


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


def _photo_loss_conv(img, gt, ssim_weight: float):
    """(1 - w) L1 + w (1 - SSIM) of two [H,W,3] images."""
    l1 = (img - gt).abs().mean()
    return (1 - ssim_weight) * l1 + ssim_weight * (1 - _ssim(img.permute(2, 0, 1), gt.permute(2, 0, 1)))


_WINDOW = [math.exp(-(k - 5.0) ** 2 / (2 * 1.5 ** 2)) for k in range(11)]
_WINDOW = [v / sum(_WINDOW) for v in _WINDOW]


def _blur_shifted(x):
    """The 11 px Gaussian blur of `_ssim` (zero padding) as weighted sums of shifted
    copies, one pass per axis: no convolution."""
    import torch.nn.functional as F

    h, w = x.shape[-2:]
    p = F.pad(x, (5, 5, 0, 0))
    y = _WINDOW[0] * p[..., :, 0:w]
    for k in range(1, 11):
        y = y + _WINDOW[k] * p[..., :, k:k + w]
    p = F.pad(y, (0, 0, 5, 5))
    z = _WINDOW[0] * p[..., 0:h, :]
    for k in range(1, 11):
        z = z + _WINDOW[k] * p[..., k:k + h, :]
    return z


def _photo_loss_shifted(img, gt, ssim_weight: float):
    """`_photo_loss_conv` without convolutions; same value and gradient."""
    import torch

    l1 = (img - gt).abs().mean()
    a, b = img.permute(2, 0, 1), gt.permute(2, 0, 1)
    x = _blur_shifted(torch.cat([a, b, a * a, b * b, a * b]))
    mu_a, mu_b, aa, bb, ab = x[0:3], x[3:6], x[6:9], x[9:12], x[12:15]
    var_a, var_b, cov = aa - mu_a ** 2, bb - mu_b ** 2, ab - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))
    return (1 - ssim_weight) * l1 + ssim_weight * (1 - s.mean())


def make_photo_loss(backend: str):
    """The image loss for a GPU backend: a function (img, gt, ssim_weight) -> scalar.

    NVIDIA: the SSIM blur is a convolution. AMD ROCm: convolutions run in AMD's MIOpen
    library, which is slow here (2 to 3 ms per blur, and SSIM needs five, plus their
    gradients). So the blur is written without convolutions and compiled into a few
    GPU kernels by torch.compile (7 times faster than the convolution on an RX 7800 XT).
    Uncompiled it still works, but slower than the convolution; used when compiling fails.
    """
    if backend != "rocm":
        return _photo_loss_conv
    import os
    from pathlib import Path

    import torch

    # Compiled kernels are kept inside the project; the default is under /tmp, which
    # is emptied on reboot (compiling takes about a minute, loading from here seconds).
    cache = Path(__file__).resolve().parents[3] / "tools" / "torch_compile_cache"
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache))
    use = [torch.compile(_photo_loss_shifted)]

    def photo_loss(img, gt, ssim_weight: float):
        try:
            return use[0](img, gt, ssim_weight)
        except Exception as err:
            if use[0] is _photo_loss_shifted:
                raise
            print(f"splat: could not compile the loss ({type(err).__name__}: {str(err)[:200]}); "
                  f"running it uncompiled, which is slower", flush=True)
            use[0] = _photo_loss_shifted
            return use[0](img, gt, ssim_weight)

    return photo_loss


def position_noise(params, scaler: float) -> None:
    """The MCMC strategy's random nudge of the Gaussian centres, in place.

    Same result as gsplat's `inject_noise_to_position`, which multiplies each 3x3
    covariance with its noise vector as a batched matrix product. On AMD (ROCm) that
    one call took 19 ms for 1.5M Gaussians, a third of a training step; written
    elementwise it takes 0.4 ms.
    """
    import torch
    from gsplat import quat_scale_to_covar_preci

    with torch.no_grad():
        opacities = torch.sigmoid(params["opacities"].flatten())
        covars, _ = quat_scale_to_covar_preci(params["quats"], torch.exp(params["scales"]),
                                              compute_covar=True, compute_preci=False, triu=False)
        # Only nearly transparent Gaussians move: the factor is 0.5 at 0.5% opacity
        # and under 0.001 above 7%.
        fade = 1 / (1 + torch.exp(-100 * ((1 - opacities) - 0.995)))
        noise = torch.randn_like(params["means"]) * fade[:, None] * scaler
        params["means"].add_((covars * noise[:, None, :]).sum(dim=-1))


def _use_fast_position_noise() -> None:
    """Point gsplat's MCMC strategy at `position_noise`. Left alone when gsplat's
    layout differs from the versions this was written for (1.4 to 1.5)."""
    from gsplat.strategy import mcmc

    if hasattr(mcmc, "inject_noise_to_position"):
        mcmc.inject_noise_to_position = (
            lambda params, optimizers=None, state=None, scaler=1.0: position_noise(params, scaler))


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


def scaled_size(w: int, h: int, percent: float) -> tuple[int, int]:
    """Frame size at `percent` of the source; never larger than the source."""
    f = min(max(percent, 1.0), 100.0) / 100.0
    return max(1, int(round(w * f))), max(1, int(round(h * f)))


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
    device = device_mod.resolve(cfg.device)
    torch.manual_seed(c.seed)
    scene = load_scene(ws)
    lidar = "source" in scene and str(scene["source"]) == "lidar"
    img_dir = ingest.images(ws) if lidar else poses.undistorted_images(ws)

    # Images stay on the CPU as uint8; one moves to the GPU per step.
    images = []
    K_train = scene["K"].copy()
    for n_img, name in enumerate(scene["names"]):
        bgr = cv2.imread(str(img_dir / str(name)))
        if bgr is None:
            raise SystemExit(f"splat: missing image {name}")
        h0, w0 = bgr.shape[:2]
        w1, h1 = scaled_size(w0, h0, c.train_scale)
        if (w1, h1) != (w0, h0):
            bgr = cv2.resize(bgr, (w1, h1), interpolation=cv2.INTER_AREA)
            K_train[n_img, 0] *= w1 / w0
            K_train[n_img, 1] *= h1 / h0
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
    w_src, h_src = (int(v) for v in scene["sizes"][0])
    print(f"splat: training at {c.train_scale:g}% of the source frames "
          f"({w_src}x{h_src} -> {images[0].shape[1]}x{images[0].shape[0]})", flush=True)
    c2w = torch.tensor(scene["c2w"], dtype=torch.float32, device=device)
    Ks = torch.tensor(K_train, dtype=torch.float32, device=device)

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
    # fused: one GPU kernel per update instead of a dozen small ones (halves the
    # optimiser time at 1.5M Gaussians; same result).
    optimizers = {
        name: torch.optim.Adam([{"params": params[name], "lr": lrs[name], "name": name}],
                               eps=1e-15, fused=True)
        for name in PARAM_NAMES
    }
    sched = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=0.01 ** (1.0 / steps))

    # ARKit poses are a few centimetres off; let each camera move a little.
    pose_delta = pose_opt = pose_sched = None
    if c.pose_opt:
        pose_delta = torch.nn.Parameter(torch.zeros(len(images), 6, device=device))
        pose_opt = torch.optim.Adam([pose_delta], lr=c.pose_lr, weight_decay=1e-6)
        pose_sched = torch.optim.lr_scheduler.ExponentialLR(pose_opt, gamma=0.01 ** (1.0 / steps))

    # Auto exposure and white balance differ per frame; let each frame have its own gain.
    log_gain = gain_opt = None
    if c.exposure_opt:
        log_gain = torch.nn.Parameter(torch.zeros(len(images), 3, device=device))
        gain_opt = torch.optim.Adam([log_gain], lr=c.exposure_lr)

    _use_fast_position_noise()
    strategy = MCMCStrategy(cap_max=cap, refine_start_iter=500,
                            refine_stop_iter=int(steps * 0.83), refine_every=100)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state()

    backend = device_mod.gpu_backend()
    photo_loss = make_photo_loss(backend)
    if backend == "rocm":
        # First call compiles (or loads) the loss; keep that out of the step timing.
        print("splat: preparing the loss for this GPU (10 s or more the first time)", flush=True)
        h0, w0 = images[0].shape[:2]
        probe = torch.full((h0, w0, 3), 0.5, device=device, requires_grad=True)
        photo_loss(probe, torch.full((h0, w0, 3), 0.4, device=device), c.ssim_weight).backward()
        del probe

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

        if log_gain is not None:
            img = img * exposure_gain(log_gain, i)
        loss = photo_loss(img, gt, c.ssim_weight)
        # Regularise only the Gaussians this frame sees. Over all of them, one that is
        # out of view gets nothing but the regulariser, and Adam turns that into a
        # full-size step: in a large scan, where each frame sees a small part, opacity
        # then falls to zero between sightings and the Gaussians die and get
        # relocated in bulk (opacity collapse).
        seen = visible_mask(info["radii"])
        if bool(seen.any()):
            loss = loss + c.opacity_reg * torch.sigmoid(params["opacities"][seen]).mean()
            loss = loss + c.scale_reg * torch.exp(params["scales"][seen]).mean()
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
        if gain_opt is not None:
            gain_opt.step()
            gain_opt.zero_grad(set_to_none=True)
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
    if pose_delta is not None:
        # The poses the splat was fitted to; scene.npz keeps the unrefined ones.
        with torch.no_grad():
            refined = torch.stack([c2w[i] @ pose_delta_matrix(pose_delta[i]) for i in range(n)])
        np.save(d / "refined_c2w.npy", refined.cpu().numpy())
    if log_gain is not None:
        # Per frame and colour channel, relative to the average frame; the splat itself
        # holds the scene at that average exposure.
        with torch.no_grad():
            gains = (log_gain - log_gain.mean(dim=0)).exp().cpu().numpy()
        np.save(d / "exposure_gain.npy", gains)
        print(f"splat: learned frame gains {gains.min():.2f} to {gains.max():.2f} "
              f"(1 = average exposure)", flush=True)
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
