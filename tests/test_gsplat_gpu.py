"""gsplat's compiled rasteriser against a plain torch one, on the GPU.

Guards the AMD build (scripts/build_amd_gsplat.sh), where the kernels are patched
for the card: a wrong wavefront size compiles and renders but returns bad gradients.
Skipped without a GPU or without gsplat.
"""
import math

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no GPU", allow_module_level=True)

from r2v import gsplat_compat  # noqa: E402

gsplat_compat.prepare()
pytest.importorskip("gsplat")

W, H = 96, 64


def _scene(n, seed):
    g = torch.Generator().manual_seed(seed)
    means = torch.rand(n, 3, generator=g) * torch.tensor([3.0, 2.0, 4.0]) + torch.tensor([-1.5, -1.0, 2.0])
    p = {
        "means": means,
        "quats": torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=-1),
        "scales": torch.rand(n, 3, generator=g) * 0.13 + 0.02,
        "opacities": torch.rand(n, generator=g) * 0.6 + 0.3,
        "colors": torch.rand(n, 3, generator=g),
    }
    K = torch.tensor([[80.0, 0, W / 2], [0, 80.0, H / 2], [0, 0, 1]])
    return {k: v.cuda().requires_grad_() for k, v in p.items()}, torch.eye(4).cuda(), K.cuda()


def _reference(p, viewmat, K):
    """Sorted alpha compositing over every Gaussian for every pixel; returns [H,W,4] (RGB, depth)."""
    from gsplat.cuda._torch_impl import _fully_fused_projection, _quat_scale_to_covar_preci

    covars, _ = _quat_scale_to_covar_preci(p["quats"], p["scales"], True, False, triu=False)
    radii, means2d, depths, conics, _ = _fully_fused_projection(p["means"], covars, viewmat[None], K[None], W, H)
    keep = (radii[0] > 0).all(dim=-1)
    order = torch.argsort(depths[0][keep])
    xy, con, z = means2d[0][keep][order], conics[0][keep][order], depths[0][keep][order]
    opac, rgb = p["opacities"][keep][order], p["colors"][keep][order]

    v, u = torch.meshgrid(torch.arange(H, device="cuda") + 0.5, torch.arange(W, device="cuda") + 0.5, indexing="ij")
    dx = u.reshape(-1, 1) - xy[:, 0]
    dy = v.reshape(-1, 1) - xy[:, 1]
    sigma = 0.5 * (con[:, 0] * dx * dx + con[:, 2] * dy * dy) + con[:, 1] * dx * dy
    alpha = torch.clamp(opac * torch.exp(-sigma), max=0.999)
    alpha = torch.where((sigma >= 0) & (alpha >= 1 / 255), alpha, torch.zeros_like(alpha))
    t_after = torch.cumprod(1 - alpha, dim=1)
    t_before = torch.cat([torch.ones_like(t_after[:, :1]), t_after[:, :-1]], dim=1)
    weight = alpha * t_before * (t_after > 1e-4)
    acc = weight.sum(dim=1, keepdim=True)
    depth = (weight @ z[:, None]) / acc.clamp(min=1e-10)
    return torch.cat([weight @ rgb, depth], dim=1).reshape(H, W, 4)


def _compiled(p, viewmat, K):
    from gsplat import rasterization

    out, _, _ = rasterization(
        means=p["means"], quats=p["quats"], scales=p["scales"], opacities=p["opacities"],
        colors=p["colors"], viewmats=viewmat[None], Ks=K[None], width=W, height=H,
        render_mode="RGB+ED", packed=False)
    return out[0]


@pytest.mark.parametrize("n,seed", [(40, 0), (400, 1), (3000, 2)])
def test_render_and_gradients_match_reference(n, seed):
    p, viewmat, K = _scene(n, seed)
    names = list(p)
    probe = torch.randn(H, W, 4, generator=torch.Generator().manual_seed(seed + 100)).cuda()

    got = _compiled(p, viewmat, K)
    want = _reference(p, viewmat, K)
    assert torch.isfinite(got).all()
    assert (got - want).abs().max() < 2e-3

    g_got = torch.autograd.grad((got * probe).sum(), [p[k] for k in names])
    g_want = torch.autograd.grad((want * probe).sum(), [p[k] for k in names])
    for name, a, b in zip(names, g_got, g_want):
        assert torch.isfinite(a).all(), name
        cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)
        assert cos > 0.999, f"{name}: gradient direction differs (cos {cos:.4f})"
        assert 0.99 < a.norm() / b.norm() < 1.01, f"{name}: gradient size differs"


def test_mcmc_relocation_runs():
    """The MCMC strategy's relocation uses its own kernel; check it returns sane values."""
    from gsplat.relocation import compute_relocation

    opac = torch.rand(1000, device="cuda") * 0.8 + 0.1
    scales = torch.rand(1000, 3, device="cuda") * 0.1 + 0.01
    ratios = torch.randint(1, 5, (1000,), device="cuda")
    binoms = torch.tensor([[float(math.comb(n, k)) for k in range(51)] for n in range(51)], device="cuda")
    new_opac, new_scales = compute_relocation(opac, scales, ratios, binoms)
    assert torch.isfinite(new_opac).all() and torch.isfinite(new_scales).all()
    assert (new_opac <= opac + 1e-6).all() and (new_opac > 0).all()
    same = ratios == 1
    assert torch.allclose(new_opac[same], opac[same], atol=1e-5)


def test_position_noise_matches_gsplat():
    """r2v's elementwise MCMC noise step against gsplat's own, same random numbers."""
    from gsplat.strategy.ops import inject_noise_to_position

    from r2v.stages.splat import position_noise

    def params():
        g = torch.Generator().manual_seed(3)
        return {"means": torch.randn(5000, 3, generator=g).cuda(),
                "quats": torch.randn(5000, 4, generator=g).cuda(),
                "scales": (torch.randn(5000, 3, generator=g) - 3).cuda(),
                "opacities": (torch.randn(5000, generator=g) * 3 - 4).cuda()}

    ours, theirs = params(), params()
    before = ours["means"].clone()
    torch.manual_seed(7)
    position_noise(ours, scaler=40.0)
    torch.manual_seed(7)
    with torch.no_grad():
        inject_noise_to_position(params=theirs, optimizers={}, state={}, scaler=40.0)
    moved = (theirs["means"] - before).abs().max()
    assert moved > 1e-3
    assert (ours["means"] - theirs["means"]).abs().max() < 1e-5 * (1 + moved)
