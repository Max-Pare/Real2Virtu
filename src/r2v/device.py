"""Which GPU backend the torch stages (splat, clean, collision) run on.

Two backends: "cuda" (NVIDIA) and "rocm" (AMD Radeon). They are different builds of
torch, so which one is available is decided at install time (`uv sync --extra gpu`
or `--extra amd`); torch addresses both as device "cuda". "auto" takes whichever
the installed torch was built for. There is no CPU path: gsplat only rasterises on a GPU.
"""
from __future__ import annotations

CHOICES = ("auto", "cuda", "rocm")
_EXTRA = {"cuda": "gpu", "rocm": "amd"}
_LABEL = {"cuda": "NVIDIA CUDA", "rocm": "AMD ROCm"}

_announced = False


def gpu_backend() -> str | None:
    """"cuda" or "rocm": the GPU backend the installed torch was built for; None for a CPU build."""
    import torch

    if getattr(torch.version, "hip", None):
        return "rocm"
    if torch.version.cuda:
        return "cuda"
    return None


def pick(choice: str = "auto") -> str:
    """Turn a --device value into the backend to use: "cuda" or "rocm"."""
    import torch

    if choice not in CHOICES:
        raise SystemExit(f"unknown device '{choice}'; choose from {', '.join(CHOICES)}")
    built = gpu_backend()
    if built is None:
        extra = _EXTRA.get(choice)
        hint = f"'uv sync --extra {extra}'" if extra else "'uv sync --extra gpu' (NVIDIA) or '--extra amd' (AMD)"
        raise SystemExit(f"device: this environment has a CPU-only build of torch "
                         f"({torch.__version__}); install with {hint}")
    if choice != "auto" and choice != built:
        raise SystemExit(f"--device {choice}: this environment has the {_LABEL[built]} build of torch "
                         f"({torch.__version__}). Install with 'uv sync --extra {_EXTRA[choice]}', "
                         f"or use --device auto.")
    if not torch.cuda.is_available():
        raise SystemExit(f"device: torch ({_LABEL[built]} build) finds no usable GPU; check the driver")
    return built


def resolve(choice: str = "auto"):
    """The torch.device for a --device value. Says once per process which backend runs."""
    import torch

    global _announced
    backend = pick(choice)
    if not _announced:
        _announced = True
        print(f"device: {_LABEL[backend]}, {torch.cuda.get_device_name(0)}", flush=True)
    return torch.device("cuda")
