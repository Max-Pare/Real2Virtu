"""Import r2v before any test module imports torch.

r2v's import prepares the process for torch on AMD (see gsplat_compat.use_system_rocm_runtime),
which only works while torch is not loaded yet.
"""
import r2v  # noqa: F401
