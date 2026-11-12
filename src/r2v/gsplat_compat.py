"""Make gsplat's first-use CUDA compile work on Windows. Call prepare() before importing gsplat.

gsplat ships no prebuilt wheel for recent torch/CUDA, so it compiles itself on first
use. On Windows that needs three fixes, all applied to this process only:

1. The MSVC and CUDA toolkit environment must be loaded (normally a "Developer
   Command Prompt"); we import it from vcvars64.bat.
2. Windows headers define a `small` macro that breaks a torch CUDA header;
   WIN32_LEAN_AND_MEAN keeps those headers out.
3. gsplat passes the GCC-only flag -Wno-attributes, which MSVC rejects.
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

_done = False


def _load_msvc_env() -> None:
    if shutil.which("cl"):
        return
    hits = sorted(glob.glob(
        r"C:\Program Files*\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat"))
    if not hits:
        return  # gsplat will report the missing compiler itself
    out = subprocess.run(f'"{hits[-1]}" >nul 2>&1 && set', shell=True,
                         capture_output=True, text=True).stdout
    for line in out.splitlines():
        key, sep, value = line.partition("=")
        if sep and key:
            os.environ[key] = value


CUDA_TOOLKIT_GLOB = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*"


def find_cuda_toolkit(torch_cuda: str | None) -> str | None:
    """Installed CUDA Toolkit folder that fits the torch build, or None.

    Several toolkits can be installed side by side, and CUDA_PATH points at the
    newest. gsplat must be compiled with the major version torch was built for:
    CUDA 13 headers do not compile with the flags torch passes for a cu12x build.
    Prefers the exact major.minor, then the newest toolkit of the same major.
    """
    if not torch_cuda:
        return None
    want = tuple(int(p) for p in torch_cuda.split(".")[:2])
    found = []
    for path in glob.glob(CUDA_TOOLKIT_GLOB):
        try:
            version = tuple(int(p) for p in Path(path).name[1:].split(".")[:2])
        except ValueError:
            continue
        if version[0] == want[0]:
            found.append((version == want, version, path))
    return max(found)[2] if found else None


def _load_cuda_env() -> None:
    import torch

    cuda = find_cuda_toolkit(torch.version.cuda)
    if cuda is None:
        installed = ", ".join(Path(p).name for p in sorted(glob.glob(CUDA_TOOLKIT_GLOB))) or "none"
        raise SystemExit(
            f"gsplat needs CUDA Toolkit {torch.version.cuda} to compile (torch is built for it); "
            f"installed: {installed}.\nRun setup.bat, or: winget install --exact --id Nvidia.CUDA "
            f"--version {torch.version.cuda}  (it installs next to other versions).")
    # Both names, overriding the machine-wide value: torch reads CUDA_HOME first.
    os.environ["CUDA_PATH"] = os.environ["CUDA_HOME"] = cuda
    os.environ["PATH"] = os.path.join(cuda, "bin") + os.pathsep + os.environ["PATH"]


def _strip_gcc_flags() -> None:
    import torch.utils.cpp_extension as ext

    original = ext._jit_compile

    def clean(value):
        if isinstance(value, list):
            return [v for v in value if v != "-Wno-attributes"]
        return value

    def patched(*args, **kwargs):
        return original(*[clean(a) for a in args], **{k: clean(v) for k, v in kwargs.items()})

    ext._jit_compile = patched


def prepare() -> None:
    global _done
    if _done or sys.platform != "win32":
        return
    _done = True
    # Keep the compiled extension inside the project. The default is under AppData,
    # which some launchers (sandboxed apps) redirect, forcing a recompile per launcher.
    cache = Path(__file__).resolve().parents[2] / "tools" / "torch_extensions"
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(cache))
    _load_msvc_env()
    _load_cuda_env()
    os.environ.setdefault("NVCC_APPEND_FLAGS", "-DWIN32_LEAN_AND_MEAN")
    os.environ.setdefault("CL", "/DWIN32_LEAN_AND_MEAN")
    _strip_gcc_flags()
