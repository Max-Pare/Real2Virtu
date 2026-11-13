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

import ctypes
import glob
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

_done = False
# Path of the ROCm runtime loaded by use_system_rocm_runtime(); None: torch uses its own.
rocm_runtime: str | None = None


def find_vcvars() -> str | None:
    """vcvars64.bat of a Visual Studio that the CUDA 12 toolkit accepts, or None.

    nvcc from CUDA 12.x refuses compilers newer than Visual Studio 2022, so a
    newer Visual Studio installed next to it must not be picked.
    """
    for year in ("2022", "2019", "2017"):
        hits = sorted(glob.glob(
            rf"C:\Program Files*\Microsoft Visual Studio\{year}\*\VC\Auxiliary\Build\vcvars64.bat"))
        if hits:
            return hits[-1]
    return None


def _load_msvc_env() -> None:
    if shutil.which("cl"):
        return
    vcvars = find_vcvars()
    if vcvars is None:
        raise SystemExit(
            "gsplat needs the Visual Studio 2022 C++ compiler to build (CUDA 12 does not accept "
            "newer versions).\nRun setup.bat, or install 'Visual Studio Build Tools 2022' with the "
            "C++ workload; it installs next to other versions.")
    out = subprocess.run(f'"{vcvars}" >nul 2>&1 && set', shell=True,
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


def point_to_gpu_id_table() -> None:
    """ROCm torch bundles a libdrm that looks for AMD's GPU name table in /opt/amdgpu
    and prints "(null): No such file or directory" when it is missing. Point it at
    the system copy. Must happen before torch is imported, so r2v/__init__ calls it."""
    if sys.platform != "linux":
        return
    table = Path("/usr/share/libdrm")
    if (table / "amdgpu.ids").exists():
        os.environ.setdefault("AMDGPU_ASIC_ID_TABLE_PATHS", str(table))


def use_system_rocm_runtime() -> str | None:
    """AMD on Linux: make torch use the installed ROCm runtime instead of its own copy.

    The ROCm torch wheel ships its own libhsa-runtime64.so (ROCR, the layer between
    HIP and the kernel driver). With the one in the rocm7.2 wheel (build
    1.18.0-rocm-rel-7.2-81), processes segfaulted at random places after GPU work:
    in its own event thread (rocr::core::Runtime::AsyncEventsLoop), in malloc, in
    Python's garbage collector. A build with 30,000 training steps crashed in the
    stage after training 3 times out of 3, and a 20 second torch-only test 16 times
    out of 16. With the distribution's runtime (hsa-rocr 7.2.4) loaded first: 0 of 2
    and 0 of 11.

    Loading the installed library with global symbols before torch makes torch's HIP
    library bind to it. Must run before torch is imported, so r2v/__init__ calls it;
    a script that imports torch first gets torch's own copy. Skipped when the
    installed ROCm and the wheel differ in major.minor version.
    R2V_ROCM_RUNTIME=bundled turns it off. Returns the path loaded, or None.
    """
    global rocm_runtime
    if rocm_runtime is not None:
        return rocm_runtime
    if sys.platform != "linux" or os.environ.get("R2V_ROCM_RUNTIME", "").lower() == "bundled":
        return None
    if "torch" in sys.modules:
        return None  # too late: torch has bound to its own copy
    try:
        spec = importlib.util.find_spec("torch")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin:
        return None
    torch_dir = Path(spec.origin).parent
    if not (torch_dir / "lib" / "libhsa-runtime64.so").exists():
        return None  # not a ROCm build of torch
    rocm_lib = Path(os.environ.get("ROCM_PATH") or "/opt/rocm") / "lib"
    system = rocm_lib / "libhsa-runtime64.so.1"
    if not system.exists():
        return None
    wheel = re.search(r"^hip\b[^=]*=\s*'(\d+)\.(\d+)", (torch_dir / "version.py").read_text(), re.M)
    installed = {tuple(m.groups()) for m in
                 (re.match(r"libamdhip64\.so\.(\d+)\.(\d+)\.", p.name) for p in rocm_lib.glob("libamdhip64.so.*.*.*")) if m}
    if not wheel or tuple(wheel.groups()) not in installed:
        return None
    try:
        ctypes.CDLL(str(system), mode=ctypes.RTLD_GLOBAL)
    except OSError:
        return None
    rocm_runtime = str(system)
    return rocm_runtime


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
