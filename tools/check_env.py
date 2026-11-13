"""Checks that everything the r2v pipeline needs is visible from a normal terminal.
Run: .venv\\Scripts\\python.exe tools\\check_env.py
"""
import glob
import shutil
import sys

import r2v  # noqa: F401  before torch: prepares the process for torch on AMD

import torch

from r2v.cli import LOCAL_COLMAP, _winget_exe
from r2v.device import gpu_backend
from r2v.gsplat_compat import CUDA_TOOLKIT_GLOB, find_cuda_toolkit, find_vcvars


def show(name, ok, detail=""):
    print(f"{'OK     ' if ok else 'MISSING'} {name}: {detail}")


show("python", True, sys.base_prefix)
show(f"torch GPU ({gpu_backend() or 'CPU-only build'})", torch.cuda.is_available(), torch.__version__)
ff = shutil.which("ffmpeg") or _winget_exe("Gyan.FFmpeg*", "ffmpeg.exe")
show("ffmpeg", bool(ff), ff or "")
show("COLMAP", LOCAL_COLMAP.exists(), str(LOCAL_COLMAP))
godot = _winget_exe("GodotEngine.GodotEngine*", "Godot*_console.exe")
show("Godot", bool(godot), godot or "")
vc = find_vcvars()
show("MSVC", bool(vc), vc or "need Visual Studio 2022 C++ tools (CUDA 12 rejects newer)")
cuda = find_cuda_toolkit(torch.version.cuda)
installed = ", ".join(sorted(glob.glob(CUDA_TOOLKIT_GLOB))) or "none"
show("CUDA toolkit", bool(cuda),
     cuda or f"need {torch.version.cuda} to match torch; installed: {installed}")
