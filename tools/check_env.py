"""Checks that everything the r2v pipeline needs is visible from a normal terminal.
Run: .venv\\Scripts\\python.exe tools\\check_env.py
"""
import glob
import shutil
import sys

import torch

from r2v.cli import LOCAL_COLMAP, _winget_exe


def show(name, ok, detail=""):
    print(f"{'OK     ' if ok else 'MISSING'} {name}: {detail}")


show("python", True, sys.base_prefix)
show("torch CUDA", torch.cuda.is_available(), torch.__version__)
ff = shutil.which("ffmpeg") or _winget_exe("Gyan.FFmpeg*", "ffmpeg.exe")
show("ffmpeg", bool(ff), ff or "")
show("COLMAP", LOCAL_COLMAP.exists(), str(LOCAL_COLMAP))
godot = _winget_exe("GodotEngine.GodotEngine*", "Godot*_console.exe")
show("Godot", bool(godot), godot or "")
vc = glob.glob(r"C:\Program Files*\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat")
show("MSVC", bool(vc), vc[-1] if vc else "")
cuda = glob.glob(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*")
show("CUDA toolkit", bool(cuda), cuda[-1] if cuda else "")
