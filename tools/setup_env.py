"""Sets up everything the r2v pipeline needs on a new Windows machine.

Started by setup.bat, which first puts a signed Python in tools\\python312_signed.
Uses the standard library only, because it runs before the venv exists.

Steps, each skipped when already done:
  uv        package manager, into tools\\uv unless one is already installed
  venv      .venv on the signed Python, packages from uv.lock (torch CUDA 12.8, gsplat, open3d)
  system    winget: ffmpeg, Godot, Visual Studio Build Tools (C++), CUDA Toolkit 12.8
  colmap    COLMAP CUDA build into tools\\colmap
  check     tools\\check_env.py
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
VENV = ROOT / ".venv"
VENV_PYTHON = VENV / "Scripts" / "python.exe"

UV_VERSION = "0.13.7"
UV_URL = f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/uv-x86_64-pc-windows-msvc.zip"
UV_SHA256 = "0cdc5b38fe5a8168ed6a25814f5ba43c21d313dd3187314fccb37b7cec46e021"

COLMAP_VERSION = "4.2.1"
COLMAP_URL = f"https://github.com/colmap/colmap/releases/download/{COLMAP_VERSION}/colmap-x64-windows-cuda.zip"
COLMAP_SHA256 = "e9c5cbd84c2ea986d2e970a2473fc2d2e6b34a2cdcf5d3df2765c319a63af881"

# 2022 exactly: nvcc from CUDA 12.8 refuses newer Visual Studio versions.
VCVARS_GLOB = r"C:\Program Files*\Microsoft Visual Studio\2022\*\VC\Auxiliary\Build\vcvars64.bat"
CUDA_GLOB = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8*"

failed: list[str] = []


def winget_exe(package_glob: str, exe: str) -> str | None:
    """winget portable installs are not always linked onto PATH; look in its package folder."""
    base = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    if base.is_dir():
        for pkg in base.glob(package_glob):
            for hit in pkg.rglob(exe):
                return str(hit)
    return None


def download(url: str, dest: Path, sha256: str) -> None:
    print(f"downloading {url}")
    digest = hashlib.sha256()
    with urllib.request.urlopen(url) as response, open(dest, "wb") as out:
        while chunk := response.read(1 << 20):
            digest.update(chunk)
            out.write(chunk)
    if digest.hexdigest() != sha256:
        dest.unlink()
        raise SystemExit(f"{dest.name}: SHA-256 does not match the expected value; not using it.")


def find_uv() -> str | None:
    local = TOOLS / "uv" / "uv.exe"
    if local.exists():
        return str(local)
    return shutil.which("uv") or winget_exe("astral-sh.uv*", "uv.exe")


def step_uv() -> str:
    print("== uv ==")
    uv = find_uv()
    if uv:
        print(f"found: {uv}")
        return uv
    archive = TOOLS / "uv.zip"
    download(UV_URL, archive, UV_SHA256)
    with zipfile.ZipFile(archive) as z:
        z.extractall(TOOLS / "uv")
    archive.unlink()
    uv = str(TOOLS / "uv" / "uv.exe")
    print(f"installed: {uv}")
    return uv


def step_venv(uv: str) -> None:
    print("== venv ==")
    python = sys.executable  # the signed Python that setup.bat started us with
    env = dict(os.environ, UV_PYTHON_DOWNLOADS="never")
    if not (VENV / "pyvenv.cfg").exists():
        subprocess.run([uv, "venv", str(VENV), "--python", python, "--prompt", "r2v"],
                       cwd=ROOT, env=env, check=True)
    else:
        home = next((line.split("=", 1)[1].strip()
                     for line in (VENV / "pyvenv.cfg").read_text().splitlines()
                     if line.startswith("home")), "")
        if Path(home).resolve() != Path(python).resolve().parent:
            raise SystemExit(
                f".venv is built on another Python ({home}).\n"
                "Delete the .venv folder and run setup again.")
        print(f"found: {VENV}")
    # --locked: install exactly what uv.lock records, and fail if it is out of date.
    subprocess.run([uv, "sync", "--locked", "--extra", "gpu", "--extra", "dev",
                    "--python", str(VENV_PYTHON)], cwd=ROOT, env=env, check=True)


def winget_install(label: str, present: bool, args: list[str]) -> None:
    if present:
        print(f"found: {label}")
        return
    print(f"installing {label} (winget may ask for administrator rights) ...")
    result = subprocess.run(
        ["winget", "install", "--exact", "--accept-package-agreements",
         "--accept-source-agreements", *args])
    if result.returncode != 0:
        print(f"FAILED: {label} (winget exit code {result.returncode})")
        failed.append(label)


def step_system() -> None:
    print("== system ==")
    if shutil.which("winget") is None:
        print("FAILED: winget not found. Install 'App Installer' from the Microsoft Store, "
              "or install ffmpeg, Godot, VS Build Tools (C++) and CUDA Toolkit 12.8 by hand.")
        failed.append("winget")
        return
    winget_install(
        "ffmpeg",
        bool(shutil.which("ffmpeg") or winget_exe("Gyan.FFmpeg*", "ffmpeg.exe")),
        ["--id", "Gyan.FFmpeg"])
    winget_install(
        "Godot",
        bool(winget_exe("GodotEngine.GodotEngine*", "Godot*_console.exe")),
        ["--id", "GodotEngine.GodotEngine"])
    # gsplat compiles its CUDA code on first use: needs the C++ compiler and the
    # CUDA Toolkit matching the torch build (cu128).
    winget_install(
        "Visual Studio Build Tools 2022 (C++)",
        bool(glob.glob(VCVARS_GLOB)),
        ["--id", "Microsoft.VisualStudio.2022.BuildTools", "--override",
         "--passive --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"])
    winget_install(
        "CUDA Toolkit 12.8",
        # Exactly this version: a newer major (13.x) cannot compile gsplat for this torch.
        # It installs next to other CUDA versions.
        bool(glob.glob(CUDA_GLOB)),
        ["--id", "Nvidia.CUDA", "--version", "12.8"])


def step_colmap() -> None:
    print("== colmap ==")
    target = TOOLS / "colmap"
    if (target / "bin" / "colmap.exe").exists():
        print(f"found: {target}")
        return
    archive = TOOLS / "colmap.zip"
    unpack = TOOLS / "_colmap_unpack"
    download(COLMAP_URL, archive, COLMAP_SHA256)
    shutil.rmtree(unpack, ignore_errors=True)
    with zipfile.ZipFile(archive) as z:
        z.extractall(unpack)
    # The archive may or may not wrap its content in a top-level folder.
    hits = list(unpack.rglob("bin/colmap.exe"))
    if not hits:
        raise SystemExit("colmap.exe not found inside the COLMAP archive.")
    shutil.rmtree(target, ignore_errors=True)
    shutil.move(str(hits[0].parents[1]), str(target))
    shutil.rmtree(unpack, ignore_errors=True)
    archive.unlink()
    print(f"installed: {target}")


def step_check() -> None:
    print("== check ==")
    result = subprocess.run([str(VENV_PYTHON), str(TOOLS / "check_env.py")],
                            cwd=ROOT, capture_output=True, text=True)
    print(result.stdout, end="")
    print(result.stderr, end="")
    if result.returncode != 0 or "MISSING" in result.stdout:
        failed.append("environment check")


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up the r2v environment.")
    parser.add_argument("--skip-system", action="store_true",
                        help="do not install ffmpeg, Godot, VS Build Tools and the CUDA Toolkit")
    parser.add_argument("--skip-colmap", action="store_true",
                        help="do not download COLMAP (only the video path needs it)")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # keep our lines in order with uv's and winget's
    if sys.platform != "win32":
        raise SystemExit("This script sets up Windows machines only.")

    uv = step_uv()
    step_venv(uv)
    if not args.skip_system:
        step_system()
    if not args.skip_colmap:
        step_colmap()
    step_check()

    if failed:
        print("\nSetup finished with problems: " + ", ".join(failed))
        print("Fix them and run setup.bat again; finished steps are skipped.")
        raise SystemExit(1)
    print(r"""
Setup complete. Try:
    .venv\Scripts\r2v.exe build --help
If Windows blocks r2v.exe (Smart App Control), use instead:
    .venv\Scripts\python.exe -m r2v build --help
The first splat training compiles gsplat's CUDA code (about 80 s).""")


if __name__ == "__main__":
    main()
