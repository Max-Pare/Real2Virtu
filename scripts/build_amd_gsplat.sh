#!/usr/bin/env bash
# Build gsplat for an AMD Radeon GPU (ROCm) from AMD's fork and install it into .venv.
#
# Needs: `uv sync --extra amd --extra dev` done first, and the ROCm SDK installed
# (Arch/CachyOS: `sudo pacman -S rocm-hip-sdk`), same major.minor as the torch wheel.
#
# The GPU architecture is read from `rocminfo` (first GPU listed). Override with
# R2V_ROCM_ARCH=gfx1101 when the wrong one is picked (e.g. an integrated GPU).
set -euo pipefail

TAG="gsplat-1.5.3b2"
REPO="https://github.com/AMD-Ecosystem/gsplat.git"

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src="$root/tools/amd_gsplat"
py="$root/.venv/bin/python"

[ -x "$py" ] || { echo "no .venv: run 'uv sync --extra amd --extra dev' first" >&2; exit 1; }
export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export PATH="$ROCM_PATH/bin:$PATH"
command -v hipcc >/dev/null || { echo "hipcc not found: install the ROCm SDK" >&2; exit 1; }
[ -e "$ROCM_PATH/include/rocprim/rocprim.hpp" ] || { echo "rocprim headers missing: install the ROCm SDK" >&2; exit 1; }

if [ ! -d "$src/.git" ]; then
    git clone --recurse-submodules --depth 1 -b "$TAG" "$REPO" "$src"
fi

cd "$src"
git checkout -q -- setup.py gsplat/cuda
# Recent torch (seen with 2.14) keeps the allocator under its CUDA name on ROCm builds.
sed -i 's/::c10::hip::HIPCachingAllocator::get()/::c10::cuda::CUDACachingAllocator::get()/' gsplat/cuda/include/Common.cuh
# setup.py takes the first gfx name rocminfo prints; R2V_ROCM_ARCH overrides it.
arch="${R2V_ROCM_ARCH:-$(rocminfo | grep -oE 'Name:\s+gfx[0-9a-z]+' | head -1 | grep -oE 'gfx[0-9a-z]+')}"
[ -n "$arch" ] || { echo "no gfx architecture found: set R2V_ROCM_ARCH" >&2; exit 1; }
echo "building for $arch"
sed -i "s/gpu_arch = get_rocm_arch()/gpu_arch = \"$arch\"/" setup.py

# Radeon cards (gfx10xx to gfx12xx) run 32-lane wavefronts; the fork assumes 64.
case "$arch" in
    gfx1[0-2][0-9a-f][0-9a-f]) "$py" "$root/scripts/amd_gsplat_wave32.py" "$src" ;;
esac

# GLM is header-only and vendored as a submodule; point the compilers at it
# instead of installing it into ~/.local as AMD's instructions do.
export CPLUS_INCLUDE_PATH="$src/gsplat/cuda/csrc/third_party/glm${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"

rm -rf build dist gsplat/hip
"$py" setup.py bdist_wheel
uv pip install --python "$py" --force-reinstall --no-deps dist/gsplat-*.whl
cd "$root"   # not the source tree, or Python imports that instead of the wheel
"$py" -c "import torch, gsplat; from gsplat.cuda._backend import _C; assert _C is not None; print('gsplat', gsplat.__version__, 'on', torch.cuda.get_device_name(0))"
