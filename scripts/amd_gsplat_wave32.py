"""Rewrite AMD's gsplat fork for 32-lane wavefronts (Radeon cards).

The fork targets AMD's datacentre GPUs, whose wavefronts ("warps") hold 64 threads,
and hardcodes that number in its reductions. Radeon RX cards run 32. Forcing them
to 64 compiles but gives wrong gradients: lane shuffles only reach half the wave.

Usage: python amd_gsplat_wave32.py <gsplat source dir>   (run by build_amd_gsplat.sh)
"""
import re
import sys
from pathlib import Path

RULES = [
    (r"threadIdx\.x % 64\b", "threadIdx.x % 32"),
    (r"\bi < 64; \+\+i", "i < 32; ++i"),
    (r"(warp_reduce<\s*(?:float|int32_t)\s*,\s*)64>", r"\g<1>32>"),
    (r"(rocprim_warpSum<(?:\s*\w+\s*,)?\s*)64>", r"\g<1>32>"),
    (r"(tiled_partition|thread_block_tile)<64>", r"\g<1><32>"),
    (r"\(block_size \+ 63\) / 64", "(block_size + 31) / 32"),
    (r"LOGICAL_WARP_SIZE = 64\b", "LOGICAL_WARP_SIZE = 32"),
    # shuffle-based tree reductions start at half the wave
    (r"for \(int offset = 32\s*; offset > 0; offset /= 2\)", "for (int offset = 16; offset > 0; offset /= 2)"),
    # a kernel specialised for "one 8x8 tile is one wave"; fall back to the general one
    (r"\(block_size == 64\) \?", "(false) ?"),
    (r"if \(block_size == 64\) \{", "if (false) {"),
]


def main(src: Path) -> None:
    cuda = src / "gsplat" / "cuda"
    files = sorted(cuda.glob("csrc/*.cu")) + sorted(cuda.glob("csrc/*.cuh")) + sorted(cuda.glob("include/*.cuh"))
    hits = [0] * len(RULES)
    for f in files:
        text = f.read_text()
        new = text
        for n, (pat, rep) in enumerate(RULES):
            new, k = re.subn(pat, rep, new)
            hits[n] += k
        if new != text:
            f.write_text(new)
    for (pat, _), k in zip(RULES, hits):
        print(f"{k:4d}  {pat}")
    if not all(hits):
        sys.exit("wave32 patch: a rule matched nothing; the fork's sources changed, review the rules")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
