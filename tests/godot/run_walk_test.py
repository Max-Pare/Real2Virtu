"""Run the headless walk test against the collision data of an exported project.

  python tests/godot/run_walk_test.py out/scan2_clean [more projects...]

Builds a small scratch project from the current godot_template plus the project's
collision.glb and scene.json (no splat: physics only), so it tests the template
as it is now, not the copy inside the exported project.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from r2v.cli import _winget_exe  # noqa: E402


def run(project: Path, godot: str) -> str:
    with tempfile.TemporaryDirectory(prefix="r2v_walk_") as tmp:
        game = Path(tmp) / "game"
        shutil.copytree(ROOT / "godot_template", game)
        (game / "assets").mkdir(exist_ok=True)
        shutil.copy2(project / "assets" / "collision.glb", game / "assets" / "collision.glb")
        shutil.copy2(project / "scene.json", game / "scene.json")
        (game / "dev").mkdir()
        shutil.copy2(Path(__file__).with_name("walk_test.gd"), game / "dev" / "walk_test.gd")
        subprocess.run([godot, "--headless", "--path", str(game), "--import"],
                       capture_output=True, text=True)
        cmd = [godot, "--headless", "--fixed-fps", "60", "--path", str(game),
               "--script", "res://dev/walk_test.gd"]
        timeout = float(os.environ.get("R2V_WALK_TIMEOUT", "900"))
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            text = out.stdout + out.stderr
        except subprocess.TimeoutExpired as e:
            # A script error leaves the test waiting for ever; show what Godot said.
            def as_text(b):
                return b.decode(errors="replace") if isinstance(b, bytes) else (b or "")
            text = as_text(e.stdout) + as_text(e.stderr) + f"\nWALKTEST TIMED OUT after {timeout:.0f}s"
        # The scratch project has no splat on purpose; that warning is expected.
        lines = text.splitlines()
        keep = []
        for i, line in enumerate(lines):
            if "Splat not imported" in line or "progress" in line:
                continue
            if "at: push_warning" in line and i and "Splat not imported" in lines[i - 1]:
                continue
            if "WALKTEST" in line or "ERROR" in line or "WARNING" in line or "   at:" in line:
                keep.append(line)
        return "\n".join(keep) or "no WALKTEST output:\n" + text[-1500:]


if __name__ == "__main__":
    godot = shutil.which("godot") or _winget_exe("GodotEngine.GodotEngine*", "Godot*_console.exe")
    if not godot:
        raise SystemExit("Godot not found")
    for arg in sys.argv[1:]:
        print(f"== {arg}")
        print(run(Path(arg), godot))
