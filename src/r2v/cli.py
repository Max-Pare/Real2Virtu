"""`r2v build`: videos in, Godot project out."""
from __future__ import annotations

import importlib
import os
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import typer

from .config import Config
from .workspace import Workspace, file_fingerprint, make_key

app = typer.Typer(add_completion=False, help=__doc__)

STAGES = ["frames", "poses", "align", "splat", "clean", "collision", "export"]
LIDAR_STAGES = ["ingest", "align", "splat", "clean", "collision", "export"]


def stages_for(source: str) -> list[str]:
    return LIDAR_STAGES if source == "lidar" else STAGES


LOCAL_COLMAP = Path(__file__).resolve().parents[2] / "tools" / "colmap" / "bin" / "colmap.exe"


def _winget_exe(package_glob: str, exe: str) -> str | None:
    """winget portable installs are not always linked onto PATH; look in its package folder."""
    base = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    if not base.is_dir():
        return None
    for pkg in base.glob(package_glob):
        for hit in pkg.rglob(exe):
            return str(hit)
    return None


def _check_tools(cfg: Config, stages: list[str]) -> None:
    # Fall back to known install locations when the default names are not on PATH.
    if cfg.poses.colmap == "colmap" and shutil.which("colmap") is None and LOCAL_COLMAP.exists():
        cfg.poses.colmap = str(LOCAL_COLMAP)
    if cfg.frames.ffmpeg == "ffmpeg" and shutil.which("ffmpeg") is None:
        cfg.frames.ffmpeg = _winget_exe("Gyan.FFmpeg*", "ffmpeg.exe") or "ffmpeg"
    need = {"frames": ("ffmpeg", cfg.frames.ffmpeg), "poses": ("COLMAP", cfg.poses.colmap)}
    for stage, (label, exe) in need.items():
        if stage in stages and shutil.which(exe) is None:
            raise SystemExit(f"{label} not found ('{exe}'). Install it or pass its path.")


@app.command()
def build(
    videos: Optional[list[Path]] = typer.Argument(None, help="Input videos of one interior."),
    lidar: Optional[Path] = typer.Option(None, help="Stray Scanner scan folder (iPhone/iPad Pro LiDAR); used instead of videos."),
    out: Path = typer.Option(Path("out/room"), help="Godot project folder to write."),
    workspace: Path = typer.Option(Path("workspace"), help="Cache folder for stage outputs."),
    camera_height: Optional[float] = typer.Option(None, help="Metres the camera was held above the floor (default guess 1.5)."),
    ceiling_height: Optional[float] = typer.Option(None, help="Real floor-to-ceiling height in metres; ceiling must be filmed."),
    scale: Optional[float] = typer.Option(None, help="Direct COLMAP-unit to metre factor (LiDAR scans are already in metres)."),
    fps: float = typer.Option(2.0, help="Frames kept per second of video. Raise for short clips."),
    steps: Optional[int] = typer.Option(None, help="Splat training steps (default 20000; LiDAR: sized to the scan)."),
    cap: Optional[int] = typer.Option(None, help="Maximum number of Gaussians (default 1500000; LiDAR: sized to the scan)."),
    train_scale: float = typer.Option(67.0, min=10.0, max=100.0, help="Percent of the source frame size used for splat training. Lower is faster and a little softer; 100 = full size."),
    fill_holes: bool = typer.Option(False, help="LiDAR only: add Gaussians where the splat is see-through on measured surfaces. Off by default: it did not improve the test scans."),
    depth_weight: float = typer.Option(0.2, help="LiDAR only: weight of the depth term in training; 0 turns it off."),
    exposure: bool = typer.Option(True, help="Learn one colour gain per frame during training, for footage with changing exposure."),
    colmap: str = typer.Option("colmap", help="COLMAP executable."),
    ffmpeg: str = typer.Option("ffmpeg", help="ffmpeg executable."),
    vocab_tree: Optional[Path] = typer.Option(None, help="COLMAP vocab tree for loop closure."),
    only: Optional[str] = typer.Option(None, help=f"Run a single stage: ingest, {', '.join(STAGES)}."),
    force: bool = typer.Option(False, help="Ignore the cache."),
) -> None:
    """Run the pipeline. Finished stages are skipped unless their inputs changed."""
    if lidar and videos:
        raise SystemExit("give either videos or --lidar, not both")
    ws = Workspace(workspace)
    # Re-run without inputs: the workspace remembers which path built it.
    source = "lidar" if lidar or (not videos and ws.stored_key("ingest")) else "video"
    stage_list = stages_for(source)
    if only and only not in stage_list:
        raise SystemExit(f"unknown stage '{only}'; choose from {', '.join(stage_list)}")

    cfg = Config(source=source)
    cfg.frames.videos = [str(v) for v in videos or []]
    cfg.frames.ffmpeg = ffmpeg
    cfg.frames.extract_fps = fps * cfg.frames.keep_one_in
    cfg.ingest.scan = str(lidar) if lidar else None
    cfg.ingest.fps = fps
    cfg.poses.colmap = colmap
    cfg.poses.vocab_tree = str(vocab_tree) if vocab_tree else None
    cfg.align.ceiling_height = ceiling_height
    cfg.align.camera_height = camera_height
    cfg.align.scale = scale
    if steps is not None:
        cfg.splat.steps = steps
    if cap is not None:
        cfg.splat.cap = cap
    cfg.splat.depth_weight = depth_weight
    cfg.splat.exposure_opt = exposure
    cfg.splat.train_scale = train_scale
    cfg.clean.fill = fill_holes
    cfg.export.out = str(out)
    if source == "lidar":
        # ARKit poses are metric but drift slightly, so: no rescaling, pose refinement on.
        if scale is None and ceiling_height is None:
            cfg.align.scale = 1.0
        cfg.splat.pose_opt = True
        # Dense start points fill the Gaussian budget early, so size it to the scan
        # unless the user chose the numbers.
        cfg.splat.auto_size = steps is None and cap is None

    for v in cfg.frames.videos:
        if not Path(v).exists():
            raise SystemExit(f"video not found: {v}")
    if lidar:
        from .stages import ingest
        ingest.check_scan(lidar)

    todo = [only] if only else stage_list
    first = stage_list[0]
    has_input = bool(lidar) if source == "lidar" else bool(cfg.frames.videos)
    key = ""
    plan = []
    for stage in stage_list:
        if stage == first and not has_input:
            # Re-run without inputs: reuse what the first stage left in the workspace.
            key = ws.stored_key(first) or ""
            if not key:
                raise SystemExit("no videos or --lidar scan given and the workspace has no frames yet")
            if only == first:
                raise SystemExit(f"{first}: needs its input again (videos or --lidar)")
            continue
        payload = asdict(getattr(cfg, stage))
        if stage == "frames":
            payload["inputs"] = [file_fingerprint(Path(v)) for v in cfg.frames.videos]
        if stage == "ingest":
            payload["inputs"] = [file_fingerprint(lidar / n) for n in ("rgb.mp4", "odometry.csv")]
        key = make_key(key, payload)
        if stage in todo and (force or not ws.is_fresh(stage, key)):
            plan.append((stage, key))
        elif stage in todo:
            print(f"{stage}: cached")

    _check_tools(cfg, [s for s, _ in plan])
    for stage, stage_key in plan:
        print(f"== {stage} ==")
        importlib.import_module(f"r2v.stages.{stage}").run(ws, cfg)
        ws.mark(stage, stage_key)


@app.command()
def stages() -> None:
    """List pipeline stages in order, for each kind of input."""
    print("video:", ", ".join(STAGES))
    print("lidar:", ", ".join(LIDAR_STAGES))


if __name__ == "__main__":
    app()
