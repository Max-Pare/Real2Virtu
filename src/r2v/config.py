"""All tunables, grouped per stage. A stage re-runs when its group changes."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FramesCfg:
    videos: list[str] = field(default_factory=list)
    ffmpeg: str = "ffmpeg"
    extract_fps: float = 6.0   # frames pulled from the video per second
    keep_one_in: int = 3       # sharpest frame of each group is kept (6/3 = 2 fps)
    long_side: int = 1600      # px; never upscales


@dataclass
class IngestCfg:
    """LiDAR path: a Stray Scanner scan folder replaces the frames and poses stages."""
    scan: str | None = None
    fps: float = 2.0             # frames kept per second
    keep_one_in: int = 3         # sharpest of this many candidates is kept
    long_side: int = 1600        # px; never upscales
    min_move: float = 0.02       # metres; a frame is dropped when the camera moved less
    min_turn_deg: float = 2.0    # ... and turned less than this since the last kept frame
    points_confidence: int = 2   # ARKit depth confidence (0-2) needed for start points
    depth_confidence: int = 1    # ... and for depth supervision
    min_depth: float = 0.1       # metres
    max_depth: float = 4.5       # metres; LiDAR is unreliable beyond this
    voxel: float = 0.02          # metres; start points are thinned to this grid
    max_points: int = 400_000


@dataclass
class PosesCfg:
    colmap: str = "colmap"
    exhaustive_max: int = 400          # above this, sequential matching is used
    vocab_tree: str | None = None      # enables loop closure for sequential matching


@dataclass
class AlignCfg:
    # Scale source, first one set wins: scale, ceiling_height, camera_height.
    scale: float | None = None           # direct COLMAP-unit to metre factor
    ceiling_height: float | None = None  # metres, floor to ceiling (ceiling must be filmed)
    camera_height: float | None = None   # metres the camera was held above the floor
    default_camera_height: float = 1.5


@dataclass
class SplatCfg:
    steps: int = 20_000
    cap: int = 1_500_000       # hard limit on Gaussians, sized for 8 GB VRAM
    auto_size: bool = False    # lower steps and cap for small captures (see splat.auto_size)
    sh_degree: int = 3
    ssim_weight: float = 0.2
    opacity_reg: float = 0.01
    scale_reg: float = 0.01
    depth_weight: float = 0.2  # L1 against sensor depth; only used when the scene has depth maps
    pose_opt: bool = False     # learn a small correction per camera (on for LiDAR poses)
    pose_lr: float = 5e-5
    seed: int = 0


@dataclass
class CollisionCfg:
    voxel: float = 0.04        # metres
    max_resolution: int = 400  # voxels per side; caps memory, coarsens large scenes
    depth_trunc: float = 8.0   # metres
    use_one_in: int = 2        # render depth from every n-th camera
    downscale: int = 2
    target_tris: int = 50_000
    min_component_tris: int = 500


@dataclass
class ExportCfg:
    project_name: str = "Real2Virtu Map"
    out: str = "out/room"
    eye_height: float | None = None  # None: use the height the camera was held at


@dataclass
class Config:
    source: str = "video"  # "video" (frames + poses) or "lidar" (ingest)
    ingest: IngestCfg = field(default_factory=IngestCfg)
    frames: FramesCfg = field(default_factory=FramesCfg)
    poses: PosesCfg = field(default_factory=PosesCfg)
    align: AlignCfg = field(default_factory=AlignCfg)
    splat: SplatCfg = field(default_factory=SplatCfg)
    collision: CollisionCfg = field(default_factory=CollisionCfg)
    export: ExportCfg = field(default_factory=ExportCfg)
