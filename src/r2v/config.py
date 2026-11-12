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
    fps: float = 2.0             # frames kept per second: the sharpest video frame of each 1/fps s
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
    # Percent of the source frame size used for training. 67 cut training time by 41%
    # for 0.1 dB PSNR on the test scan; 50 cut it by 56% for 0.6 dB. 100 = full size.
    train_scale: float = 67.0
    sh_degree: int = 3
    ssim_weight: float = 0.2
    opacity_reg: float = 0.01
    scale_reg: float = 0.01
    depth_weight: float = 0.2  # L1 against sensor depth; only used when the scene has depth maps
    pose_opt: bool = False     # learn a small correction per camera (on for LiDAR poses)
    pose_lr: float = 5e-5
    # Learn one brightness and colour gain per frame. Phones change exposure and white
    # balance while filming; without this the changes are baked into the scene as patches.
    exposure_opt: bool = True
    exposure_lr: float = 1e-2
    seed: int = 0


@dataclass
class CleanCfg:
    min_opacity: float = 0.02         # below this a Gaussian only adds haze
    # Size limits are the larger of a fixed length (metres) and a share of the
    # distance to the nearest camera, so far scenery seen through windows survives.
    max_size: float = 0.3
    max_size_per_m: float = 0.2
    needle_ratio: float = 20.0        # longest axis against the middle one
    needle_length: float = 0.25
    needle_length_per_m: float = 0.1
    isolated_gap: float = 0.15        # mean distance to the 3 nearest neighbours
    isolated_gap_per_m: float = 0.1
    # LiDAR only
    carve: bool = True                # drop Gaussians in space the sensor saw as empty
    # Add Gaussians on measured surfaces the splat leaves see-through. Off: after
    # depth-supervised training the spots it finds are mostly sensor errors (glass,
    # cables), and it lowered PSNR on both test scans (-0.1 dB at best).
    fill: bool = False
    margin: float = 0.12              # metres in front of / behind the measured surface
    margin_rel: float = 0.06          # ... or this share of the depth, whichever is larger
    carve_min_views: int = 3
    carve_min_share: float = 0.8      # share of the views seeing the Gaussian that must agree
    fill_smooth_rel: float = 0.05     # skip depth jumps larger than this share of the depth
    fill_voxel: float = 0.02          # metres between added Gaussians
    fill_min_views: int = 6           # views that must report the same uncovered spot
    fill_max: int = 300_000


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
    clean: CleanCfg = field(default_factory=CleanCfg)
    collision: CollisionCfg = field(default_factory=CollisionCfg)
    export: ExportCfg = field(default_factory=ExportCfg)
