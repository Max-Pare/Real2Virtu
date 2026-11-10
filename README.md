# Real2Virtu

Turns one or more videos of an interior, or a LiDAR scan from an iPhone/iPad Pro, into a walkable Godot 4 project.
Visuals are Gaussian splats; physics uses a hidden simplified collision mesh.
Sized for a single NVIDIA GPU with 8 GB VRAM.

## Status

Unit-tested: alignment maths, splat file writers, COLMAP reader, manifest, Godot export.

Verified on a synthetic box room (RTX 5060, 8 GB): `splat` trains with gsplat, `collision` produces a mesh matching the room within 7 cm, `export` writes a project that Godot 4.7.2 imports and renders through the GDGS addon (compute backend) upright and unmirrored, and the player stands on the floor. `frames` verified on a generated test video.

Not yet run on real video: `poses` (COLMAP 4.2.1 option names checked, never executed) and the full chain. Training quality, VRAM use at 1.5M Gaussians and timing are unmeasured.

LiDAR path verified on one real Stray Scanner scan (18 s, 66 frames kept, RTX 5060): 11.6 min training, 1.1 GB peak VRAM, rendered depth within 1.2 cm (median) of the sensor depth, exported project imports and renders in Godot 4.7.2 with the player on the floor. Not yet tried on a scan of a whole room.

Known limits: one floor level only (a flat safety floor sits at the aligned floor height); scale needs `--ceiling-height` or `--scale`.

## Requirements

- Python 3.11 or 3.12 (`uv sync --extra gpu --extra dev`); RTX 50-series needs a CUDA 12.8+ torch build
- gsplat compiles its CUDA code on first use (about 80 s): needs Visual Studio Build Tools (C++ workload) and the CUDA Toolkit matching torch. `src/r2v/gsplat_compat.py` loads that environment and works around two Windows build problems, so a plain shell is enough
- ffmpeg on PATH; COLMAP (CUDA build) on PATH, in `tools/colmap/`, or passed with `--colmap`
- Godot 4.3+ (tested on 4.7.2), Forward+ renderer

Open the exported project in the Godot editor once (or run `godot --headless --path <project> --import`) so the splat gets imported before the game runs.

Splat rendering uses GDGS 3.3.0 (MIT, ReconWorldLab), vendored in `godot_template/addons/gdgs`.

## Use

```
r2v build kitchen.mp4 hallway.mp4 --out out/flat --ceiling-height 2.65
```

Stages run in order and are cached in `workspace/`: `frames`, `poses`, `align`, `splat`, `collision`, `export`.
Re-run one stage with `--only <stage>`; ignore the cache with `--force`.

Short clips: the default keeps 2 frames per second. Raise it with `--fps` when the video is under a minute (about 100 to 300 frames in total works well).

### LiDAR scan (iPhone / iPad Pro)

```
r2v build --lidar input/my_scan --workspace workspace_scan --out out/flat
```

Record with the free Stray Scanner app and copy the scan folder (`rgb.mp4`, `odometry.csv`, `depth/`, `confidence/`) to the PC, for example through the Files app or a cable.

Stages: `ingest`, `align`, `splat`, `collision`, `export`. COLMAP and ffmpeg are not used. The scan supplies:

- camera poses and real-world scale from ARKit, so no `--camera-height` or `--ceiling-height` is needed;
- the up direction (ARKit's world is gravity-aligned);
- depth maps, used for dense start points and as a depth term during training (`--depth-weight`, 0 turns it off). Blank walls and floors, where video alone finds nothing to track, get real geometry.

Camera poses are refined slightly during training, because ARKit tracking drifts by a few centimetres.

Training size follows the scan: about 150 passes over the kept frames (6000 to 20000 steps) and 3 Gaussians per start point (300k to 1.5M). Dense start points fill the Gaussian budget early, so a short scan trained with the full budget takes several times longer for no gain. `--steps` and `--cap` override this.

Limits: one scan per build (separate scans do not share a coordinate frame). LiDAR reaches about 4.5 m; surfaces further away are learned from colour only. Use a separate workspace folder per capture.

### Scale and floor

This section applies to the video path.


COLMAP has no real-world units. Scale comes from the first of these that is given:

1. `--scale`: direct factor.
2. `--ceiling-height`: floor to ceiling in metres; the ceiling must be filmed.
3. `--camera-height`: metres the phone was held above the floor. Without any option, 1.5 m is assumed and a warning is printed.

Up direction comes from the camera orientations (phones are rarely rolled) and is then sharpened so that floor, table tops and shelves form flat layers. The floor height is first guessed from sparse points and then corrected after training from dense rendered depth, because a cluttered desk has far more feature points than a plain floor.

## Capture tips

- Keep moving; do not pan or tilt while standing still.
- Point sideways at walls and objects rather than straight along the walking direction.
- Do separate loops: one level at eye height, one tilted about 30° down (floor, table tops), one tilted up (ceiling). End each loop where it started.
- Keep the horizon level (no roll), walk slowly, lock exposure if possible.

For LiDAR scans the same tips hold, and: stay within about 4 m of the surfaces you want solid, and walk through the room rather than turning on one spot.

## Controls in the exported game

WASD or arrows to move, mouse to look, Space to jump, Shift to go faster. `F` toggles fly mode: WASD moves where you look, Space/E up, Q/Ctrl down, no collision. Esc frees the mouse.

## Output

```
out/flat/
  project.godot, main.tscn, player/, scripts/
  assets/scene.splat      Gaussians, Y-up, metres
  assets/collision.glb    collision proxy
  scene.json              manifest the Godot scene is built from; `objects` is reserved for interactables
```

## Tests

```
python -m pytest
```
