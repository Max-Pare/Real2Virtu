# Real2Virtu

Turns one or more videos of an interior, or a LiDAR scan from an iPhone/iPad Pro, into a walkable Godot 4 project.
Visuals are Gaussian splats; physics uses a hidden simplified collision mesh.
Sized for a single NVIDIA GPU with 8 GB VRAM.

## Status

Unit-tested: alignment maths, splat file writers, COLMAP reader, manifest, Godot export.

Verified on a synthetic box room (RTX 5060, 8 GB): `splat` trains with gsplat, `collision` produces a mesh matching the room within 7 cm, `export` writes a project that Godot 4.7.2 imports and renders through the GDGS addon (compute backend) upright and unmirrored, and the player stands on the floor. `frames` verified on a generated test video.

Not yet run on real video: `poses` (COLMAP 4.2.1 option names checked, never executed) and the full chain. Training quality, VRAM use at 1.5M Gaussians and timing are unmeasured.

LiDAR path verified on one real Stray Scanner scan (18 s, 66 frames kept, RTX 5060): 11.6 min training at full frame size, 1.1 GB peak VRAM, rendered depth within 1.2 cm (median) of the sensor depth, exported project imports and renders in Godot 4.7.2 with the player on the floor. A longer scan (81 s, 159 frames, 1.2M Gaussians) was built and exported in about 28 minutes; its quality has not been measured.

Not yet compared: video path against LiDAR path on the same capture.

Known limits: one floor level only (a flat safety floor sits at the aligned floor height); scale needs `--ceiling-height` or `--scale`.

## Setup on a new machine (Windows)

```
.\setup.bat
```

Needs an NVIDIA GPU with a current driver, and `winget`. It installs, skipping whatever is already there:

- a signed Python 3.12.10 from python.org in `tools\python312_signed` (unsigned Python builds get blocked by Windows Smart App Control);
- `uv`, then `.venv` with the packages recorded in `uv.lock`;
- ffmpeg, Godot, Visual Studio Build Tools (C++ workload) and CUDA Toolkit 12.8 through winget (large downloads; Windows asks for administrator rights);
- COLMAP 4.2.1 (CUDA build) in `tools\colmap`.

It ends with `tools\check_env.py`, which lists what is found. `--skip-system` leaves out the winget installs, `--skip-colmap` leaves out COLMAP (only the video path uses it). Downloads are checked against fixed SHA-256 values.

Run the tool as `.venv\Scripts\r2v.exe`. That launcher is generated on each machine and unsigned, so Smart App Control may block it; `.venv\Scripts\python.exe -m r2v` does the same and is not affected.

## Requirements

What the setup script provides, for installing by hand:

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

Stages run in order and are cached in `workspace/`: `frames`, `poses`, `align`, `splat`, `clean`, `collision`, `export`.
Re-run one stage with `--only <stage>`; ignore the cache with `--force`.

Short clips: the default keeps 2 frames per second. Raise it with `--fps` when the video is under a minute (about 100 to 300 frames in total works well).

### Training speed

`--train-scale` is the percentage of the source frame size used for splat training (default 67, range 10 to 100). Training time follows the pixel count; the number of Gaussians matters much less.

Measured on a LiDAR scan (66 frames of 1600x1200, 9900 steps, 590k Gaussians, RTX 5060), scored against the full-size photos:

| `--train-scale` | Training | PSNR | SSIM | Depth error (median) |
|---|---|---|---|---|
| 100 | 12.6 min | 26.26 dB | 0.9033 | 1.42 cm |
| 67 (default) | 7.4 min | 26.16 dB | 0.8998 | 1.40 cm |
| 50 | 5.5 min | 25.65 dB | 0.8893 | 1.43 cm |

The default is used on the video path too, where it has not been measured. Use `--train-scale 100` for the sharpest result.

Step lines in the log show steps per second and elapsed minutes. gsplat's `packed` mode gave no gain on this GPU and is not used.

### Cleanup and gaps

The `clean` stage runs after training and removes Gaussians that only add haze:

- near-transparent ones (opacity under 0.02);
- oversized blobs and long needles, judged against the distance to the nearest camera, so far scenery seen through a window stays;
- isolated ones with no neighbours;
- LiDAR only: ones floating in space that the sensor measured as empty, in at least 3 views and 80% of the views that see them.

Measured on two LiDAR scans: 32% fewer Gaussians for 0.08 dB PSNR on the short one (590,832 to 403,587), 25% fewer for 0.07 dB on the long one (1,200,000 to 902,651). The gain is a smaller file and less to draw; the picture barely changes.

`--fill-holes` (LiDAR only, off by default) adds Gaussians where the splat is see-through on a measured surface. After depth-supervised training the spots it finds are mostly sensor errors (glass, cables); every variant tried lowered PSNR, by 0.1 dB at best.

Gaps are hidden, not reconstructed. Where the splat is see-through the game shows:

- the collision mesh in colours taken from the splat (a plain version of the same surface);
- a flat floor in the room's floor colour under the scanned area;
- the room's average colour instead of black where nothing was filmed.

These only show through gaps and never cover the splat. `scene.json` holds them under `backdrop`; set it to `null` to get the plain black background back.

### LiDAR scan (iPhone / iPad Pro)

```
r2v build --lidar input/my_scan --workspace workspace_scan --out out/flat
```

Record with the free Stray Scanner app and copy the scan folder (`rgb.mp4`, `odometry.csv`, `depth/`, `confidence/`) to the PC, for example through the Files app or a cable.

Stages: `ingest`, `align`, `splat`, `clean`, `collision`, `export`. COLMAP and ffmpeg are not used. The scan supplies:

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

WASD or arrows to move, mouse to look, Space to jump, Shift to go faster. `F` toggles fly mode: WASD moves where you look, Space/E up, Q/Ctrl down, no collision. `R` returns to the start point. Esc frees the mouse.

The project uses the Jolt physics engine (bundled with Godot 4.4+). Godot's default engine wedges the player on the edges of scanned triangle meshes. If the walker is still blocked in every direction it frees itself: a small step up when that clears the obstacle, otherwise back to the last place where walking worked.

## Output

```
out/flat/
  project.godot, main.tscn, player/, scripts/
  assets/scene.splat      Gaussians, Y-up, metres
  assets/collision.glb    collision proxy with vertex colours; also shown behind the splat
  scene.json              manifest the Godot scene is built from; `backdrop` says what fills gaps, `objects` is reserved for interactables
```

## Tests

```
python -m pytest
```

Walk test (needs Godot and an exported project): drives the player through random one-second walks inside the scanned area, headless, and counts places where no direction moves it. It uses the current `godot_template` with the project's collision mesh.

```
python tests/godot/run_walk_test.py out/room
```

`R2V_WALK_TRIES=<n>` sets the number of walks (default 2000).
