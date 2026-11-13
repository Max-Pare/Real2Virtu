import json
import math

import numpy as np

from r2v import colmap_io, manifest, splat_io
from r2v.config import Config
from r2v.stages import export
from r2v.workspace import Workspace, make_key


def _gaussians(n=10, k=15):
    rng = np.random.default_rng(0)
    return dict(means=rng.normal(size=(n, 3)), scales=rng.normal(size=(n, 3)) - 3,
                quats=rng.normal(size=(n, 4)), opacities=rng.normal(size=n),
                sh0=rng.normal(size=(n, 1, 3)), shN=rng.normal(size=(n, k, 3)))


def test_ply_layout(tmp_path):
    g = _gaussians()
    splat_io.write_ply(tmp_path / "s.ply", **g)
    raw = (tmp_path / "s.ply").read_bytes()
    header, body = raw.split(b"end_header\n")
    assert b"element vertex 10" in header
    props = header.count(b"property float")
    assert props == 3 + 3 + 3 + 45 + 1 + 3 + 4
    data = np.frombuffer(body, dtype="<f4").reshape(10, props)
    assert np.allclose(data[:, :3], g["means"], atol=1e-6)
    assert np.allclose(data[:, 6:9], g["sh0"][:, 0], atol=1e-6)
    # f_rest is channel-major: first value is coefficient 0 of red.
    assert np.allclose(data[:, 9], g["shN"][:, 0, 0], atol=1e-6)
    assert np.allclose(data[:, -4:], g["quats"], atol=1e-6)


def test_splat_layout(tmp_path):
    g = _gaussians()
    splat_io.write_splat(tmp_path / "s.splat", g["means"], g["scales"], g["quats"],
                         g["opacities"], g["sh0"])
    raw = (tmp_path / "s.splat").read_bytes()
    assert len(raw) == 10 * 32
    pos = np.frombuffer(raw, dtype="<f4").reshape(10, 8)[:, :3]
    # Gaussians are reordered by visibility, so compare as sorted sets.
    want = g["means"][np.argsort(g["means"][:, 0])]
    assert np.allclose(pos[np.argsort(pos[:, 0])], want, atol=1e-6)


def test_colmap_text_model(tmp_path):
    (tmp_path / "cameras.txt").write_text("# c\n1 PINHOLE 640 480 500 510 320 240\n")
    # Identity rotation, camera centre at (1,2,3) -> t = -C.
    (tmp_path / "images.txt").write_text(
        "# i\n2 1 0 0 0 -1 -2 -3 1 v00/b.jpg\n\n1 1 0 0 0 0 0 0 1 v00/a.jpg\n1 2 -1\n")
    (tmp_path / "points3D.txt").write_text("# p\n7 0.5 0.6 0.7 10 20 30 0.1 1 2\n")
    m = colmap_io.read_model(tmp_path)
    assert list(m["names"]) == ["v00/a.jpg", "v00/b.jpg"]
    assert np.allclose(m["c2w"][1, :3, 3], [1, 2, 3])
    assert np.allclose(m["K"][0], [[500, 0, 320], [0, 510, 240], [0, 0, 1]])
    assert m["points"].shape == (1, 3) and tuple(m["colors"][0]) == (10, 20, 30)


def test_spawn_faces_camera_direction():
    c2w = np.eye(4)
    c2w[:3, 3] = [1.0, 1.5, -2.0]
    c2w[:3, 2] = [1.0, 0.0, 0.0]  # camera looks along +X
    s = manifest.spawn_from_camera(c2w)
    assert s["position"] == [1.0, 0.05, -2.0]
    yaw = math.radians(s["yaw_deg"])
    assert np.allclose([-math.sin(yaw), -math.cos(yaw)], [1.0, 0.0], atol=1e-9)


def test_spawn_avoids_furniture_and_eye_follows_camera():
    cams = np.tile(np.eye(4), (2, 1, 1))
    cams[0, :3, 3] = [0.0, 1.4, 0.0]
    cams[1, :3, 3] = [2.0, 1.4, 0.0]
    table = np.array([[0.1, 0.7, 0.0], [-0.1, 0.7, 0.1], [0.0, 0.7, -0.1]])  # under camera 0
    assert manifest.pick_spawn_camera(cams, table) == 1
    assert manifest.pick_spawn_camera(cams, np.zeros((0, 3))) == 0
    assert manifest.eye_height_from_cameras(cams) == 1.4
    cams[:, 1, 3] = 0.4
    assert manifest.eye_height_from_cameras(cams) == 1.0  # clamped to a human range


def test_cache_key_changes_with_config():
    assert make_key("", {"a": 1}) == make_key("", {"a": 1})
    assert make_key("", {"a": 1}) != make_key("", {"a": 2})
    assert make_key("x", {"a": 1}) != make_key("y", {"a": 1})


def test_export_writes_godot_project(tmp_path):
    ws = Workspace(tmp_path / "ws")
    g = _gaussians()
    splat_io.write_splat(ws.dir("splat") / "scene.splat", g["means"], g["scales"], g["quats"],
                         g["opacities"], g["sh0"])
    (ws.dir("collision") / "collision.glb").write_bytes(b"glTF")
    np.savez(ws.dir("align") / "scene.npz", c2w=np.eye(4)[None])
    cfg = Config()
    cfg.export.out = str(tmp_path / "game")
    cfg.export.project_name = 'My "Flat"'
    export.run(ws, cfg)

    out = tmp_path / "game"
    for rel in ["project.godot", "main.tscn", "scripts/world.gd", "player/player.tscn",
                "assets/scene.splat", "assets/collision.glb", "scene.json"]:
        assert (out / rel).exists(), rel
    assert "config/name=\"My 'Flat'\"" in (out / "project.godot").read_text()
    m = manifest.read(out / "scene.json")
    assert m["objects"] == [] and m["splat"]["format"] == "splat"
    assert np.allclose(m["splat"]["center"], g["means"].mean(axis=0), atol=1e-5)
    assert (out / "addons/gdgs/plugin.cfg").exists()
    assert json.loads((out / "scene.json").read_text())["collision"]["path"].startswith("res://")

    # A post-training floor/scale correction is applied to splat positions, sizes and cameras.
    (ws.dir("collision") / "refine.json").write_text('{"floor_shift": 0.5, "scale": 2.0}')
    before = np.fromfile(ws.dir("splat") / "scene.splat", dtype="<f4").reshape(-1, 8)
    export.run(ws, cfg)
    after = np.fromfile(out / "assets/scene.splat", dtype="<f4").reshape(-1, 8)
    assert np.allclose(after[:, :3], 2.0 * (before[:, :3] - [0, 0.5, 0]), atol=1e-5)
    assert np.allclose(after[:, 3:6], 2.0 * before[:, 3:6], rtol=1e-5)
    assert after[:, 6:].tobytes() == before[:, 6:].tobytes()  # colour and rotation untouched
    m = manifest.read(out / "scene.json")
    assert np.allclose(m["splat"]["center"], 2.0 * (g["means"].mean(axis=0) - [0, 0.5, 0]), atol=1e-4)


def test_export_prefers_the_full_colour_ply(tmp_path):
    ws = Workspace(tmp_path / "ws")
    g = _gaussians()
    splat_io.write_splat(ws.dir("splat") / "scene.splat", g["means"], g["scales"], g["quats"],
                         g["opacities"], g["sh0"])
    splat_io.write_ply(ws.dir("splat") / "scene.ply", **g)
    (ws.dir("collision") / "collision.glb").write_bytes(b"glTF")
    (ws.dir("collision") / "refine.json").write_text('{"floor_shift": 0.5, "scale": 2.0}')
    np.savez(ws.dir("align") / "scene.npz", c2w=np.eye(4)[None])
    cfg = Config()
    cfg.export.out = str(tmp_path / "game")
    (tmp_path / "game" / "assets").mkdir(parents=True)
    (tmp_path / "game" / "assets" / "scene.splat").write_bytes(b"left over from an older export")
    export.run(ws, cfg)

    out = tmp_path / "game"
    m = manifest.read(out / "scene.json")
    assert m["splat"]["path"] == "res://assets/scene.ply" and m["splat"]["format"] == "ply"
    assert not (out / "assets/scene.splat").exists()
    names, before = splat_io.read_ply(ws.dir("splat") / "scene.ply")
    names2, after = splat_io.read_ply(out / "assets/scene.ply")
    assert names == names2
    col = {n: i for i, n in enumerate(names)}
    xyz, sc = [col[k] for k in "xyz"], [col[f"scale_{i}"] for i in range(3)]
    assert np.allclose(after[:, xyz], 2.0 * (before[:, xyz] - [0, 0.5, 0]), atol=1e-5)
    assert np.allclose(np.exp(after[:, sc]), 2.0 * np.exp(before[:, sc]), rtol=1e-5)
    rest = [i for i in range(len(names)) if i not in xyz + sc]
    assert np.array_equal(after[:, rest], before[:, rest])  # colour, opacity, rotation untouched
    assert np.allclose(m["splat"]["center"], 2.0 * (g["means"].mean(axis=0) - [0, 0.5, 0]), atol=1e-4)


def test_dense_floor_pick_and_scale_correction():
    from r2v.config import AlignCfg
    from r2v.stages import collision

    centres = np.arange(-2.0, 3.0, 0.02) + 0.01
    area = np.zeros_like(centres)
    area[np.abs(centres - 0.0) < 0.03] = 10.0    # cluttered desk: where sparse points put the floor
    area[np.abs(centres + 0.75) < 0.03] = 6.0    # the real floor, 0.75 m lower
    area[np.abs(centres + 1.4) < 0.03] = 0.5     # faint junk under the floor
    floor = collision.pick_floor(area, centres, cam_min_y=0.9)
    assert abs(floor + 0.75) < 0.02
    assert collision.pick_floor(np.zeros_like(centres), centres, 0.9) is None

    T = collision.refine_transform(floor, cam_median_y=1.0, align_cfg=AlignCfg())
    assert abs(T["scale"] - 1.5 / 1.75) < 0.02 and abs(T["floor_shift"] + 0.75) < 0.02
    # An explicit scale from the user is never overridden; only the floor moves.
    assert collision.refine_transform(floor, 1.0, AlignCfg(scale=0.3))["scale"] == 1.0
    assert collision.refine_transform(None, 1.0, AlignCfg())["scale"] == 1.0
