//! Real2Virtu desktop: keeps a list of rooms (one r2v workspace and one Godot
//! output folder each), runs `python -m r2v build` for them and streams its log.
//! The pipeline itself stays in Python; this side only starts and watches it.

use std::collections::HashSet;
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use base64::Engine;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager, Runtime, State};
use tauri_plugin_opener::OpenerExt;

const VIDEO_STAGES: [&str; 7] = ["frames", "poses", "align", "splat", "clean", "collision", "export"];
const LIDAR_STAGES: [&str; 6] = ["ingest", "align", "splat", "clean", "collision", "export"];
const LOG_NAME: &str = "desktop.log";

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(default)]
struct Options {
    ceiling_height: Option<f64>,
    camera_height: Option<f64>,
    scale: Option<f64>,
    fps: f64,
    train_scale: f64,
    steps: Option<u64>,
    cap: Option<u64>,
    depth_weight: f64,
    exposure: bool,
    fill_holes: bool,
    device: String,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            ceiling_height: None,
            camera_height: None,
            scale: None,
            fps: 2.0,
            train_scale: 67.0,
            steps: None,
            cap: None,
            depth_weight: 0.2,
            exposure: false,
            fill_holes: false,
            device: "auto".into(),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
#[serde(default)]
struct Room {
    id: String,
    name: String,
    /// "video" or "lidar"
    source: String,
    /// Videos, or the one scan folder.
    inputs: Vec<String>,
    workspace: String,
    out: String,
    options: Options,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
#[serde(default)]
struct Settings {
    r2v_root: String,
    godot: String,
    rooms: Vec<Room>,
}

#[derive(Serialize, Clone)]
struct StageState {
    name: String,
    done: bool,
}

#[derive(Serialize, Clone, Default)]
struct RoomStats {
    frames: Option<u64>,
    gaussians: Option<u64>,
    room_height_m: Option<f64>,
    exported: bool,
}

#[derive(Serialize, Clone)]
struct RoomView {
    room: Room,
    stages: Vec<StageState>,
    stats: RoomStats,
    workspace_abs: String,
    out_abs: String,
}

#[derive(Serialize, Clone)]
struct EnvCheck {
    root: String,
    root_ok: bool,
    python: Option<String>,
    godot: Option<String>,
    ffmpeg: Option<String>,
    colmap: Option<String>,
}

#[derive(Serialize, Clone)]
struct LineEvent {
    room: String,
    line: String,
    err: bool,
}

#[derive(Serialize, Clone)]
struct ExitEvent {
    room: String,
    code: Option<i32>,
    cancelled: bool,
}

#[derive(Default)]
struct Job {
    room: String,
    pid: u32,
    cancelled: bool,
}

#[derive(Default)]
struct AppState {
    settings: Mutex<Settings>,
    job: Arc<Mutex<Option<Job>>>,
}

// ---------------------------------------------------------------- settings

fn settings_path(app: &AppHandle) -> PathBuf {
    app.path()
        .app_config_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join("settings.json")
}

fn is_r2v_root(dir: &Path) -> bool {
    fs::read_to_string(dir.join("pyproject.toml"))
        .map(|t| t.contains("name = \"r2v\""))
        .unwrap_or(false)
}

/// The r2v checkout: $R2V_ROOT, else the first folder above the executable or the
/// working directory that holds r2v's pyproject.toml.
fn detect_root() -> String {
    if let Ok(r) = std::env::var("R2V_ROOT") {
        return r;
    }
    let starts = [std::env::current_exe().ok(), std::env::current_dir().ok()];
    for start in starts.into_iter().flatten() {
        for dir in start.ancestors() {
            if is_r2v_root(dir) {
                return dir.to_string_lossy().into_owned();
            }
        }
    }
    String::new()
}

fn save(app: &AppHandle, s: &Settings) -> Result<(), String> {
    let p = settings_path(app);
    if let Some(d) = p.parent() {
        fs::create_dir_all(d).map_err(|e| e.to_string())?;
    }
    let text = serde_json::to_string_pretty(s).map_err(|e| e.to_string())?;
    fs::write(&p, text).map_err(|e| format!("could not save settings to {}: {e}", p.display()))
}

fn load(app: &AppHandle) -> Settings {
    let p = settings_path(app);
    if let Some(s) = fs::read_to_string(&p).ok().and_then(|t| serde_json::from_str::<Settings>(&t).ok()) {
        return s;
    }
    // First start: find the checkout and the workspaces already in it.
    let mut s = Settings { r2v_root: detect_root(), godot: "godot".into(), rooms: vec![] };
    discover_into(&mut s);
    let _ = save(app, &s);
    s
}

// ---------------------------------------------------------------- paths

fn resolve(root: &str, p: &str) -> PathBuf {
    let path = PathBuf::from(p);
    if path.is_absolute() || root.is_empty() {
        path
    } else {
        Path::new(root).join(path)
    }
}

fn python_exe(root: &str) -> PathBuf {
    if cfg!(windows) {
        Path::new(root).join(".venv").join("Scripts").join("python.exe")
    } else {
        Path::new(root).join(".venv").join("bin").join("python")
    }
}

/// `which`: an absolute path that exists, or the first hit on PATH.
fn find_exe(name: &str) -> Option<String> {
    let p = Path::new(name);
    if p.components().count() > 1 {
        return p.is_file().then(|| name.to_string());
    }
    let exts: Vec<String> = if cfg!(windows) {
        std::env::var("PATHEXT")
            .unwrap_or(".EXE;.BAT;.CMD".into())
            .split(';')
            .map(|e| e.to_lowercase())
            .chain(std::iter::once(String::new()))
            .collect()
    } else {
        vec![String::new()]
    };
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        for ext in &exts {
            let cand = dir.join(format!("{name}{ext}"));
            if cand.is_file() {
                return Some(cand.to_string_lossy().into_owned());
            }
        }
    }
    None
}

fn stages_for(source: &str) -> Vec<&'static str> {
    if source == "lidar" { LIDAR_STAGES.to_vec() } else { VIDEO_STAGES.to_vec() }
}

fn stamp(ws: &Path, stage: &str) -> bool {
    ws.join(stage).join(".stamp").is_file()
}

fn slug(s: &str) -> String {
    let mut out = String::new();
    for c in s.trim().chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else if !out.ends_with('_') && !out.is_empty() {
            out.push('_');
        }
    }
    let t = out.trim_end_matches('_').to_string();
    if t.is_empty() { "room".into() } else { t }
}

fn new_id() -> String {
    let n = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{n:x}")
}

/// Workspaces in the checkout that a build already wrote to, added as rooms.
fn discover_into(s: &mut Settings) -> usize {
    if s.r2v_root.is_empty() {
        return 0;
    }
    let root = PathBuf::from(&s.r2v_root);
    let known: HashSet<PathBuf> = s
        .rooms
        .iter()
        .filter_map(|r| resolve(&s.r2v_root, &r.workspace).canonicalize().ok())
        .collect();
    let Ok(entries) = fs::read_dir(&root) else { return 0 };
    let mut found: Vec<PathBuf> = entries.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
    found.sort();
    let mut added = 0;
    for dir in found {
        let lidar = stamp(&dir, "ingest");
        let video = stamp(&dir, "frames");
        if !lidar && !video {
            continue;
        }
        if dir.canonicalize().map(|c| known.contains(&c)).unwrap_or(false) {
            continue;
        }
        let folder = dir.file_name().unwrap().to_string_lossy().into_owned();
        let name = folder
            .trim_start_matches("workspace_")
            .trim_start_matches("workspace")
            .trim_start_matches("ws_")
            .to_string();
        let name = if name.is_empty() { folder.clone() } else { name };
        s.rooms.push(Room {
            id: new_id(),
            name: name.clone(),
            source: if lidar { "lidar".into() } else { "video".into() },
            inputs: vec![],
            workspace: folder,
            out: format!("out/{}", slug(&name)),
            options: Options::default(),
        });
        added += 1;
    }
    added
}

fn read_json(p: &Path) -> Option<serde_json::Value> {
    serde_json::from_str(&fs::read_to_string(p).ok()?).ok()
}

fn image_dir(ws: &Path) -> Option<PathBuf> {
    ["ingest/images", "frames/images"].iter().map(|d| ws.join(d)).find(|d| d.is_dir())
}

fn sorted_images(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = fs::read_dir(dir)
        .map(|r| {
            r.flatten()
                .map(|e| e.path())
                .filter(|p| {
                    matches!(
                        p.extension().and_then(|e| e.to_str()).map(|e| e.to_lowercase()).as_deref(),
                        Some("jpg" | "jpeg" | "png")
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}

fn view(s: &Settings, room: &Room) -> RoomView {
    let ws = resolve(&s.r2v_root, &room.workspace);
    let out = resolve(&s.r2v_root, &room.out);
    let stages = stages_for(&room.source)
        .into_iter()
        .map(|n| StageState { name: n.into(), done: stamp(&ws, n) })
        .collect();
    let stats = RoomStats {
        frames: image_dir(&ws).map(|d| sorted_images(&d).len() as u64).filter(|n| *n > 0),
        gaussians: read_json(&ws.join("clean/clean.json")).and_then(|j| j["after"].as_u64()),
        room_height_m: read_json(&ws.join("align/align.json")).and_then(|j| j["room_height_m"].as_f64()),
        exported: out.join("project.godot").is_file(),
    };
    RoomView {
        room: room.clone(),
        stages,
        stats,
        workspace_abs: ws.to_string_lossy().into_owned(),
        out_abs: out.to_string_lossy().into_owned(),
    }
}

fn find_room(s: &Settings, id: &str) -> Result<Room, String> {
    s.rooms.iter().find(|r| r.id == id).cloned().ok_or_else(|| "That room is no longer in the list.".into())
}

// ---------------------------------------------------------------- commands

#[tauri::command]
fn get_settings(state: State<AppState>) -> Settings {
    state.settings.lock().unwrap().clone()
}

#[tauri::command]
fn set_paths(app: AppHandle, state: State<AppState>, r2v_root: String, godot: String) -> Result<Settings, String> {
    let mut s = state.settings.lock().unwrap();
    s.r2v_root = r2v_root;
    s.godot = if godot.trim().is_empty() { "godot".into() } else { godot };
    save(&app, &s)?;
    Ok(s.clone())
}

#[tauri::command]
fn check_env(state: State<AppState>) -> EnvCheck {
    let s = state.settings.lock().unwrap().clone();
    let root = PathBuf::from(&s.r2v_root);
    let py = python_exe(&s.r2v_root);
    let local_colmap = root.join("tools/colmap/bin").join(if cfg!(windows) { "colmap.exe" } else { "colmap" });
    EnvCheck {
        root: s.r2v_root.clone(),
        root_ok: !s.r2v_root.is_empty() && is_r2v_root(&root),
        python: py.is_file().then(|| py.to_string_lossy().into_owned()),
        godot: find_exe(&s.godot),
        ffmpeg: find_exe("ffmpeg"),
        colmap: find_exe("colmap").or_else(|| local_colmap.is_file().then(|| local_colmap.to_string_lossy().into_owned())),
    }
}

#[tauri::command]
fn list_rooms(state: State<AppState>) -> Vec<RoomView> {
    let s = state.settings.lock().unwrap();
    s.rooms.iter().map(|r| view(&s, r)).collect()
}

#[tauri::command]
fn discover_rooms(app: AppHandle, state: State<AppState>) -> Result<usize, String> {
    let mut s = state.settings.lock().unwrap();
    let n = discover_into(&mut s);
    save(&app, &s)?;
    Ok(n)
}

#[tauri::command]
fn add_room(app: AppHandle, state: State<AppState>, name: String, source: String, inputs: Vec<String>) -> Result<RoomView, String> {
    let mut s = state.settings.lock().unwrap();
    let name = if name.trim().is_empty() { "Untitled room".to_string() } else { name.trim().to_string() };
    let base = slug(&name);
    // Each capture gets its own workspace; never share one between rooms.
    let mut tag = base.clone();
    let mut i = 2;
    while resolve(&s.r2v_root, &format!("workspace_{tag}")).exists()
        || s.rooms.iter().any(|r| r.workspace == format!("workspace_{tag}"))
    {
        tag = format!("{base}_{i}");
        i += 1;
    }
    let room = Room {
        id: new_id(),
        name,
        source: if source == "lidar" { "lidar".into() } else { "video".into() },
        inputs,
        workspace: format!("workspace_{tag}"),
        out: format!("out/{tag}"),
        options: Options::default(),
    };
    s.rooms.push(room.clone());
    save(&app, &s)?;
    Ok(view(&s, &room))
}

#[tauri::command]
fn update_room(app: AppHandle, state: State<AppState>, room: Room) -> Result<RoomView, String> {
    let mut s = state.settings.lock().unwrap();
    let slot = s.rooms.iter_mut().find(|r| r.id == room.id).ok_or("That room is no longer in the list.")?;
    *slot = room.clone();
    save(&app, &s)?;
    Ok(view(&s, &room))
}

/// Takes the room off the list. Its workspace and Godot project stay on disk.
#[tauri::command]
fn forget_room(app: AppHandle, state: State<AppState>, id: String) -> Result<(), String> {
    let mut s = state.settings.lock().unwrap();
    s.rooms.retain(|r| r.id != id);
    save(&app, &s)
}

/// A frame from the middle of the capture, as a data URL.
#[tauri::command]
fn room_photo(state: State<AppState>, id: String, index: Option<f64>) -> Option<String> {
    let s = state.settings.lock().unwrap().clone();
    let room = find_room(&s, &id).ok()?;
    let ws = resolve(&s.r2v_root, &room.workspace);
    let imgs = sorted_images(&image_dir(&ws)?);
    if imgs.is_empty() {
        return None;
    }
    let at = ((index.unwrap_or(0.5).clamp(0.0, 1.0)) * (imgs.len() - 1) as f64).round() as usize;
    let p = &imgs[at];
    let mut bytes = vec![];
    fs::File::open(p).ok()?.read_to_end(&mut bytes).ok()?;
    let mime = if p.extension().map(|e| e == "png").unwrap_or(false) { "image/png" } else { "image/jpeg" };
    Some(format!("data:{mime};base64,{}", base64::engine::general_purpose::STANDARD.encode(bytes)))
}

#[tauri::command]
fn read_log(state: State<AppState>, id: String) -> Vec<String> {
    let s = state.settings.lock().unwrap().clone();
    let Ok(room) = find_room(&s, &id) else { return vec![] };
    let text = fs::read_to_string(resolve(&s.r2v_root, &room.workspace).join(LOG_NAME)).unwrap_or_default();
    let lines: Vec<String> = text.lines().map(String::from).collect();
    // The last run only: it starts at the last command line.
    let start = lines.iter().rposition(|l| l.starts_with("$ ")).unwrap_or(0);
    lines[start..].iter().rev().take(2000).rev().cloned().collect()
}

#[tauri::command]
fn current_job(state: State<AppState>) -> Option<String> {
    state.job.lock().unwrap().as_ref().map(|j| j.room.clone())
}

fn build_args(room: &Room, ws: &Path, out: &Path, only: Option<&str>, force: bool) -> Vec<String> {
    let mut a: Vec<String> = vec!["-m".into(), "r2v".into(), "build".into()];
    let o = &room.options;
    if room.source == "lidar" {
        if let Some(scan) = room.inputs.first() {
            a.push("--lidar".into());
            a.push(scan.clone());
        }
    } else {
        a.extend(room.inputs.iter().cloned());
        if let Some(v) = o.ceiling_height { a.extend(["--ceiling-height".into(), v.to_string()]); }
        if let Some(v) = o.camera_height { a.extend(["--camera-height".into(), v.to_string()]); }
    }
    if let Some(v) = o.scale { a.extend(["--scale".into(), v.to_string()]); }
    a.extend(["--workspace".into(), ws.to_string_lossy().into_owned()]);
    a.extend(["--out".into(), out.to_string_lossy().into_owned()]);
    a.extend(["--fps".into(), o.fps.to_string()]);
    a.extend(["--train-scale".into(), o.train_scale.to_string()]);
    if let Some(v) = o.steps { a.extend(["--steps".into(), v.to_string()]); }
    if let Some(v) = o.cap { a.extend(["--cap".into(), v.to_string()]); }
    if room.source == "lidar" {
        a.extend(["--depth-weight".into(), o.depth_weight.to_string()]);
        if o.fill_holes { a.push("--fill-holes".into()); }
    }
    if o.exposure { a.push("--exposure".into()); }
    a.extend(["--device".into(), o.device.clone()]);
    if let Some(s) = only { a.extend(["--only".into(), s.into()]); }
    if force { a.push("--force".into()); }
    a
}

fn quote(s: &str) -> String {
    if s.chars().all(|c| c.is_ascii_alphanumeric() || "-_./=:".contains(c)) { s.into() } else { format!("'{s}'") }
}

/// Reads a pipe and emits one event per line. Carriage-return progress bars
/// (COLMAP, ffmpeg) keep only their last state.
fn pump<RT: Runtime, R: Read + Send + 'static>(app: AppHandle<RT>, room: String, pipe: R, err: bool, log: Arc<Mutex<Option<fs::File>>>) {
    thread::spawn(move || {
        let mut reader = BufReader::new(pipe);
        let mut buf = Vec::new();
        loop {
            buf.clear();
            match reader.read_until(b'\n', &mut buf) {
                Ok(0) | Err(_) => break,
                Ok(_) => {}
            }
            let text = String::from_utf8_lossy(&buf);
            let line = text
                .trim_end_matches(['\n', '\r'])
                .rsplit('\r')
                .find(|p| !p.trim().is_empty())
                .unwrap_or("")
                .to_string();
            if let Some(f) = log.lock().unwrap().as_mut() {
                let _ = writeln!(f, "{line}");
            }
            let _ = app.emit("build-line", LineEvent { room: room.clone(), line, err });
        }
    });
}

#[tauri::command]
fn start_build(app: AppHandle, state: State<AppState>, id: String, only: Option<String>, force: bool) -> Result<(), String> {
    start_build_on(app, state, id, only, force)
}

/// start_build for any runtime, so tests can drive it without a window.
fn start_build_on<RT: Runtime>(app: AppHandle<RT>, state: State<AppState>, id: String, only: Option<String>, force: bool) -> Result<(), String> {
    let s = state.settings.lock().unwrap().clone();
    let room = find_room(&s, &id)?;
    {
        let job = state.job.lock().unwrap();
        if let Some(j) = job.as_ref() {
            let other = s.rooms.iter().find(|r| r.id == j.room).map(|r| r.name.clone()).unwrap_or_default();
            return Err(format!("{other} is building. One build at a time: they share the GPU."));
        }
    }
    let py = python_exe(&s.r2v_root);
    if !py.is_file() {
        return Err(format!("No Python environment at {}. Set the r2v folder in Settings, or run the setup there.", py.display()));
    }
    let ws = resolve(&s.r2v_root, &room.workspace);
    let out = resolve(&s.r2v_root, &room.out);
    fs::create_dir_all(&ws).map_err(|e| format!("could not create {}: {e}", ws.display()))?;
    let args = build_args(&room, &ws, &out, only.as_deref(), force);

    let mut cmd = Command::new(&py);
    cmd.args(&args)
        .current_dir(&s.r2v_root)
        .env("PYTHONUNBUFFERED", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        // Own process group, so Stop also ends COLMAP and ffmpeg children.
        cmd.process_group(0);
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    let mut child = cmd.spawn().map_err(|e| format!("could not start {}: {e}", py.display()))?;

    let log_file = fs::OpenOptions::new().create(true).append(true).open(ws.join(LOG_NAME)).ok();
    let log = Arc::new(Mutex::new(log_file));
    let header = format!("$ r2v {}", args[2..].iter().map(|a| quote(a)).collect::<Vec<_>>().join(" "));
    if let Some(f) = log.lock().unwrap().as_mut() {
        let _ = writeln!(f, "{header}");
    }
    let _ = app.emit("build-line", LineEvent { room: id.clone(), line: header, err: false });

    *state.job.lock().unwrap() = Some(Job { room: id.clone(), pid: child.id(), cancelled: false });
    pump(app.clone(), id.clone(), child.stdout.take().unwrap(), false, log.clone());
    pump(app.clone(), id.clone(), child.stderr.take().unwrap(), true, log.clone());

    let job = state.job.clone();
    thread::spawn(move || {
        let status = child.wait();
        // Let the pipe readers flush their last lines first.
        thread::sleep(Duration::from_millis(150));
        let cancelled = job.lock().unwrap().as_ref().map(|j| j.cancelled).unwrap_or(false);
        *job.lock().unwrap() = None;
        let code = status.ok().and_then(|s| s.code());
        let _ = app.emit("build-exit", ExitEvent { room: id, code, cancelled });
    });
    Ok(())
}

#[tauri::command]
fn stop_build(state: State<AppState>) -> Result<(), String> {
    let mut job = state.job.lock().unwrap();
    let Some(j) = job.as_mut() else { return Ok(()) };
    j.cancelled = true;
    let pid = j.pid;
    #[cfg(unix)]
    {
        unsafe { libc::killpg(pid as i32, libc::SIGTERM) };
        let job = state.job.clone();
        thread::spawn(move || {
            thread::sleep(Duration::from_secs(4));
            if job.lock().unwrap().as_ref().map(|j| j.pid == pid).unwrap_or(false) {
                unsafe { libc::killpg(pid as i32, libc::SIGKILL) };
            }
        });
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        let _ = Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .creation_flags(0x0800_0000)
            .status();
    }
    Ok(())
}

#[tauri::command]
fn open_folder(app: AppHandle, path: String) -> Result<(), String> {
    if !Path::new(&path).exists() {
        return Err(format!("{path} does not exist yet."));
    }
    app.opener().open_path(path, None::<&str>).map_err(|e| e.to_string())
}

/// Opens the exported project: `play` runs the game, otherwise the editor.
/// A project that was never opened gets its assets imported first.
#[tauri::command]
fn open_godot(app: AppHandle, state: State<AppState>, id: String, play: bool) -> Result<(), String> {
    let s = state.settings.lock().unwrap().clone();
    let room = find_room(&s, &id)?;
    let out = resolve(&s.r2v_root, &room.out);
    if !out.join("project.godot").is_file() {
        return Err("This room has no Godot project yet. Build it first.".into());
    }
    let godot = find_exe(&s.godot).ok_or_else(|| format!("Godot not found ('{}'). Set its path in Settings.", s.godot))?;
    let needs_import = !out.join(".godot").is_dir();
    thread::spawn(move || {
        if needs_import {
            let _ = app.emit("godot-status", "Importing the splat into Godot. The first import of a large scan takes a minute or two.");
            let _ = Command::new(&godot).args(["--headless", "--import", "--path"]).arg(&out).status();
        }
        let mut c = Command::new(&godot);
        if !play {
            c.arg("--editor");
        }
        c.arg("--path").arg(&out);
        match c.spawn() {
            Ok(_) => { let _ = app.emit("godot-status", ""); }
            Err(e) => { let _ = app.emit("godot-status", format!("Godot did not start: {e}")); }
        }
    });
    Ok(())
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .manage(AppState::default())
        .setup(|app| {
            let s = load(app.handle());
            *app.state::<AppState>().settings.lock().unwrap() = s;
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_settings,
            set_paths,
            check_env,
            list_rooms,
            discover_rooms,
            add_room,
            update_room,
            forget_room,
            room_photo,
            read_log,
            current_job,
            start_build,
            stop_build,
            open_folder,
            open_godot,
        ])
        .build(tauri::generate_context!())
        .expect("error while building the Tauri application")
        .run(|app, event| {
            // Closing the window ends a running build instead of orphaning it on the GPU.
            if let tauri::RunEvent::Exit = event {
                let _ = stop_build(app.state::<AppState>());
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root() -> String {
        Path::new(env!("CARGO_MANIFEST_DIR")).ancestors().nth(2).unwrap().to_string_lossy().into_owned()
    }

    #[test]
    fn finds_the_checkout_and_its_workspaces() {
        let r = root();
        assert!(is_r2v_root(Path::new(&r)));
        let mut s = Settings { r2v_root: r.clone(), godot: "godot".into(), rooms: vec![] };
        let n = discover_into(&mut s);
        for room in &s.rooms {
            println!("{} {} {} {}", room.name, room.source, room.workspace, room.out);
        }
        // A second pass adds nothing.
        assert_eq!(discover_into(&mut s), 0);
        assert_eq!(s.rooms.len(), n);
    }

    #[test]
    fn lidar_args_leave_out_video_scale() {
        let mut room = Room { source: "lidar".into(), inputs: vec!["scan".into()], ..Default::default() };
        room.options = Options { ceiling_height: Some(2.6), ..Default::default() };
        let a = build_args(&room, Path::new("ws"), Path::new("out"), Some("splat"), true);
        let line = a.join(" ");
        assert!(line.starts_with("-m r2v build --lidar scan"));
        assert!(!line.contains("--ceiling-height"));
        assert!(line.contains("--depth-weight 0.2"));
        assert!(line.ends_with("--only splat --force"));
    }

    #[test]
    fn video_args_keep_inputs_and_scale() {
        let mut room = Room { source: "video".into(), inputs: vec!["a.mp4".into(), "b.mp4".into()], ..Default::default() };
        room.options = Options { ceiling_height: Some(2.65), ..Default::default() };
        let line = build_args(&room, Path::new("ws"), Path::new("out"), None, false).join(" ");
        assert!(line.starts_with("-m r2v build a.mp4 b.mp4 --ceiling-height 2.65"));
        assert!(!line.contains("--depth-weight"));
        assert!(line.contains("--train-scale 67 "));
    }

    #[test]
    fn slugs() {
        assert_eq!(slug("Living room!"), "living_room");
        assert_eq!(slug("  "), "room");
    }

    // ---- real pipeline runs, through the same commands the window calls ----
    // Ignored by default: they need the GPU environment and a workspace copy that
    // may be overwritten, named by R2V_APP_TEST_WS (relative to the checkout).
    // R2V_APP_TEST_WS=workspace_apptest cargo test real_ -- --ignored --nocapture --test-threads 1

    use std::sync::mpsc::{channel, Receiver};
    use tauri::Listener;

    fn test_app() -> (tauri::App<tauri::test::MockRuntime>, Receiver<(String, serde_json::Value)>) {
        let app = tauri::test::mock_builder()
            .manage(AppState::default())
            .build(tauri::test::mock_context(tauri::test::noop_assets()))
            .unwrap();
        let ws = std::env::var("R2V_APP_TEST_WS").expect("set R2V_APP_TEST_WS to a workspace copy");
        let room = Room {
            id: "t".into(),
            name: "apptest".into(),
            source: "lidar".into(),
            inputs: vec![],
            workspace: ws,
            out: "out/apptest".into(),
            options: Options::default(),
        };
        *app.state::<AppState>().settings.lock().unwrap() = Settings { r2v_root: root(), godot: "godot".into(), rooms: vec![room] };
        let (tx, rx) = channel();
        for name in ["build-line", "build-exit"] {
            let tx = tx.clone();
            app.listen_any(name, move |e| {
                let _ = tx.send((name.to_string(), serde_json::from_str(e.payload()).unwrap()));
            });
        }
        (app, rx)
    }

    /// Prints lines as they come; returns them and the exit payload.
    fn collect(rx: &Receiver<(String, serde_json::Value)>, mut on_line: impl FnMut(&str)) -> (Vec<String>, serde_json::Value) {
        let mut lines = vec![];
        loop {
            let (kind, v) = rx.recv_timeout(Duration::from_secs(1800)).expect("no event for 30 min");
            if kind == "build-exit" {
                return (lines, v);
            }
            let line = v["line"].as_str().unwrap().to_string();
            println!("{}{line}", if v["err"].as_bool().unwrap() { "! " } else { "  " });
            on_line(&line);
            lines.push(line);
        }
    }

    #[test]
    #[ignore]
    fn real_export_stage() {
        let (app, rx) = test_app();
        let out = Path::new(&root()).join("out/apptest");
        let _ = fs::remove_dir_all(&out);
        start_build_on(app.handle().clone(), app.state(), "t".into(), Some("export".into()), true).unwrap();
        let (lines, exit) = collect(&rx, |_| {});
        assert_eq!(exit["code"], 0, "exit {exit}");
        assert_eq!(exit["cancelled"], false);
        assert!(lines[0].starts_with("$ r2v build "));
        assert!(lines.iter().any(|l| l == "== export =="));
        assert!(out.join("project.godot").is_file());
        assert!(app.state::<AppState>().job.lock().unwrap().is_none());
        let v = list_rooms(app.state());
        assert!(v[0].stats.exported);
        // The log file holds this run, and read_log returns it.
        assert_eq!(read_log(app.state(), "t".into()), lines);
    }

    #[test]
    #[ignore]
    fn real_stop_during_splat() {
        let (app, rx) = test_app();
        start_build_on(app.handle().clone(), app.state(), "t".into(), None, true).unwrap();
        // A second build is refused while one runs.
        let again = start_build_on(app.handle().clone(), app.state(), "t".into(), None, false);
        assert!(again.unwrap_err().contains("One build at a time"));
        let handle = app.handle().clone();
        let mut stopped = false;
        let (_, exit) = collect(&rx, |line| {
            if !stopped && line.starts_with("splat: step ") {
                stopped = true;
                stop_build(handle.state::<AppState>()).unwrap();
            }
        });
        assert!(stopped, "never reached a splat step");
        assert_eq!(exit["cancelled"], true, "exit {exit}");
        thread::sleep(Duration::from_secs(1));
        let ws = std::env::var("R2V_APP_TEST_WS").unwrap();
        // Python and anything it started share the workspace path in their command line.
        let left = Command::new("pgrep").args(["-f", &format!("-m r2v build .*{ws}")]).output().unwrap();
        assert!(left.stdout.is_empty(), "processes left: {}", String::from_utf8_lossy(&left.stdout));
    }
}
