// Calls into the Rust side. Outside Tauri (plain `npm run dev` in a browser) a
// small in-memory stand-in answers instead, so the layout can be worked on
// without a GPU or a build running.

export type Source = "video" | "lidar";

export interface Options {
  ceiling_height: number | null;
  camera_height: number | null;
  scale: number | null;
  fps: number;
  train_scale: number;
  steps: number | null;
  cap: number | null;
  depth_weight: number;
  exposure: boolean;
  fill_holes: boolean;
  device: string;
}

export interface Room {
  id: string;
  name: string;
  source: Source;
  inputs: string[];
  workspace: string;
  out: string;
  options: Options;
}

export interface RoomView {
  room: Room;
  stages: { name: string; done: boolean }[];
  stats: { frames: number | null; gaussians: number | null; room_height_m: number | null; exported: boolean };
  workspace_abs: string;
  out_abs: string;
}

export interface Settings {
  r2v_root: string;
  godot: string;
  rooms: Room[];
}

export interface EnvCheck {
  root: string;
  root_ok: boolean;
  python: string | null;
  godot: string | null;
  ffmpeg: string | null;
  colmap: string | null;
}

export interface LineEvent { room: string; line: string; err: boolean }
export interface ExitEvent { room: string; code: number | null; cancelled: boolean }

const inTauri = "__TAURI_INTERNALS__" in window;

type Unlisten = () => void;

export async function call<T>(cmd: string, args: Record<string, unknown> = {}): Promise<T> {
  if (inTauri) {
    const { invoke } = await import("@tauri-apps/api/core");
    return invoke<T>(cmd, args);
  }
  return mock(cmd, args) as Promise<T>;
}

export async function on<T>(event: string, fn: (payload: T) => void): Promise<Unlisten> {
  if (inTauri) {
    const { listen } = await import("@tauri-apps/api/event");
    return listen<T>(event, (e) => fn(e.payload));
  }
  const list = (mockListeners[event] ??= []);
  list.push(fn as (p: unknown) => void);
  return () => list.splice(list.indexOf(fn as (p: unknown) => void), 1);
}

export async function pickFiles(opts: { directory: boolean; multiple: boolean; title: string; videos?: boolean }): Promise<string[]> {
  if (!inTauri) {
    return opts.directory ? ["/home/you/captures/kitchen_scan"] : ["/home/you/captures/kitchen.mp4"];
  }
  const { open } = await import("@tauri-apps/plugin-dialog");
  const res = await open({
    directory: opts.directory,
    multiple: opts.multiple,
    title: opts.title,
    filters: opts.videos ? [{ name: "Videos", extensions: ["mp4", "mov", "m4v", "mkv", "avi", "MP4", "MOV"] }] : undefined,
  });
  if (res == null) return [];
  return Array.isArray(res) ? res : [res];
}

// ---------------------------------------------------------------- stand-in

const mockListeners: Record<string, ((p: unknown) => void)[]> = {};
const emit = (event: string, payload: unknown) => (mockListeners[event] ?? []).forEach((f) => f(payload));

const defaults = (): Options => ({
  ceiling_height: null, camera_height: null, scale: null, fps: 2, train_scale: 67,
  steps: null, cap: null, depth_weight: 0.2, exposure: false, fill_holes: false, device: "auto",
});

const mockRooms: RoomView[] = [
  {
    room: { id: "a", name: "bedroom01", source: "lidar", inputs: ["input/home-room"], workspace: "ws_bedroom01", out: "out/bedroom01", options: defaults() },
    stages: ["ingest", "align", "splat", "clean", "collision", "export"].map((name) => ({ name, done: true })),
    stats: { frames: 1492, gaussians: 851609, room_height_m: 2.814, exported: true },
    workspace_abs: "/home/you/Real2Virtu/ws_bedroom01", out_abs: "/home/you/Real2Virtu/out/bedroom01",
  },
  {
    room: { id: "b", name: "Living room", source: "video", inputs: ["input/20261111_105012.mp4", "input/bedroom2.mp4"], workspace: "workspace", out: "out/salotto", options: { ...defaults(), ceiling_height: 2.7 } },
    stages: ["frames", "poses", "align", "splat", "clean", "collision", "export"].map((name, i) => ({ name, done: i < 3 })),
    stats: { frames: 214, gaussians: null, room_height_m: 2.7, exported: false },
    workspace_abs: "/home/you/Real2Virtu/workspace", out_abs: "/home/you/Real2Virtu/out/salotto",
  },
  {
    room: { id: "c", name: "bedroom02", source: "lidar", inputs: [], workspace: "workspace_bedroom02", out: "out/bedroom02", options: defaults() },
    stages: ["ingest", "align", "splat", "clean", "collision", "export"].map((name) => ({ name, done: false })),
    stats: { frames: null, gaussians: null, room_height_m: null, exported: false },
    workspace_abs: "/home/you/Real2Virtu/workspace_bedroom02", out_abs: "/home/you/Real2Virtu/out/bedroom02",
  },
];

let mockTimer: number | undefined;

function mockRun(room: string) {
  const lines = [
    "$ r2v build --workspace workspace --out out/salotto --fps 2 --train-scale 67 --device auto",
    "frames: cached", "poses: cached", "align: cached",
    "== splat ==",
    "device: ROCm runtime /opt/rocm/lib/libhsa-runtime64.so.1 (hsa-rocr 7.2.4)",
    "splat: training at 67% of the source frames (1072x804)",
    "splat: 214 frames, 61204 start points -> 1500000 Gaussians at most, 20000 steps",
    "splat: preparing the loss for this GPU (10 s or more the first time)",
  ];
  let i = 0;
  let step = 0;
  mockTimer = window.setInterval(() => {
    if (i < lines.length) {
      emit("build-line", { room, line: lines[i++], err: false });
      return;
    }
    step += 500;
    if (step > 6000) return;
    emit("build-line", {
      room,
      line: `splat: step ${step}/20000 loss 0.0${(612 - step / 40).toFixed(0)} gaussians ${Math.min(1500000, 61204 + step * 180)} peak VRAM 3.1 GB 21.4 steps/s, ${(step / 21.4 / 60).toFixed(1)} min`,
      err: false,
    });
  }, 260);
}

async function mock(cmd: string, args: Record<string, unknown>): Promise<unknown> {
  switch (cmd) {
    case "get_settings": return { r2v_root: "/home/you/Real2Virtu", godot: "godot", rooms: mockRooms.map((r) => r.room) };
    case "list_rooms": return mockRooms;
    case "check_env": return { root: "/home/you/Real2Virtu", root_ok: true, python: "/home/you/Real2Virtu/.venv/bin/python", godot: "/usr/bin/godot", ffmpeg: "/usr/bin/ffmpeg", colmap: null };
    case "room_photo": return args.id === "c" ? null : (import.meta.env.VITE_MOCK_PHOTO ?? null);
    case "read_log": return args.id === "a" ? ["$ r2v build --lidar input/home-room --workspace ws_bedroom01 --out out/bedroom01", "ingest: cached", "align: cached", "splat: cached", "clean: cached", "collision: cached", "== export ==", "export: Godot project written to /home/you/Real2Virtu/out/bedroom01"] : [];
    case "current_job": return null;
    case "start_build": mockRun(args.id as string); return null;
    case "stop_build": window.clearInterval(mockTimer); emit("build-exit", { room: "b", code: null, cancelled: true }); return null;
    case "update_room": {
      const v = mockRooms.find((r) => r.room.id === (args.room as Room).id)!;
      v.room = args.room as Room;
      return v;
    }
    case "add_room": {
      const v: RoomView = {
        room: { id: String(Date.now()), name: args.name as string, source: args.source as Source, inputs: args.inputs as string[], workspace: "workspace_new", out: "out/new", options: defaults() },
        stages: (args.source === "lidar" ? ["ingest", "align", "splat", "clean", "collision", "export"] : ["frames", "poses", "align", "splat", "clean", "collision", "export"]).map((name) => ({ name, done: false })),
        stats: { frames: null, gaussians: null, room_height_m: null, exported: false },
        workspace_abs: "/home/you/Real2Virtu/workspace_new", out_abs: "/home/you/Real2Virtu/out/new",
      };
      mockRooms.push(v);
      return v;
    }
    case "forget_room": mockRooms.splice(mockRooms.findIndex((r) => r.room.id === args.id), 1); return null;
    case "discover_rooms": return 0;
    case "set_paths": return { r2v_root: args.r2vRoot, godot: args.godot, rooms: [] };
    default: return null;
  }
}
