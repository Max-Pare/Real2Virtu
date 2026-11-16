import "@fontsource-variable/archivo/standard.css";
import "./style.css";
import { call, on, pickFiles, type EnvCheck, type ExitEvent, type LineEvent, type Room, type RoomView, type Settings, type Source } from "./api";

// ---------------------------------------------------------------- state

/** What a build has said so far, read from its log lines. */
interface Run {
  running: boolean;
  stage: string | null;       // stage in progress
  done: Set<string>;          // stages finished or cached during this run
  step: number | null;        // splat step
  steps: number | null;
  rate: number | null;        // splat steps per second
  minutes: number | null;     // splat training minutes so far
  startedAt: number;
  note: string;               // last informative log line
  failed: { stage: string | null; message: string } | null;
  cancelled: boolean;
}

const state = {
  rooms: [] as RoomView[],
  selected: null as string | null,
  view: "room" as "room" | "settings",
  photos: new Map<string, string | null>(),
  logs: new Map<string, string[]>(),
  runs: new Map<string, Run>(),
  job: null as string | null,
  godotNote: "",
  error: "",
};

/** How much of the tape each stage takes, roughly by how long it runs. */
const WEIGHT: Record<string, number> = { frames: 1.2, ingest: 1.6, poses: 2.6, align: 0.9, splat: 7, clean: 1.1, collision: 1.4, export: 1 };

const STAGE_NAME: Record<string, string> = {
  frames: "Frames", ingest: "Ingest", poses: "Poses", align: "Align", splat: "Splat", clean: "Clean", collision: "Collision", export: "Export",
};

const STAGE_DOES: Record<string, string> = {
  frames: "Picking the sharpest video frames",
  ingest: "Reading the scan and picking the sharpest frames",
  poses: "Finding where each frame was taken (COLMAP)",
  align: "Levelling the floor and setting the scale",
  splat: "Training the Gaussian splat",
  clean: "Removing haze and floating Gaussians",
  collision: "Building the walkable collision mesh",
  export: "Writing the Godot project",
};

const app = document.querySelector<HTMLDivElement>("#app")!;
const fmt = new Intl.NumberFormat("en");

// ---------------------------------------------------------------- dom helper

type Child = Node | string | null | undefined | false;
function h<K extends keyof HTMLElementTagNameMap>(tag: K, props: Record<string, unknown> = {}, ...kids: Child[]): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (v == null || v === false) continue;
    if (k.startsWith("on")) el.addEventListener(k.slice(2).toLowerCase(), v as EventListener);
    else if (k === "class") el.className = v as string;
    else if (k === "style") el.setAttribute("style", v as string);
    else if (k in el && typeof v !== "string") (el as unknown as Record<string, unknown>)[k] = v;
    else el.setAttribute(k, v === true ? "" : String(v));
  }
  for (const c of kids) if (c != null && c !== false) el.append(c);
  return el;
}

const current = () => state.rooms.find((r) => r.room.id === state.selected) ?? null;

function freshRun(): Run {
  return { running: true, stage: null, done: new Set(), step: null, steps: null, rate: null, minutes: null, startedAt: Date.now(), note: "", failed: null, cancelled: false };
}

// ---------------------------------------------------------------- log parsing

function readLine(run: Run, line: string, err: boolean) {
  let m: RegExpMatchArray | null;
  if ((m = line.match(/^== (\w+) ==$/))) {
    if (run.stage) run.done.add(run.stage);
    run.stage = m[1];
    run.step = run.steps = run.rate = run.minutes = null;
    run.note = "";
    return;
  }
  if ((m = line.match(/^(\w+): cached$/))) {
    run.done.add(m[1]);
    return;
  }
  if ((m = line.match(/^splat: step (\d+)\/(\d+) .*?([\d.]+) steps\/s, ([\d.]+) min/))) {
    run.step = +m[1];
    run.steps = +m[2];
    run.rate = +m[3];
    run.minutes = +m[4];
    return;
  }
  if (line.startsWith("$ ")) return;
  if (!err && /^\w+: /.test(line)) run.note = line.replace(/^\w+: /, "");
  if (err && line.trim() && !/^\s/.test(line) && !/Warning|warn\(/.test(line)) run.note = line;
}

// ---------------------------------------------------------------- loading

async function refresh() {
  state.rooms = await call<RoomView[]>("list_rooms");
  if (!state.selected || !current()) state.selected = state.rooms[0]?.room.id ?? null;
}

async function photo(id: string): Promise<string | null> {
  if (!state.photos.has(id)) state.photos.set(id, await call<string | null>("room_photo", { id, index: 0.5 }));
  return state.photos.get(id) ?? null;
}

async function loadLog(id: string) {
  if (state.logs.has(id) || state.runs.get(id)?.running) return;
  state.logs.set(id, await call<string[]>("read_log", { id }));
}

// ---------------------------------------------------------------- tape

/** Where the tape ends, 0 to 1, and where each stage starts. */
function tapeModel(v: RoomView) {
  const run = state.runs.get(v.room.id);
  const names = v.stages.map((s) => s.name);
  const total = names.reduce((a, n) => a + WEIGHT[n], 0);
  const starts: number[] = [];
  let acc = 0;
  for (const n of names) {
    starts.push(acc / total);
    acc += WEIGHT[n];
  }
  const done = (n: string) => (run ? run.done.has(n) || (!run.running && !run.failed && !run.cancelled && v.stages.find((s) => s.name === n)?.done) : v.stages.find((s) => s.name === n)?.done) ?? false;

  let reach = 0;
  let live: string | null = null;
  for (let i = 0; i < names.length; i++) {
    const n = names[i];
    if (done(n)) {
      reach = starts[i] + WEIGHT[n] / total;
      continue;
    }
    if (run?.running && run.stage === n) {
      live = n;
      const part = n === "splat" && run.step && run.steps ? run.step / run.steps : 0;
      reach = starts[i] + (WEIGHT[n] / total) * part;
    }
    break;
  }
  return { names, starts, total, reach, live, done };
}

function tape(v: RoomView): HTMLElement {
  const t = tapeModel(v);
  // Printed twice: grey on the track, dark on the yellow blade that covers it.
  const marks = () =>
    h("div", { class: "tape-print", "aria-hidden": "true" },
      ...t.names.map((n, i) =>
        h("span", { class: "tape-stage", style: `left:${t.starts[i] * 100}%; width:${(WEIGHT[n] / t.total) * 100}%` }, STAGE_NAME[n])));
  const bar = h("div", {
    class: "tape", "data-room": v.room.id, role: "progressbar",
    "aria-valuemin": "0", "aria-valuemax": "100", "aria-label": "Build progress",
  },
    h("div", { class: "tape-track" }, marks()),
    h("div", { class: "tape-live" }),
    h("div", { class: "tape-blade" }, marks()),
    h("div", { class: "tape-hook" }));
  patchTape(bar, v);
  return bar;
}

/** Moves the blade in place, so the CSS transition carries it to the new length. */
function patchTape(bar: HTMLElement, v: RoomView) {
  const run = state.runs.get(v.room.id);
  const t = tapeModel(v);
  bar.style.setProperty("--reach", `${t.reach * 100}%`);
  const liveFrac = t.live ? (WEIGHT[t.live] / t.total) * (1 - (run?.step && run.steps ? run.step / run.steps : 0)) : 0;
  bar.style.setProperty("--live", `${liveFrac * 100}%`);
  bar.classList.toggle("is-live", !!t.live);
  bar.classList.toggle("is-failed", !!run?.failed);
  bar.setAttribute("aria-valuenow", String(Math.round(t.reach * 100)));
}

function minutes(ms: number) {
  const m = ms / 60000;
  return m < 1 ? `${Math.max(1, Math.round(ms / 1000))} s` : `${m.toFixed(m < 10 ? 1 : 0)} min`;
}

function caption(v: RoomView): HTMLElement {
  const run = state.runs.get(v.room.id);
  const t = tapeModel(v);
  if (run?.running) {
    const stage = run.stage;
    if (!stage) return h("p", { class: "caption" }, "Checking which stages are already built.");
    let detail = run.note;
    if (stage === "splat" && run.step && run.steps) {
      const left = run.rate ? (run.steps - run.step) / run.rate / 60 : null;
      detail = `Step ${fmt.format(run.step)} of ${fmt.format(run.steps)}` +
        (run.rate ? `, ${run.rate.toFixed(0)} steps a second` : "") +
        (left != null ? `, about ${left < 1 ? "1" : Math.round(left)} min left` : "");
    }
    return h("p", { class: "caption" }, h("strong", {}, STAGE_DOES[stage] ?? stage), detail ? h("span", {}, detail) : null);
  }
  if (run?.cancelled) return h("p", { class: "caption" }, h("strong", {}, "Stopped."), h("span", {}, "Finished stages are kept; Build carries on from here."));
  if (run?.failed) {
    return h("p", { class: "caption is-failed" },
      h("strong", {}, run.failed.stage ? `${STAGE_NAME[run.failed.stage] ?? run.failed.stage} failed.` : "The build did not start."),
      h("span", {}, run.failed.message || "The log below shows why."));
  }
  if (t.names.every((n) => t.done(n))) {
    return h("p", { class: "caption" }, h("strong", {}, "Ready to walk."), h("span", {}, v.stats.exported ? `Godot project in ${v.room.out}` : ""));
  }
  if (!t.names.some((n) => t.done(n))) return h("p", { class: "caption" }, h("strong", {}, "Not built yet."), h("span", {}, "Check the capture settings, then build."));
  const next = t.names.find((n) => !t.done(n))!;
  return h("p", { class: "caption" }, h("strong", {}, `Built up to ${STAGE_NAME[t.names[t.names.indexOf(next) - 1]] ?? "the start"}.`), h("span", {}, `Build continues with ${STAGE_NAME[next].toLowerCase()}.`));
}

function updateTape() {
  const v = current();
  const slot = document.getElementById("tape-slot");
  if (!v || !slot) return;
  const run = state.runs.get(v.room.id);
  const bar = slot.querySelector<HTMLElement>(".tape");
  if (bar && bar.dataset.room === v.room.id) patchTape(bar, v);
  else slot.replaceChildren(tape(v));
  slot.querySelector(".caption")?.remove();
  slot.append(caption(v));
  const clock = document.getElementById("clock");
  if (clock) clock.textContent = run?.running ? minutes(Date.now() - run.startedAt) : "";
  updateActions();
  renderSidebarList();
}

// ---------------------------------------------------------------- sidebar

function status(v: RoomView): string {
  const run = state.runs.get(v.room.id);
  if (run?.running) {
    if (run.stage === "splat" && run.step && run.steps) return `Splat ${Math.round((run.step / run.steps) * 100)}%`;
    return run.stage ? `${STAGE_NAME[run.stage]}…` : "Starting…";
  }
  if (run?.failed) return `${run.failed.stage ? STAGE_NAME[run.failed.stage] : "Build"} failed`;
  const done = v.stages.filter((s) => s.done).length;
  if (done === v.stages.length) return v.stats.gaussians ? `${fmt.format(v.stats.gaussians)} Gaussians` : "Built";
  if (done === 0) return "Not built";
  return `${done} of ${v.stages.length} stages`;
}

function renderSidebarList() {
  const list = document.getElementById("room-list");
  if (!list) return;
  list.replaceChildren(
    ...state.rooms.map((v) => {
      const thumb = h("span", { class: "thumb" });
      const cached = state.photos.get(v.room.id);
      if (cached) thumb.style.backgroundImage = `url(${cached})`;
      else if (!state.photos.has(v.room.id)) photo(v.room.id).then((src) => src && (thumb.style.backgroundImage = `url(${src})`));
      const live = state.runs.get(v.room.id)?.running;
      return h("li", {},
        h("button", {
          class: "room-link" + (v.room.id === state.selected && state.view === "room" ? " is-current" : "") + (live ? " is-live" : ""),
          "aria-current": v.room.id === state.selected && state.view === "room" ? "page" : false,
          onclick: () => select(v.room.id),
        },
          thumb,
          h("span", { class: "room-link-text" }, h("span", { class: "room-link-name" }, v.room.name), h("span", { class: "room-link-status" }, status(v)))));
    }),
  );
}

function sidebar(): HTMLElement {
  return h("aside", { class: "sidebar" },
    h("div", { class: "brand" }, h("img", { src: "/app-icon.svg", alt: "", width: 28, height: 28 }), h("span", {}, "Real2Virtu")),
    h("nav", { "aria-label": "Rooms" }, h("ul", { id: "room-list", class: "room-list" })),
    h("div", { class: "sidebar-foot" },
      h("button", { class: "btn btn-quiet", onclick: () => openNewRoom() }, "New room"),
      h("button", { class: "btn btn-quiet" + (state.view === "settings" ? " is-current" : ""), onclick: () => { state.view = "settings"; render(); } }, "Settings")));
}

// ---------------------------------------------------------------- room page

function stat(value: string, label: string) {
  return h("div", { class: "stat" }, h("dd", {}, value), h("dt", {}, label));
}

function header(v: RoomView): HTMLElement {
  const banner = h("div", { class: "banner" });
  photo(v.room.id).then((src) => {
    if (src) {
      banner.style.backgroundImage = `url(${src})`;
      banner.classList.add("has-photo");
    }
  });
  const s = v.stats;
  const stats = [
    s.frames != null && stat(fmt.format(s.frames), "frames kept"),
    s.gaussians != null && stat(fmt.format(s.gaussians), "Gaussians"),
    s.room_height_m != null && stat(`${s.room_height_m.toFixed(2)} m`, "floor to ceiling"),
  ].filter(Boolean) as HTMLElement[];

  const nameInput = h("input", {
    class: "room-name",
    value: v.room.name,
    "aria-label": "Room name",
    spellcheck: false,
    onchange: (e: Event) => save({ ...v.room, name: (e.target as HTMLInputElement).value.trim() || v.room.name }),
    onkeydown: (e: KeyboardEvent) => e.key === "Enter" && (e.target as HTMLInputElement).blur(),
  });
  return h("header", { class: "room-head" },
    banner,
    h("div", { class: "room-head-text" },
      h("p", { class: "room-kind" }, v.room.source === "lidar" ? "LiDAR scan" : `${v.room.inputs.length || "No"} video${v.room.inputs.length === 1 ? "" : "s"}`),
      nameInput,
      stats.length ? h("dl", { class: "stats" }, ...stats) : null));
}

function actions(v: RoomView): HTMLElement {
  const box = h("div", { id: "actions", class: "actions" });
  fillActions(box, v);
  return box;
}

function updateActions() {
  const box = document.getElementById("actions");
  const v = current();
  if (box && v) fillActions(box, v);
}

function fillActions(box: HTMLElement, v: RoomView) {
  const run = state.runs.get(v.room.id);
  const busyElsewhere = state.job && state.job !== v.room.id;
  const rerun = document.getElementById("rerun-stage") as HTMLSelectElement | null;
  const forceBox = document.getElementById("rerun-force") as HTMLInputElement | null;
  const primary = run?.running
    ? h("button", { class: "btn btn-stop", onclick: stop }, "Stop build")
    : h("button", {
        class: "btn btn-primary",
        disabled: !!busyElsewhere,
        title: busyElsewhere ? "Another room is building. One build at a time: they share the GPU." : false,
        onclick: () => build(v, rerun?.value || null, forceBox?.checked ?? false),
      }, "Build");
  box.replaceChildren(
    primary,
    h("button", { class: "btn", disabled: !v.stats.exported, onclick: () => godot(v, true) }, "Walk around"),
    h("button", { class: "btn", disabled: !v.stats.exported, onclick: () => godot(v, false) }, "Open in Godot"),
    h("button", { class: "btn btn-quiet", onclick: () => call("open_folder", { path: v.stats.exported ? v.out_abs : v.workspace_abs }).catch(showError) }, "Show files"),
    h("span", { id: "clock", class: "clock", "aria-live": "off" }, run?.running ? minutes(Date.now() - run.startedAt) : ""),
  );
  if (state.godotNote) box.append(h("p", { class: "godot-note", role: "status" }, state.godotNote));
}

// Settings that change what the pipeline does. Each field saves on change.
function field(label: string, hint: string | null, control: HTMLElement, id: string) {
  control.id = id;
  return h("div", { class: "field" }, h("label", { for: id }, label), control, hint ? h("p", { class: "hint" }, hint) : null);
}

function num(value: number | null, opts: { step: string; min?: string; max?: string; placeholder?: string }, set: (n: number | null) => void) {
  return h("input", {
    type: "number", inputmode: "decimal", value: value ?? "", step: opts.step, min: opts.min, max: opts.max, placeholder: opts.placeholder,
    onchange: (e: Event) => {
      const t = (e.target as HTMLInputElement).value.trim();
      set(t === "" ? null : Number(t));
    },
  });
}

function check(label: string, hint: string, value: boolean, set: (b: boolean) => void, id = `chk-${label.replace(/\W+/g, "-").toLowerCase()}`) {
  return h("div", { class: "field field-check" },
    h("input", { type: "checkbox", id, checked: value, onchange: (e: Event) => set((e.target as HTMLInputElement).checked) }),
    h("label", { for: id }, label),
    h("p", { class: "hint" }, hint));
}

const TRAIN_NOTE: Record<number, string> = {
  100: "Sharpest. On a 66-frame scan: 12.6 min, 26.26 dB.",
  67: "Default. On a 66-frame scan: 7.4 min, 26.16 dB.",
  50: "On a 66-frame scan: 5.5 min, 25.65 dB.",
};

function trainNote(p: number) {
  return TRAIN_NOTE[p] ?? "Training time follows the pixel count: lower is faster and a little softer.";
}

function captureForm(v: RoomView): HTMLElement {
  const r = v.room;
  const o = r.options;
  const set = (patch: Partial<typeof o>) => save({ ...r, options: { ...o, ...patch } });
  const lidar = r.source === "lidar";

  const inputs = h("div", { class: "field" },
    h("span", { class: "label" }, lidar ? "Scan folder" : "Videos"),
    r.inputs.length
      ? h("ul", { class: "inputs" }, ...r.inputs.map((p, i) => h("li", {},
          h("span", { class: "path", title: p }, p.split(/[\\/]/).pop() ?? p),
          h("button", { class: "link", "aria-label": `Remove ${p}`, onclick: () => save({ ...r, inputs: r.inputs.filter((_, j) => j !== i) }) }, "Remove"))))
      : h("p", { class: "hint" }, v.stages[0].done ? "Not needed: the workspace already holds the frames." : lidar ? "Choose the Stray Scanner folder (rgb.mp4, odometry.csv, depth/, confidence/)." : "Add one or more videos of this interior."),
    h("button", {
      class: "btn btn-small",
      onclick: async () => {
        const got = await pickFiles(lidar ? { directory: true, multiple: false, title: "Choose the scan folder" } : { directory: false, multiple: true, title: "Add videos", videos: true });
        if (got.length) save({ ...r, inputs: lidar ? got.slice(0, 1) : [...r.inputs, ...got] });
      },
    }, lidar ? (r.inputs.length ? "Choose another folder" : "Choose folder") : "Add videos"));

  // Video captures have no real-world units; the first scale source given wins.
  const scaleMode = o.scale != null ? "scale" : o.ceiling_height != null ? "ceiling" : o.camera_height != null ? "camera" : "guess";
  const modes = [
    ["ceiling", "Ceiling height", "ceiling_height", "Floor to ceiling, in metres. The ceiling must be filmed."],
    ["camera", "Camera height", "camera_height", "How high the phone was held, in metres."],
    ["scale", "Scale factor", "scale", "Direct factor from COLMAP units to metres."],
    ["guess", "Guess", null, "Assumes the phone was held at 1.5 m. A warning is printed."],
  ] as const;
  const scaleBox = h("fieldset", { class: "field scale" },
    h("legend", {}, "Scale"),
    ...modes.map(([mode, label, key]) =>
      h("div", { class: "scale-row" },
        h("input", {
          type: "radio", name: "scale-mode", id: `scale-${mode}`, checked: scaleMode === mode,
          onchange: () => set({
            ceiling_height: mode === "ceiling" ? (o.ceiling_height ?? 2.6) : null,
            camera_height: mode === "camera" ? (o.camera_height ?? 1.5) : null,
            scale: mode === "scale" ? (o.scale ?? 1) : null,
          }).then(() => render()),
        }),
        h("label", { for: `scale-${mode}` }, label),
        key && scaleMode === mode
          ? num(o[key], { step: "0.01", min: "0.01" }, (n) => set({ [key]: n } as Partial<typeof o>))
          : null)),
    h("p", { class: "hint" }, modes.find(([m]) => m === scaleMode)![3]));

  const train = h("input", { type: "range", min: "10", max: "100", step: "1", value: String(o.train_scale) });
  const trainOut = h("output", { for: "train-scale" }, `${o.train_scale}%`);
  const trainHint = h("p", { class: "hint" }, trainNote(o.train_scale));
  train.addEventListener("input", () => {
    trainOut.textContent = `${train.value}%`;
    trainHint.textContent = trainNote(+train.value);
  });
  train.addEventListener("change", () => set({ train_scale: +train.value }));
  train.id = "train-scale";

  const device = h("select", { onchange: (e: Event) => set({ device: (e.target as HTMLSelectElement).value }) },
    ...[["auto", "Whatever torch was built for"], ["cuda", "NVIDIA (CUDA)"], ["rocm", "AMD Radeon (ROCm)"]].map(([val, label]) =>
      h("option", { value: val, selected: o.device === val }, label)));

  return h("section", { class: "panel capture", "aria-labelledby": "capture-title" },
    h("h2", { id: "capture-title" }, "Capture"),
    inputs,
    lidar ? null : scaleBox,
    field(lidar ? "Frames per second of scan" : "Frames per second of video", lidar ? "The sharpest frame of each slot is kept." : "Raise for clips under a minute; 100 to 300 frames in total works well.",
      num(o.fps, { step: "0.5", min: "0.1" }, (n) => set({ fps: n ?? 2 })), "fps"),
    h("h2", {}, "Training"),
    h("div", { class: "field" },
      h("div", { class: "label-row" }, h("label", { for: "train-scale" }, "Training size"), trainOut),
      train, trainHint),
    h("div", { class: "field-pair" },
      field("Steps", null, num(o.steps, { step: "500", min: "500", placeholder: lidar ? "Sized to scan" : "20,000" }, (n) => set({ steps: n })), "steps"),
      field("Gaussian limit", null, num(o.cap, { step: "100000", min: "10000", placeholder: lidar ? "Sized to scan" : "1,500,000" }, (n) => set({ cap: n })), "cap")),
    lidar ? field("Depth weight", "How strongly training follows the sensor depth. 0 turns it off.", num(o.depth_weight, { step: "0.05", min: "0" }, (n) => set({ depth_weight: n ?? 0 })), "depth") : null,
    check("Even out exposure", "Learns a colour gain per frame for footage whose brightness changes. Lowered quality on the first real scan.", o.exposure, (b) => set({ exposure: b })),
    lidar ? check("Fill see-through spots", "Adds Gaussians where the splat is see-through on measured surfaces. Did not help the test scans.", o.fill_holes, (b) => set({ fill_holes: b })) : null,
    field("GPU", null, device, "device"),
    h("details", { class: "advanced" },
      h("summary", {}, "Folders and single stages"),
      field("Workspace", "Cache for stage results. One per capture.", h("input", { value: r.workspace, spellcheck: false, onchange: (e: Event) => save({ ...r, workspace: (e.target as HTMLInputElement).value }) }), "ws"),
      field("Godot project", "Written by the export stage. A re-export keeps your main.tscn.", h("input", { value: r.out, spellcheck: false, onchange: (e: Event) => save({ ...r, out: (e.target as HTMLInputElement).value }) }), "out"),
      field("Run only", "Build runs every stage whose settings changed. Pick one to run it alone.",
        h("select", {}, h("option", { value: "" }, "All stages"), ...v.stages.map((s) => h("option", { value: s.name }, STAGE_NAME[s.name]))), "rerun-stage"),
      check("Ignore the cache", "Re-runs stages even when nothing changed.", false, () => {}, "rerun-force"),
      h("button", { class: "link link-danger", onclick: () => forget(v) }, "Remove from list"),
      h("p", { class: "hint" }, "Removing keeps the workspace and the Godot project on disk.")));
}

function logPanel(v: RoomView): HTMLElement {
  const pre = h("div", { id: "log", class: "log", role: "log", "aria-label": "Build log", tabindex: "0" });
  for (const line of state.logs.get(v.room.id) ?? []) pre.append(logLine(line, false));
  requestAnimationFrame(() => (pre.scrollTop = pre.scrollHeight));
  return h("section", { class: "panel log-panel", "aria-labelledby": "log-title" },
    h("h2", { id: "log-title" }, "Log"),
    pre.childElementCount ? pre : h("div", { id: "log", class: "log is-empty" }, "The pipeline's output shows here while it builds."));
}

function logLine(line: string, err: boolean): HTMLElement {
  let cls = "ln";
  if (line.startsWith("$ ")) cls += " ln-cmd";
  else if (/^== \w+ ==$/.test(line)) cls += " ln-stage";
  else if (/^\w+: cached$/.test(line)) cls += " ln-cached";
  else if (/WARNING|Warning/.test(line)) cls += " ln-warn";
  else if (/Error|Traceback|error:|not found/.test(line)) cls += " ln-err";
  else if (err) cls += " ln-stderr";
  return h("div", { class: cls }, line.startsWith("$ ") ? line : line.replace(/^== (\w+) ==$/, (_, s) => STAGE_NAME[s] ?? s));
}

function roomPage(v: RoomView): HTMLElement {
  const page = h("main", { class: "room" },
    header(v),
    h("section", { class: "progress", "aria-label": "Build" }, h("div", { id: "tape-slot" }), actions(v)),
    h("div", { class: "panels" }, captureForm(v), logPanel(v)));
  const slot = page.querySelector("#tape-slot")!;
  slot.append(tape(v), caption(v));
  return page;
}

function emptyPage(): HTMLElement {
  return h("main", { class: "empty" },
    h("h1", {}, "Turn a room into a place you can walk around."),
    h("p", {}, "Start from phone videos of an interior, or a LiDAR scan made with Stray Scanner on an iPhone or iPad Pro. Real2Virtu trains a Gaussian splat, builds a collision mesh and writes a Godot 4 project."),
    h("div", { class: "actions" },
      h("button", { class: "btn btn-primary", onclick: () => openNewRoom("video") }, "Add videos"),
      h("button", { class: "btn", onclick: () => openNewRoom("lidar") }, "Add a LiDAR scan")));
}

// ---------------------------------------------------------------- settings page

async function settingsPage(): Promise<HTMLElement> {
  const s = await call<Settings>("get_settings");
  const env = await call<EnvCheck>("check_env");
  const root = h("input", { value: s.r2v_root, spellcheck: false, id: "root" });
  const godotPath = h("input", { value: s.godot, spellcheck: false, id: "godot" });
  const note = h("p", { class: "hint", role: "status" });
  const row = (name: string, found: string | null, missing: string) =>
    h("li", { class: found ? "ok" : "missing" }, h("span", { class: "env-name" }, name), h("span", { class: "env-where" }, found ?? missing));
  return h("main", { class: "settings" },
    h("h1", {}, "Settings"),
    h("section", { class: "panel" },
      h("h2", {}, "Where things are"),
      field("r2v folder", "The Real2Virtu checkout with its .venv. Relative room folders are inside it.", root, "root"),
      field("Godot", "Command name or full path. Godot 4.3 or newer.", godotPath, "godot"),
      h("div", { class: "actions" },
        h("button", { class: "btn btn-primary", onclick: async () => {
          await call("set_paths", { r2vRoot: root.value.trim(), godot: godotPath.value.trim() });
          await refresh();
          render();
        } }, "Save"),
        h("button", { class: "btn", onclick: async () => {
          const got = await pickFiles({ directory: true, multiple: false, title: "Choose the r2v folder" });
          if (got[0]) root.value = got[0];
        } }, "Choose folder"))),
    h("section", { class: "panel" },
      h("h2", {}, "What the pipeline needs"),
      h("ul", { class: "env" },
        row("r2v checkout", env.root_ok ? env.root : null, env.root ? `No r2v pyproject.toml in ${env.root}` : "Not set"),
        row("Python environment", env.python, "Missing. Run the setup in the r2v folder (README, Setup)."),
        row("Godot", env.godot, "Not found. Needed to walk around and open projects."),
        row("ffmpeg", env.ffmpeg, "Not found. Needed for videos, not for LiDAR scans."),
        row("COLMAP", env.colmap, "Not found. Needed for videos, not for LiDAR scans."))),
    h("section", { class: "panel" },
      h("h2", {}, "Existing workspaces"),
      h("p", { class: "hint" }, "Adds every workspace in the r2v folder that a build already wrote to."),
      h("div", { class: "actions" },
        h("button", { class: "btn", onclick: async () => {
          const n = await call<number>("discover_rooms");
          note.textContent = n ? `Added ${n} room${n === 1 ? "" : "s"}.` : "No new workspaces found.";
          await refresh();
          renderSidebarList();
        } }, "Find rooms"), note)));
}

// ---------------------------------------------------------------- new room dialog

function openNewRoom(kind: Source = "video") {
  let source: Source = kind;
  let picked: string[] = [];
  const name = h("input", { id: "new-name", placeholder: "Kitchen", spellcheck: false, autocomplete: "off" });
  const pickedList = h("ul", { class: "inputs" });
  const createBtn = h("button", { class: "btn btn-primary", type: "submit", disabled: true }, "Create room");
  const pickBtn = h("button", { class: "btn", type: "button" });
  const help = h("p", { class: "hint" });

  const sync = () => {
    pickBtn.textContent = source === "lidar" ? (picked.length ? "Choose another folder" : "Choose scan folder") : "Add videos";
    help.textContent = source === "lidar"
      ? "The folder copied from Stray Scanner: rgb.mp4, odometry.csv, depth/ and confidence/. Scale and the up direction come from the scan."
      : "One or more videos of the same interior. Walk slowly, keep the horizon level, film the floor and ceiling too.";
    pickedList.replaceChildren(...picked.map((p) => h("li", {}, h("span", { class: "path", title: p }, p.split(/[\\/]/).pop() ?? p))));
    createBtn.disabled = picked.length === 0;
    tabs.forEach((t) => t.setAttribute("aria-checked", String(t.dataset.kind === source)));
  };

  const tabs = (["video", "lidar"] as const).map((k) =>
    h("button", { type: "button", role: "radio", class: "seg", "data-kind": k, onclick: () => { source = k; picked = []; sync(); } }, k === "video" ? "Videos" : "LiDAR scan"));

  pickBtn.addEventListener("click", async () => {
    const got = await pickFiles(source === "lidar" ? { directory: true, multiple: false, title: "Choose the scan folder" } : { directory: false, multiple: true, title: "Add videos", videos: true });
    if (!got.length) return;
    picked = source === "lidar" ? got.slice(0, 1) : [...picked, ...got];
    if (!name.value) name.value = (picked[0].split(/[\\/]/).pop() ?? "").replace(/\.[^.]+$/, "").replace(/[_-]+/g, " ");
    sync();
  });

  const dlg = h("dialog", { class: "dialog", "aria-labelledby": "new-title" },
    h("form", { method: "dialog", onsubmit: async (e: Event) => {
      e.preventDefault();
      const v = await call<RoomView>("add_room", { name: name.value, source, inputs: picked });
      dlg.close();
      await refresh();
      select(v.room.id);
    } },
      h("h2", { id: "new-title" }, "New room"),
      h("div", { class: "segmented", role: "radiogroup", "aria-label": "Capture" }, ...tabs),
      help,
      pickedList,
      pickBtn,
      field("Name", null, name, "new-name"),
      h("div", { class: "actions" }, createBtn, h("button", { class: "btn btn-quiet", type: "button", onclick: () => dlg.close() }, "Cancel"))));
  dlg.addEventListener("close", () => dlg.remove());
  document.body.append(dlg);
  sync();
  dlg.showModal();
}

// ---------------------------------------------------------------- actions

function showError(e: unknown) {
  state.error = String(e);
  const bar = document.getElementById("error");
  if (bar) {
    bar.replaceChildren(h("span", {}, state.error), h("button", { class: "link", onclick: () => { state.error = ""; bar.hidden = true; } }, "Dismiss"));
    bar.hidden = false;
  }
}

async function save(room: Room) {
  try {
    const i = state.rooms.findIndex((r) => r.room.id === room.id);
    if (room.workspace !== state.rooms[i]?.room.workspace) state.photos.delete(room.id);
    const v = await call<RoomView>("update_room", { room });
    if (i >= 0) state.rooms[i] = v;
    renderSidebarList();
  } catch (e) {
    showError(e);
  }
}

async function forget(v: RoomView) {
  if (!window.confirm(`Remove ${v.room.name} from the list? Its files stay on disk.`)) return;
  await call("forget_room", { id: v.room.id });
  state.selected = null;
  await refresh();
  render();
}

async function build(v: RoomView, only: string | null, force: boolean) {
  const run = freshRun();
  state.runs.set(v.room.id, run);
  state.logs.set(v.room.id, []);
  state.job = v.room.id;
  const log = document.getElementById("log");
  if (log) {
    log.classList.remove("is-empty");
    log.replaceChildren();
  }
  updateTape();
  try {
    await call("start_build", { id: v.room.id, only, force });
  } catch (e) {
    run.running = false;
    run.failed = { stage: null, message: String(e) };
    state.job = null;
    updateTape();
  }
}

async function stop() {
  await call("stop_build");
}

async function godot(v: RoomView, play: boolean) {
  try {
    await call("open_godot", { id: v.room.id, play });
  } catch (e) {
    showError(e);
  }
}

async function select(id: string) {
  state.selected = id;
  state.view = "room";
  await loadLog(id);
  render();
}

// ---------------------------------------------------------------- render

let lastPage = "";
async function render() {
  const main = state.view === "settings"
    ? await settingsPage()
    : current() ? roomPage(current()!) : emptyPage();
  // Re-rendering the same page (a settings change) keeps the scroll position.
  const page = `${state.view}:${state.selected}`;
  const keep = page === lastPage ? document.querySelector(".stage")?.scrollTop ?? 0 : 0;
  lastPage = page;
  app.replaceChildren(
    sidebar(),
    h("div", { class: "stage" },
      h("div", { id: "error", class: "error-bar", role: "alert", hidden: !state.error }, state.error),
      main));
  document.querySelector(".stage")!.scrollTop = keep;
  renderSidebarList();
}

// ---------------------------------------------------------------- events

on<LineEvent>("build-line", ({ room, line, err }) => {
  const run = state.runs.get(room) ?? (state.runs.set(room, freshRun()), state.runs.get(room)!);
  readLine(run, line, err);
  const lines = state.logs.get(room) ?? (state.logs.set(room, []), state.logs.get(room)!);
  lines.push(line);
  if (room === state.selected && state.view === "room") {
    const log = document.getElementById("log");
    if (log) {
      log.classList.remove("is-empty");
      if (log.textContent === "The pipeline's output shows here while it builds.") log.replaceChildren();
      const stick = log.scrollHeight - log.scrollTop - log.clientHeight < 40;
      log.append(logLine(line, err));
      if (log.childElementCount > 4000) log.firstElementChild?.remove();
      if (stick) log.scrollTop = log.scrollHeight;
    }
  }
  scheduleTape();
});

on<ExitEvent>("build-exit", async ({ room, code, cancelled }) => {
  const run = state.runs.get(room);
  state.job = null;
  if (run) {
    run.running = false;
    run.cancelled = cancelled;
    if (!cancelled && code !== 0) {
      const lines = state.logs.get(room) ?? [];
      const last = [...lines].reverse().find((l) => l.trim() && !l.startsWith(" ") && !/^\s*File /.test(l)) ?? "";
      run.failed = { stage: run.stage, message: last };
    } else if (!cancelled && run.stage) {
      run.done.add(run.stage);
    }
  }
  state.photos.delete(room);
  await refresh();
  if (run && !run.failed && !run.cancelled) state.runs.delete(room);
  if (room === state.selected && state.view === "room") {
    // Stats and stage stamps changed on disk.
    const v = current();
    const head = document.querySelector(".room-head");
    if (v && head) head.replaceWith(header(v));
  }
  updateTape();
});

on<string>("godot-status", (msg) => {
  state.godotNote = msg;
  updateActions();
});

let tapeQueued = false;
function scheduleTape() {
  if (tapeQueued) return;
  tapeQueued = true;
  requestAnimationFrame(() => {
    tapeQueued = false;
    updateTape();
  });
}

// The clock ticks once a second while something builds.
window.setInterval(() => {
  const clock = document.getElementById("clock");
  const run = state.selected ? state.runs.get(state.selected) : undefined;
  if (clock && run?.running) clock.textContent = minutes(Date.now() - run.startedAt);
}, 1000);

(async () => {
  await refresh();
  state.job = await call<string | null>("current_job");
  if (state.selected) await loadLog(state.selected);
  render();
})();
