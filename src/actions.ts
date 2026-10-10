import * as api from "./api";
import { ApiError, inTauri } from "./api";
import { store } from "./store";
import type { ExportSettings, Job, MatteSettings, ObjectInfo, ProjectState, SystemStatus, Timeline } from "./types";

const VIDEO_EXT = /\.(mp4|mkv|mov)$/i;

export function showError(e: unknown): void {
  if (e instanceof ApiError) store.set({ error: e.toAppError() });
  else store.set({ error: { message: "Something went wrong.", hints: ["Details are in the app log"] } });
  console.error(e);
}

export function toast(message: string): void {
  store.set({ toast: message });
  window.setTimeout(() => {
    if (store.get().toast === message) store.set({ toast: null });
  }, 3500);
}

async function guard<T>(fn: () => Promise<T>): Promise<T | undefined> {
  try {
    return await fn();
  } catch (e) {
    showError(e);
    return undefined;
  }
}

export const bump = (): void => store.set((s) => ({ renderKey: s.renderKey + 1 }));

export async function refreshTimeline(): Promise<void> {
  if (!store.get().project) return;
  const tl = await guard(() => api.get<Timeline>("/timeline"));
  if (tl) store.set({ timeline: tl });
}

export async function refreshProject(): Promise<void> {
  const p = await guard(() => api.get<ProjectState>("/project"));
  if (p) store.set({ project: p });
}

export async function afterEdit(): Promise<void> {
  bump();
  await refreshTimeline();
}

// ------------------------------------------------------------------ boot / events / status
export async function boot(): Promise<void> {
  try {
    await api.initBackend();
  } catch (e) {
    showError(e);
    return;
  }
  const es = new EventSource(api.url("/events"));
  es.onmessage = (m) => handleEvent(JSON.parse(m.data) as { type: string; [k: string]: unknown });
  await pollStatus();
  window.setInterval(pollStatus, 5000);
  store.set({ ready: true });
  if (inTauri()) {
    const { getCurrentWebview } = await import("@tauri-apps/api/webview");
    await getCurrentWebview().onDragDropEvent((ev) => {
      const t = ev.payload.type;
      if (t === "enter" || t === "over") store.set({ dropActive: true });
      else if (t === "leave") store.set({ dropActive: false });
      else if (t === "drop") {
        store.set({ dropActive: false });
        const path = ev.payload.paths.find((p: string) => VIDEO_EXT.test(p));
        if (path) void importPath(path);
        else showError(new ApiError("That file type is not supported.", ["Drop an .mp4, .mkv or .mov file"]));
      }
    });
  }
}

async function pollStatus(): Promise<void> {
  try {
    const st = await api.get<SystemStatus>("/system/status");
    store.set({ gpu: st.gpu, engines: st.engines, activeEngine: st.active_engine.name, engineProvider: st.active_engine.provider, engineAccelerated: st.active_engine.accelerated });
  } catch {
    /* transient */
  }
}

function handleEvent(ev: { type: string; [k: string]: unknown }): void {
  switch (ev.type) {
    case "processing_progress":
    case "export_progress": {
      const job = ev.job as Job;
      store.set((s) => ({ jobs: { ...s.jobs, [job.id]: job } }));
      if (ev.type === "processing_progress" && job.kind === "auto-roto" && job.status === "running" && job.progress > 0.1) {
        // frames appear as they are tracked
        if (Math.round(job.progress * 100) % 10 === 0) void refreshTimeline();
      }
      break;
    }
    case "job_finished": {
      const job = ev.job as Job;
      store.set((s) => ({ jobs: { ...s.jobs, [job.id]: job } }));
      if (job.status === "failed" && job.error) store.set({ error: job.error });
      if (job.lane === "cache" && job.status === "completed") void refreshProject().then(refreshTimeline);
      if (job.lane === "process") {
        if (job.status === "completed" && job.kind === "auto-roto") store.set({ analysis: "done" });
        void afterEdit();
        void refreshProject();
        if (job.status === "completed") toast(job.kind === "auto-roto" ? "Auto roto finished" : "Tracking finished");
      }
      if (job.lane === "export" && job.status === "completed") {
        const w = (job.result.warnings as string[] | undefined) ?? [];
        toast(w.length ? `Export completed with a warning: ${w[0]}` : "Export completed successfully.");
      }
      break;
    }
    case "error":
      store.set({ error: { message: String(ev.message), hints: (ev.hints as string[]) ?? [] } });
      break;
    default:
      break;
  }
}

// ------------------------------------------------------------------ project lifecycle
async function openState(p: ProjectState): Promise<void> {
  store.set({
    project: p, timeline: null, frame: 0, displayedFrame: -1, playing: false, detections: p.objects,
    analysis: p.objects.length ? "done" : "idle", renderKey: store.get().renderKey + 1, busy: null,
  });
  await refreshTimeline();
  if (p.recovered) toast("Recovered unsaved changes from autosave");
}

export async function importPath(path: string): Promise<void> {
  store.set({ busy: "Reading video…" });
  const p = await guard(() => api.post<ProjectState>("/video/import", { path }));
  store.set({ busy: null });
  if (!p) return;
  await openState(p);
  await runAnalysis(true);
}

export async function importFile(file: File): Promise<void> {
  if (!VIDEO_EXT.test(file.name)) {
    showError(new ApiError("That file type is not supported.", ["Drop an .mp4, .mkv or .mov file"]));
    return;
  }
  store.set({ busy: "Uploading video…" });
  const p = await guard(() => api.upload<ProjectState>("/video/upload", file));
  store.set({ busy: null });
  if (!p) return;
  await openState(p);
  await runAnalysis(true);
}

async function pick(kind: "video" | "project" | "folder" | "save-video", extra: { default_name?: string; ext?: string } = {}): Promise<string> {
  const r = await guard(() => api.post<{ path: string }>("/system/pick", { kind, ...extra }));
  return r?.path ?? "";
}

export const browseVideo = async (): Promise<void> => {
  const path = await pick("video");
  if (path) await importPath(path);
};

export async function openProject(): Promise<void> {
  const path = await pick("project");
  if (!path) return;
  store.set({ busy: "Opening project…" });
  const p = await guard(() => api.post<ProjectState>("/project/open", { path }));
  store.set({ busy: null });
  if (p) await openState(p);
}

export async function saveProject(): Promise<void> {
  if (!store.get().project) return;
  if (await guard(() => api.post("/project/save"))) toast("Project saved");
}

export async function saveProjectAs(): Promise<void> {
  if (!store.get().project) return;
  const path = await pick("folder");
  if (!path) return;
  const name = store.get().project!.project.name;
  const p = await guard(() => api.post<ProjectState>("/project/save-as", { path: `${path}/${name}` }));
  if (p) {
    await openState(p);
    toast("Project saved");
  }
}

export async function newProject(): Promise<void> {
  if (store.get().project) await guard(() => api.post("/project/close"));
  store.set({ project: null, timeline: null, detections: [], analysis: "idle", frame: 0, jobs: {} });
}

// ------------------------------------------------------------------ analysis / auto roto
export async function runAnalysis(autoStart: boolean): Promise<void> {
  store.set({ analysis: "running" });
  const r = await guard(() => api.post<{ frame: number; objects: ObjectInfo[]; dominant: string | null }>("/segmentation/analyze", {}));
  if (!r) {
    store.set({ analysis: "idle" });
    return;
  }
  store.set({ detections: r.objects, frame: r.frame });
  if (!r.objects.length) {
    store.set({
      analysis: "needs-selection",
      error: {
        message: "Object detection failed.",
        hints: ["Select the object manually", "Use Brush", "Use Lasso"],
      },
    });
    return;
  }
  if (autoStart && r.dominant) await startAutoRoto([r.dominant], r.frame);
  else store.set({ analysis: "needs-selection" });
}

export async function startAutoRoto(ids: string[], frame?: number): Promise<void> {
  const f = frame ?? store.get().detections[0]?.frame ?? store.get().frame;
  const job = await guard(() => api.post<Job>("/rotoscope/auto", { frame: f, object_ids: ids }));
  if (job) store.set((s) => ({ jobs: { ...s.jobs, [job.id]: job }, analysis: "running", frame: f }));
}

export async function toggleObject(id: string): Promise<void> {
  store.set((s) => ({ detections: s.detections.map((o) => (o.id === id ? { ...o, selected: !o.selected } : o)) }));
}

export async function propagate(direction: "forward" | "backward" | "both", recalc = false): Promise<void> {
  const { frame } = store.get();
  const job = await guard(() => api.post<Job>("/tracking/propagate", { frame, direction, recalc }));
  if (job) store.set((s) => ({ jobs: { ...s.jobs, [job.id]: job } }));
}

export async function clearFrameMask(): Promise<void> {
  if (await guard(() => api.post("/mask/clear", { frame: store.get().frame }))) await afterEdit();
}

// ------------------------------------------------------------------ undo / redo / settings
export async function undo(): Promise<void> {
  const r = await guard(() => api.post<{ label: string | null }>("/mask/undo"));
  if (r?.label) {
    toast(`Undo ${r.label}`);
    await afterEdit();
  }
}

export async function redo(): Promise<void> {
  const r = await guard(() => api.post<{ label: string | null }>("/mask/redo"));
  if (r?.label) {
    toast(`Redo ${r.label}`);
    await afterEdit();
  }
}

let matteTimer: number | undefined;
export function setMatte(patch: Partial<MatteSettings>): void {
  const p = store.get().project;
  if (!p) return;
  store.set({ project: { ...p, maskSettings: { ...p.maskSettings, ...patch } } });
  window.clearTimeout(matteTimer);
  matteTimer = window.setTimeout(async () => {
    await guard(() => api.patch("/project/settings", { maskSettings: store.get().project?.maskSettings }));
    bump();
  }, 150);
}

export async function setWorkArea(a: number, b: number): Promise<void> {
  const r = await guard(() => api.patch<{ workArea: { in: number; out: number } }>("/project/settings", { workArea: { in: a, out: b } }));
  const p = store.get().project;
  if (r && p) store.set({ project: { ...p, workArea: r.workArea } });
}

export async function setEngine(name: string): Promise<void> {
  if (await guard(() => api.patch("/project/settings", { engine: name }))) {
    await refreshProject();
    store.set({ activeEngine: name });
  }
}

// ------------------------------------------------------------------ export
export async function pickExportPath(kind: string): Promise<string> {
  const base = store.get().project?.project.name ?? "roto";
  if (kind === "after-effects") return pick("folder").then((p) => (p ? `${p}/${base}_RotoExport` : ""));
  if (kind === "mov-alpha") return pick("save-video", { default_name: `${base}_alpha.mov`, ext: ".mov" });
  return pick("save-video", { default_name: `${base}_roto.mp4`, ext: ".mp4" });
}

export async function queueExport(kind: string, output: string, settings: Partial<ExportSettings>): Promise<boolean> {
  const job = await guard(() => api.post<Job>("/export", { kind, output, settings }));
  if (!job) return false;
  store.set((s) => ({ jobs: { ...s.jobs, [job.id]: job }, queueOpen: true }));
  return true;
}

export async function cancelJob(id: string): Promise<void> {
  await guard(() => api.post(`/jobs/${id}/cancel`));
}

// ------------------------------------------------------------------ playback
export function setFrame(f: number): void {
  const n = store.get().project?.project.frame_count ?? 1;
  store.set({ frame: Math.max(0, Math.min(n - 1, Math.round(f))) });
}

export const stepFrame = (d: number): void => setFrame(store.get().frame + d);

export function togglePlay(): void {
  if (!store.get().project) return;
  store.set((s) => ({ playing: !s.playing }));
}

let raf = 0;
let lastTick = 0;
export function startPlaybackLoop(): void {
  const loop = (t: number): void => {
    const s = store.get();
    if (s.playing && s.project) {
      const fps = s.project.project.fps;
      const wa = s.project.workArea;
      // advance only when the previous frame has been drawn (drops frames instead of stalling)
      if (t - lastTick >= 1000 / fps && s.displayedFrame === s.frame) {
        lastTick = t;
        const next = s.frame + 1 > wa.out ? wa.in : s.frame + 1;
        store.set({ frame: next });
        for (let k = 1; k <= 4; k++) {
          const im = new Image();
          im.src = api.url(`/render/${Math.min(wa.out, next + k)}`, { mode: s.viewMode, q: s.previewQuality, k: s.renderKey });
        }
      }
    }
    raf = requestAnimationFrame(loop);
  };
  cancelAnimationFrame(raf);
  raf = requestAnimationFrame(loop);
}
