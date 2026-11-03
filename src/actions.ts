import * as api from "./api";
import { ApiError, inTauri } from "./api";
import { store } from "./store";
import type { AutoUpdateSettings, CompState, ExportSettings, Job, Layer, MatteSettings, ObjectInfo, ProjectState, SystemStatus, Timeline, VideoProbe } from "./types";

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

// ------------------------------------------------------------------ layers <-> timeline time
export const activeLayer = (): Layer | null => {
  const c = store.get().comp;
  return c ? (c.layers.find((l) => l.id === c.active) ?? null) : null;
};

/** Comp frame -> the layer's own source frame (what the roto tools and renders use), or null if it is off screen. */
export function layerLocal(L: Layer, cf: number, comp: CompState): number | null {
  if (cf < L.start || cf > L.end) return null;
  return Math.min(L.trimOut, L.trimIn + Math.round(((cf - L.start) * L.fps) / comp.fps));
}

export const compFrameOf = (L: Layer, local: number, comp: CompState): number =>
  L.start + Math.round(((local - L.trimIn) * comp.fps) / L.fps);

export function localFrameNow(warn = true): number | null {
  const { comp, frame } = store.get();
  const L = activeLayer();
  if (!comp || !L) return null;
  const lf = layerLocal(L, frame, comp);
  if (lf === null && warn) toast("Move the playhead onto the selected layer first");
  return lf;
}

/** URL of the image the viewport shows at a comp frame (null when the selected layer is not on screen there). */
export function renderUrl(cf: number): string | null {
  const s = store.get();
  if (!s.comp) return null;
  if (s.viewMode === "composite") return api.url(`/comp/render/${cf}`, { q: s.previewQuality, k: s.renderKey });
  const L = activeLayer();
  const lf = L ? layerLocal(L, cf, s.comp) : null;
  if (!L || lf === null) return null;
  return api.url(`/render/${lf}`, { mode: s.viewMode, q: s.previewQuality, k: s.renderKey, layer: L.id });
}

// ------------------------------------------------------------------ state sync
export async function refreshTimeline(): Promise<void> {
  if (!store.get().project) return;
  const tl = await guard(() => api.get<Timeline>("/timeline"));
  if (tl) store.set({ timeline: tl });
}

function applyState(p: ProjectState): void {
  api.setLayer(p.comp.active);
  store.set({ project: p, comp: p.comp });
}

export async function refreshProject(): Promise<void> {
  const p = await guard(() => api.get<ProjectState>("/project"));
  if (p) applyState(p);
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
        if (path) void beginImport(path);
        else showError(new ApiError("That file type is not supported.", ["Drop an .mp4, .mkv or .mov file"]));
      }
    });
  }
}

async function pollStatus(): Promise<void> {
  try {
    const st = await api.get<SystemStatus>("/system/status");
    store.set({ gpu: st.gpu, engines: st.engines, activeEngine: st.active_engine.name, engineProvider: st.active_engine.provider, engineAccelerated: st.active_engine.accelerated, engineReady: st.active_engine.ready });
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
      if (job.lane === "auto") refreshSoon();           // silent: the picture and timeline catch up, no toasts
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
  applyState(p);
  store.set({
    timeline: null, frame: 0, displayedFrame: -1, playing: false, detections: p.objects,
    analysis: p.objects.length ? "done" : "idle", renderKey: store.get().renderKey + 1, busy: null,
  });
  await refreshTimeline();
  if (p.recovered) toast("Recovered unsaved changes from autosave");
}

export interface ImportOptions {
  name: string;
  trimIn: number;
  trimOut: number;
  asLayer: boolean;
  start: number;
  autoRoto: boolean;
  aiModel: string;
}

/** Step 1 of every import: read the video's metadata and open the import dialog (nothing is imported yet). */
export async function beginImport(path: string, preferLayer?: boolean): Promise<void> {
  store.set({ busy: "Reading video…" });
  const info = await guard(() => api.post<VideoProbe>("/video/probe", { path }));
  store.set({ busy: null });
  if (info) store.set({ importDraft: { path, info, preferLayer: preferLayer ?? !!store.get().comp } });
}

/** Browser dev mode: upload the file so the dialog can preview it (the desktop app passes real paths instead). */
export async function stageFile(file: File): Promise<void> {
  if (!VIDEO_EXT.test(file.name)) {
    showError(new ApiError("That file type is not supported.", ["Drop an .mp4, .mkv or .mov file"]));
    return;
  }
  store.set({ busy: "Uploading video…" });
  const r = await guard(() => api.upload<{ path: string; info: VideoProbe }>("/video/stage", file));
  store.set({ busy: null });
  if (r) store.set({ importDraft: { path: r.path, info: r.info, preferLayer: !!store.get().comp } });
}

/** Step 2: the user pressed "Add to timeline" in the dialog. */
export async function confirmImport(o: ImportOptions): Promise<void> {
  const d = store.get().importDraft;
  if (!d) return;
  store.set({ importDraft: null, busy: "Importing…" });
  const body = { path: d.path, trimIn: o.trimIn, trimOut: o.trimOut, name: o.name, aiModel: o.aiModel };
  if (o.asLayer && store.get().comp) {
    const p = await guard(() => api.post<ProjectState>("/layers/add", { ...body, start: o.start }));
    store.set({ busy: null });
    if (!p) return;
    applyState(p);
    store.set({ detections: p.objects, analysis: "idle", frame: o.start, renderKey: store.get().renderKey + 1 });
    await refreshTimeline();
    toast(`Layer added at frame ${o.start}`);
  } else {
    const p = await guard(() => api.post<ProjectState>("/video/import", body));
    store.set({ busy: null });
    if (!p) return;
    await openState(p);
  }
  if (o.autoRoto) await runAnalysis(true);
}

export const cancelImport = (): void => store.set({ importDraft: null });

/** Dropping a video opens the import dialog (add as a layer when a project is open, like After Effects). */
export const droppedPath = (path: string): Promise<void> => beginImport(path);
export const importFile = (file: File): Promise<void> => stageFile(file);

async function pick(kind: "video" | "project" | "folder" | "save-video", extra: { default_name?: string; ext?: string } = {}): Promise<string> {
  const r = await guard(() => api.post<{ path: string }>("/system/pick", { kind, ...extra }));
  return r?.path ?? "";
}

export async function browseVideo(asLayer = false): Promise<void> {
  const path = await pick("video");
  if (path) await beginImport(path, asLayer);
}

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
  const c = store.get().comp;
  if (!c) return;
  const path = await pick("folder");
  if (!path) return;
  const p = await guard(() => api.post<ProjectState>("/project/save-as", { path: `${path}/${c.name}` }));
  if (p) {
    await openState(p);
    toast("Project saved");
  }
}

export async function newProject(): Promise<void> {
  if (store.get().project) await guard(() => api.post("/project/close"));
  api.setLayer("");
  store.set({ project: null, comp: null, timeline: null, detections: [], analysis: "idle", frame: 0, jobs: {} });
}

// ------------------------------------------------------------------ layers
export async function selectLayer(id: string): Promise<void> {
  if (store.get().comp?.active === id) return;
  const p = await guard(() => api.post<ProjectState>(`/layers/${id}/activate`));
  if (!p) return;
  applyState(p);
  store.set({ detections: p.objects, analysis: p.objects.length ? "done" : "idle", timeline: null });
  bump();
  await refreshTimeline();
}

export async function patchLayer(id: string, patch: Partial<Pick<Layer, "name" | "start" | "trimIn" | "trimOut" | "visible" | "opacity" | "scale" | "x" | "y" | "useMask">>): Promise<void> {
  const p = await guard(() => api.patch<ProjectState>(`/layers/${id}`, patch));
  if (!p) return;
  applyState(p);
  bump();
  await refreshTimeline();
}

export async function reorderLayers(order: string[]): Promise<void> {
  const p = await guard(() => api.post<ProjectState>("/layers/reorder", { order }));
  if (p) {
    applyState(p);
    bump();
  }
}

export async function moveLayer(id: string, dir: -1 | 1): Promise<void> {
  const ids = (store.get().comp?.layers ?? []).map((l) => l.id);
  const i = ids.indexOf(id);
  const j = i + dir;
  if (i < 0 || j < 0 || j >= ids.length) return;
  [ids[i], ids[j]] = [ids[j], ids[i]];
  await reorderLayers(ids);
}

export async function deleteLayer(id: string): Promise<void> {
  const p = await guard(() => api.del<ProjectState>(`/layers/${id}`));
  if (!p) return;
  applyState(p);
  store.set({ detections: p.objects, analysis: p.objects.length ? "done" : "idle", timeline: null });
  bump();
  await refreshTimeline();
}

/** Cut the selected layer at the playhead. which: head (set In), tail (set Out) or reset. */
export async function trimActive(which: "in" | "out" | "reset"): Promise<void> {
  const L = activeLayer();
  const c = store.get().comp;
  if (!L || !c) return;
  if (which === "reset") return patchLayer(L.id, { trimIn: 0, trimOut: L.frames - 1 });
  const lf = localFrameNow();
  if (lf === null) return;
  if (which === "in") await patchLayer(L.id, { trimIn: lf, start: compFrameOf(L, lf, c) === L.start && c.layers.length === 1 ? 0 : compFrameOf(L, lf, c) });
  else await patchLayer(L.id, { trimOut: lf });
}

export async function setCompWorkArea(a: number, b: number): Promise<void> {
  const c = await guard(() => api.post<CompState>("/comp/work-area", { start: a, end: b }));
  const p = store.get().project;
  if (c && p) store.set({ comp: c, project: { ...p, comp: c } });
}

// ------------------------------------------------------------------ analysis / auto roto
export async function runAnalysis(autoStart: boolean): Promise<void> {
  const L = activeLayer();
  const c = store.get().comp;
  store.set({ analysis: "running" });
  const r = await guard(() => api.post<{ frame: number; objects: ObjectInfo[]; dominant: string | null }>("/segmentation/analyze", {}));
  if (!r || !L || !c) {
    store.set({ analysis: "idle" });
    return;
  }
  store.set({ detections: r.objects, frame: compFrameOf(L, r.frame, c) });
  await refreshProject();
  if (!r.objects.length) {
    store.set({ analysis: "needs-selection", error: { message: "Object detection failed.", hints: ["Pick another AI model (Subjects panel)", "Drag a box with the Object tool", "Or use Brush / Lasso"] } });
    return;
  }
  if (autoStart && r.dominant) await startAutoRoto([r.dominant], r.frame);
  else store.set({ analysis: "needs-selection" });
}

export async function startAutoRoto(ids: string[], localFrame?: number): Promise<void> {
  const L = activeLayer();
  const c = store.get().comp;
  const f = localFrame ?? store.get().detections[0]?.frame ?? localFrameNow(false) ?? 0;
  const job = await guard(() => api.post<Job>("/rotoscope/auto", { frame: f, object_ids: ids }));
  if (job) store.set((s) => ({ jobs: { ...s.jobs, [job.id]: job }, analysis: "running", frame: L && c ? compFrameOf(L, f, c) : s.frame }));
}

export async function toggleObject(id: string): Promise<void> {
  store.set((s) => ({ detections: s.detections.map((o) => (o.id === id ? { ...o, selected: !o.selected } : o)) }));
}

export async function setAiModel(model: string): Promise<void> {
  if (await guard(() => api.patch("/project/settings", { aiModel: model }))) {
    await refreshProject();
    toast(model === "auto" ? "AI model: Auto (decided on the next detection)" : `AI model: ${model}`);
  }
}

export async function propagate(direction: "forward" | "backward" | "both", recalc = false): Promise<void> {
  const frame = localFrameNow();
  if (frame === null) return;
  const job = await guard(() => api.post<Job>("/tracking/propagate", { frame, direction, recalc }));
  if (job) store.set((s) => ({ jobs: { ...s.jobs, [job.id]: job } }));
}

export async function clearFrameMask(): Promise<void> {
  const frame = localFrameNow();
  if (frame === null) return;
  if (await guard(() => api.post("/mask/clear", { frame }))) await afterEdit();
}

// ------------------------------------------------------------------ AI after an edit (background, never blocks)
let autoTimer: number | undefined;

/** After the user applies a tool, let the AI carry the correction to nearby frames. Debounced, runs on the server in
 * the background, and is cancelled by the server the moment another edit arrives. */
export function scheduleAutoUpdate(localFrame: number): void {
  const au = store.get().autoUpdate;
  window.clearTimeout(autoTimer);
  if (!au.enabled || (au.forward <= 0 && au.backward <= 0)) return;
  autoTimer = window.setTimeout(() => {
    const cur = store.get().autoUpdate;
    if (!cur.enabled) return;
    void api.post("/mask/auto-update", { frame: localFrame, forward: cur.forward, backward: cur.backward }).catch(() => undefined);
  }, 900);
}

export function setAutoUpdate(patch: Partial<AutoUpdateSettings>): void {
  const next = { ...store.get().autoUpdate, ...patch };
  store.set({ autoUpdate: next });
  try {
    localStorage.setItem("zelfbrush.autoUpdate", JSON.stringify(next));
  } catch {
    /* storage unavailable */
  }
}

/** Show the "AI working" chip while a request that runs the model is in flight. */
export async function trackAi<T>(fn: () => Promise<T>): Promise<T> {
  store.set((s) => ({ aiPending: s.aiPending + 1 }));
  try {
    return await fn();
  } finally {
    store.set((s) => ({ aiPending: Math.max(0, s.aiPending - 1) }));
  }
}

let refreshTimer: number | undefined;
function refreshSoon(): void {
  window.clearTimeout(refreshTimer);
  refreshTimer = window.setTimeout(() => void afterEdit(), 150);
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

export async function setEngine(name: string): Promise<void> {
  if (await guard(() => api.patch("/project/settings", { engine: name }))) {
    await refreshProject();
    store.set({ activeEngine: name });
  }
}

// ------------------------------------------------------------------ export
export async function pickExportPath(kind: string, depthFormat = "mp4"): Promise<string> {
  const base = store.get().comp?.name ?? "zelf-brush";
  if (kind === "after-effects") return pick("folder").then((p) => (p ? `${p}/${base}_AE` : ""));
  if (kind === "depth" && depthFormat === "png16") return pick("folder").then((p) => (p ? `${p}/${base}_depth` : ""));
  if (kind === "depth") return pick("save-video", { default_name: `${base}_depth.mp4`, ext: ".mp4" });
  if (kind === "mov-alpha") return pick("save-video", { default_name: `${base}_alpha.mov`, ext: ".mov" });
  return pick("save-video", { default_name: `${base}.mp4`, ext: ".mp4" });
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
  const n = store.get().comp?.duration ?? 1;
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
    if (s.playing && s.comp) {
      const wa = s.comp.workArea;
      // advance only when the previous frame has been drawn (drops frames instead of stalling)
      if (t - lastTick >= 1000 / s.comp.fps && s.displayedFrame === s.frame) {
        lastTick = t;
        const next = s.frame + 1 > wa.out ? wa.in : s.frame + 1;
        store.set({ frame: next });
        for (let k = 1; k <= 4; k++) {
          const u = renderUrl(Math.min(wa.out, next + k));
          if (u) new Image().src = u;
        }
      }
    }
    raf = requestAnimationFrame(loop);
  };
  cancelAnimationFrame(raf);
  raf = requestAnimationFrame(loop);
}
