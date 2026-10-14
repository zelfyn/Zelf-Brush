import { useSyncExternalStore } from "react";
import type { AppError, CompState, EngineInfo, GpuInfo, Job, ObjectInfo, ProjectState, Timeline, Tool, ViewMode } from "./types";

export interface AppState {
  ready: boolean;
  project: ProjectState | null;
  comp: CompState | null;
  timeline: Timeline | null;
  frame: number;
  displayedFrame: number;
  playing: boolean;
  tool: Tool;
  viewMode: ViewMode;
  previewQuality: 25 | 50 | 75 | 100;
  brush: { size: number; hardness: number; strength: number };
  lasso: { mode: "add" | "remove"; shape: "freehand" | "polygon"; aiRefine: boolean };
  objectMode: "add" | "subtract";
  renderKey: number;
  detections: ObjectInfo[];
  analysis: "idle" | "running" | "needs-selection" | "done";
  jobs: Record<string, Job>;
  gpu: GpuInfo | null;
  engines: EngineInfo[];
  activeEngine: string;
  engineProvider: string;
  engineAccelerated: boolean;
  error: AppError | null;
  toast: string | null;
  dialog: null | "export";
  exportKind: "mp4" | "mov-alpha" | "after-effects" | "depth";
  queueOpen: boolean;
  busy: string | null;
  dropActive: boolean;
}

const initial: AppState = {
  ready: false, project: null, comp: null, timeline: null, frame: 0, displayedFrame: -1, playing: false,
  tool: "brush", viewMode: "overlay", previewQuality: 100,
  brush: { size: 40, hardness: 0.7, strength: 1 },
  lasso: { mode: "add", shape: "freehand", aiRefine: true },
  objectMode: "add", renderKey: 0, detections: [], analysis: "idle", jobs: {},
  gpu: null, engines: [], activeEngine: "", engineProvider: "CPU", engineAccelerated: false, error: null, toast: null, dialog: null, exportKind: "mp4",
  queueOpen: false, busy: null, dropActive: false,
};

let state = initial;
const listeners = new Set<() => void>();

export const store = {
  get: (): AppState => state,
  set(patch: Partial<AppState> | ((s: AppState) => Partial<AppState>)): void {
    state = { ...state, ...(typeof patch === "function" ? patch(state) : patch) };
    listeners.forEach((l) => l());
  },
  subscribe(l: () => void): () => void {
    listeners.add(l);
    return () => listeners.delete(l);
  },
};

/** Selectors must return primitives or stable references (never fresh objects). */
export function useApp<T>(sel: (s: AppState) => T): T {
  return useSyncExternalStore(store.subscribe, () => sel(state));
}
