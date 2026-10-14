export type Tool = "brush" | "eraser" | "lasso" | "object" | "pan";
export type ViewMode = "original" | "mask" | "transparent" | "overlay" | "edge" | "depth" | "composite";

export interface MatteSettings {
  edgeRefine: number;
  detail: number;
  smooth: number;
  feather: number;
  shift: number;
  decontaminate: boolean;
  temporal: number;
}

export interface ExportSettings {
  kind: string;
  background: string;
  scale: number;
  codec: string;
  crf: number;
  movCodec: string;
  range: "full" | "work";
  copySource: boolean;
  hwEncode: "auto" | "cpu" | "nvenc";
  depthFormat: "mp4" | "png16";
  depthInvert: boolean;
  depthSmooth: number;
}

export interface ObjectInfo {
  id: string;
  label: string;
  confidence: number;
  bbox: [number, number, number, number];
  frame: number;
  selected: boolean;
}

export interface Keyframe {
  frame: number;
  kind: "ai" | "user";
}

export interface Layer {
  id: string; name: string; dir: string;
  start: number; end: number; length: number; trimIn: number; trimOut: number; frames: number;
  visible: boolean; opacity: number; scale: number; x: number; y: number; useMask: boolean;
  filename: string; width: number; height: number; fps: number;
  aiModel: "auto" | "general" | "anime" | "fast"; aiModelResolved: string | null;
  hasMask: boolean; hasAudio: boolean;
}

export interface CompState {
  name: string; root: string; width: number; height: number; fps: number; duration: number;
  layers: Layer[]; active: string; workArea: { in: number; out: number };
  exportSettings: ExportSettings; proxy: { width: number; height: number };
}

export interface ProjectState {
  comp: CompState;
  project: {
    name: string; source: string; filename: string; width: number; height: number; fps: number;
    duration: number; frame_count: number; codec: string; size_bytes: number; has_audio: boolean;
  };
  proxy: { width: number; height: number };
  engine: string;
  objects: ObjectInfo[];
  keyframes: Keyframe[];
  maskSettings: MatteSettings;
  exportSettings: ExportSettings;
  workArea: { in: number; out: number };
  root: string;
  recovered: boolean;
}

export interface Timeline {
  n_frames: number;
  keyframes: Keyframe[];
  confidence: Record<string, number>;
  warnings: number[];
  masked: [number, number][];
  cache: { frames: number; complete: boolean };
  can_undo: boolean;
  can_redo: boolean;
}

export interface Job {
  id: string;
  lane: "cache" | "process" | "export";
  kind: string;
  title: string;
  status: "waiting" | "running" | "completed" | "failed" | "cancelled";
  progress: number;
  message: string;
  error: AppError | null;
  result: Record<string, unknown>;
}

export interface GpuInfo {
  detected: boolean;
  name: string | null;
  vram_total_mb: number;
  vram_used_mb: number;
  cuda_available: boolean;
  mode: "cpu" | "cuda" | "directml";
  message: string;
  directml_available?: boolean;
}

export interface EngineInfo {
  name: string;
  available: boolean;
  description: string;
  reason: string;
}

export interface SystemStatus {
  models?: { segmentation: string[]; depth: boolean };
  gpu: GpuInfo;
  engines: EngineInfo[];
  active_engine: { name: string; description: string; provider: string; accelerated: boolean };
}

export interface AppError {
  message: string;
  hints: string[];
  stage?: string | null;
  exit_code?: number | null;
  detail?: string | null;
  log?: string | null;
}
