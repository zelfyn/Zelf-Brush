import { useApp } from "../store";

export function StatusBar() {
  const p = useApp((s) => s.project);
  const comp = useApp((s) => s.comp);
  const frame = useApp((s) => s.frame);
  const gpu = useApp((s) => s.gpu);
  const tl = useApp((s) => s.timeline);
  const jobs = useApp((s) => s.jobs);
  const engine = useApp((s) => s.activeEngine);
  const provider = useApp((s) => s.engineProvider);
  const accel = useApp((s) => s.engineAccelerated);
  const open = useApp((s) => s.queueOpen);
  const active = Object.values(jobs).filter((j) => j.status === "running");
  const job = active[0];
  const vram = gpu && gpu.detected ? `${(gpu.vram_used_mb / 1024).toFixed(1)} / ${(gpu.vram_total_mb / 1024).toFixed(0)} GB` : "";
  return (
    <div className="status">
      {p ? (
        <>
          <span>Frame <b className="num">{frame}</b></span>
          <span><b>{comp?.width ?? p.project.width}×{comp?.height ?? p.project.height}</b></span>
          <span><b>{(comp?.fps ?? p.project.fps).toFixed((comp?.fps ?? p.project.fps) % 1 ? 3 : 0)}</b> fps</span>
        </>
      ) : <span>No project</span>}
      <span title={accel ? "AI inference runs on the GPU" : gpu?.detected ? "A GPU exists but the active AI engine is running on the CPU. Pick a GPU engine in Properties > Engine, or check the AI model is installed." : gpu?.message}>
        {accel
          ? <>GPU: <b>{gpu?.name ?? "GPU"}</b> {vram} · <b>{provider}</b> (AI on GPU)</>
          : gpu?.detected ? <>GPU: <b>{gpu.name}</b> · AI running on CPU</> : "No NVIDIA GPU detected · CPU mode"}
      </span>
      <span>Engine: <b>{engine}</b></span>
      {tl && <span>Cache: <b className="num">{tl.cache.frames}/{tl.cache.needed ?? tl.n_frames}</b>{tl.cache.complete ? "" : " …"}</span>}
      <span className="spacer" />
      {job && <span>{job.title}: {job.message} <span className="bar"><i style={{ width: `${job.progress * 100}%` }} /></span></span>}
      <button onClick={() => import("../store").then((m) => m.store.set({ queueOpen: !open }))}>Queue</button>
    </div>
  );
}
