import * as api from "../api";
import { cancelJob, clearFrameMask, propagate, runAnalysis, setEngine, setMatte, startAutoRoto, toggleObject } from "../actions";
import { store, useApp } from "../store";
import type { MatteSettings } from "../types";

function Slider(p: { label: string; value: number; min: number; max: number; step: number; fmt?: (v: number) => string; onChange: (v: number) => void }) {
  return (
    <div className="row">
      <label>{p.label}</label>
      <input type="range" min={p.min} max={p.max} step={p.step} value={p.value} onChange={(e) => p.onChange(Number(e.target.value))} aria-label={p.label} />
      <output>{p.fmt ? p.fmt(p.value) : p.value}</output>
    </div>
  );
}

const fmtBytes = (b: number) => (b > 1e9 ? `${(b / 1e9).toFixed(2)} GB` : `${(b / 1e6).toFixed(1)} MB`);
const fmtDur = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}.${String(Math.round((s % 1) * 10)).padStart(1, "0")}`;

export function Properties() {
  const project = useApp((s) => s.project);
  const tool = useApp((s) => s.tool);
  const brush = useApp((s) => s.brush);
  const lasso = useApp((s) => s.lasso);
  const objectMode = useApp((s) => s.objectMode);
  const dets = useApp((s) => s.detections);
  const analysis = useApp((s) => s.analysis);
  const jobs = useApp((s) => s.jobs);
  const engines = useApp((s) => s.engines);
  const engine = useApp((s) => s.activeEngine);
  const running = Object.values(jobs).filter((j) => j.lane === "process" && (j.status === "running" || j.status === "waiting"));

  if (!project) {
    return (
      <div className="props"><div className="sec"><h3>Properties</h3><p className="hint">Import a video to see its details and tool settings here.</p></div></div>
    );
  }
  const m = project.project;
  const ms: MatteSettings = project.maskSettings;
  const selected = dets.filter((d) => d.selected).map((d) => d.id);
  const isPaint = tool === "brush" || tool === "eraser";

  return (
    <div className="props">
      <div className="sec">
        <h3>Source</h3>
        <img className="thumb" alt="Video thumbnail" src={api.url("/thumbnail", { n: m.name })} />
        <div className="row kv"><span>File</span><span title={m.filename}>{m.filename}</span></div>
        <div className="row kv"><span>Resolution</span><span>{m.width}×{m.height}</span></div>
        <div className="row kv"><span>Frame rate</span><span>{m.fps.toFixed(3)} fps</span></div>
        <div className="row kv"><span>Duration</span><span>{fmtDur(m.duration)} · {m.frame_count} frames</span></div>
        <div className="row kv"><span>Codec</span><span>{m.codec}</span></div>
        <div className="row kv"><span>Size</span><span>{fmtBytes(m.size_bytes)}</span></div>
      </div>

      <div className="sec">
        <h3>Subjects</h3>
        {analysis === "running" && running.length === 0 && <p className="hint">Looking for subjects…</p>}
        {dets.length === 0 && analysis !== "running" && <p className="hint">No subjects found yet.</p>}
        {dets.map((d) => (
          <label className="obj" key={d.id}>
            <input type="checkbox" checked={d.selected} onChange={() => void toggleObject(d.id)} />
            <span>{d.label}</span>
            <span className="conf num">{Math.round(d.confidence * 100)}%</span>
          </label>
        ))}
        <div className="btns">
          <button className="primary" disabled={!selected.length || running.length > 0} onClick={() => void startAutoRoto(selected)}>Auto roto</button>
          <button disabled={running.length > 0} onClick={() => void runAnalysis(false)}>Detect again</button>
        </div>
        {running.map((j) => (
          <div key={j.id} className="row kv" style={{ gridTemplateColumns: "1fr auto" }}>
            <span>{j.title}: {j.message}</span>
            <button onClick={() => void cancelJob(j.id)}>Cancel</button>
          </div>
        ))}
      </div>

      <div className="sec">
        <h3>{{ brush: "Brush", eraser: "Eraser", lasso: "Lasso", object: "Object selection", pan: "Pan" }[tool]}</h3>
        {isPaint && (
          <>
            <Slider label="Size" value={brush.size} min={2} max={300} step={1} fmt={(v) => `${v}px`} onChange={(v) => store.set({ brush: { ...brush, size: v } })} />
            <Slider label="Hardness" value={brush.hardness} min={0} max={1} step={0.05} fmt={(v) => `${Math.round(v * 100)}%`} onChange={(v) => store.set({ brush: { ...brush, hardness: v } })} />
            <Slider label="Strength" value={brush.strength} min={0.05} max={1} step={0.05} fmt={(v) => `${Math.round(v * 100)}%`} onChange={(v) => store.set({ brush: { ...brush, strength: v } })} />
            <p className="hint">{tool === "brush" ? "Paint areas to keep." : "Paint areas to remove."} Sizes are in proxy pixels.</p>
          </>
        )}
        {tool === "lasso" && (
          <>
            <div className="seg"><button className={lasso.mode === "add" ? "on" : ""} onClick={() => store.set({ lasso: { ...lasso, mode: "add" } })}>Keep</button><button className={lasso.mode === "remove" ? "on" : ""} onClick={() => store.set({ lasso: { ...lasso, mode: "remove" } })}>Remove</button></div>{" "}
            <div className="seg"><button className={lasso.shape === "freehand" ? "on" : ""} onClick={() => store.set({ lasso: { ...lasso, shape: "freehand" } })}>Freehand</button><button className={lasso.shape === "polygon" ? "on" : ""} onClick={() => store.set({ lasso: { ...lasso, shape: "polygon" } })}>Polygon</button></div>
            <label className="chk"><input type="checkbox" checked={lasso.aiRefine} onChange={(e) => store.set({ lasso: { ...lasso, aiRefine: e.target.checked } })} />AI refine to edges</label>
            <p className="hint">{lasso.shape === "polygon" ? "Click points, double-click or press Enter to close, Esc to cancel." : "Drag around the area, release to apply."}</p>
          </>
        )}
        {tool === "object" && (
          <>
            <div className="seg"><button className={objectMode === "add" ? "on" : ""} onClick={() => store.set({ objectMode: "add" })}>Add</button><button className={objectMode === "subtract" ? "on" : ""} onClick={() => store.set({ objectMode: "subtract" })}>Subtract</button></div>
            <p className="hint">Click an object to isolate it. Hold Alt to subtract.</p>
          </>
        )}
        {tool === "pan" && <p className="hint">Drag to move the view. Scroll to zoom. Middle mouse always pans.</p>}
      </div>

      <div className="sec">
        <h3>Matte</h3>
        <Slider label="Edge refine" value={ms.edgeRefine} min={0} max={1} step={0.05} fmt={(v) => `${Math.round(v * 100)}%`} onChange={(v) => setMatte({ edgeRefine: v })} />
        <Slider label="Detail" value={ms.detail} min={0} max={1} step={0.05} fmt={(v) => `${Math.round(v * 100)}%`} onChange={(v) => setMatte({ detail: v })} />
        <Slider label="Smooth" value={ms.smooth} min={0} max={1} step={0.05} fmt={(v) => `${Math.round(v * 100)}%`} onChange={(v) => setMatte({ smooth: v })} />
        <Slider label="Feather" value={ms.feather} min={0} max={30} step={0.5} fmt={(v) => `${v}px`} onChange={(v) => setMatte({ feather: v })} />
        <Slider label="Expand / contract" value={ms.shift} min={-30} max={30} step={1} fmt={(v) => `${v > 0 ? "+" : ""}${v}`} onChange={(v) => setMatte({ shift: v })} />
        <Slider label="Stabilise" value={ms.temporal} min={0} max={1} step={0.05} fmt={(v) => `${Math.round(v * 100)}%`} onChange={(v) => setMatte({ temporal: v })} />
        <label className="chk"><input type="checkbox" checked={ms.decontaminate} onChange={(e) => setMatte({ decontaminate: e.target.checked })} />Remove background colour from edges</label>
        <p className="hint">Matte settings never change the stored masks. Stabilise is applied on export.</p>
      </div>

      <div className="sec">
        <h3>Current frame</h3>
        <div className="btns">
          <button onClick={() => void propagate("forward")}>Propagate forward</button>
          <button onClick={() => void propagate("backward")}>Propagate backward</button>
          <button className="primary" onClick={() => void propagate("both")}>Propagate both</button>
          <button onClick={() => void propagate("forward", true)} title="Re-track every frame after this one, using later corrections as anchors">Recalculate from here</button>
          <button onClick={() => void clearFrameMask()}>Clear mask</button>
        </div>
        <p className="hint">Propagation stops at the next frame you corrected yourself.</p>
      </div>

      <div className="sec">
        <h3>Engine</h3>
        <select value={project.engine === "auto" ? "auto" : project.engine} onChange={(e) => void setEngine(e.target.value)} aria-label="Segmentation engine">
          <option value="auto">Auto (best available)</option>
          {engines.map((e) => <option key={e.name} value={e.name} disabled={!e.available}>{e.name}{e.available ? "" : " (not installed)"}</option>)}
        </select>
        <p className="hint">{engines.find((e) => e.name === engine)?.description}</p>
        {engines.filter((e) => !e.available && e.reason).map((e) => <p className="hint" key={e.name}>{e.name}: {e.reason}</p>)}
      </div>
    </div>
  );
}
