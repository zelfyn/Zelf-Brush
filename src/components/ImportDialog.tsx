import { useEffect, useState } from "react";
import * as api from "../api";
import { cancelImport, confirmImport } from "../actions";
import { store, useApp } from "../store";

function tc(frame: number, fps: number): string {
  const total = Math.floor(frame / fps);
  const ff = Math.floor(frame - total * fps);
  return `${String(Math.floor(total / 60)).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}:${String(ff).padStart(2, "0")}`;
}

/** Shown after choosing or dropping a video, BEFORE it reaches the timeline: preview, cut, name and placement. */
export function ImportDialog() {
  const d = useApp((s) => s.importDraft);
  const hasProject = useApp((s) => s.comp !== null);
  const [pos, setPos] = useState(0);
  const [shown, setShown] = useState(0);
  const [tin, setTin] = useState(0);
  const [tout, setTout] = useState(0);
  const [name, setName] = useState("");
  const [asLayer, setAsLayer] = useState(false);
  const [start, setStart] = useState(0);
  const [auto, setAuto] = useState(true);
  const [model, setModel] = useState("auto");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!d) return;
    const n = d.info.frame_count;
    setPos(0); setShown(0); setTin(0); setTout(n - 1);
    setName(d.info.filename.replace(/\.[^.]+$/, ""));
    const lay = d.preferLayer && !!store.get().comp;
    setAsLayer(lay);
    setStart(store.get().frame);
    setAuto(!lay);
    setModel("auto");
  }, [d]);

  useEffect(() => {
    const t = window.setTimeout(() => setShown(pos), 90);   // don't decode a frame for every pixel of slider travel
    return () => window.clearTimeout(t);
  }, [pos]);

  useEffect(() => setLoading(true), [shown, d]);

  if (!d) return null;
  const { info } = d;
  const n = info.frame_count;
  const keep = tout - tin + 1;
  const fps = info.fps;

  const onIn = (v: number) => { const x = Math.min(v, tout); setTin(x); setPos(x); };
  const onOut = (v: number) => { const x = Math.max(v, tin); setTout(x); setPos(x); };
  const submit = () => void confirmImport({ name: name.trim() || info.filename, trimIn: tin, trimOut: tout, asLayer: asLayer && hasProject, start, autoRoto: auto, aiModel: model });

  return (
    <div className="scrim" onMouseDown={(e) => e.target === e.currentTarget && cancelImport()}
      onKeyDown={(e) => { if (e.key === "Escape") cancelImport(); if (e.key === "Enter" && (e.target as HTMLElement).tagName !== "BUTTON") submit(); }}>
      <div className="dialog wide" role="dialog" aria-label="Import video">
        <h2>Import video</h2>
        <div className="hint" style={{ margin: "-4px 0 10px" }}>
          {info.filename} · {info.width}×{info.height} · {info.fps.toFixed(3)} fps · {n} frames · {info.codec}{info.has_audio ? " · audio" : ""}
        </div>

        <div className="previewbox">
          <img src={api.url("/video/frame", { path: d.path, index: shown, w: 640 })} alt="Video preview" onLoad={() => setLoading(false)} onError={() => setLoading(false)} />
          {loading && <span className="previewbusy">…</span>}
        </div>

        <div className="field slim"><label>Preview</label>
          <div className="inline" style={{ flexWrap: "nowrap" }}>
            <input type="range" min={0} max={n - 1} value={pos} onChange={(e) => setPos(Number(e.target.value))} aria-label="Preview position" style={{ flex: 1 }} />
            <span className="num tcl">{tc(pos, fps)} · {pos}</span>
          </div>
        </div>

        <div className="field slim"><label>Start (In)</label>
          <div className="inline" style={{ flexWrap: "nowrap" }}>
            <input type="range" min={0} max={n - 1} value={tin} onChange={(e) => onIn(Number(e.target.value))} aria-label="In point" style={{ flex: 1 }} />
            <span className="num tcl">{tc(tin, fps)} · {tin}</span>
            <button onClick={() => onIn(pos)} title="Use the preview position as the start">Set here</button>
          </div>
        </div>
        <div className="field slim"><label>End (Out)</label>
          <div className="inline" style={{ flexWrap: "nowrap" }}>
            <input type="range" min={0} max={n - 1} value={tout} onChange={(e) => onOut(Number(e.target.value))} aria-label="Out point" style={{ flex: 1 }} />
            <span className="num tcl">{tc(tout, fps)} · {tout}</span>
            <button onClick={() => onOut(pos)} title="Use the preview position as the end">Set here</button>
          </div>
        </div>
        <div className="hint keepline"><b>{keep}</b> of {n} frames kept · {(keep / fps).toFixed(2)} s{keep < n ? " · the rest is never processed" : ""}</div>

        <div className="field"><label>Layer name</label>
          <input type="text" value={name} onChange={(e) => setName(e.target.value)} aria-label="Layer name" />
        </div>
        {hasProject && (
          <div className="field"><label>Place</label>
            <div className="inline">
              <label><input type="radio" checked={asLayer} onChange={() => { setAsLayer(true); setAuto(false); }} /> New layer on top</label>
              <label><input type="radio" checked={!asLayer} onChange={() => { setAsLayer(false); setAuto(true); }} /> Replace the project</label>
            </div>
          </div>
        )}
        {asLayer && hasProject && (
          <div className="field"><label>Starts at frame</label>
            <input type="number" min={0} value={start} onChange={(e) => setStart(Math.max(0, Number(e.target.value)))} aria-label="Start frame" style={{ width: 110 }} />
          </div>
        )}
        <div className="field"><label>AI model</label>
          <select value={model} onChange={(e) => setModel(e.target.value)} aria-label="AI model">
            <option value="auto">Auto (detects anime)</option>
            <option value="general">General (best edges)</option>
            <option value="anime">Anime / illustration</option>
            <option value="fast">Fast (lower quality)</option>
          </select>
        </div>
        <label className="chk"><input type="checkbox" checked={auto} onChange={(e) => setAuto(e.target.checked)} />Detect the subject and start roto after importing</label>

        <div className="footer">
          <button onClick={cancelImport}>Cancel</button>
          <button className="primary" onClick={submit}>Add to timeline</button>
        </div>
      </div>
    </div>
  );
}
