import { useState } from "react";
import { pickExportPath, queueExport } from "../actions";
import { store, useApp } from "../store";
import type { ExportSettings } from "../types";

const KINDS = [
  { id: "mp4", label: "MP4" },
  { id: "mov-alpha", label: "MOV with alpha" },
  { id: "after-effects", label: "After Effects" },
] as const;

export function ExportDialog() {
  const project = useApp((s) => s.project);
  const kind = useApp((s) => s.exportKind);
  const [s, setS] = useState<ExportSettings>(() => store.get().project!.exportSettings);
  const [out, setOut] = useState("");
  const [bgMode, setBgMode] = useState<"white" | "black" | "custom">(
    s.background.toUpperCase() === "#FFFFFF" ? "white" : s.background === "#000000" ? "black" : "custom",
  );
  if (!project) return null;
  const set = (p: Partial<ExportSettings>) => setS((x) => ({ ...x, ...p }));
  const m = project.project;
  const w = Math.round((m.width * s.scale) / 2) * 2, h = Math.round((m.height * s.scale) / 2) * 2;
  const wa = project.workArea;
  const close = () => store.set({ dialog: null });

  const submit = async () => {
    if (!out) return;
    const bg = bgMode === "white" ? "#FFFFFF" : bgMode === "black" ? "#000000" : s.background;
    if (await queueExport(kind, out, { ...s, background: bg })) close();
  };

  return (
    <div className="scrim" onMouseDown={(e) => e.target === e.currentTarget && close()}>
      <div className="dialog" role="dialog" aria-label="Export">
        <h2>Export</h2>
        <div className="tabs">
          {KINDS.map((k) => <button key={k.id} className={kind === k.id ? "on" : ""} onClick={() => { store.set({ exportKind: k.id }); setOut(""); }}>{k.label}</button>)}
        </div>

        <div className="field"><label>Range</label>
          <div className="inline">
            <label><input type="radio" checked={s.range === "full"} onChange={() => set({ range: "full" })} /> Whole clip</label>
            <label><input type="radio" checked={s.range === "work"} onChange={() => set({ range: "work" })} /> Work area ({wa.in}–{wa.out})</label>
          </div>
        </div>
        <div className="field"><label>Resolution</label>
          <div className="inline">
            <select value={s.scale} onChange={(e) => set({ scale: Number(e.target.value) })}>
              {[1, 0.75, 0.5, 0.25].map((v) => <option key={v} value={v}>{Math.round(v * 100)}% of source</option>)}
            </select>
            <span className="num" style={{ color: "var(--dim)" }}>{w}×{h} · {m.fps.toFixed(3)} fps</span>
          </div>
        </div>

        {kind === "mp4" && (
          <>
            <div className="field"><label>Background</label>
              <div className="inline">
                {(["white", "black", "custom"] as const).map((b) => <label key={b}><input type="radio" checked={bgMode === b} onChange={() => setBgMode(b)} /> {b[0].toUpperCase() + b.slice(1)}</label>)}
                {bgMode === "custom" && (<><input type="color" className="swatch" value={s.background} onChange={(e) => set({ background: e.target.value.toUpperCase() })} aria-label="Background colour" /><input type="text" size={8} value={s.background} onChange={(e) => set({ background: e.target.value })} aria-label="Background hex" /></>)}
              </div>
            </div>
            <div className="field"><label>Codec</label>
              <select value={s.codec} onChange={(e) => set({ codec: e.target.value })}><option value="h264">H.264 (most compatible)</option><option value="h265">H.265 / HEVC (smaller)</option></select>
            </div>
            <div className="field"><label>Encoder</label>
              <select value={s.hwEncode ?? "auto"} onChange={(e) => set({ hwEncode: e.target.value as ExportSettings["hwEncode"] })}>
                <option value="auto">Auto (NVIDIA GPU if available, else CPU)</option>
                <option value="cpu">CPU only (most compatible)</option>
              </select>
            </div>
            <div className="field"><label>Quality</label>
              <div className="inline"><input type="range" min={12} max={32} value={s.crf} onChange={(e) => set({ crf: Number(e.target.value) })} /><span className="num">CRF {s.crf}</span><span style={{ color: "var(--dim)" }}>lower = better</span></div>
            </div>
            <p className="hint">MP4 cannot carry transparency, so the subject is composited onto the background. Audio is kept.</p>
          </>
        )}
        {(kind === "mov-alpha" || kind === "after-effects") && (
          <div className="field"><label>Alpha codec</label>
            <select value={s.movCodec} onChange={(e) => set({ movCodec: e.target.value })}>
              <option value="prores4444">ProRes 4444 (best for editing)</option>
              <option value="qtrle">QuickTime Animation (lossless, large)</option>
              <option value="png">PNG in MOV (lossless)</option>
            </select>
          </div>
        )}
        {kind === "after-effects" && (
          <>
            <label className="chk"><input type="checkbox" checked={s.copySource} onChange={(e) => set({ copySource: e.target.checked })} />Copy the source video into the package</label>
            <p className="hint">Creates a folder with the alpha movie, the footage and comp.jsx. Run the script in After Effects to build the composition. This is not an .aep file.</p>
          </>
        )}

        <div className="field"><label>{kind === "after-effects" ? "Folder" : "Save to"}</label>
          <div className="inline" style={{ flexWrap: "nowrap" }}>
            <input type="text" readOnly value={out} placeholder="Choose where to save" style={{ flex: 1, minWidth: 0 }} />
            <button onClick={async () => setOut((await pickExportPath(kind)) || out)}>Browse…</button>
          </div>
        </div>
        <div className="footer"><button onClick={close}>Cancel</button><button className="primary" disabled={!out} onClick={() => void submit()}>Add to queue</button></div>
      </div>
    </div>
  );
}
