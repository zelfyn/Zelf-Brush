import { useEffect } from "react";
import { openLog } from "./api";
import { boot, browseVideo, importFile, openProject, redo, saveProject, saveProjectAs, setCompWorkArea, setFrame, startPlaybackLoop, stepFrame, togglePlay, trimActive, undo } from "./actions";
import { ExportDialog } from "./components/ExportDialog";
import { MenuBar } from "./components/MenuBar";
import { Properties } from "./components/Properties";
import { QueuePanel } from "./components/QueuePanel";
import { StatusBar } from "./components/StatusBar";
import { Timeline } from "./components/Timeline";
import { ToolRail } from "./components/ToolRail";
import { Viewport } from "./components/Viewport";
import { store, useApp } from "./store";
import type { Tool, ViewMode } from "./types";

const TOOL_KEYS: Record<string, Tool> = { b: "brush", e: "eraser", l: "lasso", w: "object", h: "pan" };
const VIEW_KEYS: Record<string, ViewMode> = { o: "original", m: "mask", t: "transparent", v: "overlay", d: "depth", c: "composite" };

function onKey(e: KeyboardEvent): void {
  const el = e.target as HTMLElement;
  if (el.tagName === "INPUT" && (el as HTMLInputElement).type === "text") return;
  if (el.tagName === "SELECT") return;
  const k = e.key.toLowerCase();
  if (e.ctrlKey || e.metaKey) {
    if (k === "z") { e.preventDefault(); void (e.shiftKey ? redo() : undo()); }
    else if (k === "y") { e.preventDefault(); void redo(); }
    else if (k === "s") { e.preventDefault(); void (e.shiftKey ? saveProjectAs() : saveProject()); }
    else if (k === "o") { e.preventDefault(); void openProject(); }
    return;
  }
  const s = store.get();
  if (e.altKey && (e.key === "[" || e.key === "]")) { e.preventDefault(); void trimActive(e.key === "[" ? "in" : "out"); return; }
  if (k === " ") { e.preventDefault(); togglePlay(); }
  else if (e.key === "ArrowLeft") { e.preventDefault(); stepFrame(e.shiftKey ? -10 : -1); }
  else if (e.key === "ArrowRight") { e.preventDefault(); stepFrame(e.shiftKey ? 10 : 1); }
  else if (e.key === "Home") setFrame(0);
  else if (e.key === "End") setFrame(1e9);
  else if (k === "i" && s.comp) void setCompWorkArea(s.frame, Math.max(s.frame, s.comp.workArea.out));
  else if (k === "p" && s.comp) void setCompWorkArea(Math.min(s.frame, s.comp.workArea.in), s.frame);
  else if (k === "[") store.set({ brush: { ...s.brush, size: Math.max(2, Math.round(s.brush.size * 0.85)) } });
  else if (k === "]") store.set({ brush: { ...s.brush, size: Math.min(300, Math.round(s.brush.size / 0.85) + 1) } });
  else if (k === "e" && s.tool === "eraser") store.set({ viewMode: "edge" });
  else if (TOOL_KEYS[k]) store.set({ tool: TOOL_KEYS[k] });
  else if (VIEW_KEYS[k]) store.set({ viewMode: VIEW_KEYS[k] });
}

export function App() {
  const ready = useApp((s) => s.ready);
  const project = useApp((s) => s.project);
  const dialog = useApp((s) => s.dialog);
  const queueOpen = useApp((s) => s.queueOpen);
  const error = useApp((s) => s.error);
  const toast = useApp((s) => s.toast);
  const rows = useApp((s) => s.comp?.layers.length ?? 1);

  useEffect(() => {
    void boot();
    startPlaybackLoop();
    window.addEventListener("keydown", onKey);
    const stopBrowserDrop = (e: DragEvent) => e.preventDefault();
    window.addEventListener("dragover", stopBrowserDrop);
    window.addEventListener("drop", stopBrowserDrop);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <div className="app" style={{ ["--tl-h" as string]: `${Math.min(340, Math.max(170, 34 + 24 + rows * 28 + 14))}px` }}>
      <MenuBar />
      <ToolRail />
      <Viewport />
      <Properties />
      <Timeline />
      <StatusBar />
      {dialog === "export" && project && <ExportDialog />}
      {queueOpen && <QueuePanel />}
      {toast && <div className="toast" role="status">{toast}</div>}
      {!ready && !error && <div className="scrim"><div className="dialog"><h2>Starting Zelf-Brush…</h2><p className="hint">Waiting for the processing backend.</p></div></div>}
      {error && (
        <div className="scrim">
          <div className="dialog" role="alertdialog" aria-label="Error">
            <h2>{error.message}</h2>
            {(error.stage || error.exit_code != null || error.detail) && (
              <div className="hint" style={{ userSelect: "text", margin: "0 0 10px" }}>
                {error.stage && <div>Stage: <b>{error.stage}</b>{error.exit_code != null ? ` · FFmpeg exit code ${error.exit_code}` : ""}</div>}
                {error.detail && <div style={{ wordBreak: "break-word" }}>{error.detail}</div>}
              </div>
            )}
            {error.hints.length > 0 && <><span style={{ color: "var(--dim)" }}>Try:</span><ul>{error.hints.map((h) => <li key={h}>{h}</li>)}</ul></>}
            <div className="footer">
              {!ready && <button onClick={() => window.location.reload()}>Retry</button>}
              {error.log && <button onClick={() => void openLog(error.log!).catch(() => undefined)}>Open Log</button>}
              {(error.message.includes("Object detection") || error.stage === "object-selection") && <button onClick={() => { store.set({ error: null, tool: "lasso" }); }}>Use Lasso</button>}
              <button className="primary" onClick={() => store.set({ error: null })}>OK</button>
            </div>
          </div>
        </div>
      )}
      <input type="file" id="pick-file" hidden accept=".mp4,.mkv,.mov" onChange={(e) => { const f = e.target.files?.[0]; if (f) void importFile(f); }} />
      <span hidden>{String(typeof browseVideo)}</span>
    </div>
  );
}
