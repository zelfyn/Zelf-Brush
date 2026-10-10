import { store, useApp } from "../store";
import type { Tool } from "../types";

const ICONS: Record<Tool, JSX.Element> = {
  brush: <svg viewBox="0 0 24 24"><path d="M14 4l6 6-8 8H6v-6z" /><path d="M4 21c3 0 3-3 6-3" /></svg>,
  eraser: <svg viewBox="0 0 24 24"><path d="M8 20h12" /><path d="M4.5 14.5l9-9a2 2 0 013 0l3 3a2 2 0 010 3L11 20H7z" /></svg>,
  lasso: <svg viewBox="0 0 24 24"><path d="M12 4c5 0 9 2.3 9 5.5S17 15 12 15 3 12.8 3 9.5 7 4 12 4z" /><path d="M8 15c-1 2 0 4 3 5" /></svg>,
  object: <svg viewBox="0 0 24 24"><path d="M5 5h3M16 5h3M5 19h3M16 19h3M5 5v3M5 16v3M19 5v3M19 16v3" /><circle cx="12" cy="12" r="3" /></svg>,
  pan: <svg viewBox="0 0 24 24"><path d="M9 11V5a1.5 1.5 0 013 0v5M12 10V4a1.5 1.5 0 013 0v7M15 11V6a1.5 1.5 0 013 0v8a7 7 0 01-7 7c-3 0-4.5-1.5-6-4l-2-3a1.5 1.5 0 012.5-1.5L9 15" /></svg>,
};
const TOOLS: { id: Tool; label: string; key: string }[] = [
  { id: "brush", label: "Brush (keep)", key: "B" },
  { id: "eraser", label: "Eraser (remove)", key: "E" },
  { id: "lasso", label: "Lasso", key: "L" },
  { id: "object", label: "Object selection", key: "W" },
  { id: "pan", label: "Pan", key: "H" },
];

export function ToolRail() {
  const tool = useApp((s) => s.tool);
  return (
    <div className="tools" role="toolbar" aria-label="Tools">
      {TOOLS.map((t) => (
        <button key={t.id} className={`tool${tool === t.id ? " active" : ""}`} title={`${t.label} (${t.key})`} aria-label={t.label} aria-pressed={tool === t.id} onClick={() => store.set({ tool: t.id })}>
          {ICONS[t.id]}
        </button>
      ))}
    </div>
  );
}
