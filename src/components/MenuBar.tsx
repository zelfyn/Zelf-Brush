import { useEffect, useRef, useState } from "react";
import { browseVideo, newProject, openProject, redo, saveProject, saveProjectAs, trimActive, undo } from "../actions";
import { store, useApp } from "../store";
import type { ViewMode } from "../types";

interface Item { label: string; kbd?: string; run?: () => void; disabled?: boolean; sep?: boolean }

export function MenuBar() {
  const project = useApp((s) => s.project);
  const tl = useApp((s) => s.timeline);
  const [open, setOpen] = useState<string | null>(null);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const h = (e: MouseEvent) => { if (!ref.current?.contains(e.target as Node)) setOpen(null); };
    window.addEventListener("mousedown", h);
    return () => window.removeEventListener("mousedown", h);
  }, []);
  const exp = (k: "mp4" | "mov-alpha" | "after-effects" | "depth") => () => store.set({ dialog: "export", exportKind: k });
  const view = (m: ViewMode) => () => store.set({ viewMode: m });
  const menus: Record<string, Item[]> = {
    File: [
      { label: "New project from video…", run: () => void newProject().then(() => browseVideo(false)) },
      { label: "Add video as layer…", run: () => void browseVideo(true), disabled: !project },
      { label: "Open project…", kbd: "Ctrl+O", run: () => void openProject() },
      { sep: true, label: "" },
      { label: "Save", kbd: "Ctrl+S", run: () => void saveProject(), disabled: !project },
      { label: "Save as…", kbd: "Ctrl+Shift+S", run: () => void saveProjectAs(), disabled: !project },
    ],
    Edit: [
      { label: "Undo", kbd: "Ctrl+Z", run: () => void undo(), disabled: !tl?.can_undo },
      { label: "Redo", kbd: "Ctrl+Shift+Z", run: () => void redo(), disabled: !tl?.can_redo },
      { sep: true, label: "" },
      { label: "Trim layer start to playhead", kbd: "Alt+[", run: () => void trimActive("in"), disabled: !project },
      { label: "Trim layer end to playhead", kbd: "Alt+]", run: () => void trimActive("out"), disabled: !project },
    ],
    View: [
      { label: "Original", kbd: "O", run: view("original") },
      { label: "Mask", kbd: "M", run: view("mask") },
      { label: "Transparent", kbd: "T", run: view("transparent") },
      { label: "Overlay", kbd: "V", run: view("overlay") },
      { label: "Problem areas", kbd: "E", run: view("edge") },
      { label: "Depth map", kbd: "D", run: view("depth") },
      { label: "Composite (all layers)", kbd: "C", run: view("composite") },
    ],
    Export: [
      { label: "MP4 with background…", run: exp("mp4"), disabled: !project },
      { label: "MOV with alpha…", run: exp("mov-alpha"), disabled: !project },
      { label: "After Effects package…", run: exp("after-effects"), disabled: !project },
      { sep: true, label: "" },
      { label: "Depth map…", run: exp("depth"), disabled: !project },
    ],
  };
  return (
    <div className="menubar" ref={ref}>
      <span className="brand">Zelf-Brush</span>
      {Object.entries(menus).map(([name, items]) => (
        <div className="menu" key={name}>
          <button className={open === name ? "open" : ""} onClick={() => setOpen(open === name ? null : name)} onMouseEnter={() => open && setOpen(name)}>{name}</button>
          {open === name && (
            <div className="menu-pop" role="menu">
              {items.map((it, i) => it.sep ? <hr key={i} /> : (
                <button key={it.label} disabled={it.disabled} role="menuitem" onClick={() => { setOpen(null); it.run?.(); }}>
                  <span>{it.label}</span><span className="kbd">{it.kbd}</span>
                </button>
              ))}
            </div>
          )}
        </div>
      ))}
      <span className="spacer" />
      {project && <span style={{ color: "var(--dim)" }}>{project.project.name}</span>}
    </div>
  );
}
