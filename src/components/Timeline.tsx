import { useEffect, useMemo, useRef, useState } from "react";
import { compFrameOf, moveLayer, patchLayer, selectLayer, setCompWorkArea, setFrame, stepFrame, togglePlay } from "../actions";
import { useApp } from "../store";
import type { Layer } from "../types";

function timecode(f: number, fps: number): string {
  const total = Math.floor(f / fps);
  const ff = Math.floor(f - total * fps);
  return `${String(Math.floor(total / 60)).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}:${String(ff).padStart(2, "0")}`;
}

const PAD = 10;
const RULER = 24;
export const ROW = 28;

type Mode = "none" | "scrub" | "move" | "trimIn" | "trimOut" | "waIn" | "waOut";
interface Geo { start: number; trimIn: number; trimOut: number }

export function Timeline() {
  const project = useApp((s) => s.project);
  const comp = useApp((s) => s.comp);
  const tl = useApp((s) => s.timeline);
  const frame = useApp((s) => s.frame);
  const playing = useApp((s) => s.playing);
  const cvs = useRef<HTMLCanvasElement>(null);
  const cell = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(600);
  const [live, setLive] = useState<Record<string, Geo>>({});   // layer geometry while a drag is in progress
  const drag = useRef<{ mode: Mode; id: string; x0: number; orig: Geo; cur: Geo } | null>(null);

  const layers = comp?.layers ?? [];
  const dur = Math.max(1, comp?.duration ?? 1);
  const fps = comp?.fps ?? 24;
  const wa = comp?.workArea ?? { in: 0, out: 0 };
  const contentH = RULER + Math.max(1, layers.length) * ROW + 6;
  const px = (width - PAD * 2) / dur;                               // pixels per comp frame
  const xOf = (f: number) => PAD + f * px;

  useEffect(() => {
    const el = cell.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setWidth(el.clientWidth));
    ro.observe(el);
    setWidth(el.clientWidth);
    return () => ro.disconnect();
  }, []);

  const geoOf = (L: Layer): Geo => live[L.id] ?? { start: L.start, trimIn: L.trimIn, trimOut: L.trimOut };
  const spanOf = (L: Layer): [number, number] => {
    const g = geoOf(L);
    const len = Math.max(1, Math.round(((g.trimOut - g.trimIn + 1) * (comp?.fps ?? 24)) / L.fps));
    return [g.start, g.start + len];                                 // [first frame, one past the last]
  };

  const active = useMemo(() => layers.find((l) => l.id === comp?.active) ?? null, [layers, comp?.active]);

  useEffect(() => {
    const cv = cvs.current;
    if (!cv || !comp) return;
    const dpr = window.devicePixelRatio || 1;
    cv.width = Math.round(width * dpr);
    cv.height = Math.round(contentH * dpr);
    const g = cv.getContext("2d")!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, width, contentH);
    const css = getComputedStyle(document.documentElement);
    const c = (k: string) => css.getPropertyValue(k).trim();
    g.font = "11px " + c("--font");
    g.textBaseline = "middle";

    // ruler
    g.fillStyle = c("--dim");
    const pxPerSec = px * fps;
    const step = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600].find((s) => s * pxPerSec >= 70) ?? 1200;
    for (let t = 0; t * fps < dur; t += step) {
      const x = xOf(t * fps);
      g.fillRect(x, RULER - 7, 1, 7);
      g.fillText(timecode(t * fps, fps).slice(0, 5), x + 3, 9);
    }
    // work area (comp)
    g.fillStyle = "rgba(110,168,254,.22)";
    g.fillRect(xOf(wa.in), 0, Math.max(2, xOf(wa.out + 1) - xOf(wa.in)), RULER);
    g.fillStyle = c("--accent");
    g.fillRect(xOf(wa.in) - 1, 0, 3, RULER);
    g.fillRect(xOf(wa.out + 1) - 2, 0, 3, RULER);

    // layer rows
    layers.forEach((L, i) => {
      const y = RULER + i * ROW;
      g.fillStyle = i % 2 ? "rgba(255,255,255,.015)" : "rgba(0,0,0,.12)";
      g.fillRect(0, y, width, ROW);
      const [a, b] = spanOf(L);
      const x0 = xOf(a), x1 = xOf(b);
      const isActive = L.id === comp.active;
      g.globalAlpha = L.visible ? 1 : 0.4;
      g.fillStyle = isActive ? "#2c4a73" : "#2a3140";
      g.strokeStyle = isActive ? c("--accent") : "#3b4456";
      g.lineWidth = 1;
      g.beginPath();
      g.roundRect(x0, y + 3, Math.max(4, x1 - x0), ROW - 6, 4);
      g.fill();
      g.stroke();
      // matte coverage
      g.fillStyle = "#3f8f65";
      if (isActive && tl) {
        for (const [m0, m1] of tl.masked) {
          const lo = compFrameOf(L, Math.max(m0, L.trimIn), comp), hi = compFrameOf(L, Math.min(m1, L.trimOut), comp) + 1;
          if (hi > lo) g.fillRect(xOf(lo), y + ROW - 8, Math.max(1, xOf(hi) - xOf(lo)), 3);
        }
      } else if (L.hasMask) {
        g.fillRect(x0 + 2, y + ROW - 8, Math.max(1, x1 - x0 - 4), 3);
      }
      // trim grips
      g.fillStyle = isActive ? c("--accent") : "#566179";
      g.fillRect(x0, y + 5, 3, ROW - 10);
      g.fillRect(x1 - 3, y + 5, 3, ROW - 10);
      // name
      g.save();
      g.beginPath();
      g.rect(x0 + 6, y, Math.max(0, x1 - x0 - 12), ROW);
      g.clip();
      g.fillStyle = c("--text");
      g.fillText(L.name, x0 + 8, y + 11);
      g.restore();
      g.globalAlpha = 1;
      // active layer: confidence, warnings, keyframes
      if (isActive && tl) {
        for (const [k, v] of Object.entries(tl.confidence)) {
          const cf = compFrameOf(L, Number(k), comp);
          if (cf < a || cf >= b) continue;
          g.fillStyle = v >= 0.75 ? "#5bc489" : v >= 0.55 ? "#f0b429" : "#ff6b4a";
          g.fillRect(xOf(cf), y + ROW - 5, Math.max(1, px), 2);
        }
        g.fillStyle = c("--warn");
        for (const f of tl.warnings) {
          const cf = compFrameOf(L, f, comp);
          if (cf >= a && cf < b) g.fillRect(xOf(cf), y + 4, Math.max(1, px), 5);
        }
        for (const k of tl.keyframes) {
          const cf = compFrameOf(L, k.frame, comp);
          if (cf < a || cf >= b) continue;
          const x = xOf(cf) + px / 2, yy = y + 14;
          g.fillStyle = k.kind === "user" ? c("--user") : c("--ai");
          g.beginPath();
          g.moveTo(x, yy - 5); g.lineTo(x + 5, yy); g.lineTo(x, yy + 5); g.lineTo(x - 5, yy);
          g.closePath();
          g.fill();
        }
      }
    });

    // playhead
    const hx = xOf(frame) + px / 2;
    g.fillStyle = "#fff";
    g.fillRect(hx - 1, RULER - 8, 2, contentH - RULER + 8);
    g.beginPath();
    g.moveTo(hx - 6, RULER - 10); g.lineTo(hx + 6, RULER - 10); g.lineTo(hx, RULER - 2);
    g.closePath();
    g.fill();
  }, [width, contentH, comp, layers, tl, frame, live, wa.in, wa.out, dur, fps, px]);

  const posOf = (e: React.PointerEvent) => {
    const r = cvs.current!.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  };
  const frameAt = (x: number) => Math.max(0, Math.min(dur - 1, Math.floor((x - PAD) / px)));

  const hit = (x: number, y: number): { mode: Mode; layer?: Layer } => {
    if (y < RULER) {
      if (Math.abs(x - xOf(wa.in)) < 7) return { mode: "waIn" };
      if (Math.abs(x - xOf(wa.out + 1)) < 7) return { mode: "waOut" };
      return { mode: "scrub" };
    }
    const L = layers[Math.floor((y - RULER) / ROW)];
    if (!L) return { mode: "scrub" };
    const [a, b] = spanOf(L);
    const x0 = xOf(a), x1 = xOf(b);
    if (x < x0 - 6 || x > x1 + 6) return { mode: "scrub", layer: L };
    if (Math.abs(x - x0) < 7) return { mode: "trimIn", layer: L };
    if (Math.abs(x - x1) < 7) return { mode: "trimOut", layer: L };
    return { mode: "move", layer: L };
  };

  const down = (e: React.PointerEvent) => {
    if (!comp) return;
    cvs.current!.setPointerCapture(e.pointerId);
    const { x, y } = posOf(e);
    const h = hit(x, y);
    if (h.layer && h.layer.id !== comp.active) void selectLayer(h.layer.id);
    if (h.mode === "scrub") setFrame(frameAt(x));
    const g0 = h.layer ? { start: h.layer.start, trimIn: h.layer.trimIn, trimOut: h.layer.trimOut } : { start: 0, trimIn: 0, trimOut: 0 };
    drag.current = { mode: h.mode, id: h.layer?.id ?? "", x0: x, orig: g0, cur: g0 };
  };

  const move = (e: React.PointerEvent) => {
    const { x, y } = posOf(e);
    const d = drag.current;
    if (!d || d.mode === "none") {
      const h = hit(x, y);
      cvs.current!.style.cursor = h.mode === "trimIn" || h.mode === "trimOut" || h.mode === "waIn" || h.mode === "waOut" ? "ew-resize" : h.mode === "move" ? "grab" : "default";
      return;
    }
    const df = Math.round((x - d.x0) / px);                             // comp frames moved
    const L = layers.find((l) => l.id === d.id);
    if (d.mode === "scrub") return setFrame(frameAt(x));
    if (d.mode === "waIn") return void setCompWorkArea(Math.min(frameAt(x), wa.out), wa.out);
    if (d.mode === "waOut") return void setCompWorkArea(wa.in, Math.max(frameAt(x), wa.in));
    if (!L) return;
    const ratio = L.fps / fps;                                           // source frames per comp frame
    let cur: Geo = d.orig;
    if (d.mode === "move") cur = { ...d.orig, start: Math.max(0, d.orig.start + df) };
    else if (d.mode === "trimIn") {
      const tin = Math.max(0, Math.min(d.orig.trimOut, d.orig.trimIn + Math.round(df * ratio)));
      cur = { start: layers.length === 1 ? 0 : Math.max(0, d.orig.start + Math.round((tin - d.orig.trimIn) / ratio)), trimIn: tin, trimOut: d.orig.trimOut };
    } else if (d.mode === "trimOut") {
      cur = { ...d.orig, trimOut: Math.max(d.orig.trimIn, Math.min(L.frames - 1, d.orig.trimOut + Math.round(df * ratio))) };
    }
    d.cur = cur;
    setLive({ [L.id]: cur });
  };

  const up = () => {
    const d = drag.current;
    drag.current = null;
    if (!d || !d.id || d.mode === "scrub" || d.mode === "none") return;
    setLive({});
    const { orig, cur } = d;
    if (cur.start === orig.start && cur.trimIn === orig.trimIn && cur.trimOut === orig.trimOut) return;
    void patchLayer(d.id, d.mode === "move" ? { start: cur.start } : d.mode === "trimIn" ? { trimIn: cur.trimIn, start: cur.start } : { trimOut: cur.trimOut });
  };

  const jump = (dir: 1 | -1) => {
    if (!comp || !active || !tl) return;
    const marks = [...new Set(tl.keyframes.map((k) => compFrameOf(active, k.frame, comp)).concat(tl.warnings.map((f) => compFrameOf(active, f, comp))))].sort((a, b) => a - b);
    const t = dir > 0 ? marks.find((f) => f > frame) : [...marks].reverse().find((f) => f < frame);
    if (t !== undefined) setFrame(t);
  };

  return (
    <div className="timeline">
      <div className="transport">
        <button onClick={() => jump(-1)} title="Previous marker" disabled={!project}>⏮</button>
        <button onClick={() => stepFrame(-1)} title="Previous frame (←)" disabled={!project}>◀</button>
        <button className="primary" style={{ width: 44 }} onClick={togglePlay} title="Play / pause (Space)" disabled={!project}>{playing ? "❚❚" : "▶"}</button>
        <button onClick={() => stepFrame(1)} title="Next frame (→)" disabled={!project}>▶</button>
        <button onClick={() => jump(1)} title="Next marker" disabled={!project}>⏭</button>
        <span className="tc">{timecode(frame, fps)}</span>
        <span className="dim num">frame {frame} / {Math.max(0, dur - 1)}</span>
        <span className="spacer" />
        <span className="legend">
          <span><i style={{ background: "var(--ai)" }} />AI</span>
          <span><i style={{ background: "var(--user)" }} />Correction</span>
          <span><i style={{ background: "var(--warn)" }} />⚠ Correction recommended</span>
        </span>
      </div>
      <div className="tlscroll">
        <div className="tllayers" style={{ height: contentH }}>
          <div style={{ height: RULER }} className="tlhead">Layers</div>
          {layers.map((L, i) => (
            <div key={L.id} className={`tlrow${L.id === comp?.active ? " active" : ""}`} style={{ height: ROW }}>
              <button className="icon" title={L.visible ? "Hide layer" : "Show layer"} aria-label={`${L.visible ? "Hide" : "Show"} ${L.name}`}
                onClick={() => void patchLayer(L.id, { visible: !L.visible })}>{L.visible ? "◉" : "○"}</button>
              <button className="name" title={L.name} onClick={() => void selectLayer(L.id)}>{L.name}</button>
              <button className="icon" title="Move up" disabled={i === 0} onClick={() => void moveLayer(L.id, -1)}>▲</button>
              <button className="icon" title="Move down" disabled={i === layers.length - 1} onClick={() => void moveLayer(L.id, 1)}>▼</button>
            </div>
          ))}
        </div>
        <div className="tlcell" ref={cell}>
          <canvas ref={cvs} style={{ width, height: contentH, touchAction: "none", display: "block" }} onPointerDown={down} onPointerMove={move} onPointerUp={up} />
        </div>
      </div>
    </div>
  );
}
