import { useCallback, useEffect, useRef, useState } from "react";
import { setFrame, setWorkArea, stepFrame, togglePlay } from "../actions";
import { useApp } from "../store";

function timecode(f: number, fps: number): string {
  const total = Math.floor(f / fps);
  const ff = Math.floor(f - total * fps);
  const m = Math.floor(total / 60), s = total % 60;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}:${String(ff).padStart(2, "0")}`;
}

const PAD = 14;

export function Timeline() {
  const project = useApp((s) => s.project);
  const tl = useApp((s) => s.timeline);
  const frame = useApp((s) => s.frame);
  const playing = useApp((s) => s.playing);
  const ref = useRef<HTMLCanvasElement>(null);
  const wrap = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const drag = useRef<"none" | "scrub" | "in" | "out">("none");

  const n = project?.project.frame_count ?? 0;
  const fps = project?.project.fps ?? 24;
  const wa = project?.workArea ?? { in: 0, out: 0 };
  const xOf = useCallback((f: number) => PAD + (n > 1 ? (f / (n - 1)) * (size.w - PAD * 2) : 0), [n, size.w]);
  const fOf = (x: number) => (n > 1 ? ((x - PAD) / (size.w - PAD * 2)) * (n - 1) : 0);

  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const cv = ref.current;
    if (!cv || !size.w) return;
    const dpr = window.devicePixelRatio || 1;
    cv.width = Math.round(size.w * dpr);
    cv.height = Math.round(size.h * dpr);
    const g = cv.getContext("2d")!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, size.w, size.h);
    const css = getComputedStyle(document.documentElement);
    const c = (k: string) => css.getPropertyValue(k).trim();
    const top = 22, trackH = 34, rulerY = 4;
    if (!project) return;
    // ruler
    g.fillStyle = c("--dim");
    g.font = "11px " + c("--font");
    g.textBaseline = "top";
    const pxPerSec = (size.w - PAD * 2) / Math.max(1, (n - 1) / fps);
    const step = [1, 2, 5, 10, 15, 30, 60, 120, 300].find((s) => s * pxPerSec >= 70) ?? 600;
    for (let t = 0; t * fps <= n - 1; t += step) {
      const x = xOf(t * fps);
      g.fillRect(x, rulerY + 10, 1, 6);
      g.fillText(timecode(t * fps, fps).slice(0, 5), x + 3, rulerY);
    }
    // track + work area
    g.fillStyle = c("--bg0");
    g.fillRect(PAD, top, size.w - PAD * 2, trackH);
    g.fillStyle = "#1e2a3a";
    g.fillRect(xOf(wa.in), top, Math.max(2, xOf(wa.out) - xOf(wa.in)), trackH);
    // masked ranges
    g.fillStyle = "#2f5b45";
    for (const [a, b] of tl?.masked ?? []) g.fillRect(xOf(a), top + trackH - 10, Math.max(1, xOf(b) - xOf(a)), 6);
    // confidence strip
    if (tl) {
      const bw = Math.max(1, (size.w - PAD * 2) / Math.max(1, n));
      for (const [k, v] of Object.entries(tl.confidence)) {
        g.fillStyle = v >= 0.75 ? "#5bc489" : v >= 0.55 ? "#f0b429" : "#ff6b4a";
        g.fillRect(xOf(Number(k)), top + trackH - 3, bw, 3);
      }
      // warnings
      g.fillStyle = c("--warn");
      for (const f of tl.warnings) g.fillRect(xOf(f), top + 2, Math.max(1, bw), 8);
      // keyframes
      for (const k of tl.keyframes) {
        const x = xOf(k.frame), y = top + 16;
        g.fillStyle = k.kind === "user" ? c("--user") : c("--ai");
        g.beginPath();
        g.moveTo(x, y - 7); g.lineTo(x + 6, y); g.lineTo(x, y + 7); g.lineTo(x - 6, y);
        g.closePath();
        g.fill();
      }
    }
    // work-area handles
    g.fillStyle = c("--accent");
    for (const f of [wa.in, wa.out]) g.fillRect(xOf(f) - 2, top - 4, 4, trackH + 8);
    // playhead
    const px = xOf(frame);
    g.fillStyle = "#fff";
    g.fillRect(px - 1, rulerY + 14, 2, top + trackH - rulerY - 10);
    g.beginPath();
    g.moveTo(px - 6, rulerY + 14); g.lineTo(px + 6, rulerY + 14); g.lineTo(px, rulerY + 22);
    g.closePath();
    g.fill();
  }, [size, project, tl, frame, wa.in, wa.out, n, fps, xOf]);

  const pos = (e: React.PointerEvent) => e.clientX - ref.current!.getBoundingClientRect().left;

  const down = (e: React.PointerEvent) => {
    if (!project) return;
    ref.current!.setPointerCapture(e.pointerId);
    const x = pos(e);
    drag.current = Math.abs(x - xOf(wa.in)) < 6 ? "in" : Math.abs(x - xOf(wa.out)) < 6 ? "out" : "scrub";
    if (drag.current === "scrub") setFrame(fOf(x));
  };
  const move = (e: React.PointerEvent) => {
    if (drag.current === "none") return;
    const f = Math.max(0, Math.min(n - 1, Math.round(fOf(pos(e)))));
    if (drag.current === "scrub") setFrame(f);
    else if (drag.current === "in") void setWorkArea(Math.min(f, wa.out), wa.out);
    else void setWorkArea(wa.in, Math.max(f, wa.in));
  };

  const jump = (dir: 1 | -1) => {
    const all = [...new Set((tl?.keyframes ?? []).map((k) => k.frame).concat(tl?.warnings ?? []))].sort((a, b) => a - b);
    const t = dir > 0 ? all.find((f) => f > frame) : [...all].reverse().find((f) => f < frame);
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
        <span className="dim num">frame {frame} / {Math.max(0, n - 1)}</span>
        <span className="spacer" />
        <span className="legend">
          <span><i style={{ background: "var(--ai)" }} />AI</span>
          <span><i style={{ background: "var(--user)" }} />Correction</span>
          <span><i style={{ background: "var(--warn)" }} />⚠ Correction recommended</span>
        </span>
      </div>
      <div className="tlwrap" ref={wrap}>
        <canvas ref={ref} onPointerDown={down} onPointerMove={move} onPointerUp={() => (drag.current = "none")} style={{ touchAction: "none" }} />
      </div>
    </div>
  );
}
