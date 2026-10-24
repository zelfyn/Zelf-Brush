import { useCallback, useEffect, useRef, useState } from "react";
import * as api from "../api";
import { activeLayer, afterEdit, browseVideo, cancelJob, importFile, localFrameNow, renderUrl, scheduleAutoUpdate, showError, trackAi } from "../actions";
import { store, useApp } from "../store";
import type { ViewMode } from "../types";

type Pt = [number, number];
const MODES: { id: ViewMode; label: string; key: string }[] = [
  { id: "composite", label: "Composite", key: "C" },
  { id: "original", label: "Original", key: "O" },
  { id: "mask", label: "Mask", key: "M" },
  { id: "transparent", label: "Transparent", key: "T" },
  { id: "overlay", label: "Overlay", key: "V" },
  { id: "edge", label: "Problem areas", key: "E" },
  { id: "depth", label: "Depth", key: "D" },
];

function checker(): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = c.height = 16;
  const g = c.getContext("2d")!;
  g.fillStyle = "#3a3f48";
  g.fillRect(0, 0, 16, 16);
  g.fillStyle = "#2b2f36";
  g.fillRect(0, 0, 8, 8);
  g.fillRect(8, 8, 8, 8);
  return c;
}
const CHECK = checker();
const NEEDS_CHECKER: ViewMode[] = ["transparent", "composite"];

/** The roto tools always act on the SELECTED layer. In the composite view a click is in comp space, so map it into that
 * layer's own proxy pixels (undoing its fit, scale and position). `k` converts a length (brush size) the same way. */
function layerMap(): { to: (p: Pt) => Pt; k: number } | null {
  const s = store.get();
  const L = activeLayer();
  if (!s.comp || !s.project || !L) return null;
  if (s.viewMode !== "composite") return { to: (p) => p, k: 1 };
  const c = s.comp;
  const sx = c.proxy.width / c.width;
  const fit = Math.min(c.width / L.width, c.height / L.height);
  const dw = L.width * fit * L.scale * sx, dh = L.height * fit * L.scale * sx;
  const ox = (c.proxy.width - dw) / 2 + L.x * sx, oy = (c.proxy.height - dh) / 2 + L.y * sx;
  const lw = s.project.proxy.width, lh = s.project.proxy.height;
  return { to: (q) => [((q[0] - ox) / dw) * lw, ((q[1] - oy) / dh) * lh], k: lw / dw };
}

export function Viewport() {
  const project = useApp((s) => s.project);
  const comp = useApp((s) => s.comp);
  const frame = useApp((s) => s.frame);
  const mode = useApp((s) => s.viewMode);
  const quality = useApp((s) => s.previewQuality);
  const renderKey = useApp((s) => s.renderKey);
  const tool = useApp((s) => s.tool);
  const busy = useApp((s) => s.busy);
  const dropActive = useApp((s) => s.dropActive);
  const aiPending = useApp((s) => s.aiPending);
  const jobs = useApp((s) => s.jobs);
  const autoJob = Object.values(jobs).find((j) => j.lane === "auto" && (j.status === "running" || j.status === "waiting"));

  const stageRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const view = useRef({ s: 1, tx: 0, ty: 0 });
  const fit = useRef(true);
  const hover = useRef<Pt | null>(null);
  const stroke = useRef<Pt[]>([]);
  const poly = useRef<Pt[]>([]);
  const boxSel = useRef<{ a: Pt; b: Pt } | null>(null);
  const active = useRef<"none" | "paint" | "lasso" | "pan" | "box">("none");
  const panStart = useRef({ x: 0, y: 0, tx: 0, ty: 0 });
  const pending = useRef(false);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const [offLayer, setOffLayer] = useState(false);

  const dims = mode === "composite" ? comp?.proxy : project?.proxy;
  const W = dims?.width ?? 0;
  const H = dims?.height ?? 0;

  const draw = useCallback(() => {
    const cv = canvasRef.current;
    if (!cv || !size.w) return;
    const dpr = window.devicePixelRatio || 1;
    if (cv.width !== Math.round(size.w * dpr) || cv.height !== Math.round(size.h * dpr)) {
      cv.width = Math.round(size.w * dpr);
      cv.height = Math.round(size.h * dpr);
    }
    const g = cv.getContext("2d")!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, size.w, size.h);
    if (!W) return;
    const v = view.current;
    const st = store.get();
    if (NEEDS_CHECKER.includes(st.viewMode)) {
      g.save();
      g.beginPath();
      g.rect(v.tx, v.ty, W * v.s, H * v.s);
      g.clip();
      g.fillStyle = g.createPattern(CHECK, "repeat")!;
      g.fillRect(v.tx, v.ty, W * v.s, H * v.s);
      g.restore();
    }
    if (imgRef.current) {
      g.imageSmoothingQuality = "high";
      g.drawImage(imgRef.current, v.tx, v.ty, W * v.s, H * v.s);
    }
    const toScreen = (p: Pt): Pt => [p[0] * v.s + v.tx, p[1] * v.s + v.ty];
    const t = st.tool;
    if ((t === "brush" || t === "eraser") && stroke.current.length) {
      g.strokeStyle = t === "brush" ? "rgba(91,196,137,.55)" : "rgba(255,107,74,.55)";
      g.lineCap = g.lineJoin = "round";
      g.lineWidth = st.brush.size * v.s;
      g.beginPath();
      stroke.current.forEach((p, i) => {
        const [x, y] = toScreen(p);
        if (i) g.lineTo(x, y);
        else g.moveTo(x, y);
      });
      if (stroke.current.length === 1) g.lineTo(...toScreen(stroke.current[0]));
      g.stroke();
    }
    if (t === "lasso" && poly.current.length) {
      g.strokeStyle = st.lasso.mode === "add" ? "#5bc489" : "#ff6b4a";
      g.fillStyle = st.lasso.mode === "add" ? "rgba(91,196,137,.15)" : "rgba(255,107,74,.15)";
      g.lineWidth = 1.5;
      g.setLineDash([5, 4]);
      g.beginPath();
      poly.current.forEach((p, i) => {
        const [x, y] = toScreen(p);
        if (i) g.lineTo(x, y);
        else g.moveTo(x, y);
      });
      if (active.current === "lasso" || st.lasso.shape === "freehand") g.closePath();
      g.fill();
      g.stroke();
      g.setLineDash([]);
    }
    if (boxSel.current) {
      const [ax, ay] = toScreen(boxSel.current.a);
      const [bx, by] = toScreen(boxSel.current.b);
      const sub = st.objectMode === "subtract";
      g.strokeStyle = sub ? "#ff6b4a" : "#6ea8fe";
      g.fillStyle = sub ? "rgba(255,107,74,.12)" : "rgba(110,168,254,.12)";
      g.lineWidth = 1.5;
      g.setLineDash([6, 4]);
      g.fillRect(Math.min(ax, bx), Math.min(ay, by), Math.abs(bx - ax), Math.abs(by - ay));
      g.strokeRect(Math.min(ax, bx), Math.min(ay, by), Math.abs(bx - ax), Math.abs(by - ay));
      g.setLineDash([]);
    }
    if (hover.current && (t === "brush" || t === "eraser")) {
      const [x, y] = toScreen(hover.current);
      g.strokeStyle = "#fff";
      g.lineWidth = 1;
      g.beginPath();
      g.arc(x, y, (st.brush.size * v.s) / 2, 0, Math.PI * 2);
      g.stroke();
    }
  }, [size, W, H]);

  const schedule = useCallback(() => {
    if (pending.current) return;
    pending.current = true;
    requestAnimationFrame(() => {
      pending.current = false;
      draw();
    });
  }, [draw]);

  const fitView = useCallback(() => {
    if (!W || !size.w) return;
    const s = Math.min(size.w / W, size.h / H) * 0.96;
    view.current = { s, tx: (size.w - W * s) / 2, ty: (size.h - H * s) / 2 };
    fit.current = true;
    schedule();
  }, [W, H, size, schedule]);

  useEffect(() => {
    const el = stageRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    if (fit.current) fitView();
    else schedule();
  }, [size, W, H, fitView, schedule]);

  // load the image for the current comp frame (the old image stays visible until the new one is ready)
  useEffect(() => {
    if (!project || !comp) {
      imgRef.current = null;
      setOffLayer(false);
      schedule();
      return;
    }
    const u = renderUrl(frame);
    if (!u) {
      imgRef.current = null;
      setOffLayer(true);
      store.set({ displayedFrame: frame });
      schedule();
      return;
    }
    setOffLayer(false);
    let stale = false;
    const im = new Image();
    im.onload = () => {
      if (stale) return;
      imgRef.current = im;
      store.set({ displayedFrame: frame });
      schedule();
    };
    im.onerror = () => {
      if (!stale) store.set({ displayedFrame: frame });
    };
    im.src = u;
    return () => {
      stale = true;
    };
  }, [project, comp, frame, mode, quality, renderKey, schedule]);

  useEffect(schedule, [tool, schedule]);

  const toImg = (e: { clientX: number; clientY: number }): Pt => {
    const r = canvasRef.current!.getBoundingClientRect();
    const v = view.current;
    return [(e.clientX - r.left - v.tx) / v.s, (e.clientY - r.top - v.ty) / v.s];
  };

  const send = async (path: string, body: Record<string, unknown>, localFrame?: number, usesAi = false) => {
    try {
      await (usesAi ? trackAi(() => api.post(path, body)) : api.post(path, body));
      if (localFrame !== undefined) scheduleAutoUpdate(localFrame);   // the edit is applied; the AI updates neighbours later
    } catch (err) {
      showError(err);
    }
    stroke.current = [];
    poly.current = [];
    boxSel.current = null;
    active.current = "none";
    await afterEdit();
    schedule();
  };

  const closePolygon = () => {
    const st = store.get();
    const lf = localFrameNow();
    const m = layerMap();
    if (poly.current.length >= 3 && lf !== null && m)
      void send("/mask/lasso", { frame: lf, polygon: poly.current.map(m.to), mode: st.lasso.mode, ai_refine: st.lasso.aiRefine }, lf, false);
    else poly.current = [];
    schedule();
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (store.get().tool !== "lasso" || !poly.current.length) return;
      if (e.key === "Enter") closePolygon();
      if (e.key === "Escape") {
        poly.current = [];
        schedule();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const onDown = (e: React.PointerEvent) => {
    if (!project) return;
    const st = store.get();
    const p = toImg(e);
    canvasRef.current!.setPointerCapture(e.pointerId);
    if (e.button === 1 || st.tool === "pan") {
      active.current = "pan";
      panStart.current = { x: e.clientX, y: e.clientY, tx: view.current.tx, ty: view.current.ty };
      return;
    }
    if (e.button !== 0) return;
    if (st.viewMode === "depth") {
      store.set({ viewMode: "overlay" });   // the depth view is read-only: switch to a paintable view instead of ignoring the click
      return;
    }
    if (st.tool === "brush" || st.tool === "eraser") {
      active.current = "paint";
      stroke.current = [p];
    } else if (st.tool === "lasso") {
      if (st.lasso.shape === "freehand") {
        active.current = "lasso";
        poly.current = [p];
      } else {
        poly.current = [...poly.current, p];
      }
    } else if (st.tool === "object") {
      active.current = "box";
      boxSel.current = { a: p, b: p };
    }
    schedule();
  };

  const onMove = (e: React.PointerEvent) => {
    const p = toImg(e);
    hover.current = p;
    if (active.current === "pan") {
      view.current = { ...view.current, tx: panStart.current.tx + e.clientX - panStart.current.x, ty: panStart.current.ty + e.clientY - panStart.current.y };
      fit.current = false;
    } else if (active.current === "paint" || active.current === "lasso") {
      const arr = active.current === "paint" ? stroke.current : poly.current;
      const last = arr[arr.length - 1];
      if (!last || Math.hypot(p[0] - last[0], p[1] - last[1]) >= 1.5 / view.current.s) arr.push(p);
    } else if (active.current === "box" && boxSel.current) {
      boxSel.current = { ...boxSel.current, b: p };
    }
    schedule();
  };

  const onUp = () => {
    const st = store.get();
    if (active.current === "paint" && stroke.current.length) {
      const lf = localFrameNow();
      const m = layerMap();
      if (lf === null || !m) {
        stroke.current = [];
        active.current = "none";
        schedule();
        return;
      }
      void send("/mask/stroke", {
        frame: lf, mode: st.tool === "brush" ? "add" : "remove", points: stroke.current.map(m.to),
        size: Math.max(1, st.brush.size * m.k), hardness: st.brush.hardness, strength: st.brush.strength,
      }, lf, false);
    } else if (active.current === "lasso") {
      active.current = "none";
      closePolygon();
    } else if (active.current === "box" && boxSel.current) {
      const { a, b } = boxSel.current;
      const lf = localFrameNow();
      const m = layerMap();
      const dragged = Math.hypot((b[0] - a[0]) * view.current.s, (b[1] - a[1]) * view.current.s) > 8;
      boxSel.current = null;
      if (lf === null || !m) {
        active.current = "none";
        schedule();
        return;
      }
      const mode = st.objectMode === "subtract" ? "subtract" : "add";
      const la = m.to(a), lb = m.to(b);
      if (dragged) {
        const x = Math.round(Math.min(la[0], lb[0])), y = Math.round(Math.min(la[1], lb[1]));
        void send("/segmentation/select", { frame: lf, mode, box: [x, y, Math.round(Math.abs(lb[0] - la[0])), Math.round(Math.abs(lb[1] - la[1]))] }, lf, true);
      } else {
        void send("/segmentation/select", { frame: lf, mode, points: [la] }, lf, true);
      }
    } else {
      active.current = "none";
    }
  };

  const onWheel = (e: React.WheelEvent) => {
    const v = view.current;
    const r = canvasRef.current!.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    const s2 = Math.min(40, Math.max(0.05, v.s * Math.exp(-e.deltaY * 0.0015)));
    view.current = { s: s2, tx: mx - ((mx - v.tx) / v.s) * s2, ty: my - ((my - v.ty) / v.s) * s2 };
    fit.current = false;
    schedule();
  };

  const cursor = tool === "pan" ? "grab" : tool === "object" ? "crosshair" : "none";

  return (
    <div className="view">
      <div className="viewbar">
        <div className="seg" role="group" aria-label="Viewport mode">
          {MODES.map((m) => (
            <button key={m.id} className={mode === m.id ? "on" : ""} title={`${m.label} (${m.key})`} onClick={() => store.set({ viewMode: m.id })}>
              {m.label}
            </button>
          ))}
        </div>
        <span className="spacer" />
        {(aiPending > 0 || autoJob) && (
          <span className="chip" role="status" aria-live="polite">
            <i />{aiPending > 0 ? "AI detecting…" : autoJob?.message || autoJob?.title || "AI updating…"}
            {autoJob && aiPending === 0 && <button onClick={() => void cancelJob(autoJob.id)} title="Stop the background update">Stop</button>}
          </span>
        )}
        <label title="Resolution used for preview frames. Masks are edited at proxy size; export uses the original.">
          Preview{" "}
          <select value={quality} onChange={(e) => store.set({ previewQuality: Number(e.target.value) as 25 | 50 | 75 | 100 })}>
            {[25, 50, 75, 100].map((q) => (
              <option key={q} value={q}>{q}%</option>
            ))}
          </select>
        </label>
        <button onClick={fitView} title="Fit to window">Fit</button>
      </div>
      <div className="stage" ref={stageRef} onDragOver={(e) => e.preventDefault()} onDrop={(e) => {
        e.preventDefault();
        const f = e.dataTransfer.files[0];
        if (f) void importFile(f);
      }}>
        <canvas
          ref={canvasRef}
          style={{ cursor, touchAction: "none" }}
          onPointerDown={onDown}
          onPointerMove={onMove}
          onPointerUp={onUp}
          onPointerLeave={() => {
            hover.current = null;
            schedule();
          }}
          onDoubleClick={() => {
            if (store.get().tool === "lasso" && store.get().lasso.shape === "polygon") closePolygon();
          }}
          onWheel={onWheel}
          onContextMenu={(e) => e.preventDefault()}
        />
        {offLayer && project && (
          <div className="empty" style={{ pointerEvents: "none" }}>
            <div className="drop" style={{ borderStyle: "solid", maxWidth: 420 }}>
              <p style={{ margin: 0 }}>The playhead is outside the selected layer. Move it onto the layer, or choose <b>Composite</b>.</p>
            </div>
          </div>
        )}
        {!project && (
          <div className="empty" onDragOver={(e) => e.preventDefault()} onDrop={(e) => {
            e.preventDefault();
            const f = e.dataTransfer.files[0];
            if (f) void importFile(f);
          }}>
            <div className={`drop${dropActive ? " active" : ""}`}>
              <h2>Drop a video to start</h2>
              <p>MP4, MKV or MOV. Your original file is never modified.</p>
              <button className="primary" onClick={() => void browseVideo()}>Choose video…</button>
            </div>
          </div>
        )}
        {busy && <div className="busy">{busy}</div>}
      </div>
    </div>
  );
}
