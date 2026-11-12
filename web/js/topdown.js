// A top-down picture on a canvas: north up, east right, metres. A view is a
// centre (metres east and north) and a scale (CSS px per metre). Layers are
// drawn in order over a grid: `arrows` (a filled triangle along each item's
// heading, with an optional label), `dots` (a small translucent disc, with a
// heading tick when given, and an optional radius in px) and `paths` (polylines,
// each with an optional end marker: a dot, or an X), so a cloud of planned
// starts or of flown tracks reads as density.
// Drag pans; the wheel zooms about the cursor; `+` and `-` zoom, the arrow keys
// pan, and `0`, double-click or fit() frame every item. Hover reports the
// nearest item within HIT_PX as onHover(item | null, {x, y}); click reports it
// as onPick(item). Both pass the layer's item, whose `data` is the caller's.
// Colours are CSS colour strings from the items; the grid and labels take the
// canvas's --chart-grid, --chart-label and --chart-halo (else the page's
// --line, --muted and --panel-solid). No dependencies.

const ARROW_NOSE_PX = 8;
const ARROW_TAIL_PX = 6;
const ARROW_HALF_PX = 5;
const LABEL_OFFSET_PX = 10;
const DOT_RADIUS_PX = 2;
const DOT_ALPHA = 0.7;
const TICK_PX = 9;
const HIT_PX = 10;
const PATH_HIT_PX = 8; // a path is hit within this distance of its nearest segment
const PATH_WIDTH_PX = 1.5;
const PATH_ALPHA = 0.5;
const END_DOT_PX = 3;
const END_X_PX = 4;
const DRAG_SLOP_PX = 4;
const GRID_PX = 90; // the grid's spacing, about this many CSS px
const BAR_PX = 100; // the scale bar's length, about this many CSS px
const BAR_MARGIN_PX = 12;
const MIN_SPAN_M = 200; // a view is at least this wide and high
const FIT_MARGIN = 0.12; // the margin on each side of the framed items, as a share of their span
const MIN_SCALE = 0.002; // CSS px per metre: 500 m per px
const MAX_SCALE = 50;
const ZOOM_STEP = 1.25;
const WHEEL_RATE = 0.0015; // zoom factor per CSS px of wheel delta
const WHEEL_UNIT_PX = [1, 16, 400]; // wheel deltas by deltaMode: px, lines, pages
const PAN_KEY_PX = 60;
const LABEL_FONT = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
const DEG = Math.PI / 180;

const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
const cssVar = (css, name, fallback) => css.getPropertyValue(name).trim() || fallback;

// The 1-2-5 step (metres) at or above x.
function niceAbove(x) {
  const exp = 10 ** Math.floor(Math.log10(x));
  const m = x / exp;
  return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 5 ? 5 : 10) * exp;
}

// The multiples of `step` from lo to hi.
function* multiples(lo, hi, step) {
  for (let k = Math.ceil(lo / step); k * step <= hi; k++) yield k * step;
}

// A metre value on a `step` grid, with the decimals the step needs.
function metres(v, step) {
  const dp = step >= 1 ? 0 : Math.ceil(-Math.log10(step));
  const x = Math.abs(v) < step / 1e6 ? 0 : v;
  return `${x.toFixed(dp)} m`;
}

// Text with a halo, so it reads over the grid and the other items.
function haloText(ctx, text, x, y, color, halo) {
  ctx.lineJoin = "round";
  ctx.lineWidth = 3;
  ctx.strokeStyle = halo;
  ctx.strokeText(text, x, y);
  ctx.fillStyle = color;
  ctx.fillText(text, x, y);
}

// The distance, CSS px, from (x, y) to the segment from a to b (both {x, y}).
function segmentDistance(a, b, x, y) {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const len2 = dx * dx + dy * dy;
  const t = len2 === 0 ? 0 : clamp(((x - a.x) * dx + (y - a.y) * dy) / len2, 0, 1);
  return Math.hypot(a.x + t * dx - x, a.y + t * dy - y);
}

// The [e, n] bounds of a polyline, in metres: { e0, e1, n0, n1 }.
function boundsOf(pts) {
  const b = { e0: Infinity, e1: -Infinity, n0: Infinity, n1: -Infinity };
  for (const [e, n] of pts) {
    b.e0 = Math.min(b.e0, e);
    b.e1 = Math.max(b.e1, e);
    b.n0 = Math.min(b.n0, n);
    b.n1 = Math.max(b.n1, n);
  }
  return b;
}

// The [e, n] places of a layer: each path's points, else each item's position.
function* placesOf(layer) {
  for (const it of layer.items) {
    if (layer.kind === "paths") yield* it.pts;
    else yield [it.e, it.n];
  }
}

export class TopDown {
  #canvas;
  #ctx;
  #onHover;
  #onPick;
  #ac = new AbortController();
  #ro;
  #layers = [];
  #w = 0; // CSS px
  #h = 0;
  #dpr = 1;
  #cx = 0; // the view centre, metres
  #cy = 0;
  #scale = 1; // CSS px per metre
  #fitPending = false; // fit() ran while the canvas had no size
  #raf = 0;
  #drag = null; // { x, y, startX, startY, moved } while a pointer drags
  #boxes = new WeakMap(); // path item -> its bounds in metres (set by setLayers)
  #hoverFrame = 0; // the hover test queued for the next frame
  #hoverAt = null; // the pointer position it tests

  constructor(canvas, { onHover, onPick } = {}) {
    this.#canvas = canvas;
    this.#ctx = canvas.getContext("2d");
    this.#onHover = onHover;
    this.#onPick = onPick;
    canvas.tabIndex = 0;
    canvas.style.cursor = "grab";
    canvas.style.touchAction = "none";
    const { signal } = this.#ac;
    canvas.addEventListener("pointerdown", (e) => this.#down(e), { signal });
    canvas.addEventListener("pointermove", (e) => this.#move(e), { signal });
    canvas.addEventListener("pointerup", (e) => this.#up(e), { signal });
    canvas.addEventListener("pointercancel", () => this.#release(), { signal });
    canvas.addEventListener("pointerleave", (e) => { this.#cancelHover(); if (!this.#drag) this.#onHover?.(null, this.#pos(e)); }, { signal });
    canvas.addEventListener("wheel", (e) => this.#wheel(e), { passive: false, signal });
    canvas.addEventListener("dblclick", () => this.fit(), { signal });
    canvas.addEventListener("keydown", (e) => this.#key(e), { signal });
    this.#ro = new ResizeObserver(() => {
      this.#sync();
      if (this.#fitPending) this.fit();
      else this.#schedule();
    });
    this.#ro.observe(canvas);
  }

  // The layers to show, drawn in order.
  setLayers(layers) {
    this.#layers = layers;
    this.#boxes = new WeakMap();
    for (const layer of layers) {
      if (layer.kind !== "paths") continue;
      for (const item of layer.items) if (item.pts.length) this.#boxes.set(item, boundsOf(item.pts));
    }
    this.#schedule();
  }

  // Frame every item of every layer, with a margin; a window of at least
  // MIN_SPAN_M in each direction (so one point gets a sensible view).
  fit() {
    this.#sync();
    if (!this.#w || !this.#h) {
      this.#fitPending = true;
      return;
    }
    this.#fitPending = false;
    let e0 = Infinity;
    let e1 = -Infinity;
    let n0 = Infinity;
    let n1 = -Infinity;
    for (const layer of this.#layers) {
      for (const [e, n] of placesOf(layer)) {
        e0 = Math.min(e0, e);
        e1 = Math.max(e1, e);
        n0 = Math.min(n0, n);
        n1 = Math.max(n1, n);
      }
    }
    if (e0 > e1) {
      e0 = n0 = -MIN_SPAN_M / 2;
      e1 = n1 = MIN_SPAN_M / 2;
    }
    const spanE = Math.max(e1 - e0, MIN_SPAN_M) * (1 + 2 * FIT_MARGIN);
    const spanN = Math.max(n1 - n0, MIN_SPAN_M) * (1 + 2 * FIT_MARGIN);
    this.#cx = (e0 + e1) / 2;
    this.#cy = (n0 + n1) / 2;
    this.#scale = clamp(Math.min(this.#w / spanE, this.#h / spanN), MIN_SCALE, MAX_SCALE);
    this.#schedule();
  }

  destroy() {
    this.#ac.abort();
    this.#ro.disconnect();
    cancelAnimationFrame(this.#raf);
    this.#raf = 0;
    this.#cancelHover();
    this.#layers = [];
  }

  // ---- drawing ----

  #schedule() {
    if (!this.#raf) {
      this.#raf = requestAnimationFrame(() => {
        this.#raf = 0;
        this.#draw();
      });
    }
  }

  // The backing store matches the CSS size at the device pixel ratio.
  #sync() {
    const { clientWidth: w, clientHeight: h } = this.#canvas;
    const dpr = window.devicePixelRatio || 1;
    this.#w = w;
    this.#h = h;
    this.#dpr = dpr;
    const bw = Math.round(w * dpr);
    const bh = Math.round(h * dpr);
    if (this.#canvas.width !== bw || this.#canvas.height !== bh) {
      this.#canvas.width = bw;
      this.#canvas.height = bh;
    }
  }

  #draw() {
    this.#sync();
    const ctx = this.#ctx;
    const w = this.#w;
    const h = this.#h;
    ctx.setTransform(this.#dpr, 0, 0, this.#dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    if (!w || !h) return;
    // Read from the canvas, so a page can theme one picture (--chart-*); each
    // falls back to the page's own variable.
    const css = getComputedStyle(this.#canvas);
    const theme = {
      line: cssVar(css, "--chart-grid", "") || cssVar(css, "--line", "rgba(90,130,170,0.22)"),
      muted: cssVar(css, "--chart-label", "") || cssVar(css, "--muted", "#7d8da0"),
      text: cssVar(css, "--text", "#dbe4ef"),
      halo: cssVar(css, "--chart-halo", "") || cssVar(css, "--panel-solid", "#101620"),
    };
    this.#grid(theme);
    for (const layer of this.#layers) {
      if (layer.kind === "arrows") this.#arrows(layer.items, theme);
      else if (layer.kind === "dots") this.#dots(layer.items);
      else if (layer.kind === "paths") this.#paths(layer.items);
      else throw new Error(`topdown: unknown layer kind ${layer.kind}`);
    }
    this.#scaleBar(theme);
  }

  #grid({ line, muted, halo }) {
    const ctx = this.#ctx;
    const w = this.#w;
    const h = this.#h;
    const s = this.#scale;
    const step = niceAbove(GRID_PX / s);
    const es = [...multiples(this.#cx - w / 2 / s, this.#cx + w / 2 / s, step)];
    const ns = [...multiples(this.#cy - h / 2 / s, this.#cy + h / 2 / s, step)];
    ctx.beginPath();
    for (const e of es) {
      const x = Math.round(this.#screenX(e)) + 0.5;
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
    }
    for (const n of ns) {
      const y = Math.round(this.#screenY(n)) + 0.5;
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
    }
    ctx.lineWidth = 1;
    ctx.strokeStyle = line;
    ctx.stroke();
    ctx.font = LABEL_FONT;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const e of es) haloText(ctx, metres(e, step), this.#screenX(e), 4, muted, halo);
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    for (const n of ns) haloText(ctx, metres(n, step), 6, this.#screenY(n), muted, halo);
  }

  #arrows(items, { halo }) {
    const ctx = this.#ctx;
    ctx.font = LABEL_FONT;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    for (const it of items) {
      const { x, y } = this.#screen(it.e, it.n);
      // Along the heading, on screen: (sin, -cos) (north is up). Across it: (cos, sin).
      const sin = Math.sin(it.heading_deg * DEG);
      const cos = Math.cos(it.heading_deg * DEG);
      ctx.beginPath();
      ctx.moveTo(x + ARROW_NOSE_PX * sin, y - ARROW_NOSE_PX * cos);
      ctx.lineTo(x - ARROW_TAIL_PX * sin + ARROW_HALF_PX * cos, y + ARROW_TAIL_PX * cos + ARROW_HALF_PX * sin);
      ctx.lineTo(x - ARROW_TAIL_PX * sin - ARROW_HALF_PX * cos, y + ARROW_TAIL_PX * cos - ARROW_HALF_PX * sin);
      ctx.closePath();
      ctx.fillStyle = it.color;
      ctx.fill();
      if (it.label) haloText(ctx, it.label, x + LABEL_OFFSET_PX, y, it.color, halo);
    }
  }

  #dots(items) {
    const ctx = this.#ctx;
    ctx.globalAlpha = DOT_ALPHA;
    ctx.lineWidth = 1;
    for (const it of items) {
      const { x, y } = this.#screen(it.e, it.n);
      ctx.fillStyle = it.color;
      ctx.beginPath();
      ctx.arc(x, y, it.radius ?? DOT_RADIUS_PX, 0, 2 * Math.PI);
      ctx.fill();
      if (it.heading_deg != null) {
        ctx.strokeStyle = it.color;
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x + TICK_PX * Math.sin(it.heading_deg * DEG), y - TICK_PX * Math.cos(it.heading_deg * DEG));
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
  }

  // Polylines in screen space, each with an optional end marker at its last point.
  #paths(items) {
    const ctx = this.#ctx;
    ctx.lineJoin = "round";
    for (const it of items) {
      ctx.strokeStyle = it.color;
      ctx.fillStyle = it.color;
      ctx.lineWidth = it.width ?? PATH_WIDTH_PX;
      ctx.globalAlpha = it.alpha ?? PATH_ALPHA;
      ctx.beginPath();
      it.pts.forEach(([e, n], i) => {
        const { x, y } = this.#screen(e, n);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      if (it.end && it.pts.length) {
        const [e, n] = it.pts[it.pts.length - 1];
        const { x, y } = this.#screen(e, n);
        ctx.globalAlpha = 1;
        ctx.beginPath();
        if (it.end === "x") {
          ctx.moveTo(x - END_X_PX, y - END_X_PX);
          ctx.lineTo(x + END_X_PX, y + END_X_PX);
          ctx.moveTo(x + END_X_PX, y - END_X_PX);
          ctx.lineTo(x - END_X_PX, y + END_X_PX);
          ctx.stroke();
        } else {
          ctx.arc(x, y, END_DOT_PX, 0, 2 * Math.PI);
          ctx.fill();
        }
      }
    }
    ctx.globalAlpha = 1;
  }

  #scaleBar({ text, halo }) {
    const ctx = this.#ctx;
    const step = niceAbove(BAR_PX / this.#scale);
    const len = step * this.#scale;
    const x0 = BAR_MARGIN_PX;
    const y1 = this.#h - BAR_MARGIN_PX;
    ctx.beginPath();
    ctx.moveTo(x0, y1 - 5);
    ctx.lineTo(x0, y1);
    ctx.lineTo(x0 + len, y1);
    ctx.lineTo(x0 + len, y1 - 5);
    ctx.lineWidth = 2;
    ctx.strokeStyle = text;
    ctx.stroke();
    ctx.font = LABEL_FONT;
    ctx.textAlign = "left";
    ctx.textBaseline = "bottom";
    haloText(ctx, metres(step, step), x0, y1 - 8, text, halo);
  }

  // ---- view and coordinates ----

  #screenX(e) {
    return this.#w / 2 + (e - this.#cx) * this.#scale;
  }

  #screenY(n) {
    return this.#h / 2 - (n - this.#cy) * this.#scale;
  }

  #screen(e, n) {
    return { x: this.#screenX(e), y: this.#screenY(n) };
  }

  #world(x, y) {
    return { e: this.#cx + (x - this.#w / 2) / this.#scale, n: this.#cy - (y - this.#h / 2) / this.#scale };
  }

  // Move the view by `de` east and `dn` north, metres.
  #shift(de, dn) {
    this.#cx += de;
    this.#cy += dn;
    this.#schedule();
  }

  // Zoom by `factor` keeping the point under (x, y), CSS px, where it is.
  #zoomAt(x, y, factor) {
    const under = this.#world(x, y);
    this.#scale = clamp(this.#scale * factor, MIN_SCALE, MAX_SCALE);
    this.#cx = under.e - (x - this.#w / 2) / this.#scale;
    this.#cy = under.n + (y - this.#h / 2) / this.#scale;
    this.#schedule();
  }

  // The item under (x, y), CSS px, the nearest within HIT_PX over every layer; a
  // path counts within PATH_HIT_PX of its nearest segment.
  #hit(x, y) {
    let best = null;
    let bestD = HIT_PX;
    for (const layer of this.#layers) {
      for (const item of layer.items) {
        let d;
        if (layer.kind === "paths") {
          const box = this.#boxes.get(item);
          if (!box || !this.#near(box, x, y)) continue;
          d = this.#pathDistance(item.pts, x, y);
          if (d > PATH_HIT_PX) continue;
        } else {
          const p = this.#screen(item.e, item.n);
          d = Math.hypot(p.x - x, p.y - y);
        }
        if (d <= bestD) {
          best = item;
          bestD = d;
        }
      }
    }
    return best;
  }

  // Whether (x, y), CSS px, is within PATH_HIT_PX of the screen box of `b`.
  #near(b, x, y) {
    const xa = this.#screenX(b.e0);
    const xb = this.#screenX(b.e1);
    const ya = this.#screenY(b.n0);
    const yb = this.#screenY(b.n1);
    return x >= Math.min(xa, xb) - PATH_HIT_PX && x <= Math.max(xa, xb) + PATH_HIT_PX
      && y >= Math.min(ya, yb) - PATH_HIT_PX && y <= Math.max(ya, yb) + PATH_HIT_PX;
  }

  // The distance, CSS px, from (x, y) to the polyline `pts` ([e, n] metres).
  #pathDistance(pts, x, y) {
    let best = Infinity;
    let prev = null;
    for (const [e, n] of pts) {
      const p = this.#screen(e, n);
      best = Math.min(best, prev ? segmentDistance(prev, p, x, y) : Math.hypot(p.x - x, p.y - y));
      prev = p;
    }
    return best;
  }

  // ---- pointer, wheel and keys ----

  // The pointer in CSS px from the canvas's content box.
  #pos(e) {
    const r = this.#canvas.getBoundingClientRect();
    return { x: e.clientX - r.left - this.#canvas.clientLeft, y: e.clientY - r.top - this.#canvas.clientTop };
  }

  #down(e) {
    if (e.button !== 0) return;
    const p = this.#pos(e);
    this.#drag = { x: p.x, y: p.y, startX: p.x, startY: p.y, moved: false };
    this.#canvas.setPointerCapture(e.pointerId);
    this.#canvas.style.cursor = "grabbing";
  }

  #move(e) {
    const p = this.#pos(e);
    const d = this.#drag;
    if (!d) {
      this.#hoverLater(p);
      return;
    }
    this.#cancelHover();
    // The content follows the pointer.
    this.#shift(-(p.x - d.x) / this.#scale, (p.y - d.y) / this.#scale);
    if (Math.hypot(p.x - d.startX, p.y - d.startY) > DRAG_SLOP_PX) d.moved = true;
    d.x = p.x;
    d.y = p.y;
    this.#onHover?.(null, p);
  }

  // A hover test per animation frame, at the pointer's latest position.
  #hoverLater(p) {
    this.#hoverAt = p;
    if (this.#hoverFrame) return;
    this.#hoverFrame = requestAnimationFrame(() => {
      this.#hoverFrame = 0;
      const at = this.#hoverAt;
      this.#onHover?.(this.#hit(at.x, at.y), at);
    });
  }

  #cancelHover() {
    cancelAnimationFrame(this.#hoverFrame);
    this.#hoverFrame = 0;
  }

  #up(e) {
    const d = this.#drag;
    if (!d) return;
    this.#release();
    if (d.moved) return;
    const p = this.#pos(e);
    const item = this.#hit(p.x, p.y);
    if (item) this.#onPick?.(item);
  }

  #release() {
    this.#drag = null;
    this.#canvas.style.cursor = "grab";
  }

  #wheel(e) {
    e.preventDefault();
    const p = this.#pos(e);
    const px = e.deltaY * (WHEEL_UNIT_PX[e.deltaMode] ?? 1);
    this.#zoomAt(p.x, p.y, Math.exp(-px * WHEEL_RATE));
  }

  #key(e) {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const cx = this.#w / 2;
    const cy = this.#h / 2;
    const pan = PAN_KEY_PX / this.#scale;
    switch (e.key) {
      case "+":
      case "=":
        this.#zoomAt(cx, cy, ZOOM_STEP);
        break;
      case "-":
        this.#zoomAt(cx, cy, 1 / ZOOM_STEP);
        break;
      case "ArrowLeft":
        this.#shift(-pan, 0);
        break;
      case "ArrowRight":
        this.#shift(pan, 0);
        break;
      case "ArrowUp":
        this.#shift(0, pan);
        break;
      case "ArrowDown":
        this.#shift(0, -pan);
        break;
      case "0":
        this.fit();
        break;
      default:
        return;
    }
    e.preventDefault();
  }
}
