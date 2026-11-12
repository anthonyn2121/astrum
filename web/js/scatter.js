// A scatter plot on a canvas (the sweep page's Compare runs). The caller owns
// the data and what a point's `data` means: this draws the points, reports the
// point under the pointer (onHover) and the point picked by a click or by the
// arrow keys (onPick). Colours are plain CSS colour strings; the grid, label and
// legend colours come from the canvas's --chart-grid, --chart-label and
// --chart-halo (else the page's --line, --muted and --panel-solid). Points are
// drawn as small stars: a soft glow in their colour and a brighter core.

const MARGIN = { left: 60, right: 14, top: 12, bottom: 44 };
const POINT_R = 5;
const SELECTED_R = 8;
const HIT_PX = 10;
const TARGET_TICKS = 6;
const FONT_PX = 11;
const GLOW_PX = 9;
const CORE_R = 1.8;

let probe = null;

// The canvas colour of `name` read on `el`, or of `fallback` when `name` is unset there.
function chartColour(el, name, fallback) {
  return getComputedStyle(el).getPropertyValue(name).trim() ? themeColour(name, el) : themeColour(fallback, el);
}

// A theme colour (a CSS custom property) as a canvas colour string.
export function themeColour(name, el = document.documentElement) {
  probe ??= document.createElement("canvas").getContext("2d");
  probe.fillStyle = getComputedStyle(el).getPropertyValue(name).trim();
  return probe.fillStyle;
}

// The 1-2-5 step that splits `span` into about `count` intervals.
function niceStep(span, count) {
  const raw = span / count;
  const exp = 10 ** Math.floor(Math.log10(raw));
  const f = raw / exp;
  return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * exp;
}

// The ticks of an axis over [min, max] (min < max): the first and last one
// enclose the range.
function niceTicks(min, max) {
  const step = niceStep(max - min, TARGET_TICKS);
  const first = Math.floor(min / step);
  const last = Math.ceil(max / step);
  const ticks = [];
  for (let i = first; i <= last; i++) ticks.push(Number((i * step).toPrecision(12)));
  return { ticks, lo: ticks[0], hi: ticks[ticks.length - 1] };
}

// The axis for `values`: a range with no width (all equal) is padded first.
function axis(values) {
  let min = values.length ? Math.min(...values) : 0;
  let max = values.length ? Math.max(...values) : 1;
  if (!(max > min)) {
    const pad = Math.abs(min) * 0.1 || 1;
    min -= pad;
    max += pad;
  }
  return niceTicks(min, max);
}

// Decimals a tick label needs at this step.
const decimals = (step) => Math.max(0, Math.ceil(-Math.log10(step) - 1e-9));

// A tick's label: a whole number without decimals, else the step's decimals.
const tickLabel = (t, step) => (Number.isInteger(t) ? String(t) : t.toFixed(decimals(step)));

export class Scatter {
  #canvas;
  #ctx;
  #onHover;
  #onPick;
  #data = { xLabel: "", yLabel: "", points: [], legend: [] };
  #frame = 0;
  #observer;

  constructor(canvas, { onHover, onPick } = {}) {
    this.#canvas = canvas;
    this.#ctx = canvas.getContext("2d");
    this.#onHover = onHover;
    this.#onPick = onPick;
    canvas.tabIndex = 0;
    this.#observer = new ResizeObserver(() => this.#schedule());
    this.#observer.observe(canvas);
    canvas.addEventListener("pointermove", this.#onMove);
    canvas.addEventListener("pointerleave", this.#onLeave);
    canvas.addEventListener("click", this.#onClick);
    canvas.addEventListener("keydown", this.#onKey);
    this.#schedule();
  }

  // { xLabel, yLabel, points: [{ x, y, color, selected?, data? }], legend?:
  // [{ label, color }] } (the legend is drawn in the plot's top right corner).
  setData({ xLabel = "", yLabel = "", points = [], legend = [] }) {
    this.#data = {
      xLabel,
      yLabel,
      points: points.map((p) => ({ ...p, selected: !!p.selected })),
      legend,
    };
    this.#schedule();
  }

  destroy() {
    cancelAnimationFrame(this.#frame);
    this.#observer.disconnect();
    this.#canvas.removeEventListener("pointermove", this.#onMove);
    this.#canvas.removeEventListener("pointerleave", this.#onLeave);
    this.#canvas.removeEventListener("click", this.#onClick);
    this.#canvas.removeEventListener("keydown", this.#onKey);
  }

  #schedule() {
    if (this.#frame) return;
    this.#frame = requestAnimationFrame(() => {
      this.#frame = 0;
      this.#draw();
    });
  }

  // The plot's geometry for a canvas of w × h CSS pixels.
  #layout(w, h) {
    const plot = { left: MARGIN.left, top: MARGIN.top, right: w - MARGIN.right, bottom: h - MARGIN.bottom };
    const xa = axis(this.#data.points.map((p) => p.x));
    const ya = axis(this.#data.points.map((p) => p.y));
    const sx = (x) => plot.left + ((x - xa.lo) / (xa.hi - xa.lo)) * (plot.right - plot.left);
    const sy = (y) => plot.bottom - ((y - ya.lo) / (ya.hi - ya.lo)) * (plot.bottom - plot.top);
    return { plot, xa, ya, sx, sy };
  }

  #draw() {
    const canvas = this.#canvas;
    const ctx = this.#ctx;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (!w || !h) return;
    const dpr = window.devicePixelRatio || 1;
    const bw = Math.round(w * dpr);
    const bh = Math.round(h * dpr);
    if (canvas.width !== bw || canvas.height !== bh) {
      canvas.width = bw;
      canvas.height = bh;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const { plot, xa, ya, sx, sy } = this.#layout(w, h);
    if (plot.right <= plot.left || plot.bottom <= plot.top) return;

    const style = getComputedStyle(canvas);
    const colour = {
      text: themeColour("--text", canvas),
      muted: chartColour(canvas, "--chart-label", "--muted"),
      line: chartColour(canvas, "--chart-grid", "--line"),
      accent: themeColour("--accent", canvas),
      panel: chartColour(canvas, "--chart-halo", "--panel-solid"),
      star: chartColour(canvas, "--star", "--text"),
    };
    const font = style.fontFamily;
    ctx.font = `${FONT_PX}px ${font}`;
    ctx.lineWidth = 1;

    // Grid lines and tick labels.
    ctx.strokeStyle = colour.line;
    ctx.fillStyle = colour.muted;
    const xStep = xa.ticks.length > 1 ? xa.ticks[1] - xa.ticks[0] : 1;
    const yStep = ya.ticks.length > 1 ? ya.ticks[1] - ya.ticks[0] : 1;
    ctx.textBaseline = "top";
    ctx.textAlign = "center";
    for (const t of xa.ticks) {
      const x = Math.round(sx(t)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(x, plot.top);
      ctx.lineTo(x, plot.bottom);
      ctx.stroke();
      ctx.fillText(tickLabel(t, xStep), x, plot.bottom + 6);
    }
    ctx.textBaseline = "middle";
    ctx.textAlign = "right";
    for (const t of ya.ticks) {
      const y = Math.round(sy(t)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(plot.left, y);
      ctx.lineTo(plot.right, y);
      ctx.stroke();
      ctx.fillText(tickLabel(t, yStep), plot.left - 6, y);
    }

    // The axes' frame and titles.
    ctx.strokeStyle = colour.muted;
    ctx.beginPath();
    ctx.moveTo(plot.left + 0.5, plot.top);
    ctx.lineTo(plot.left + 0.5, plot.bottom + 0.5);
    ctx.lineTo(plot.right, plot.bottom + 0.5);
    ctx.stroke();
    ctx.fillStyle = colour.text;
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.font = `12px ${font}`;
    ctx.fillText(this.#data.xLabel, (plot.left + plot.right) / 2, h - 4);
    ctx.save();
    ctx.translate(14, (plot.top + plot.bottom) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textBaseline = "middle";
    ctx.fillText(this.#data.yLabel, 0, 0);
    ctx.restore();
    ctx.font = `${FONT_PX}px ${font}`;

    // Points, the selected one last.
    const selected = [];
    for (const p of this.#data.points) {
      if (p.selected) {
        selected.push(p);
        continue;
      }
      this.#dot(p, sx(p.x), sy(p.y), colour);
    }
    for (const p of selected) {
      const x = sx(p.x);
      const y = sy(p.y);
      this.#dot(p, x, y, colour);
      ctx.save();
      ctx.shadowColor = colour.accent;
      ctx.shadowBlur = 12;
      ctx.beginPath();
      ctx.arc(x, y, SELECTED_R, 0, 2 * Math.PI);
      ctx.strokeStyle = colour.accent;
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.restore();
      ctx.beginPath();
      ctx.arc(x, y, SELECTED_R + 4, 0, 2 * Math.PI);
      ctx.globalAlpha = 0.35;
      ctx.strokeStyle = colour.accent;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    this.#drawLegend(plot, colour);
  }

  // A point as a small star: its colour with a soft glow, then a brighter core.
  #dot(p, x, y, colour) {
    const ctx = this.#ctx;
    ctx.save();
    ctx.shadowColor = p.color;
    ctx.shadowBlur = GLOW_PX;
    ctx.beginPath();
    ctx.arc(x, y, POINT_R, 0, 2 * Math.PI);
    ctx.fillStyle = p.color;
    ctx.globalAlpha = 0.9;
    ctx.fill();
    ctx.restore();
    ctx.beginPath();
    ctx.arc(x, y, CORE_R, 0, 2 * Math.PI);
    ctx.fillStyle = colour.star;
    ctx.globalAlpha = 0.85;
    ctx.fill();
    ctx.globalAlpha = 1;
  }

  #drawLegend(plot, colour) {
    const legend = this.#data.legend;
    if (!legend.length) return;
    const ctx = this.#ctx;
    const pad = 8;
    const row = 16;
    const labelW = Math.max(...legend.map((item) => ctx.measureText(item.label).width));
    const boxW = pad * 2 + 10 + 6 + labelW;
    const boxH = pad * 2 + legend.length * row - (row - 10);
    const x0 = plot.right - boxW - 6;
    const y0 = plot.top + 6;
    ctx.globalAlpha = 0.85;
    ctx.fillStyle = colour.panel;
    ctx.fillRect(x0, y0, boxW, boxH);
    ctx.globalAlpha = 1;
    ctx.strokeStyle = colour.line;
    ctx.strokeRect(x0 + 0.5, y0 + 0.5, boxW - 1, boxH - 1);
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    legend.forEach((item, i) => {
      const cy = y0 + pad + i * row + 5;
      ctx.beginPath();
      ctx.arc(x0 + pad + 5, cy, POINT_R, 0, 2 * Math.PI);
      ctx.fillStyle = item.color;
      ctx.fill();
      ctx.fillStyle = colour.text;
      ctx.fillText(item.label, x0 + pad + 16, cy);
    });
  }

  // The point nearest `at` (CSS pixels) within HIT_PX, if any.
  #hit(at) {
    const w = this.#canvas.clientWidth;
    const h = this.#canvas.clientHeight;
    const { sx, sy } = this.#layout(w, h);
    let best = null;
    let bestD = HIT_PX * HIT_PX;
    for (const p of this.#data.points) {
      const dx = sx(p.x) - at.x;
      const dy = sy(p.y) - at.y;
      const d = dx * dx + dy * dy;
      if (d < bestD) {
        best = p;
        bestD = d;
      }
    }
    return best;
  }

  // Make `p` the selected point and report it.
  #pick(p) {
    for (const q of this.#data.points) q.selected = q === p;
    this.#schedule();
    this.#onPick?.(p.data);
  }

  #onMove = (e) => {
    const at = { x: e.offsetX, y: e.offsetY };
    this.#onHover?.(this.#hit(at)?.data ?? null, at);
  };

  #onLeave = (e) => {
    this.#onHover?.(null, { x: e.offsetX, y: e.offsetY });
  };

  #onClick = (e) => {
    const p = this.#hit({ x: e.offsetX, y: e.offsetY });
    if (p) this.#pick(p);
  };

  // Left and right step through the points in x order from the selected one.
  #onKey = (e) => {
    const step = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
    if (!step) return;
    e.preventDefault();
    const order = [...this.#data.points].sort((a, b) => a.x - b.x);
    if (!order.length) return;
    const at = order.findIndex((p) => p.selected);
    const next = at < 0 ? (step > 0 ? 0 : order.length - 1) : Math.min(order.length - 1, Math.max(0, at + step));
    this.#pick(order[next]);
  };
}
