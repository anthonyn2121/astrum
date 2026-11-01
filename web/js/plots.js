// Plots dock: stacked, time-synced telemetry charts for the selected aircraft.
//
// One uPlot chart per active channel (altitude, airspeed, load factor, … and
// relative geometry against the pinned target), stacked in the dock body. All
// charts share an absolute sim-time x axis (ticks labelled M:SS from the start
// of the timeline) and a synced hover cursor. main.js owns the playhead and the
// selection and calls `update()` every animation frame; the dock only does
// work when what it shows changed:
//
//  * series arrays are rebuilt when the selection, target, channel set or the
//    store's data change — once per file load, at most ~4 Hz while live frames
//    arrive — and handed to the existing charts with `setData`; a chart is
//    only recreated when its series layout changes;
//  * the playhead is a positioned DOM line per chart, not a canvas redraw;
//  * the row readouts (value at the playhead, or under the hover cursor) are
//    interpolated from the store and written only when their text changes.
//
// A click on a chart seeks (`onSeek`), a horizontal drag zooms every chart to
// that span, a double-click returns to the full timeline — which, unzoomed,
// keeps growing with a live run. Frame data reaches the DOM only through
// `textContent` and style properties.

import uPlot from "../vendor/uPlot.esm.js";
import { relativeGeometry } from "./geometry.js";

const RAD2DEG = 180 / Math.PI;
// Live data refresh cadence: a full rebuild of 36 000-point series per frame
// would be wasted work, the eye can't follow faster than this anyway.
const LIVE_REFRESH_MS = 250;
const MIN_ROW_H = 56;   // px per chart row; the body scrolls beyond that
const HINT_ROW_H = 30;  // px for a relative row waiting for a target
const CLICK_PX = 4;     // pointer travel below this is a click (seek), not a zoom drag
const MIN_ZOOM_S = 0.05;
// Max time gap (s) between a selected-aircraft sample and the target sample it
// is paired with for relative channels; the sim publishes every aircraft on
// the same tick, so pairs normally match exactly.
const REL_TOL_S = 0.25;
const DEFAULT_CHANNELS = ["alt", "tas", "nz"];
const FONT = "11px system-ui, -apple-system, \"Segoe UI\", Roboto, sans-serif";
// x tick spacings (s): round clock steps, so M:SS labels land on round values.
const X_INCRS = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200];
// Hit points and ammo change in steps; a sloped line between samples would lie.
const STEPPED = uPlot.paths.stepped({ align: 1 });

function wrap360(deg) {
  return ((deg % 360) + 360) % 360;
}

// Channel table in menu order. `get` reads a store sample, `rel` a
// relativeGeometry result; `dp` = readout decimals; `wrap` = the value jumps
// by 360° (a gap is inserted there); `step` = drawn as a step line.
const DEFS = [
  { key: "alt", label: "Altitude (HAE)", unit: "m", dp: 0, get: (s) => s.alt_hae_m },
  { key: "vs", label: "Vertical speed", unit: "m/s", dp: 1, get: (s) => s.vel_u_mps },
  { key: "tas", label: "True airspeed", unit: "m/s", dp: 1, get: (s) => s.tas_mps },
  { key: "gs", label: "Ground speed", unit: "m/s", dp: 1, get: (s) => Math.hypot(s.vel_e_mps, s.vel_n_mps) },
  { key: "mach", label: "Mach", unit: "", dp: 2, get: (s) => s.mach },
  { key: "nz", label: "Load factor", unit: "g", dp: 2, get: (s) => s.nz_g },
  { key: "aoa", label: "Angle of attack", unit: "°", dp: 1, get: (s) => s.alpha_rad * RAD2DEG },
  { key: "hdg", label: "Heading", unit: "°", dp: 0, wrap: true, get: (s) => wrap360(s.psi_rad * RAD2DEG) },
  { key: "pitch", label: "Pitch", unit: "°", dp: 1, get: (s) => s.theta_rad * RAD2DEG },
  { key: "roll", label: "Roll", unit: "°", dp: 1, wrap: true, get: (s) => s.phi_rad * RAD2DEG },
  { key: "p", label: "Roll rate", unit: "°/s", dp: 1, get: (s) => s.p_rad_s * RAD2DEG },
  { key: "q", label: "Pitch rate", unit: "°/s", dp: 1, get: (s) => s.q_rad_s * RAD2DEG },
  { key: "r", label: "Yaw rate", unit: "°/s", dp: 1, get: (s) => s.r_rad_s * RAD2DEG },
  { key: "hp", label: "Hit points", unit: "", dp: 0, step: true, get: (s) => s.hit_points },
  { key: "ammo", label: "Ammo", unit: "", dp: 0, step: true, get: (s) => s.ammo },
  { key: "range", label: "Range", unit: "m", dp: 0, relative: true, hint: "range", rel: (g) => g.range },
  { key: "closure", label: "Closure", unit: "m/s", dp: 1, relative: true, hint: "closure", rel: (g) => g.closure },
  { key: "ata", label: "ATA", unit: "°", dp: 0, relative: true, hint: "ATA", rel: (g) => g.ata * RAD2DEG },
  { key: "aspect", label: "Aspect", unit: "°", dp: 0, relative: true, hint: "aspect", rel: (g) => g.aspect * RAD2DEG },
  { key: "relalt", label: "Relative altitude", unit: "m", dp: 0, relative: true, hint: "relative altitude", rel: (g) => g.relAlt },
];
const DEF = new Map(DEFS.map((d) => [d.key, d]));

// Public channel list, in menu order.
export const CHANNELS = Object.freeze(DEFS.map(({ key, label, unit, relative = false }) =>
  Object.freeze({ key, label, unit, relative })));

function el(tag, className, text) {
  const e = document.createElement(tag);
  if (className) e.className = className;
  if (text != null) e.textContent = text;
  return e;
}

function clamp(v, lo, hi) {
  return Math.min(Math.max(v, lo), hi);
}

function sanitizeKeys(keys) {
  const out = [];
  for (const k of keys ?? []) if (DEF.has(k) && !out.includes(k)) out.push(k);
  return out;
}

function readTheme() {
  const cs = getComputedStyle(document.documentElement);
  const v = (name, fallback) => cs.getPropertyValue(name).trim() || fallback;
  return {
    bg: v("--bg", "#0a0e14"),
    text: v("--text", "#dbe4ef"),
    muted: v("--muted", "#7d8da0"),
    line: v("--line", "rgba(90,130,170,0.22)"),
    live: v("--live", "#ff5a4d"),
  };
}

// Readout text; never "-0.0".
function fmtValue(v, dp) {
  if (v == null || !Number.isFinite(v)) return "—";
  const s = v.toFixed(dp);
  return +s === 0 ? (0).toFixed(dp) : s;
}

// y tick label: compact, no float noise (0.30000000000000004).
function fmtTick(v) {
  if (v == null) return "";
  const a = Math.abs(v);
  if (a >= 10000) return `${+(v / 1000).toFixed(a >= 100000 ? 0 : 1)}k`;
  return String(+v.toFixed(3));
}

// x tick label: seconds since the start of the timeline as M:SS (M:SS.s when
// the ticks are less than a second apart).
function fmtClock(rel, incr) {
  const sign = rel < 0 ? "-" : "";
  const a = Math.abs(rel);
  if (incr < 1) {
    const tenths = Math.round(a * 10);
    const m = Math.floor(tenths / 600);
    const sec = ((tenths - m * 600) / 10).toFixed(1);
    return `${sign}${m}:${sec.padStart(4, "0")}`;
  }
  const whole = Math.round(a);
  const m = Math.floor(whole / 60);
  return `${sign}${m}:${String(whole - m * 60).padStart(2, "0")}`;
}

// Full-timeline x range; never empty (uPlot ignores a zero-width scale).
function fullRange(bounds) {
  if (!bounds || bounds.empty) return [0, 1];
  if (bounds.span < 1e-3) return [bounds.min - 0.5, bounds.min + 0.5];
  return [bounds.min, bounds.max];
}

// Union of two tracks' sample times in one merge walk. `sa[i]` / `sb[i]` are
// the samples at x[i], or undefined where that track has none — uPlot bridges
// undefined points (null would break the line). `sb` is only built when
// `withB` (filled with undefined if there is no track b).
function mergeTracks(a, b, withB) {
  const na = a ? a.length : 0;
  const nb = withB && b ? b.length : 0;
  const cap = na + nb;
  const x = new Float64Array(cap);
  const sa = new Array(cap);
  const sb = withB ? new Array(cap) : null;
  let i = 0, j = 0, n = 0;
  let pa = na ? a.at(0) : null;
  let pb = nb ? b.at(0) : null;
  while (pa || pb) {
    if (pa && (!pb || pa.t <= pb.t)) {
      x[n] = pa.t;
      sa[n] = pa;
      if (pb && pb.t === pa.t) {
        sb[n] = pb;
        pb = ++j < nb ? b.at(j) : null;
      } else if (sb) {
        sb[n] = undefined;
      }
      pa = ++i < na ? a.at(i) : null;
    } else {
      x[n] = pb.t;
      sa[n] = undefined;
      sb[n] = pb;
      pb = ++j < nb ? b.at(j) : null;
    }
    n++;
  }
  sa.length = n;
  if (sb) sb.length = n;
  return { x: x.subarray(0, n), sa, sb };
}

// One channel's y values over aligned samples. A non-finite value is a gap
// (null); for wrapping angles a jump of more than half a turn between
// neighbours is also nulled, so the line breaks instead of drawing a
// full-height spike across the chart.
function column(samples, def) {
  const n = samples.length;
  const ys = new Array(n);
  let prev = null;
  for (let i = 0; i < n; i++) {
    const s = samples[i];
    if (s === undefined) { ys[i] = undefined; continue; }
    const v = def.get(s);
    if (!Number.isFinite(v)) { ys[i] = prev = null; continue; }
    ys[i] = def.wrap && prev !== null && Math.abs(v - prev) > 180 ? null : v;
    prev = v;
  }
  return ys;
}

// Relative channels (selected → target) at the selected track's sample times.
// One merge walk keeps a cursor into the target track and pairs each sample
// with the nearest target sample; no partner within REL_TOL_S is a gap.
function relativeColumns(a, b, defs) {
  const na = a ? a.length : 0;
  const nb = b ? b.length : 0;
  const x = new Float64Array(na);
  const cols = defs.map(() => new Array(na));
  let j = 0;
  let cur = nb ? b.at(0) : null;
  let next = nb > 1 ? b.at(1) : null;
  for (let i = 0; i < na; i++) {
    const s = a.at(i);
    const t = s.t;
    x[i] = t;
    // Advance until `cur` is the last target sample at or before t.
    while (next && next.t <= t) {
      cur = next;
      j++;
      next = j + 1 < nb ? b.at(j + 1) : null;
    }
    let m = cur;
    if (next && (!m || Math.abs(next.t - t) < Math.abs(m.t - t))) m = next;
    if (!m || Math.abs(m.t - t) > REL_TOL_S) {
      for (const col of cols) col[i] = null;
      continue;
    }
    const g = relativeGeometry(s, m);
    for (let k = 0; k < defs.length; k++) {
      const v = defs[k].rel(g);
      cols[k][i] = Number.isFinite(v) ? v : null;
    }
  }
  return { x, cols };
}

let instances = 0;

export class Plots {
  // toolbar / body: the dock's #plotsToolbar and #plotsBody elements (owned
  // by this module). channels: initial channel keys (unknown keys dropped).
  // onSeek(simTime): a click on a chart. onChannelsChange(keys): the user
  // added or removed a channel.
  constructor({ toolbar, body, channels = DEFAULT_CHANNELS, onSeek = () => {}, onChannelsChange = () => {} }) {
    this.toolbar = toolbar;
    this.body = body;
    this.onSeek = onSeek;
    this.onChannelsChange = onChannelsChange;

    this._keys = sanitizeKeys(channels);
    this._theme = readTheme();
    // uPlot's sync registry is global; a per-instance key keeps it private.
    this._syncKey = `astrum-plots-${++instances}`;
    this._visible = true;

    this._rows = [];           // in display order
    this._layoutKey = null;    // selection/target presence + channel keys
    this._dirty = true;        // force a data rebuild on the next update
    this._built = { sel: null, tgt: null, selTrack: null, tgtTrack: null, frameCount: -1, mode: null, at: 0 };
    this._selColor = null;
    this._tgtColor = null;
    this._bounds = null;
    this._xFull = [0, 1];
    this._t0 = 0;              // tick labels count from here (bounds.min)
    this._zoom = null;         // [min, max] sim s, or null = follow the bounds
    this._hoverT = null;       // sim time under the synced cursor, or null
    this._valT = NaN;          // readout cache key: time + frame count
    this._valFc = -1;

    this._hint = el("p", "plots-hint");
    this._hint.hidden = true;
    body.replaceChildren(this._hint);

    this._buildToolbar();

    // Width and height both feed the chart sizes; the body's size changes with
    // the dock resize handle and with the window.
    this._ro = new ResizeObserver(() => { if (this._visible) this._sizeRows(); });
    this._ro.observe(body);
  }

  // Every animation frame. bounds = store.bounds(); colorOf(id) → CSS colour.
  update(store, { playhead, bounds, selectedId = null, targetId = null, colorOf }) {
    if (!this._visible) return;
    const sel = selectedId ?? null;
    const tgt = sel != null && targetId != null && targetId !== sel ? targetId : null;
    this._bounds = bounds;
    this._t0 = bounds && !bounds.empty ? bounds.min : 0;
    this._xFull = fullRange(bounds);

    const layoutKey = sel == null ? "none" : `${tgt != null}|${this._keys.join(",")}`;
    if (layoutKey !== this._layoutKey) {
      this._layoutKey = layoutKey;
      this._layout(sel != null, tgt != null);
    }
    if (sel == null) return;

    const selColor = colorOf ? colorOf(sel) : this._theme.text;
    const tgtColor = tgt == null ? null : colorOf ? colorOf(tgt) : this._theme.live;
    const selTrack = store.track(sel)?.samples ?? null;
    const tgtTrack = tgt != null ? store.track(tgt)?.samples ?? null : null;
    const fc = store.frameCount;
    const b = this._built;
    let rebuild = this._dirty || sel !== b.sel || tgt !== b.tgt
      || selTrack !== b.selTrack || tgtTrack !== b.tgtTrack || store.mode !== b.mode
      || selColor !== this._selColor || tgtColor !== this._tgtColor;
    if (!rebuild && fc !== b.frameCount) {
      rebuild = store.mode !== "live" || performance.now() - b.at >= LIVE_REFRESH_MS;
    }
    if (rebuild) {
      // A new run or file (not merely another aircraft): the zoom window is stale.
      if (store.mode !== b.mode || fc < b.frameCount || (sel === b.sel && selTrack !== b.selTrack)) {
        this._zoom = null;
      }
      this._selColor = selColor;
      this._tgtColor = tgtColor;
      this._rebuild(sel, tgt, selTrack, tgtTrack);
      Object.assign(b, { sel, tgt, selTrack, tgtTrack, frameCount: fc, mode: store.mode, at: performance.now() });
      this._dirty = false;
      this._valFc = -1;
    }

    this._updateReadouts(store, sel, tgt, fc, playhead);
    this._updatePlayheads(playhead);
  }

  // false while the dock is hidden: update() then does nothing. true: re-measure
  // the body and refresh everything on the next update().
  setVisible(visible) {
    this._visible = !!visible;
    if (!this._visible) {
      this._closeMenu(false);
      this._hoverT = null;
      return;
    }
    this._sizeRows();
    this._dirty = true;
    this._valFc = -1;
  }

  // ---- rows ----

  // Rebuild the row list for the current channels; rows whose series layout
  // is unchanged keep their chart.
  _layout(hasSel, hasTgt) {
    const old = new Map(this._rows.map((r) => [r.def.key, r]));
    const specs = [];
    if (hasSel) {
      for (const key of this._keys) {
        const def = DEF.get(key);
        specs.push({ def, chart: !def.relative || hasTgt, n: !def.relative && hasTgt ? 2 : 1 });
      }
    }
    // Only the bottom chart carries x tick labels.
    const lastChart = specs.findLastIndex((s) => s.chart);
    const rows = specs.map((s, i) => {
      const isLast = i === lastChart;
      const sig = s.chart ? `chart|${s.n}|${isLast}` : "hint";
      const prev = old.get(s.def.key);
      if (prev && prev.sig === sig) {
        old.delete(s.def.key);
        return prev;
      }
      return this._makeRow(s.def, sig, s.chart, s.n, isLast);
    });
    for (const row of old.values()) this._dropRow(row);
    this._rows = rows;

    this._hint.textContent = !hasSel ? "Select an aircraft to plot its telemetry"
      : rows.length ? "" : "Add a channel with + Channel";
    this._hint.hidden = rows.length > 0;
    this.body.append(...rows.map((r) => r.el)); // appending moves reused rows into order

    this._hoverT = null;
    this._dirty = true;
    this._valFc = -1;
    this._sizeRows();
  }

  _makeRow(def, sig, chart, n, isLast) {
    const rowEl = el("div", chart ? "plot-row" : "plot-row plot-row-hint");
    const head = el("div", "plot-head");
    const label = el("div", "plot-label", def.label);
    if (def.unit) label.append(" ", el("span", "plot-unit", def.unit));
    head.append(label);
    const row = {
      def, sig, chart, n, isLast, el: rowEl,
      vals: [], valText: [], chartEl: null, u: null, ph: null, phPx: undefined, w: 0, h: 0, rowH: 0,
    };
    if (chart) {
      const vals = el("div", "plot-values");
      for (let k = 0; k < n; k++) {
        const span = el("span", "plot-val", "—");
        vals.append(span);
        row.vals.push(span);
        row.valText.push("—");
      }
      head.append(vals);
      row.chartEl = el("div", "plot-chart");
      rowEl.append(head, row.chartEl);
    } else {
      rowEl.append(head, el("div", "plot-note", `Pin a target to plot ${def.hint}`));
    }
    return row;
  }

  _dropRow(row) {
    if (row.u) row.u.destroy();
    row.u = null;
    row.el.remove();
  }

  // Split the body height between the chart rows (hint rows are short) and
  // size the charts, creating any that don't exist yet.
  _sizeRows() {
    const rows = this._rows;
    if (!rows.length) return;
    const H = this.body.clientHeight;
    if (!H) return; // hidden
    const charts = rows.filter((r) => r.chart);
    const avail = H - (rows.length - charts.length) * HINT_ROW_H;
    const h = Math.max(MIN_ROW_H, Math.floor(avail / Math.max(1, charts.length)));
    for (const r of rows) {
      const rh = r.chart ? h : HINT_ROW_H;
      if (r.rowH !== rh) {
        r.rowH = rh;
        r.el.style.height = `${rh}px`;
      }
    }
    if (!charts.length) return;
    const w = charts[0].chartEl.clientWidth;
    if (!w) return;
    for (const r of charts) {
      if (!r.u) {
        this._createChart(r, w, h);
        this._dirty = true;
      } else if (r.w !== w || r.h !== h) {
        r.u.setSize({ width: w, height: h });
      }
      r.w = w;
      r.h = h;
    }
  }

  _createChart(row, w, h) {
    const t = this._theme;
    const line = (stroke, extra) => ({
      stroke, width: 1.5, points: { show: false }, ...(row.def.step ? { paths: STEPPED } : null), ...extra,
    });
    const series = [{}];
    if (row.def.relative) {
      series.push(line(t.live));
    } else {
      // Colours are read at draw time, so a reused chart follows a new selection.
      series.push(line(() => this._selColor ?? t.text));
      if (row.n === 2) series.push(line(() => this._tgtColor ?? t.live, { width: 1.25, dash: [5, 4] }));
    }
    const grid = { stroke: t.line, width: 1 };
    const xAxis = {
      stroke: t.muted, font: FONT, grid, space: 64, incrs: X_INCRS,
      size: row.isLast ? 20 : 2, gap: 4,
      ticks: { show: row.isLast, stroke: t.line, width: 1, size: 3 },
      splits: (u, axisIdx, min, max, incr) => this._xSplits(min, max, incr),
      values: row.isLast
        ? (u, splits, axisIdx, space, incr) => splits.map((v) => fmtClock(v - this._t0, incr))
        : (u, splits) => splits.map(() => ""),
    };
    const yAxis = {
      stroke: t.muted, font: FONT, grid, size: 46, gap: 4, space: 18,
      ticks: { stroke: t.line, width: 1, size: 3 },
      values: (u, splits) => splits.map(fmtTick),
    };
    const u = new uPlot({
      width: w,
      height: h,
      legend: { show: false },
      padding: [6, 12, row.isLast ? 0 : 4, 0],
      scales: { x: { time: false, auto: false } },
      series,
      axes: [xAxis, yAxis],
      cursor: {
        y: false,
        sync: { key: this._syncKey, scales: ["x", null] },
        // Drag draws the selection box on every chart; the zoom itself is
        // applied in _onRelease so all charts get the same range. Double-click
        // is ours too (reset to the full bounds, not to the data extent).
        drag: { x: true, y: false, setScale: false },
        bind: { dblclick: () => null },
        points: { size: 6, width: 1.5, fill: t.bg, stroke: (self, i) => self.series[i]._stroke },
      },
      hooks: { setCursor: [(self) => this._onCursor(self)] },
    }, series.map(() => []), row.chartEl);

    const ph = el("div", "plot-playhead");
    ph.style.display = "none";
    u.over.append(ph);
    u.over.addEventListener("mousedown", (e) => {
      if (e.button !== 0) return;
      const x0 = e.clientX;
      // Registered after uPlot's own document mouseup (added in its mousedown),
      // so the selection boxes are final when this runs.
      const onUp = (ev) => {
        document.removeEventListener("mouseup", onUp);
        this._onRelease(row, x0, ev.clientX);
      };
      document.addEventListener("mouseup", onUp);
    });
    u.over.addEventListener("dblclick", (e) => {
      if (e.button === 0) this._resetZoom();
    });
    row.u = u;
    row.ph = ph;
    row.phPx = undefined;
  }

  // x ticks counted from the timeline start, so labels are round M:SS values.
  _xSplits(min, max, incr) {
    const out = [];
    if (!(incr > 0) || !Number.isFinite(min) || !Number.isFinite(max)) return out;
    const t0 = this._t0;
    let k = Math.ceil((min - t0) / incr - 1e-9);
    for (let v = t0 + k * incr; v <= max + 1e-9 && out.length < 1000; v = t0 + ++k * incr) out.push(v);
    return out;
  }

  // ---- data ----

  _rebuild(sel, tgt, selTrack, tgtTrack) {
    const zoom = this._zoom;
    if (zoom && (zoom[1] <= this._xFull[0] || zoom[0] >= this._xFull[1])) this._zoom = null;
    const abs = [], rel = [];
    for (const row of this._rows) if (row.u) (row.def.relative ? rel : abs).push(row);
    const [lo, hi] = this._zoom ?? this._xFull;

    if (abs.length) {
      const withTgt = abs[0].n === 2;
      const { x, sa, sb } = mergeTracks(selTrack, tgtTrack, withTgt);
      for (const row of abs) {
        row.vals[0].style.color = this._selColor;
        row.vals[0].title = `Aircraft ${sel}`;
        const data = [x, column(sa, row.def)];
        if (withTgt) {
          data.push(column(sb, row.def));
          row.vals[1].style.color = this._tgtColor;
          row.vals[1].title = `Target ${tgt}`;
        }
        row.u.setData(data, false);
        row.u.setScale("x", { min: lo, max: hi });
      }
    }
    if (rel.length) {
      const { x, cols } = relativeColumns(selTrack, tgtTrack, rel.map((r) => r.def));
      rel.forEach((row, k) => {
        row.vals[0].style.color = this._theme.live;
        row.vals[0].title = `Aircraft ${sel} → target ${tgt}`;
        row.u.setData([x, cols[k]], false);
        row.u.setScale("x", { min: lo, max: hi });
      });
    }
  }

  // Row readouts at the hover time (while hovering) or the playhead.
  _updateReadouts(store, sel, tgt, fc, playhead) {
    const t = this._hoverT ?? playhead;
    if (t === this._valT && fc === this._valFc) return;
    this._valT = t;
    this._valFc = fc;
    const a = store.sampleAt(sel, t);
    const me = a && a.inRange ? a.s : null;
    let bandit = null;
    if (tgt != null) {
      const b = store.sampleAt(tgt, t);
      bandit = b && b.inRange ? b.s : null;
    }
    let g; // relative geometry, computed once if any row needs it
    for (const row of this._rows) {
      if (!row.chart) continue;
      const d = row.def;
      if (d.relative) {
        if (g === undefined) g = me && bandit ? relativeGeometry(me, bandit) : null;
        this._setVal(row, 0, g ? d.rel(g) : null);
      } else {
        this._setVal(row, 0, me ? d.get(me) : null);
        if (row.n === 2) this._setVal(row, 1, bandit ? d.get(bandit) : null);
      }
    }
  }

  _setVal(row, i, v) {
    const text = fmtValue(v, row.def.dp);
    if (row.valText[i] !== text) {
      row.valText[i] = text;
      row.vals[i].textContent = text;
    }
  }

  // Move each chart's playhead line; unzoomed, the view spans the whole
  // timeline, so a playhead a refresh ahead of the data is pinned to the edge.
  _updatePlayheads(playhead) {
    const zoomed = this._zoom !== null;
    for (const row of this._rows) {
      const u = row.u;
      if (!u) continue;
      const sx = u.scales.x;
      let v = null;
      if (sx.min != null && sx.max != null && Number.isFinite(playhead)) {
        let px = u.valToPos(playhead, "x");
        const maxPx = u.valToPos(sx.max, "x");
        if (Number.isFinite(px) && Number.isFinite(maxPx)) {
          if (!zoomed) px = clamp(px, 0, maxPx);
          if (px >= -0.5 && px <= maxPx + 0.5) v = Math.round(px * 2) / 2;
        }
      }
      if (v === row.phPx) continue;
      if (v === null) {
        row.ph.style.display = "none";
      } else {
        if (row.phPx == null) row.ph.style.display = "";
        row.ph.style.transform = `translateX(${v}px)`;
      }
      row.phPx = v;
    }
  }

  // ---- interaction ----

  _onCursor(u) {
    const left = u.cursor.left;
    this._hoverT = left != null && left >= 0 ? u.posToVal(left, "x") : null;
  }

  _onRelease(row, x0, x1) {
    const u = row.u;
    if (!u) return; // the row was rebuilt mid-drag
    const r = u.over.getBoundingClientRect();
    const toPx = (x) => clamp(x - r.left, 0, r.width);
    // Even a few pixels of travel left a selection box on every chart.
    for (const other of this._rows) other.u?.setSelect({ left: 0, top: 0, width: 0, height: 0 }, false);
    if (Math.abs(x1 - x0) < CLICK_PX) {
      let t = u.posToVal(toPx(x0), "x");
      const b = this._bounds;
      if (b && !b.empty) t = clamp(t, b.min, b.max);
      if (Number.isFinite(t)) this.onSeek(t);
      return;
    }
    const lo = u.posToVal(toPx(Math.min(x0, x1)), "x");
    const hi = u.posToVal(toPx(Math.max(x0, x1)), "x");
    if (hi - lo >= MIN_ZOOM_S) {
      this._zoom = [lo, hi];
      this._applyX();
    }
  }

  _resetZoom() {
    this._zoom = null;
    this._applyX();
  }

  _applyX() {
    const [lo, hi] = this._zoom ?? this._xFull;
    for (const row of this._rows) row.u?.setScale("x", { min: lo, max: hi });
  }

  // ---- toolbar: channel chips and the "+ Channel" menu ----

  _buildToolbar() {
    this._chips = el("div", "plots-chips");
    const add = this._addBtn = el("button", "mini plots-add", "+ Channel");
    add.type = "button";
    add.setAttribute("aria-haspopup", "menu");
    add.setAttribute("aria-expanded", "false");
    add.setAttribute("aria-controls", "plotsMenu");
    this.toolbar.replaceChildren(this._chips, add);

    // The menu opens upwards over the 3-D view; it lives on <body> so the
    // dock's overflow and backdrop-filter can't clip or re-anchor it.
    const menu = this._menu = el("div");
    menu.id = "plotsMenu";
    menu.setAttribute("role", "menu");
    menu.setAttribute("aria-label", "Add channel");
    menu.hidden = true;
    document.body.append(menu);
    this._menuOpen = false;

    this._chips.addEventListener("click", (e) => {
      const btn = e.target.closest("button[data-key]");
      if (btn) this._removeChannel(btn.dataset.key);
    });
    add.addEventListener("click", () => {
      if (this._menuOpen) this._closeMenu(true); else this._openMenu();
    });
    add.addEventListener("keydown", (e) => {
      if ((e.key === "ArrowDown" || e.key === "ArrowUp") && !this._menuOpen) {
        e.preventDefault();
        this._openMenu();
      }
    });
    menu.addEventListener("click", (e) => {
      const item = e.target.closest("button[data-key]");
      if (item) this._addChannel(item.dataset.key);
    });
    menu.addEventListener("keydown", (e) => this._menuKey(e));
    this._onDocPointer = (e) => {
      if (!menu.contains(e.target) && !add.contains(e.target)) this._closeMenu(false);
    };
    // Capture phase on window: Escape closes the menu without also reaching
    // the app's global shortcuts.
    this._onEscape = (e) => {
      if (e.key !== "Escape") return;
      e.preventDefault();
      e.stopPropagation();
      this._closeMenu(true);
    };
    this._onWinResize = () => this._closeMenu(false);

    this._renderChips();
  }

  _renderChips() {
    this._chips.replaceChildren(...this._keys.map((key) => {
      const d = DEF.get(key);
      const chip = el("span", d.relative ? "plots-chip is-relative" : "plots-chip");
      chip.append(el("span", "plots-chip-label", d.label));
      const x = el("button", "plots-chip-x", "×");
      x.type = "button";
      x.dataset.key = key;
      x.title = `Remove ${d.label}`;
      x.setAttribute("aria-label", `Remove ${d.label}`);
      chip.append(x);
      return chip;
    }));
    this._addBtn.disabled = this._keys.length === CHANNELS.length;
  }

  _channelsChanged() {
    this._renderChips();
    this.onChannelsChange(this._keys.slice());
  }

  _addChannel(key) {
    if (!DEF.has(key) || this._keys.includes(key)) return;
    this._keys = [...this._keys, key];
    this._closeMenu(true);
    this._channelsChanged();
  }

  _removeChannel(key) {
    const i = this._keys.indexOf(key);
    if (i < 0) return;
    this._keys = this._keys.filter((k) => k !== key);
    this._channelsChanged();
    // Keep keyboard focus in the toolbar: the next chip's ×, else "+ Channel".
    const xs = this._chips.querySelectorAll("button[data-key]");
    (xs[i] ?? xs[i - 1] ?? this._addBtn).focus();
  }

  _openMenu() {
    const active = new Set(this._keys);
    const item = (c) => {
      const b = el("button", "plots-menu-item");
      b.type = "button";
      b.dataset.key = c.key;
      b.setAttribute("role", "menuitem");
      b.append(el("span", "", c.label));
      if (c.unit) b.append(el("span", "plot-unit", c.unit));
      return b;
    };
    const abs = CHANNELS.filter((c) => !c.relative && !active.has(c.key)).map(item);
    const rel = CHANNELS.filter((c) => c.relative && !active.has(c.key)).map(item);
    if (!abs.length && !rel.length) return;
    const menu = this._menu;
    menu.replaceChildren(...abs);
    if (rel.length) {
      const head = el("div", "plots-menu-head", "vs target");
      head.id = "plotsMenuRelHead";
      const group = el("div", "plots-menu-group");
      group.setAttribute("role", "group");
      group.setAttribute("aria-labelledby", head.id);
      group.append(...rel);
      menu.append(head, group);
    }
    menu.hidden = false;
    this._menuOpen = true;
    this._addBtn.setAttribute("aria-expanded", "true");

    const r = this._addBtn.getBoundingClientRect();
    menu.style.maxHeight = `${Math.max(120, r.top - 12)}px`;
    menu.style.bottom = `${window.innerHeight - r.top + 4}px`;
    menu.style.left = `${clamp(r.left, 4, Math.max(4, window.innerWidth - menu.offsetWidth - 4))}px`;

    document.addEventListener("pointerdown", this._onDocPointer, true);
    window.addEventListener("keydown", this._onEscape, true);
    window.addEventListener("resize", this._onWinResize);
    menu.querySelector("button")?.focus();
  }

  _closeMenu(focusButton) {
    if (!this._menuOpen) return;
    this._menuOpen = false;
    this._menu.hidden = true;
    this._menu.replaceChildren();
    this._addBtn.setAttribute("aria-expanded", "false");
    document.removeEventListener("pointerdown", this._onDocPointer, true);
    window.removeEventListener("keydown", this._onEscape, true);
    window.removeEventListener("resize", this._onWinResize);
    if (focusButton) this._addBtn.focus();
  }

  // Arrow keys / Home / End move between items; Tab leaves the menu.
  _menuKey(e) {
    const items = [...this._menu.querySelectorAll("button")];
    const i = items.indexOf(document.activeElement);
    let next = null;
    if (e.key === "ArrowDown") next = items[(i + 1) % items.length];
    else if (e.key === "ArrowUp") next = items[i <= 0 ? items.length - 1 : i - 1];
    else if (e.key === "Home") next = items[0];
    else if (e.key === "End") next = items[items.length - 1];
    else if (e.key === "Tab") {
      e.preventDefault();
      this._closeMenu(true);
      return;
    }
    if (next) {
      e.preventDefault();
      next.focus();
    }
  }
}
