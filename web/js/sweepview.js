// The sweep page (sweep.html?id=<id>): one sweep flown from the Sweeps builder.
// Its state, counts and runs come from GET /api/sweeps/<id> (polled while it
// runs), its results table from GET /api/sweeps/<id>/results (refetched when
// the counts change, and once when it ends). Stop is DELETE /api/sweeps/current.
// Each run links to its bag in the viewer (view.html?sweep=<id>&run=<n>).
//
// Everything the server sends (names, reasons, statuses, cells) is untrusted
// and reaches the DOM as text only.

import { ApiError, getSweep, getSweepResults, stopSweep } from "./api.js";
import { $, h, showError } from "./dom.js";
import { fmtTime } from "./format.js";
import { Scatter, themeColour } from "./scatter.js";
import { SweepTracks } from "./sweeptracks.js";

const POLL_MS = 1000;
const RETRY_MAX_MS = 15000; // the longest wait before a failed read is retried
const RESULTS_TRIES = 5; // tries at the results once the sweep has ended
const RUN_STATES = ["queued", "running", "ok", "failed", "stopped"];
// Columns left out of the min / mean / max rows.
const NO_STATS = new Set(["run", "seed", "sample", "wall_s", "speed_x", "status", "bag"]);
const collator = new Intl.Collator(undefined, { numeric: true, sensitivity: "base" });

const el = {
  loadError: $("loadError"), content: $("content"), sweepId: $("sweepId"), scenario: $("scenario"),
  stateBadge: $("stateBadge"), stateReason: $("stateReason"), progress: $("progress"), progressText: $("progressText"),
  chipOk: $("chipOk"), chipFailed: $("chipFailed"), chipRunning: $("chipRunning"), chipQueued: $("chipQueued"),
  chipStopped: $("chipStopped"), mode: $("mode"), jobs: $("jobs"), wall: $("wall"), out: $("out"),
  stopBtn: $("stopBtn"), actionError: $("actionError"), pollError: $("pollError"), pollNote: $("pollNote"), strip: $("strip"),
  statusFilter: $("statusFilter"), resultsCount: $("resultsCount"), resultsNote: $("resultsNote"),
  resultsError: $("resultsError"), resultsWrap: $("resultsWrap"), results: $("results"),
  xAxis: $("xAxis"), yAxis: $("yAxis"), colourBy: $("colourBy"), compareEmpty: $("compareEmpty"),
  compareWrap: $("compareWrap"), scatter: $("scatter"), scatterTip: $("scatterTip"), ring: $("ring"),
};

const state = {
  id: null,
  view: null,        // the last SweepView
  wall: null,        // { s, at }: the wall time as read, and when (ms since the epoch)
  squareOf: new Map(), // run -> its square in the strip
  rows: new Map(), // run -> its table row: { tr, cells, replay, texts, classes, href }
  selected: null, // the selected run (1-based)
  results: null, // the last {columns, rows}
  resultsJson: null, // the last results as JSON: an identical reply renders nothing
  version: 0, // bumped with each new results: the footer's key
  columnsKey: null, // the columns the table rows were built for
  headKey: null, // the columns and sort the header was built for
  footKey: null, // the results version and filter the footer was built for
  resultsKey: null, // done / running counts at the last table fetch
  resultsFinal: false,
  resultsSeq: 0,
  sort: { col: "run", dir: 1 },
  filter: "all",
  plot: { x: null, y: null, colour: null }, // the user's picks (null: the default)
  shown: null,       // the axes and colour plotted, after defaults
  timer: 0,
  pollFails: 0, // failed polls in a row: the backoff
  polling: false, // a poll is in flight
  gone: false, // the sweep is unknown (404): polling has stopped
};

const allRuns = new SweepTracks({ onSelect: (n) => select(n) });

// The table's parts are patched in place, not rebuilt on each refresh.
const thead = h("thead");
const tbody = h("tbody");
const tfoot = h("tfoot");
el.results.replaceChildren(thead, tbody, tfoot);

// ---- the summary ----

// The orbit progress ring (decoration: the <progress> beside it carries the value):
// a track, an arc for the done fraction and a glowing satellite at the arc's head,
// with done / total in the middle. Built once; render() moves the arc and satellite.
const SVG_NS = "http://www.w3.org/2000/svg";
const RING_R = 50;
const RING_C = 2 * Math.PI * RING_R;

function svg(tag, attrs = {}, ...children) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  node.append(...children);
  return node;
}

const ring = (() => {
  const arc = svg("circle", {
    class: "ring-arc", cx: 60, cy: 60, r: RING_R, "stroke-dasharray": RING_C, "stroke-dashoffset": RING_C,
  });
  const sat = svg("g", { class: "ring-sat" },
    svg("circle", { class: "ring-sat-halo", cx: 60, cy: 60 - RING_R, r: 7 }),
    svg("circle", { class: "ring-sat-core", cx: 60, cy: 60 - RING_R, r: 3.2 }));
  const done = svg("tspan", { class: "ring-done" });
  const total = svg("tspan", { class: "ring-total", dx: 2 });
  el.ring.append(svg("svg", { viewBox: "0 0 120 120", width: 132, height: 132 },
    svg("circle", { class: "ring-orbit", cx: 60, cy: 60, r: 38 }),
    svg("circle", { class: "ring-track", cx: 60, cy: 60, r: RING_R }),
    arc,
    sat,
    svg("text", { class: "ring-text", x: 60, y: 60 }, done, total),
    svg("text", { class: "ring-label", x: 60, y: 76 }, "done")));
  return { arc, sat, done, total };
})();

function renderRing(done, total, sweepState) {
  const f = total > 0 ? Math.min(1, done / total) : 0;
  ring.arc.setAttribute("stroke-dashoffset", String(RING_C * (1 - f)));
  ring.sat.style.transform = `rotate(${f * 360}deg)`;
  ring.done.textContent = String(done);
  ring.total.textContent = `/${total}`;
  el.ring.dataset.state = sweepState;
}

function render(view) {
  state.view = view;
  const running = view.state === "running";
  el.sweepId.textContent = view.id;
  el.scenario.textContent = view.scenario;
  el.stateBadge.className = `state-badge ${view.state}`;
  el.stateBadge.textContent = view.state;
  el.stateReason.textContent = view.reason || "";
  const done = view.ok + view.failed + view.stopped;
  el.progress.max = view.total;
  el.progress.value = done;
  el.progressText.textContent = `${done} of ${view.total} done`;
  renderRing(done, view.total, view.state);
  el.chipOk.textContent = `ok ${view.ok}`;
  el.chipFailed.textContent = `failed ${view.failed}`;
  el.chipRunning.textContent = `running ${view.running}`;
  el.chipQueued.textContent = `queued ${view.queued}`;
  el.chipStopped.textContent = `stopped ${view.stopped}`;
  el.mode.textContent = view.in_process ? "in-process" : "processes";
  el.jobs.textContent = String(view.jobs);
  state.wall = { s: view.wall_s ?? 0, at: Date.now() };
  el.wall.textContent = fmtTime(state.wall.s);
  el.out.textContent = view.out;
  el.stopBtn.hidden = !running;
  renderStrip(view.runs);
  allRuns.update(view);
  refreshResults(view, running);
}

// The clock between polls: a running sweep's wall time, counted up locally.
setInterval(() => {
  const view = state.view;
  if (view?.state === "running" && state.wall && !document.hidden) {
    el.wall.textContent = fmtTime(state.wall.s + (Date.now() - state.wall.at) / 1000);
  }
}, 1000);

// ---- the run strip ----

const runLabel = (run) => `run ${run.run}: ${run.status}${run.wall_s == null ? "" : `, ${run.wall_s.toFixed(1)} s`}`;

// One square per run, made once; later polls change only the squares whose status changed.
function renderStrip(runs) {
  if (state.squareOf.size !== runs.length) {
    state.squareOf = new Map();
    el.strip.replaceChildren(...runs.map((run) => {
      const square = h("button", { type: "button", class: "run-cell", "aria-pressed": "false" });
      square.addEventListener("click", () => select(run.run, { scroll: true }));
      state.squareOf.set(run.run, square);
      return square;
    }));
    paint(state.selected, true);
  }
  for (const run of runs) {
    const square = state.squareOf.get(run.run);
    if (square.dataset.status !== run.status) {
      for (const s of RUN_STATES) square.classList.toggle(s, s === run.status);
      square.dataset.status = run.status;
    }
    const label = runLabel(run);
    if (square.getAttribute("aria-label") !== label) square.setAttribute("aria-label", label);
    const title = run.reason ? `${label} — ${run.reason}` : label;
    if (square.title !== title) square.title = title;
  }
}

// Mark run `n`'s square and table row as selected, or not.
function paint(n, on) {
  const square = state.squareOf.get(n);
  if (square) {
    square.classList.toggle("selected", on);
    square.setAttribute("aria-pressed", String(on));
  }
  state.rows.get(n)?.tr.classList.toggle("selected", on);
}

// Select run `n` (from the strip, the table, the scatter, the paths or the arrow
// keys). Only a click on a strip square scrolls its table row into view.
function select(n, { scroll = false } = {}) {
  if (n !== state.selected) {
    const prev = state.selected;
    state.selected = n;
    paint(prev, false);
    paint(n, true);
    allRuns.setSelected(n);
    drawPlot();
  }
  if (scroll) state.rows.get(n)?.tr.scrollIntoView({ block: "nearest" });
}

// ---- the results ----

// The table is refetched when the counts change while the sweep runs, and once
// when it ends.
function refreshResults(view, running) {
  const key = `${view.ok + view.failed + view.stopped}/${view.running}`;
  if (running ? key === state.resultsKey : state.resultsFinal) return;
  if (!running) state.resultsFinal = true;
  state.resultsKey = key;
  loadResults();
}

// A failed fetch once the sweep has ended is retried with the poll's backoff, up to
// RESULTS_TRIES tries; while it runs, the next poll asks again.
async function loadResults(tries = 1) {
  const seq = ++state.resultsSeq;
  try {
    const results = await getSweepResults(state.id);
    if (seq !== state.resultsSeq) return;
    showError(el.resultsError, "");
    showError(el.resultsNote, "");
    const json = JSON.stringify(results);
    if (json !== state.resultsJson) {
      state.resultsJson = json;
      state.results = results;
      state.version++;
      renderCompare();
      renderTable();
    }
  } catch (err) {
    if (seq !== state.resultsSeq) return;
    if (err instanceof ApiError && err.status === 404) {
      showError(el.resultsNote, "No results yet: the table appears once the sweep has written them.");
    } else if (state.view?.state === "running") {
      state.resultsKey = null;
      showError(el.resultsError, `Cannot read the results: ${err.message || err}`);
    } else if (tries < RESULTS_TRIES) {
      showError(el.resultsNote, "Connection problem — retrying…");
      setTimeout(() => loadResults(tries + 1), backoff(tries));
    } else {
      showError(el.resultsError, `Cannot read the results: ${err.message || err}`);
    }
  }
}

const statusOf = (row, at) => String(row[at] ?? "");

// Whether a row passes the status filter: ok, failed, or other (the rest).
function passes(status) {
  if (state.filter === "all") return true;
  if (state.filter === "ok") return status.startsWith("ok");
  if (state.filter === "failed") return status.startsWith("failed");
  return !status.startsWith("ok") && !status.startsWith("failed");
}

// The rows the filter keeps, sorted: numbers numerically, text naturally,
// empty cells last whichever way the sort runs.
function visibleRows() {
  const { columns, rows } = state.results;
  const statusAt = columns.indexOf("status");
  const at = columns.indexOf(state.sort.col);
  const dir = state.sort.dir;
  return rows
    .filter((row) => passes(statusOf(row, statusAt)))
    .sort((a, b) => {
      const x = a[at], y = b[at];
      if (x == null || y == null) return (x == null) - (y == null);
      const c = typeof x === "number" && typeof y === "number" ? x - y : collator.compare(String(x), String(y));
      return c * dir;
    });
}

// A cell's text: a dash for no value; a whole number without decimals; other numbers
// to 3 decimals, or to 3 significant digits below 0.001.
function cellText(v) {
  if (v == null) return "–";
  if (typeof v !== "number") return String(v);
  if (Number.isInteger(v)) return String(v);
  return Math.abs(v) < 0.001 ? v.toPrecision(3) : v.toFixed(3);
}

function sortBy(column) {
  state.sort = state.sort.col === column ? { col: column, dir: -state.sort.dir } : { col: column, dir: 1 };
  renderTable();
}

// The table: the header is rebuilt only when its columns or sort change; one row per
// run is patched in place; rows move only when the visible order changes; the footer
// is rebuilt only when the results or the filter change. The bag column is not shown;
// a row that was flown and has a bag links to its replay.
function renderTable() {
  if (!state.results) return;
  const { columns, rows } = state.results;
  const shown = columns.map((_, i) => i).filter((i) => columns[i] !== "bag");
  const statusAt = columns.indexOf("status");
  const bagAt = columns.indexOf("bag");
  const runAt = columns.indexOf("run");
  const visible = visibleRows();

  const columnsKey = columns.join("\u0000");
  if (columnsKey !== state.columnsKey) {
    state.columnsKey = columnsKey;
    state.headKey = null;
    state.rows.clear();
    tbody.replaceChildren();
  }
  const headKey = `${state.sort.col}\u0000${state.sort.dir}`;
  if (headKey !== state.headKey) {
    state.headKey = headKey;
    renderHead(columns, shown);
  }

  const keep = new Set(visible.map((row) => row[runAt]));
  for (const row of visible) {
    const run = row[runAt];
    let entry = state.rows.get(run);
    if (!entry) {
      entry = newRow(run, shown.length);
      state.rows.set(run, entry);
    }
    patchRow(entry, row, shown, statusAt, bagAt, run);
  }
  for (const [run, entry] of state.rows) if (!keep.has(run)) entry.tr.remove();
  visible.forEach((row, i) => {
    const tr = state.rows.get(row[runAt]).tr;
    if (tbody.children[i] !== tr) tbody.insertBefore(tr, tbody.children[i] ?? null);
  });

  const footKey = `${state.version}\u0000${state.filter}`;
  if (footKey !== state.footKey) {
    state.footKey = footKey;
    renderFoot(columns, rows, visible, shown);
  }
  el.resultsWrap.hidden = false;
  el.resultsCount.textContent = `${visible.length} of ${rows.length} runs`;
}

// The header: a sort button per shown column, and the Replay column. Focus stays on
// the sort button it was on across the rebuild.
function renderHead(columns, shown) {
  const focused = thead.contains(document.activeElement) ? document.activeElement.dataset.column : null;
  thead.replaceChildren(h("tr", {},
    ...shown.map((i) => {
      const name = columns[i];
      const button = h("button", { type: "button", class: "sort", text: name, "data-column": name });
      button.addEventListener("click", () => sortBy(name));
      const aria = state.sort.col !== name ? "none" : state.sort.dir > 0 ? "ascending" : "descending";
      return h("th", { scope: "col", "aria-sort": aria }, button);
    }),
    h("th", { scope: "col", text: "Replay" })));
  if (focused != null) [...thead.querySelectorAll("button")].find((b) => b.dataset.column === focused)?.focus();
}

// The cells of a new row for run `run`, patched by patchRow from then on.
function newRow(run, count) {
  const tr = h("tr", { "data-run": String(run) });
  const cells = Array.from({ length: count }, () => h("td"));
  const replay = h("td");
  tr.append(...cells, replay);
  if (run === state.selected) tr.classList.add("selected");
  return { tr, cells, replay, texts: [], classes: [], href: "" };
}

// Set a row's cells from its values: only a cell whose text or class changed is touched.
function patchRow(entry, row, shown, statusAt, bagAt, run) {
  const status = statusOf(row, statusAt);
  shown.forEach((i, j) => {
    const v = row[i];
    const text = cellText(v);
    const cls = i === statusAt ? "status" : typeof v === "number" ? "num" : "";
    const td = entry.cells[j];
    if (entry.classes[j] !== cls) {
      td.className = cls;
      entry.classes[j] = cls;
    }
    if (entry.texts[j] !== text) {
      td.textContent = text;
      entry.texts[j] = text;
      if (i === statusAt) td.title = status;
    }
  });
  const bag = bagAt >= 0 ? row[bagAt] : null;
  const replayable = (status.startsWith("ok") || status.startsWith("stopped")) && bag != null && String(bag) !== "";
  const href = replayable ? `./view.html?sweep=${encodeURIComponent(state.id)}&run=${encodeURIComponent(String(run))}` : "";
  if (entry.href !== href) {
    entry.href = href;
    entry.replay.replaceChildren(...(href ? [h("a", { href, text: "Replay" })] : []));
  }
}

// The footer: min, mean and max over the visible ok rows, for each numeric column
// the stats do not skip (NO_STATS).
function renderFoot(columns, rows, visible, shown) {
  const statusAt = columns.indexOf("status");
  const okRows = visible.filter((row) => statusOf(row, statusAt) === "ok");
  const stat = (i, pick) => {
    const column = columns[i];
    if (NO_STATS.has(column)) return "";
    const numeric = rows.some((row) => typeof row[i] === "number");
    if (!numeric) return "";
    const values = okRows.map((row) => row[i]).filter((v) => typeof v === "number");
    return values.length ? cellText(pick(values)) : "–";
  };
  const min = (vs) => vs.reduce((a, b) => Math.min(a, b));
  const max = (vs) => vs.reduce((a, b) => Math.max(a, b));
  const mean = (vs) => vs.reduce((a, b) => a + b, 0) / vs.length;
  const foot = (label, pick) => h("tr", {},
    ...shown.map((i, k) => (k === 0
      ? h("th", { scope: "row", text: label })
      : h("td", { class: "num", text: stat(i, pick) }))));
  tfoot.replaceChildren(foot("min", min), foot("mean", mean), foot("max", max));
}

// ---- compare runs ----

// Numeric columns no plot uses as an axis or colour: the seed and sample, the
// bag, the wall time and the speed.
const NOT_PLOTTED = new Set(["seed", "sample", "bag", "wall_s", "speed_x", "status"]);
// The y axis and the colour leave out `run` too: an index, not a result.
const NOT_Y = new Set([...NOT_PLOTTED, "run"]);
// The numeric fleet metrics in catalog order (sim_time_s aside): the y axis
// defaults to the first of them with a number, else to sim_time_s.
const METRIC_COLUMNS = [
  "min_separation_m", "min_alt_hae_m", "max_bank_deg", "max_nz_g", "failsafe_entries",
  "controller_losses", "command_refusals", "hits", "kills",
];

const scatter = new Scatter(el.scatter, { onHover: showTip, onPick: (data) => select(data.run) });

// The columns with a number in at least one row, minus `skip`.
function numericColumns(columns, rows, skip) {
  return columns.filter((name, i) => !skip.has(name) && rows.some((row) => typeof row[i] === "number"));
}

// Refill the three selects (rebuilt only when their options change, so an open
// popup survives a refresh), keeping the user's picks that still exist, then plot.
function renderCompare() {
  const { columns, rows } = state.results;
  const xs = numericColumns(columns, rows, NOT_PLOTTED);
  const ys = numericColumns(columns, rows, NOT_Y);
  const colours = ["status", ...ys];
  const choose = (kind, options, fallback) => (options.includes(state.plot[kind]) ? state.plot[kind] : fallback);
  state.shown = {
    x: choose("x", xs, xs.find((c) => c.startsWith("aircraft.")) ?? (xs.includes("run") ? "run" : xs[0])),
    y: choose("y", ys, METRIC_COLUMNS.find((c) => ys.includes(c)) ?? (ys.includes("sim_time_s") ? "sim_time_s" : ys[0])),
    colour: choose("colour", colours, "status"),
  };
  fillSelect(el.xAxis, xs, state.shown.x);
  fillSelect(el.yAxis, ys, state.shown.y);
  fillSelect(el.colourBy, colours, state.shown.colour);
  drawPlot();
}

function fillSelect(select, options, value) {
  const key = options.join("\n");
  if (select.dataset.options !== key) {
    select.replaceChildren(...options.map((o) => h("option", { value: o, text: o })));
    select.dataset.options = key;
  }
  select.value = value ?? "";
  select.disabled = options.length === 0;
}

const hexRgb = (hex) => {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
};

// The colour at t in [0, 1] on the ramp from `from` to `to` (both "#rrggbb").
function rampColour(from, to) {
  const a = hexRgb(from);
  const b = hexRgb(to);
  return (t) => `rgb(${a.map((c, i) => Math.round(c + (b[i] - c) * t)).join(", ")})`;
}

// The points of the shown axes: every run with a number for both, coloured by
// status (ok, failed, other) or along the ramp of a numeric colour column; the
// selected run is flagged.
function drawPlot() {
  const shown = state.shown;
  if (!shown) return;
  const { columns, rows } = state.results;
  const xi = columns.indexOf(shown.x);
  const yi = columns.indexOf(shown.y);
  const ci = shown.colour === "status" ? -1 : columns.indexOf(shown.colour);
  const runAt = columns.indexOf("run");
  const statusAt = columns.indexOf("status");
  const placed = rows.filter((row) => typeof row[xi] === "number" && typeof row[yi] === "number");

  let colourOf;
  let legend;
  if (ci < 0) {
    const ok = themeColour("--ok");
    const failed = themeColour("--live");
    const other = themeColour("--warn");
    colourOf = (row) => {
      const status = statusOf(row, statusAt);
      return status.startsWith("ok") ? ok : status.startsWith("failed") ? failed : other;
    };
    legend = [{ label: "ok", color: ok }, { label: "failed", color: failed }, { label: "other", color: other }];
  } else {
    const numbers = placed.map((row) => row[ci]).filter((v) => typeof v === "number");
    const lo = numbers.length ? Math.min(...numbers) : 0;
    const hi = numbers.length ? Math.max(...numbers) : 0;
    const ramp = rampColour(themeColour("--accent"), themeColour("--warn"));
    const muted = themeColour("--muted");
    colourOf = (row) => {
      const v = row[ci];
      return typeof v === "number" ? ramp(hi > lo ? (v - lo) / (hi - lo) : 0) : muted;
    };
    legend = numbers.length
      ? [{ label: `min ${cellText(lo)}`, color: ramp(0) }, { label: `max ${cellText(hi)}`, color: ramp(1) }]
      : [];
  }

  const points = placed.map((row) => ({
    x: row[xi],
    y: row[yi],
    color: colourOf(row),
    selected: row[runAt] === state.selected,
    data: { run: row[runAt], status: statusOf(row, statusAt), x: row[xi], y: row[yi] },
  }));
  scatter.setData({ xLabel: shown.x ?? "", yLabel: shown.y ?? "", points, legend });

  const chosen = points.find((p) => p.selected);
  el.scatter.setAttribute("aria-label", points.length
    ? `${shown.y} against ${shown.x}, ${points.length} runs${chosen ? `, run ${chosen.data.run} selected` : ""}`
    : "Scatter plot of the runs");
  el.compareEmpty.hidden = points.length > 0;
  el.compareWrap.hidden = points.length === 0;
  if (!points.length) showTip(null, null);
}

// The hover text, beside the pointer: text only.
function showTip(data, at) {
  el.scatterTip.hidden = !data;
  if (!data) return;
  const { x, y } = state.shown;
  el.scatterTip.replaceChildren(
    h("div", { text: `run ${data.run} · ${data.status}` }),
    h("div", { text: `${x} = ${cellText(data.x)}` }),
    h("div", { text: `${y} = ${cellText(data.y)}` }));
  el.scatterTip.classList.toggle("left", at.x > el.compareWrap.clientWidth / 2);
  el.scatterTip.style.left = `${at.x}px`;
  el.scatterTip.style.top = `${at.y}px`;
}

// ---- stop ----

el.stopBtn.addEventListener("click", async () => {
  el.stopBtn.disabled = true;
  showError(el.actionError, "");
  try {
    render(await stopSweep());
  } catch (err) {
    showError(el.actionError, `stop: ${err.message || err}`);
  }
  el.stopBtn.disabled = false;
});

el.statusFilter.addEventListener("change", () => {
  state.filter = el.statusFilter.value;
  renderTable();
});
el.results.addEventListener("click", (e) => {
  const row = e.target.closest("tbody tr[data-run]");
  if (row) select(Number(row.dataset.run));
});

el.xAxis.addEventListener("change", () => { state.plot.x = el.xAxis.value; renderCompare(); });
el.yAxis.addEventListener("change", () => { state.plot.y = el.yAxis.value; renderCompare(); });
el.colourBy.addEventListener("change", () => { state.plot.colour = el.colourBy.value; renderCompare(); });

// ---- polling ----

// The wait before retry `n` (1-based) after a failed read: 1 s, 2 s, 4 s, capped.
const backoff = (n) => Math.min(POLL_MS * 2 ** (n - 1), RETRY_MAX_MS);

async function poll() {
  clearTimeout(state.timer);
  state.timer = 0;
  if (document.hidden || state.polling || state.gone) return; // a hidden tab resumes on visibilitychange
  state.polling = true;
  let view;
  try {
    view = await getSweep(state.id);
  } catch (err) {
    state.polling = false;
    if (err instanceof ApiError && err.status === 404) {
      state.gone = true;
      showError(el.pollError, `Cannot read the sweep: ${err.message || err}`);
      return;
    }
    state.pollFails++;
    showError(el.pollNote, "Connection problem — retrying…");
    keepPolling(backoff(state.pollFails));
    return;
  }
  state.polling = false;
  state.pollFails = 0;
  showError(el.pollNote, "");
  render(view);
  keepPolling(POLL_MS);
}

function keepPolling(delay) {
  if (state.view?.state === "running") state.timer = setTimeout(poll, delay);
}

document.addEventListener("visibilitychange", () => {
  if (!document.hidden && state.view?.state === "running") poll();
});

// The page is leaving: stop the poll and the canvases' loops and observers.
window.addEventListener("pagehide", () => {
  clearTimeout(state.timer);
  allRuns.destroy();
  scatter.destroy();
});

// ---- boot ----

function fail(msg) {
  showError(el.loadError, msg);
  el.loadError.append(" ", h("a", { href: "./sweeps.html", text: "Back to the sweeps" }));
}

async function boot() {
  const id = new URLSearchParams(location.search).get("id");
  if (!id) {
    fail("No sweep id in the address: open a sweep from the Sweeps page.");
    return;
  }
  state.id = id;
  try {
    render(await getSweep(id));
  } catch (err) {
    fail(err.message || String(err));
    return;
  }
  el.content.hidden = false;
  keepPolling(POLL_MS);
}

boot();
