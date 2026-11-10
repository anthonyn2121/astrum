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

const POLL_MS = 1000;
const RUN_STATES = ["queued", "running", "ok", "failed", "stopped"];
// Columns with whole-number cells, shown unformatted; the rest show 3 decimals.
const INTEGER_COLUMNS = new Set(["run", "seed", "sample"]);
// Columns left out of the min / mean / max rows.
const NO_STATS = new Set(["run", "seed", "sample", "wall_s", "speed_x", "status", "bag"]);
const collator = new Intl.Collator(undefined, { numeric: true, sensitivity: "base" });

const el = {
  loadError: $("loadError"), content: $("content"), sweepId: $("sweepId"), scenario: $("scenario"),
  stateBadge: $("stateBadge"), stateReason: $("stateReason"), progress: $("progress"), progressText: $("progressText"),
  chipOk: $("chipOk"), chipFailed: $("chipFailed"), chipRunning: $("chipRunning"), chipQueued: $("chipQueued"),
  chipStopped: $("chipStopped"), mode: $("mode"), jobs: $("jobs"), wall: $("wall"), out: $("out"),
  stopBtn: $("stopBtn"), actionError: $("actionError"), pollError: $("pollError"), strip: $("strip"),
  statusFilter: $("statusFilter"), resultsCount: $("resultsCount"), resultsNote: $("resultsNote"),
  resultsError: $("resultsError"), resultsWrap: $("resultsWrap"), results: $("results"),
};

const state = {
  id: null,
  view: null,        // the last SweepView
  wall: null,        // { s, at }: the wall time as read, and when (ms since the epoch)
  squares: 0,        // runs with a square in the strip
  selected: null,    // the selected run (1-based)
  results: null,     // the last {columns, rows}
  resultsKey: null,  // done / running counts at the last table fetch
  resultsFinal: false,
  resultsSeq: 0,
  sort: { col: "run", dir: 1 },
  filter: "all",
  timer: 0,
};

// ---- the summary ----

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
  refreshResults(view, running);
}

// The clock between polls: a running sweep's wall time, counted up locally.
setInterval(() => {
  const view = state.view;
  if (view?.state === "running" && state.wall) {
    el.wall.textContent = fmtTime(state.wall.s + (Date.now() - state.wall.at) / 1000);
  }
}, 1000);

// ---- the run strip ----

const runLabel = (run) => `run ${run.run}: ${run.status}${run.wall_s == null ? "" : `, ${run.wall_s.toFixed(1)} s`}`;

// One square per run, made once; later polls only change their classes and names.
function renderStrip(runs) {
  if (state.squares !== runs.length) {
    el.strip.replaceChildren(...runs.map((run) => {
      const square = h("button", { type: "button", class: "run-cell", "data-run": String(run.run), "aria-pressed": "false" });
      square.addEventListener("click", () => select(run.run));
      return square;
    }));
    state.squares = runs.length;
  }
  runs.forEach((run) => {
    const square = el.strip.children[run.run - 1];
    for (const s of RUN_STATES) square.classList.toggle(s, s === run.status);
    const label = runLabel(run);
    if (square.getAttribute("aria-label") !== label) square.setAttribute("aria-label", label);
    const title = run.reason ? `${label} — ${run.reason}` : label;
    if (square.title !== title) square.title = title;
  });
  markSelected();
}

function markSelected() {
  for (const square of el.strip.children) {
    const on = Number(square.dataset.run) === state.selected;
    square.classList.toggle("selected", on);
    square.setAttribute("aria-pressed", String(on));
  }
  for (const row of el.results.querySelectorAll("tbody tr")) {
    row.classList.toggle("selected", Number(row.dataset.run) === state.selected);
  }
}

// Select run `n` (from the strip or the table): its row is brought into view.
function select(n) {
  state.selected = n;
  markSelected();
  el.results.querySelector(`tbody tr[data-run="${n}"]`)?.scrollIntoView({ block: "nearest" });
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

async function loadResults() {
  const seq = ++state.resultsSeq;
  try {
    const results = await getSweepResults(state.id);
    if (seq !== state.resultsSeq) return;
    state.results = results;
    showError(el.resultsError, "");
    showError(el.resultsNote, "");
    renderTable();
  } catch (err) {
    if (seq !== state.resultsSeq) return;
    if (err instanceof ApiError && err.status === 404) {
      showError(el.resultsNote, "No results yet: the table appears once the sweep has written them.");
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

function cellText(column, v) {
  if (v == null) return "–";
  if (typeof v === "number") return INTEGER_COLUMNS.has(column) ? String(v) : v.toFixed(3);
  return String(v);
}

function sortBy(column) {
  state.sort = state.sort.col === column ? { col: column, dir: -state.sort.dir } : { col: column, dir: 1 };
  renderTable();
}

// The whole table: header, the visible rows, and min / mean / max over the
// visible `ok` rows. The bag column is not shown; a row that was flown and has
// a bag links to its replay.
function renderTable() {
  if (!state.results) return;
  const { columns, rows } = state.results;
  const shown = columns.map((_, i) => i).filter((i) => columns[i] !== "bag");
  const statusAt = columns.indexOf("status");
  const bagAt = columns.indexOf("bag");
  const runAt = columns.indexOf("run");
  const visible = visibleRows();
  const okRows = visible.filter((row) => statusOf(row, statusAt) === "ok");

  const head = h("tr", {},
    ...shown.map((i) => {
      const name = columns[i];
      const button = h("button", { type: "button", class: "sort", text: name });
      button.addEventListener("click", () => sortBy(name));
      const aria = state.sort.col !== name ? "none" : state.sort.dir > 0 ? "ascending" : "descending";
      return h("th", { scope: "col", "aria-sort": aria }, button);
    }),
    h("th", { scope: "col", text: "Replay" }));

  const body = visible.map((row) => {
    const run = row[runAt];
    const status = statusOf(row, statusAt);
    const bag = bagAt >= 0 ? row[bagAt] : null;
    const replayable = (status.startsWith("ok") || status.startsWith("stopped")) && bag != null && String(bag) !== "";
    const cells = shown.map((i) => {
      const v = row[i];
      const cls = i === statusAt ? "status" : typeof v === "number" ? "num" : "";
      return h("td", { class: cls, title: i === statusAt ? status : null, text: cellText(columns[i], v) });
    });
    cells.push(h("td", {}, replayable
      ? h("a", { href: `./view.html?sweep=${encodeURIComponent(state.id)}&run=${run}`, text: "Replay" })
      : null));
    return h("tr", { "data-run": String(run) }, ...cells);
  });

  const stat = (i, pick) => {
    const column = columns[i];
    if (NO_STATS.has(column)) return "";
    const numeric = rows.some((row) => typeof row[i] === "number");
    if (!numeric) return "";
    const values = okRows.map((row) => row[i]).filter((v) => typeof v === "number");
    return values.length ? cellText(column, pick(values)) : "–";
  };
  const min = (vs) => vs.reduce((a, b) => Math.min(a, b));
  const max = (vs) => vs.reduce((a, b) => Math.max(a, b));
  const mean = (vs) => vs.reduce((a, b) => a + b, 0) / vs.length;
  const foot = (label, pick) => h("tr", {},
    ...shown.map((i, k) => (k === 0
      ? h("th", { scope: "row", text: label })
      : h("td", { class: "num", text: stat(i, pick) }))));

  el.results.replaceChildren(
    h("thead", {}, head),
    h("tbody", {}, ...body),
    h("tfoot", {}, foot("min", min), foot("mean", mean), foot("max", max)));
  el.resultsWrap.hidden = false;
  el.resultsCount.textContent = `${visible.length} of ${rows.length} runs`;
  markSelected();
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

// ---- polling ----

async function poll() {
  clearTimeout(state.timer);
  if (document.hidden) return; // resumed by visibilitychange
  try {
    render(await getSweep(state.id));
    showError(el.pollError, "");
  } catch (err) {
    showError(el.pollError, `Cannot read the sweep: ${err.message || err}`);
  }
  keepPolling();
}

function keepPolling() {
  if (state.view?.state === "running") state.timer = setTimeout(poll, POLL_MS);
}

document.addEventListener("visibilitychange", () => {
  if (!document.hidden && state.view?.state === "running") poll();
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
  keepPolling();
}

boot();
