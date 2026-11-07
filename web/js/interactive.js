// The Interactive setup page: environment → scenario → a catalog controller
// (and its parameters) per aircraft → speed / max time → POST /api/runs, then
// on to the viewer (view.html?run=<id>). One live run at a time: while one is
// active it is shown, with links to open or stop it, instead of a new start.
//
// The form mirrors the server's checks (types, bounds, choices, fleet ids)
// so most mistakes show before a request; the server's verdict is shown next
// to the parameter or aircraft it names. Scenario names, summaries, errors
// and catalog text are untrusted: they reach the DOM as text only.

import {
  getEnvironments, getScenarios, getScenario, getControllers, getCurrentRun, startRun, stopRun, ApiError,
} from "./api.js";

const ACTIVE_POLL_MS = 2000;
const TARGET_UNITS = { heading_deg: "°", alt_hae_m: "m", tas_mps: "m/s" };

const $ = (id) => document.getElementById(id);

// A DOM element; string children become text nodes.
function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v === true ? "" : String(v));
  }
  for (const c of children) if (c != null) node.append(c);
  return node;
}

const el = {
  active: $("activeRun"), activeId: $("activeId"), activeMeta: $("activeMeta"), activeOpen: $("activeOpen"),
  activeStop: $("activeStop"), activeError: $("activeError"), loadError: $("loadError"),
  form: $("setup"), fields: $("setupFields"), envList: $("envList"), scenarioList: $("scenarioList"),
  scenarioEmpty: $("scenarioEmpty"), fleet: $("fleet"), fleetEmpty: $("fleetEmpty"),
  speedSeg: $("speedSeg"), maxTime: $("maxTime"), maxTimeHint: $("maxTimeHint"), maxTimeError: $("maxTimeError"),
  formError: $("formError"), runBtn: $("runBtn"), runNote: $("runNote"),
};

const state = {
  environments: [],
  scenarios: [],
  controllers: [],
  envId: null,
  scenario: null,  // the chosen ScenarioEntry
  doc: null,       // its ScenarioDoc (initial conditions, sim settings)
  rows: new Map(), // aircraft id -> { aircraft, select, paramsBox, error, fields: Map name -> field }
  speed: 1,
  activeRun: null, // the active RunView, if any
  starting: false,
};

function showError(node, msg) {
  node.textContent = msg || "";
  node.hidden = !msg;
}

// ---- catalog lists ----

function renderEnvironments() {
  el.envList.replaceChildren();
  for (const env of state.environments) {
    const input = h("input", { type: "radio", name: "env", value: env.id });
    input.checked = env.id === state.envId;
    input.addEventListener("change", () => { state.envId = env.id; renderScenarios(); selectScenario(null); });
    el.envList.append(h("label", { class: "choice" },
      input,
      h("span", { class: "choice-body" },
        h("span", { class: "choice-title", text: env.name }),
        h("span", { class: "choice-text muted", text: env.description }))));
  }
}

function fleetSummary(aircraft) {
  const counts = new Map();
  for (const a of aircraft) counts.set(a.model, (counts.get(a.model) || 0) + 1);
  const models = [...counts].map(([m, n]) => (n > 1 ? `${m} ×${n}` : m)).join(", ");
  return `${aircraft.length} aircraft${models ? `: ${models}` : ""}`;
}

function renderScenarios() {
  el.scenarioList.replaceChildren();
  const list = state.scenarios.filter((s) => s.environment === state.envId);
  el.scenarioEmpty.hidden = list.length > 0;
  for (const s of list) {
    const invalid = typeof s.error === "string";
    const input = h("input", { type: "radio", name: "scenario", value: s.name, disabled: invalid });
    input.checked = state.scenario?.name === s.name;
    input.addEventListener("change", () => selectScenario(s));
    const body = h("span", { class: "choice-body" },
      h("span", { class: "choice-title", text: s.name }),
      s.summary ? h("span", { class: "choice-text", text: s.summary }) : null,
      invalid
        ? h("span", { class: "choice-text field-error", text: s.error })
        : h("span", { class: "choice-text muted", text: fleetSummary(s.aircraft) }));
    el.scenarioList.append(h("label", { class: invalid ? "choice disabled" : "choice", title: s.path }, input, body));
  }
}

async function selectScenario(entry) {
  state.scenario = entry;
  state.doc = null;
  state.rows = new Map();
  el.fleet.replaceChildren();
  el.fleetEmpty.hidden = !!entry;
  el.fleetEmpty.textContent = "Pick a scenario first.";
  el.maxTimeHint.textContent = "";
  showError(el.formError, "");
  syncRunButton();
  if (!entry) return;
  try {
    const doc = await getScenario(entry.name);
    if (state.scenario !== entry) return; // another one was picked meanwhile
    state.doc = doc;
  } catch (e) {
    if (state.scenario !== entry) return;
    el.fleetEmpty.hidden = false;
    el.fleetEmpty.textContent = `Cannot read scenario ${entry.name}: ${e.message || e}`;
    return;
  }
  const max = state.doc.scenario?.sim?.max_time_s;
  el.maxTimeHint.textContent = Number.isFinite(max)
    ? `empty: the scenario's ${max > 0 ? `${max} s` : "no limit"}; 0 = until stopped`
    : "empty: the scenario's limit; 0 = until stopped";
  renderFleet();
  syncRunButton();
}

// ---- fleet: a controller (and its parameters) per aircraft ----

function renderFleet() {
  const fleet = state.scenario.aircraft;
  for (const a of fleet) {
    const flies = state.controllers.filter((c) => Array.isArray(c.kinds) && c.kinds.includes(a.kind));
    const select = h("select", { id: `ctl-${a.id}`, "aria-label": `Controller for aircraft ${a.id}` },
      h("option", { value: "", text: "none — uncommanded" }),
      ...flies.map((c) => h("option", { value: c.id, text: c.name, title: c.description })));
    const paramsBox = h("div", { class: "params" });
    const error = h("div", { class: "field-error", role: "alert", hidden: true });
    const row = { aircraft: a, select, paramsBox, error, fields: new Map(), controller: null };
    select.addEventListener("change", () => {
      row.controller = state.controllers.find((c) => c.id === select.value) || null;
      renderParams(row);
      showError(row.error, "");
      showError(el.formError, "");
    });
    const team = Number.isInteger(a.team) ? `team ${a.team}` : null;
    el.fleet.append(h("div", { class: "fleet-row" },
      h("div", { class: "fleet-head" },
        h("span", { class: "fleet-id", text: `#${a.id}` }),
        h("span", { class: "fleet-model", text: a.model }),
        h("span", { class: "tag", text: a.kind }),
        team ? h("span", { class: "muted small", text: team }) : null,
        h("label", { class: "fleet-ctl" }, h("span", { class: "muted small", text: "Controller" }), select)),
      error,
      paramsBox));
    state.rows.set(a.id, row);
  }
}

// An aircraft's initial value of a parameter target (a scenario aircraft
// field: heading_deg, alt_hae_m, tas_mps); headings wrapped to [0, 360).
function initialOf(aircraftId, target) {
  const spec = state.doc?.scenario?.aircraft?.find((s) => s.id === aircraftId);
  if (!spec || !(target in TARGET_UNITS) || !Number.isFinite(spec[target])) return null;
  const v = spec[target];
  return target === "heading_deg" ? ((v % 360) + 360) % 360 : v;
}

const inBounds = (p, x) => !(Number.isFinite(p.min) && x < p.min) && !(Number.isFinite(p.max) && x > p.max);

// The catalog default of an `aircraft` param, if it is a valid choice for this
// row (another aircraft of the fleet) — what the server would use.
function validAircraftDefault(row, p) {
  const d = p.default;
  return Number.isInteger(d) && d !== row.aircraft.id && state.scenario.aircraft.some((o) => o.id === d) ? d : null;
}

// Whose initial conditions a target param starts from: the aircraft picked in
// the controller's `aircraft` param (a formation follower holds its leader's
// altitude), else — none picked, or no such param — the controlled aircraft.
function targetSource(row) {
  for (const [, f] of row.fields) {
    if (f.spec.type !== "aircraft") continue;
    const id = Number(f.input.value);
    return f.input.value !== "" && Number.isInteger(id) ? id : row.aircraft.id;
  }
  return row.aircraft.id;
}

// Fill each untouched target param from its source aircraft's initial value
// (the catalog default when there is none or it is out of bounds).
function prefillTargets(row) {
  const source = targetSource(row);
  for (const [, f] of row.fields) {
    const p = f.spec;
    if (!p.target || f.edited) continue;
    const from = p.target === "alt_hae_m" ? source : row.aircraft.id;
    const initial = initialOf(from, p.target);
    const use = initial != null && inBounds(p, initial);
    f.input.value = use ? String(Math.round(initial * 100) / 100) : p.default != null ? String(p.default) : "";
    f.from.textContent = use ? `from #${from}'s start` : "";
    check(row, p);
  }
}

function renderParams(row) {
  row.fields = new Map();
  row.paramsBox.replaceChildren();
  const c = row.controller;
  if (!c) return;
  row.paramsBox.append(h("p", { class: "muted small ctl-desc", text: c.description }));
  const grid = h("div", { class: "param-grid" });
  for (const p of c.params || []) {
    const id = `p-${row.aircraft.id}-${p.name}`;
    const label = h("label", { for: id, class: "field-label", text: p.unit ? `${p.label} (${p.unit})` : p.label });
    let input;
    if (p.type === "choice") {
      input = h("select", { id }, ...(p.choices || []).map((ch) => h("option", { value: ch, text: ch })));
      if (p.default != null) input.value = String(p.default);
    } else if (p.type === "aircraft") {
      const others = state.scenario.aircraft.filter((o) => o.id !== row.aircraft.id);
      input = h("select", { id },
        h("option", { value: "", text: "— pick an aircraft —" }),
        ...others.map((o) => h("option", { value: String(o.id), text: `#${o.id} · ${o.model}` })));
      const d = validAircraftDefault(row, p);
      input.value = d != null ? String(d) : "";
    } else {
      input = h("input", {
        id, type: "number", inputmode: "decimal",
        step: p.type === "integer" ? "1" : "any",
        min: Number.isFinite(p.min) ? p.min : null,
        max: Number.isFinite(p.max) ? p.max : null,
      });
      if (p.default != null) input.value = String(p.default);
    }
    const error = h("span", { class: "field-error", hidden: true });
    const from = h("span", {});
    const field = h("div", { class: "field" }, label, input);
    const f = { spec: p, input, error, from, edited: false };
    if (p.target && p.type !== "aircraft" && p.type !== "choice") {
      const hint = h("span", { class: "muted small hint" }, from);
      if (p.default != null) {
        const reset = h("button", { type: "button", class: "mini", title: "Set it to the catalog default", text: "reset" });
        reset.addEventListener("click", () => {
          input.value = String(p.default);
          f.edited = true;
          from.textContent = "";
          check(row, p);
        });
        hint.append(` default ${p.default} `, reset);
      }
      field.append(hint);
    }
    field.append(error);
    input.addEventListener("change", () => {
      f.edited = true;
      if (p.target) from.textContent = "";
      check(row, p);
      if (p.type === "aircraft") prefillTargets(row); // the leader's altitude follows it
    });
    row.fields.set(p.name, f);
    grid.append(field);
  }
  if (!(c.params || []).length) grid.append(h("p", { class: "muted small", text: "No parameters." }));
  row.paramsBox.append(grid);
  prefillTargets(row);
}

// ---- validation (mirrors ControllerSpec::render) ----

// The value to send for one parameter: { value } (undefined = the catalog
// default), or { error }.
function paramValue(row, p) {
  const raw = row.fields.get(p.name).input.value.trim();
  const fleetIds = state.scenario.aircraft.map((a) => a.id);
  if (raw === "") {
    // An aircraft select left on (or set back to) "— pick an aircraft —"
    // asks for a pick, whatever the catalog default; elsewhere empty means
    // the catalog default.
    if (p.type === "aircraft") return { error: "pick an aircraft" };
    return p.default != null ? { value: undefined } : { error: "required" };
  }
  switch (p.type) {
    case "choice":
      return (p.choices || []).includes(raw) ? { value: raw } : { error: `must be one of ${(p.choices || []).join(", ")}` };
    case "aircraft": {
      const id = Number(raw);
      if (!Number.isInteger(id)) return { error: "pick an aircraft" };
      if (id === row.aircraft.id) return { error: "cannot be the controlled aircraft itself" };
      if (!fleetIds.includes(id)) return { error: `aircraft ${id} is not in the scenario` };
      return { value: id };
    }
    default: {
      const x = Number(raw);
      if (!Number.isFinite(x)) return { error: "must be a number" };
      if (p.type === "integer" && !Number.isInteger(x)) return { error: "must be a whole number" };
      const lo = Number.isFinite(p.min) ? p.min : -Infinity, hi = Number.isFinite(p.max) ? p.max : Infinity;
      if (x < lo || x > hi) {
        return { error: Number.isFinite(lo) && Number.isFinite(hi) ? `must be within [${lo}, ${hi}]`
          : Number.isFinite(lo) ? `must be at least ${lo}` : `must be at most ${hi}` };
      }
      return { value: x };
    }
  }
}

function check(row, p) {
  const f = row.fields.get(p.name);
  const { error } = paramValue(row, p);
  showError(f.error, error || "");
  f.input.setAttribute("aria-invalid", String(!!error));
  return !error;
}

function maxTimeValue() {
  const raw = el.maxTime.value.trim();
  if (raw === "") return { value: undefined };
  const x = Number(raw);
  if (!Number.isFinite(x) || x < 0) return { error: "must be 0 (until stopped) or a positive number of seconds" };
  return { value: x };
}

// The POST /api/runs body, or null after marking every problem.
function buildRequest() {
  let ok = true;
  const assignments = [];
  for (const row of state.rows.values()) {
    if (!row.controller) continue;
    const params = {};
    for (const p of row.controller.params || []) {
      if (!check(row, p)) { ok = false; continue; }
      const { value } = paramValue(row, p);
      if (value !== undefined) params[p.name] = value;
    }
    assignments.push({ aircraft: row.aircraft.id, controller: row.controller.id, params });
  }
  const max = maxTimeValue();
  showError(el.maxTimeError, max.error || "");
  if (max.error) ok = false;
  if (!ok) return null;
  const body = { scenario: state.scenario.name, assignments, speed: state.speed };
  if (max.value !== undefined) body.max_time_s = max.value;
  return body;
}

// Show the server's verdict next to what it names ("aircraft 2: controller
// param `leader`: …"), else above the Run button.
function showServerError(msg) {
  const m = /^aircraft (\d+): (?:controller param `([^`]+)`)?/.exec(msg);
  const row = m ? state.rows.get(Number(m[1])) : null;
  const field = row && m[2] ? row.fields.get(m[2]) : null;
  if (field) {
    showError(field.error, msg);
    field.input.setAttribute("aria-invalid", "true");
    field.input.focus();
  } else if (row) {
    showError(row.error, msg);
    row.select.focus();
  } else {
    showError(el.formError, msg);
  }
}

// ---- run ----

function syncRunButton() {
  const active = !!state.activeRun;
  el.fields.disabled = active;
  el.runBtn.disabled = active || state.starting || !state.scenario || !state.doc;
  el.runNote.textContent = active ? "Another run is active."
    : state.starting ? "Starting…" : "";
}

async function submit(e) {
  e.preventDefault();
  if (el.runBtn.disabled) return;
  showError(el.formError, "");
  for (const row of state.rows.values()) showError(row.error, "");
  const body = buildRequest();
  if (!body) {
    showError(el.formError, "Fix the marked fields first.");
    return;
  }
  state.starting = true;
  syncRunButton();
  try {
    const run = await startRun(body);
    location.href = `./view.html?run=${encodeURIComponent(run.id)}`;
    return;
  } catch (err) {
    showServerError(err.message || String(err));
    if (err instanceof ApiError && err.status === 409) pollActive();
  }
  state.starting = false;
  syncRunButton();
}

// ---- the active run ----

let activeTimer = 0;

async function pollActive() {
  clearTimeout(activeTimer);
  if (document.hidden) return; // resumed by visibilitychange
  try {
    const run = await getCurrentRun();
    state.activeRun = run && run.state !== "finished" ? run : null;
  } catch {
    state.activeRun = null;
  }
  renderActive();
  activeTimer = setTimeout(pollActive, ACTIVE_POLL_MS);
}

function renderActive() {
  const run = state.activeRun;
  el.active.hidden = !run;
  syncRunButton();
  if (!run) return;
  el.activeId.textContent = run.id;
  const ctl = Array.isArray(run.assignments) ? run.assignments.length : 0;
  el.activeMeta.textContent = [
    run.scenario ? `scenario ${run.scenario}` : "inline scenario",
    run.state,
    ctl ? `${ctl} controller${ctl > 1 ? "s" : ""}` : "uncommanded",
  ].join(" · ");
  el.activeOpen.href = `./view.html?run=${encodeURIComponent(run.id)}`;
}

el.activeStop.addEventListener("click", async () => {
  el.activeStop.disabled = true;
  el.activeStop.textContent = "Stopping…";
  try {
    await stopRun();
    showError(el.activeError, "");
  } catch (err) {
    showError(el.activeError, `stop: ${err.message || err}`);
  }
  el.activeStop.disabled = false;
  el.activeStop.textContent = "Stop it";
  pollActive();
});

document.addEventListener("visibilitychange", () => { if (!document.hidden) pollActive(); });

// ---- boot ----

for (const b of el.speedSeg.querySelectorAll("[data-speed]")) {
  b.addEventListener("click", () => {
    state.speed = Number(b.dataset.speed);
    for (const o of el.speedSeg.querySelectorAll("[data-speed]")) {
      o.setAttribute("aria-pressed", String(o === b));
    }
  });
}
el.maxTime.addEventListener("change", () => showError(el.maxTimeError, maxTimeValue().error || ""));
el.form.addEventListener("submit", submit);

async function boot() {
  pollActive();
  try {
    const [environments, scenarios, controllers] = await Promise.all([
      getEnvironments(), getScenarios(), getControllers(),
    ]);
    state.environments = Array.isArray(environments) ? environments : [];
    state.scenarios = Array.isArray(scenarios) ? scenarios : [];
    state.controllers = Array.isArray(controllers) ? controllers : [];
  } catch (e) {
    showError(el.loadError, `Cannot load the studio catalogs: ${e.message || e}`);
    return;
  }
  state.envId = state.environments[0]?.id ?? null;
  renderEnvironments();
  renderScenarios();
}

boot();
