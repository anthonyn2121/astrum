// The Interactive setup page: environment → scenario → a catalog controller
// (and its parameters) per aircraft → speed / max time → POST /api/runs, then
// on to the viewer (view.html?run=<id>). One live run at a time: while one is
// active it is shown, with links to open or stop it, instead of a new start.
//
// The controller rows are ControllerForm (ctlform.js); this page keeps the
// flow around them. `?scenario=<name>` picks that scenario as a click would.
// Scenario names, summaries, errors and catalog text are untrusted: they reach
// the DOM as text only.

import {
  getEnvironments, getScenarios, getScenario, getControllers, getCurrentRun, startRun, stopRun, ApiError,
} from "./api.js";
import { $, h, showError } from "./dom.js";
import { ControllerForm } from "./ctlform.js";

const ACTIVE_POLL_MS = 2000;

const el = {
  active: $("activeRun"), activeId: $("activeId"), activeMeta: $("activeMeta"), activeOpen: $("activeOpen"),
  activeStop: $("activeStop"), activeError: $("activeError"), loadError: $("loadError"),
  form: $("setup"), fields: $("setupFields"), envList: $("envList"), scenarioError: $("scenarioError"),
  scenarioList: $("scenarioList"), scenarioEmpty: $("scenarioEmpty"), fleet: $("fleet"), fleetEmpty: $("fleetEmpty"),
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
  speed: 1,
  activeRun: null, // the active RunView, if any
  starting: false,
};

let form = null; // the controller form; made once the catalogs are loaded

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

// The listing resolves each aircraft's kind (a known model may leave it out of
// the file); the controller form filters on it, so the doc carries it too.
function withKinds(doc, entry) {
  const kinds = new Map(entry.aircraft.map((a) => [a.id, a.kind]));
  const aircraft = doc.scenario.aircraft.map((a) => ({ ...a, kind: kinds.get(a.id) }));
  return { ...doc, scenario: { ...doc.scenario, aircraft } };
}

async function selectScenario(entry) {
  state.scenario = entry;
  state.doc = null;
  form.setScenario(null);
  el.fleetEmpty.hidden = !!entry;
  el.fleetEmpty.textContent = "Pick a scenario first.";
  el.maxTimeHint.textContent = "";
  showError(el.formError, "");
  syncRunButton();
  if (!entry) return;
  try {
    const doc = await getScenario(entry.name);
    if (state.scenario !== entry) return; // another one was picked meanwhile
    state.doc = withKinds(doc, entry);
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
  form.setScenario(state.doc);
  syncRunButton();
}

// `?scenario=<name>`: pick that scenario as a click would; when it is not
// listed or not valid, say why and pick nothing.
function selectFromUrl(name) {
  const s = state.scenarios.find((x) => x.name === name);
  if (!s || typeof s.error === "string") {
    const reason = s ? s.error : "no such scenario in the config directory";
    showError(el.scenarioError, `Scenario "${name}" is not available: ${reason}`);
    return;
  }
  state.envId = s.environment;
  renderEnvironments();
  selectScenario(s);
  renderScenarios();
}

// ---- run ----

// The POST /api/runs body, or null after marking every problem.
function buildRequest() {
  const { ok, assignments } = form.assignments();
  const max = maxTimeValue();
  showError(el.maxTimeError, max.error || "");
  if (!ok || max.error) return null;
  const body = { scenario: state.scenario.name, assignments, speed: state.speed };
  if (max.value !== undefined) body.max_time_s = max.value;
  return body;
}

function maxTimeValue() {
  const raw = el.maxTime.value.trim();
  if (raw === "") return { value: undefined };
  const x = Number(raw);
  if (!Number.isFinite(x) || x < 0) return { error: "must be 0 (until stopped) or a positive number of seconds" };
  return { value: x };
}

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
    const msg = err.message || String(err);
    if (!form.showServerError(msg)) showError(el.formError, msg);
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
el.fleet.addEventListener("controllerchange", () => showError(el.formError, ""));
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
  form = new ControllerForm({ host: el.fleet, controllers: state.controllers });
  renderEnvironments();
  renderScenarios();
  const want = new URLSearchParams(location.search).get("scenario");
  if (want !== null) selectFromUrl(want);
}

boot();
