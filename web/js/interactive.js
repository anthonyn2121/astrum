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
  getControllers, getCurrentRun, startRun, stopRun, ApiError,
} from "./api.js";
import { $, h, showError } from "./dom.js";
import { teamColor } from "./format.js";
import { ControllerForm } from "./ctlform.js";
import { ScenarioPicker } from "./scenariopicker.js";

const ACTIVE_POLL_MS = 2000;

const el = {
  active: $("activeRun"), activeId: $("activeId"), activeMeta: $("activeMeta"), activeOpen: $("activeOpen"),
  activeStop: $("activeStop"), activeError: $("activeError"), loadError: $("loadError"),
  form: $("setup"), fields: $("setupFields"), fleet: $("fleet"), fleetEmpty: $("fleetEmpty"),
  speedSeg: $("speedSeg"), maxTime: $("maxTime"), maxTimeHint: $("maxTimeHint"), maxTimeError: $("maxTimeError"),
  formError: $("formError"), runBtn: $("runBtn"), runNote: $("runNote"),
  envList: $("envList"), planEnv: $("planEnv"), planScenario: $("planScenario"), planSpeed: $("planSpeed"),
  planMax: $("planMax"), planFleet: $("planFleet"), planLocked: $("planLocked"),
};

// The four constellation steps, in order (css/studio.css lights them).
const steps = {
  env: el.envList.closest(".step"), scenario: $("scenarioList").closest(".step"),
  fleet: el.fleet.closest(".step"), run: el.maxTime.closest(".step"),
};

const state = {
  controllers: [],
  scenario: null,  // the picked ScenarioEntry
  doc: null,       // its ScenarioDoc (initial conditions, sim settings)
  speed: 1,
  activeRun: null, // the active RunView, if any
  starting: false,
};

let form = null; // the controller form; made once the catalogs are loaded

const picker = new ScenarioPicker({
  envList: $("envList"), scenarioList: $("scenarioList"), scenarioEmpty: $("scenarioEmpty"),
  scenarioError: $("scenarioError"), onChange: onPick,
});

// ---- the scenario ----

// The picker's report of each pick (scenariopicker.js): the scenario, and its
// doc, which is null while it is read and when nothing is picked.
function onPick(entry, doc, error) {
  state.scenario = entry;
  state.doc = doc;
  form.setScenario(doc);
  el.fleetEmpty.hidden = !!entry && !error;
  el.fleetEmpty.textContent = error ? `Cannot read scenario ${entry.name}: ${error}` : "Pick a scenario first.";
  const max = doc?.scenario?.sim?.max_time_s;
  el.maxTimeHint.textContent = !doc ? ""
    : Number.isFinite(max) ? `empty: the scenario's ${max > 0 ? `${max} s` : "no limit"}; 0 = until stopped`
    : "empty: the scenario's limit; 0 = until stopped";
  showError(el.formError, "");
  sync();
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

// Everything derived from the state, redone on every change: the Run button
// and the locked form, the steps' is-done / is-current, the flight plan.
function sync() {
  const active = !!state.activeRun;
  el.fields.disabled = active;
  el.runBtn.disabled = active || state.starting || !state.scenario || !state.doc;
  el.runNote.textContent = active ? "Another run is active."
    : state.starting ? "Starting…" : "";
  el.planLocked.hidden = !active;
  syncSteps();
  renderPlan();
}

// Environment: one is picked; Scenario and Controllers (optional): its doc is
// read; Run: never done. The first step not done is the current one.
function syncSteps() {
  const env = !!el.envList.querySelector("input:checked");
  const loaded = !!state.scenario && !!state.doc;
  const done = [[steps.env, env], [steps.scenario, loaded], [steps.fleet, loaded], [steps.run, false]];
  let current = null;
  for (const [step, isDone] of done) {
    step.classList.toggle("is-done", isDone);
    if (!isDone && !current) current = step;
  }
  for (const [step] of done) step.classList.toggle("is-current", step === current);
}

// The flight plan card: what Launch would start.
function renderPlan() {
  const env = el.envList.querySelector("input:checked")?.closest(".choice")?.querySelector(".choice-title");
  el.planEnv.textContent = env ? env.textContent : "—";
  el.planScenario.textContent = state.scenario
    ? state.doc ? state.scenario.name : `${state.scenario.name} (reading…)` : "—";
  el.planSpeed.textContent = state.speed === 0 ? "Max (as fast as the host allows)"
    : state.speed === 1 ? "1× (real time)" : `${state.speed}×`;
  const max = maxTimeValue();
  const limit = state.doc?.scenario?.sim?.max_time_s;
  el.planMax.textContent = max.error ? "invalid"
    : max.value === 0 ? "until stopped"
    : max.value !== undefined ? `${max.value} s`
    : !state.doc ? "—"
    : !Number.isFinite(limit) ? "scenario's limit"
    : limit > 0 ? `${limit} s (scenario)` : "no limit (scenario)";
  const aircraft = state.doc ? [...state.doc.scenario.aircraft].sort((a, b) => a.id - b.id) : [];
  el.planFleet.replaceChildren(...aircraft.map((a) => {
    // The row's controller select (ctlform.js), when the form has one.
    const select = $(`ctl-${a.id}`);
    const ctl = select && select.value !== "" ? select.selectedOptions[0]?.textContent : null;
    const style = Number.isInteger(a.team) ? `--team: ${teamColor(a.team)}` : null;
    return h("li", { class: "plan-ac", style },
      h("span", { class: "plan-dot", "aria-hidden": "true" }),
      h("span", { class: "plan-ac-id", text: `#${a.id}` }),
      h("span", { class: "plan-ac-model", text: a.model }),
      h("span", { class: ctl ? "plan-ac-ctl" : "plan-ac-ctl muted", text: ctl || "uncommanded" }));
  }));
  if (!aircraft.length) {
    el.planFleet.append(h("li", { class: "muted small", text: state.scenario ? "Reading the scenario…" : "Pick a scenario." }));
  }
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
  sync();
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
  sync();
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
  sync();
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
    sync();
  });
}
el.maxTime.addEventListener("change", () => showError(el.maxTimeError, maxTimeValue().error || ""));
el.maxTime.addEventListener("input", sync);
el.envList.addEventListener("change", sync);
el.fleet.addEventListener("controllerchange", () => {
  showError(el.formError, "");
  sync();
});
el.form.addEventListener("submit", submit);

async function boot() {
  pollActive();
  try {
    const [controllers] = await Promise.all([getControllers(), picker.load()]);
    state.controllers = Array.isArray(controllers) ? controllers : [];
  } catch (e) {
    showError(el.loadError, `Cannot load the studio catalogs: ${e.message || e}`);
    return;
  }
  form = new ControllerForm({ host: el.fleet, controllers: state.controllers });
  sync();
  const want = new URLSearchParams(location.search).get("scenario");
  if (want !== null) picker.pick(want);
}

boot();
