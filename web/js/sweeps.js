// The Sweeps builder: a saved base scenario (environment → scenario, as on
// Interactive), its controllers (optional: without them a sweep flies
// in-process), the run settings, and the plan: what the sweep would fly, as
// POST /api/sweeps/plan says it, re-planned as the form changes. Save writes
// config/sweep.<name>.toml for `astrum sweep`; `?sweep=<name>` (or the Saved
// sweeps list) opens one. The variation joins the request through
// variationFields().
//
// Names, summaries, plans and errors from the server and catalogs are
// untrusted: they reach the DOM as text only.

import { ApiError, getControllers, getSavedSweep, getSavedSweeps, planSweep, saveSweep } from "./api.js";
import { $, h, showError } from "./dom.js";
import { ControllerForm } from "./ctlform.js";
import { ScenarioPicker } from "./scenariopicker.js";

const PLAN_DEBOUNCE_MS = 400;
const PLAN_ROWS = 20;        // the plan table's rows; the rest is counted
const DEFAULT_MAX_TIME_S = 60;

const el = {
  savedSweeps: $("savedSweeps"), savedError: $("savedError"), savedEmpty: $("savedEmpty"), savedList: $("savedList"),
  loadError: $("loadError"), form: $("sweep"), fleet: $("fleet"), fleetEmpty: $("fleetEmpty"), fleetError: $("fleetError"),
  maxTime: $("maxTime"), maxTimeError: $("maxTimeError"), repeats: $("repeats"), repeatsError: $("repeatsError"),
  jobs: $("jobs"), jobsError: $("jobsError"), runTimeout: $("runTimeout"), runTimeoutError: $("runTimeoutError"),
  planNote: $("planNote"), planBusy: $("planBusy"), planError: $("planError"), planSummary: $("planSummary"),
  planTrim: $("planTrim"), planWrap: $("planWrap"), planTable: $("planTable"), planMore: $("planMore"),
  checkTrims: $("checkTrims"), saveForm: $("saveForm"), saveName: $("saveName"), saveSummary: $("saveSummary"),
  saveError: $("saveError"), saveSubmit: $("saveSubmit"), saveReplace: $("saveReplace"), saveResult: $("saveResult"),
};

const state = {
  scenario: null,    // the picked ScenarioEntry
  doc: null,         // its ScenarioDoc
  seq: 0,            // bumped by every change and every plan: a reply to an older one is ignored
  timer: 0,          // the debounced plan
  savedJson: "null", // the request last saved or opened (the unsaved-changes guard compares it)
};

let form = null; // the controller form; made once the catalogs are loaded

const picker = new ScenarioPicker({
  envList: $("envList"), scenarioList: $("scenarioList"), scenarioEmpty: $("scenarioEmpty"),
  scenarioError: $("scenarioError"), onChange: onPick,
});

// ---- the scenario ----

// The picker's report of each pick: the scenario, and its doc (null while it
// is read, and when nothing is picked).
function onPick(entry, doc, error) {
  state.scenario = entry;
  state.doc = doc;
  form.setScenario(doc);
  el.fleetEmpty.hidden = !!entry && !error;
  el.fleetEmpty.textContent = error ? `Cannot read scenario ${entry.name}: ${error}` : "Pick a scenario first.";
  showError(el.fleetError, "");
  if (doc) el.maxTime.value = String(defaultMaxTime(doc));
  changed();
}

// The max sim time a new sweep starts from: the scenario's, when it has one.
function defaultMaxTime(doc) {
  const max = doc.scenario?.sim?.max_time_s;
  return Number.isFinite(max) && max > 0 ? max : DEFAULT_MAX_TIME_S;
}

// ---- the request ----

// A settings field's number, or undefined after showing why it is refused.
function numberOf(input, errorNode, { integer = false, min = 0, exclusive = false }) {
  const raw = input.value.trim();
  const x = Number(raw);
  let msg = "";
  if (raw === "" || !Number.isFinite(x)) msg = "must be a number";
  else if (integer && !Number.isInteger(x)) msg = "must be a whole number";
  else if (exclusive ? x <= min : x < min) msg = exclusive ? `must be more than ${min}` : `must be at least ${min}`;
  showError(errorNode, msg);
  input.setAttribute("aria-invalid", String(!!msg));
  return msg ? undefined : x;
}

// The variation's request fields (perturb, vary, samples, sample_seed,
// min_separation_m, min_alt_hae_m); there is no variation yet.
function variationFields() {
  return {};
}

// The POST /api/sweeps/plan body for the form: the picked scenario, its
// controllers and the run settings, plus the variation's fields. Null while no
// scenario is picked or a field is not valid; every problem is marked.
function buildRequest() {
  if (!state.doc) return null;
  const { ok, assignments } = form.assignments();
  const maxTime = numberOf(el.maxTime, el.maxTimeError, { min: 0, exclusive: true });
  const repeats = numberOf(el.repeats, el.repeatsError, { integer: true, min: 1 });
  const jobs = numberOf(el.jobs, el.jobsError, { integer: true, min: 0 });
  const runTimeout = numberOf(el.runTimeout, el.runTimeoutError, { min: 0 });
  if (!ok || [maxTime, repeats, jobs, runTimeout].includes(undefined)) return null;
  return {
    scenario: state.scenario.name,
    assignments,
    max_time_s: maxTime,
    repeats,
    jobs,
    run_timeout_s: runTimeout,
    ...variationFields(),
  };
}

// Why no request can be made, for the buttons that need one.
const whyNot = () => (state.doc ? "Fix the marked fields first." : "Pick a scenario first.");

// The request as the unsaved-changes guard compares it.
const snapshot = () => JSON.stringify(buildRequest() ?? null);

// ---- the plan ----

// Forget the plan on screen: it is out of date, or the request was refused.
function clearPlan() {
  el.planBusy.hidden = true;
  showError(el.planError, "");
  el.planSummary.textContent = "";
  showError(el.planTrim, "");
  el.planTable.replaceChildren();
  el.planWrap.hidden = true;
  showError(el.planMore, "");
}

// The form changed: the plan is out of date. Plan again once the form has
// settled, or say what is missing.
function changed() {
  state.seq += 1;
  clearTimeout(state.timer);
  clearPlan();
  const request = buildRequest();
  showError(el.planNote, request ? "" : state.doc ? "Fix the marked fields to see the plan." : "Pick a scenario to see its plan.");
  if (!request) return;
  el.planBusy.hidden = false;
  state.timer = setTimeout(() => send(request, false), PLAN_DEBOUNCE_MS);
}

// Plan `request`; with `check`, the runs' aircraft trims are built and checked too.
async function send(request, check) {
  clearTimeout(state.timer);
  const seq = ++state.seq;
  el.planBusy.hidden = false;
  try {
    const reply = await planSweep(check ? { ...request, check_trim: true } : request);
    if (seq === state.seq) renderPlan(reply);
  } catch (err) {
    if (seq === state.seq) {
      clearPlan();
      showError(el.planError, err.message || String(err));
    }
  } finally {
    if (seq === state.seq) el.planBusy.hidden = true;
  }
}

const formatValue = (v) => (typeof v === "number" ? v.toFixed(3) : String(v));

// A POST /api/sweeps/plan reply: the summary line, the first rows, the rest
// counted, and the trim check when it was asked for.
function renderPlan(plan) {
  showError(el.planError, "");
  el.planSummary.textContent = `${plan.runs} runs · ${plan.in_process ? "in-process" : "with controllers (processes)"}`;
  const withSample = plan.rows.some((r) => r.sample != null);
  const shown = plan.rows.slice(0, PLAN_ROWS);
  el.planTable.replaceChildren(
    h("thead", {}, h("tr", {},
      h("th", { text: "run" }),
      h("th", { text: "seed" }),
      withSample ? h("th", { text: "sample" }) : null,
      ...plan.keys.map((k) => h("th", { text: k })))),
    h("tbody", {}, ...shown.map((r) => h("tr", {},
      h("td", { text: String(r.run) }),
      h("td", { text: String(r.seed) }),
      withSample ? h("td", { text: r.sample == null ? "" : String(r.sample) }) : null,
      ...plan.keys.map((k) => h("td", { text: formatValue(r.values[k]) }))))));
  el.planWrap.hidden = false;
  const more = plan.rows.length - shown.length;
  showError(el.planMore, more > 0 ? `… and ${more} more` : "");
  showError(el.planTrim, plan.trim
    ? `Every run's aircraft trim (${plan.trim.trims} builds in ${plan.trim.wall_s.toFixed(1)} s)`
    : "");
}

// ---- save ----

// Save the request as `name`; `overwrite` replaces the sweep file that exists.
async function save(overwrite) {
  showError(el.saveError, "");
  const request = buildRequest();
  if (!request) {
    showError(el.saveError, whyNot());
    return;
  }
  el.saveSubmit.disabled = true;
  el.saveReplace.disabled = true;
  try {
    const saved = await saveSweep(el.saveName.value.trim(), {
      request,
      summary: el.saveSummary.value.trim(),
      overwrite,
    });
    state.savedJson = JSON.stringify(request);
    el.saveReplace.hidden = true;
    el.saveResult.replaceChildren(
      h("p", {}, "Saved as ", h("code", { text: saved.path })),
      h("p", {}, "Run it with ", h("code", { text: `astrum sweep ${saved.path}` })));
    history.replaceState(null, "", `?sweep=${encodeURIComponent(saved.name)}`);
    refreshSaved();
  } catch (err) {
    showError(el.saveError, err.message || String(err));
    el.saveReplace.hidden = !(err instanceof ApiError && err.status === 409);
  } finally {
    el.saveSubmit.disabled = false;
    el.saveReplace.disabled = false;
  }
}

// ---- saved sweeps ----

// List the saved sweeps, the invalid ones too, with why.
async function refreshSaved() {
  try {
    const entries = await getSavedSweeps();
    showError(el.savedError, "");
    el.savedEmpty.hidden = entries.length > 0;
    el.savedList.replaceChildren(...entries.map(savedItem));
  } catch (err) {
    showError(el.savedError, `Cannot list the saved sweeps: ${err.message || err}`);
  }
}

function savedItem(entry) {
  const button = h("button", { type: "button", text: "Open" });
  button.addEventListener("click", () => openSweep(entry.name));
  return h("li", { class: "saved-item" },
    h("div", { class: "saved-text" },
      h("span", { class: "saved-name", text: entry.name }),
      entry.summary ? h("span", { class: "small", text: entry.summary }) : null,
      entry.base ? h("span", { class: "muted small", text: `base ${entry.base}` }) : null,
      entry.error ? h("span", { class: "field-error", text: entry.error }) : null),
    button);
}

// Open the saved sweep `name`: its scenario, controllers, run settings and the
// name and summary it was saved under.
async function openSweep(name) {
  showError(el.savedError, "");
  let saved;
  try {
    saved = await getSavedSweep(name);
  } catch (err) {
    showError(el.savedError, err.message || String(err));
    return;
  }
  const { request } = saved;
  if (!(await picker.pick(request.scenario))) return; // the picker says why
  showError(el.fleetError, form.setAssignments(request.assignments).join("; "));
  el.maxTime.value = String(request.max_time_s);
  el.repeats.value = String(request.repeats);
  el.jobs.value = String(request.jobs);
  el.runTimeout.value = String(request.run_timeout_s);
  el.saveName.value = saved.name;
  el.saveSummary.value = saved.summary;
  el.saveReplace.hidden = true;
  showError(el.saveError, "");
  el.saveResult.replaceChildren();
  history.replaceState(null, "", `?sweep=${encodeURIComponent(saved.name)}`);
  changed();
  state.savedJson = snapshot();
}

// ---- wiring ----

el.form.addEventListener("change", changed); // any edit: the scenario, controllers, parameters, settings
el.form.addEventListener("submit", (e) => e.preventDefault());
el.checkTrims.addEventListener("click", () => {
  const request = buildRequest();
  if (!request) {
    showError(el.planError, whyNot());
    return;
  }
  send(request, true);
});
el.saveForm.addEventListener("submit", (e) => {
  e.preventDefault();
  save(false);
});
el.saveReplace.addEventListener("click", () => save(true));

// Unsaved changes: the request differs from the one last saved or opened.
window.addEventListener("beforeunload", (e) => {
  if (snapshot() === state.savedJson) return;
  e.preventDefault();
  e.returnValue = "";
});

// ---- boot ----

async function boot() {
  refreshSaved();
  try {
    const [controllers] = await Promise.all([getControllers(), picker.load()]);
    form = new ControllerForm({ host: el.fleet, controllers: Array.isArray(controllers) ? controllers : [] });
  } catch (e) {
    showError(el.loadError, `Cannot load the studio catalogs: ${e.message || e}`);
    return;
  }
  changed();
  const want = new URLSearchParams(location.search).get("sweep");
  if (want) openSweep(want);
}

boot();
