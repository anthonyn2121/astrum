// The Sweeps builder: a saved base scenario (environment → scenario, as on
// Interactive), its controllers (optional: without them a sweep flies
// in-process), the run settings, and the plan: what the sweep would fly, as
// POST /api/sweeps/plan says it, re-planned as the form changes. Save writes
// config/sweep.<name>.toml for `astrum sweep`; `?sweep=<name>` (or the Saved
// sweeps list) opens one. The variation (variation.js) joins the request.
// Run sweep flies the plan on screen (POST /api/sweeps) and opens its page
// (sweep.html); a sweep that is running is shown above the form.
//
// Names, summaries, plans and errors from the server and catalogs are
// untrusted: they reach the DOM as text only.

import {
  ApiError, getControllers, getCurrentSweep, getSavedSweep, getSavedSweeps, getSweeps, planSweep, saveSweep,
  startSweep, stopSweep,
} from "./api.js";
import { $, h, showError } from "./dom.js";
import { ControllerForm } from "./ctlform.js";
import { ScenarioPicker } from "./scenariopicker.js";
import { VariationForm } from "./variation.js";
import { StartCloud } from "./startcloud.js";

const PLAN_DEBOUNCE_MS = 400;
const PLAN_ROWS = 20;        // the plan table's rows; the rest is counted
const DEFAULT_MAX_TIME_S = 60;
const ACTIVE_POLL_MS = 2000;

const el = {
  savedError: $("savedError"), savedEmpty: $("savedEmpty"), savedList: $("savedList"),
  loadError: $("loadError"), form: $("sweep"), fleet: $("fleet"), fleetEmpty: $("fleetEmpty"), fleetError: $("fleetError"),
  maxTime: $("maxTime"), maxTimeError: $("maxTimeError"), repeats: $("repeats"), repeatsError: $("repeatsError"),
  jobs: $("jobs"), jobsError: $("jobsError"), runTimeout: $("runTimeout"), runTimeoutError: $("runTimeoutError"),
  planNote: $("planNote"), planBusy: $("planBusy"), planError: $("planError"), planBody: $("planBody"),
  planSummary: $("planSummary"), planTrim: $("planTrim"), planWrap: $("planWrap"), planTable: $("planTable"),
  planMore: $("planMore"), checkTrims: $("checkTrims"), saveForm: $("saveForm"), saveName: $("saveName"),
  saveSummary: $("saveSummary"), saveError: $("saveError"), saveSubmit: $("saveSubmit"),
  saveReplace: $("saveReplace"), saveResult: $("saveResult"), variation: $("variation"),
  startPanel: $("startPanel"), startLegend: $("startLegend"), startTip: $("startTip"),
  startCanvas: $("startCanvas"), startFit: $("startFit"),
  runSweep: $("runSweep"), activeSweep: $("activeSweep"), activeSweepId: $("activeSweepId"),
  activeSweepProgress: $("activeSweepProgress"), activeSweepOpen: $("activeSweepOpen"),
  activeSweepStop: $("activeSweepStop"), activeSweepError: $("activeSweepError"),
  activeSweepStatus: $("activeSweepStatus"),
  recentError: $("recentError"), recentEmpty: $("recentEmpty"), recentList: $("recentList"),
};

const state = {
  scenario: null,    // the picked ScenarioEntry
  doc: null,         // its ScenarioDoc
  loading: false,    // the picked scenario's doc is being read
  seq: 0,            // bumped by every change and every plan: a reply to an older one is ignored
  timer: 0,          // the debounced plan
  handledJson: null, // the request the last change handled (null: none, or the plan was cleared)
  savedJson: null,   // the request last saved or opened; null until a pick's first plan (the guard's baseline)
  plannedJson: null, // the request the plan on screen is for (null: none, or out of date)
  openTok: 0,        // bumped by each openSweep: an older open or save that finishes returns
  pickTok: 0,        // bumped by each pick that is not openSweep's own: an open in flight is stale
  starting: false,   // a sweep start is in flight
  activeSweep: null, // the running sweep, if any
  activeMisses: 0,   // polls in a row that failed
  leaving: false,    // a sweep was started: leaving the page is not an unsaved change
};

let form = null; // the controller form; made before the picker loads
let variation = null; // the variation form; made with the controller form
let opening = null; // the scenario name openSweep is picking (its own pick is not a user's)

const picker = new ScenarioPicker({
  envList: $("envList"), scenarioList: $("scenarioList"), scenarioEmpty: $("scenarioEmpty"),
  scenarioError: $("scenarioError"), onChange: onPick,
});
const cloud = new StartCloud({ canvas: el.startCanvas, legend: el.startLegend, tip: el.startTip, fitBtn: el.startFit });

// ---- the scenario ----

// The picker's report of each pick: the scenario, and its doc (null while it
// is read, and when nothing is picked). The plan is cleared; the first plan
// for the pick is the baseline of the unsaved-changes guard.
function onPick(entry, doc, error) {
  if (!entry || entry.name !== opening) state.pickTok += 1;
  if (entry) state.savedJson = null;
  state.scenario = entry;
  state.doc = doc;
  state.loading = !!entry && !doc && !error;
  form.setScenario(doc);
  variation.setScenario(doc);
  el.fleetEmpty.hidden = !!entry && !error;
  el.fleetEmpty.textContent = error ? `Cannot read scenario ${entry.name}: ${error}` : "Pick a scenario first.";
  showError(el.fleetError, "");
  el.startPanel.hidden = !doc;
  if (doc) el.maxTime.value = String(defaultMaxTime(doc));
  clearPlan();
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
  const fields = variation.fields();
  if (!ok || !fields || [maxTime, repeats, jobs, runTimeout].includes(undefined)) return null;
  return {
    scenario: state.scenario.name,
    assignments,
    max_time_s: maxTime,
    repeats,
    jobs,
    run_timeout_s: runTimeout,
    ...fields,
  };
}

// Why no request can be made, for the buttons that need one.
const whyNot = () => (state.doc ? "Fix the marked fields first." : "Pick a scenario first.");

// The request as the unsaved-changes guard compares it.
const snapshot = () => JSON.stringify(buildRequest() ?? null);

// ---- the plan ----

// The plan on screen is out of date (dimmed) until its reply arrives.
function setStale(stale) {
  el.planBody.classList.toggle("is-stale", stale);
  el.planBody.setAttribute("aria-busy", String(stale));
}

// Forget the plan on screen: the scenario changed, or the request was refused
// or failed. The next change is always handled.
function clearPlan() {
  state.plannedJson = null;
  state.handledJson = null;
  setStale(false);
  el.planBusy.hidden = true;
  showError(el.planError, "");
  el.planSummary.textContent = "";
  showError(el.planTrim, "");
  el.planTable.replaceChildren();
  el.planWrap.hidden = true;
  showError(el.planMore, "");
  cloud.set(state.doc, null);
}

// The form changed. A request that differs from the last one marks the plan on
// screen stale and plans again once the form has settled; no request (a field
// is invalid, or no scenario) clears the plan and says why. An unchanged
// request (a field blurred, say) changes nothing.
function changed() {
  const request = buildRequest();
  const json = JSON.stringify(request ?? null);
  if (json === state.handledJson) return;
  clearTimeout(state.timer);
  state.seq += 1;
  state.plannedJson = null;
  if (request) {
    showError(el.planNote, "");
    setStale(true);
    el.planBusy.hidden = false;
    state.timer = setTimeout(() => send(request, false), PLAN_DEBOUNCE_MS);
  } else {
    clearPlan();
    showError(el.planNote, state.doc
      ? "Fix the marked fields to see the plan."
      : state.loading ? "Loading the scenario…" : "Pick a scenario to see its plan.");
  }
  state.handledJson = json;
  syncRunSweep();
}

// Plan `request`; with `check`, the runs' aircraft trims are built and checked too.
async function send(request, check) {
  clearTimeout(state.timer);
  const seq = ++state.seq;
  el.planBusy.hidden = false;
  if (check) el.checkTrims.disabled = true;
  try {
    const reply = await planSweep(check ? { ...request, check_trim: true } : request);
    if (seq === state.seq) {
      renderPlan(reply);
      state.plannedJson = JSON.stringify(request);
      if (state.savedJson === null) state.savedJson = JSON.stringify(request);
    }
  } catch (err) {
    if (seq === state.seq) {
      clearPlan();
      showError(el.planError, err.message || String(err));
    }
  } finally {
    if (check) el.checkTrims.disabled = false;
    if (seq === state.seq) {
      el.planBusy.hidden = true;
      syncRunSweep();
    }
  }
}

const formatValue = (v) => {
  if (typeof v !== "number") return String(v);
  return v !== 0 && Math.abs(v) < 0.001 ? v.toPrecision(3) : v.toFixed(3);
};

// A POST /api/sweeps/plan reply: the summary line, the first rows, the rest
// counted, and the trim check when it was asked for. It replaces the plan on
// screen and clears its stale marking.
function renderPlan(plan) {
  setStale(false);
  showError(el.planError, "");
  cloud.set(state.doc, plan);
  el.planSummary.textContent = `${plan.runs} ${plan.runs === 1 ? "run" : "runs"} · ${plan.in_process ? "in-process" : "with controllers (processes)"}`;
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
// A save that finishes after an open started leaves the form's result alone.
async function save(overwrite) {
  showError(el.saveError, "");
  const request = buildRequest();
  if (!request) {
    showError(el.saveError, whyNot());
    return;
  }
  const token = state.openTok;
  el.saveSubmit.disabled = true;
  el.saveReplace.disabled = true;
  try {
    const saved = await saveSweep(el.saveName.value.trim(), {
      request,
      summary: el.saveSummary.value.trim(),
      overwrite,
    });
    if (token === state.openTok) {
      state.savedJson = JSON.stringify(request);
      el.saveReplace.hidden = true;
      el.saveResult.replaceChildren(
        h("p", {}, "Saved as ", h("code", { text: saved.path })),
        h("p", {}, "Run it with ", h("code", { text: `astrum sweep ${saved.path}` })));
      history.replaceState(null, "", `?sweep=${encodeURIComponent(saved.name)}`);
    }
    refreshSaved();
  } catch (err) {
    if (token === state.openTok) {
      showError(el.saveError, err.message || String(err));
      el.saveReplace.hidden = !(err instanceof ApiError && err.status === 409);
    }
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
// name and summary it was saved under. An open is stale (and returns without
// touching the form) when another open or a user's pick started meanwhile.
async function openSweep(name) {
  showError(el.savedError, "");
  const token = ++state.openTok;
  const picks = state.pickTok;
  let saved;
  try {
    saved = await getSavedSweep(name);
  } catch (err) {
    if (token === state.openTok) showError(el.savedError, err.message || String(err));
    return;
  }
  if (token !== state.openTok || picks !== state.pickTok) return;
  const { request } = saved;
  opening = request.scenario;
  let picked;
  try {
    picked = await picker.pick(request.scenario);
  } finally {
    opening = null;
  }
  if (!picked || token !== state.openTok) return; // the picker says why
  showError(el.fleetError, form.setAssignments(request.assignments).join("; "));
  variation.setFields(request);
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

// ---- running sweeps ----

let activeTimer = 0;

// Why the plan on screen cannot be flown, or "" when it can: a sweep is
// running, the form is not a valid request, or the plan is not for it.
function runBlocker(request) {
  if (state.activeSweep) return "A sweep is already running: stop it or open it first.";
  if (!request) return whyNot();
  if (state.plannedJson !== JSON.stringify(request)) return "The plan on screen is not for this request.";
  return "";
}

// Run sweep: enabled for the plan on screen, and not while a sweep runs or a
// start is in flight.
function syncRunSweep() {
  const why = runBlocker(buildRequest());
  el.runSweep.disabled = !!why || state.starting;
  el.runSweep.title = why;
}

// Start the sweep the form asks for and open its page. A refusal shows its
// message in the plan; a 409 also links to the sweep that runs.
async function runSweep() {
  const request = buildRequest();
  const why = runBlocker(request);
  if (why) {
    showError(el.planError, why);
    return;
  }
  state.starting = true;
  syncRunSweep();
  try {
    const sweep = await startSweep(request);
    state.leaving = true;
    location.href = `./sweep.html?id=${encodeURIComponent(sweep.id)}`;
    return;
  } catch (err) {
    showError(el.planError, err.message || String(err));
    if (err instanceof ApiError && err.status === 409) linkRunning();
  }
  state.starting = false;
  syncRunSweep();
}

async function linkRunning() {
  const sweep = await getCurrentSweep().catch(() => null);
  if (!sweep || sweep.state !== "running") return;
  el.planError.append(" ", h("a", { href: `./sweep.html?id=${encodeURIComponent(sweep.id)}`, text: "Open the running sweep" }));
}

// The running sweep (if any) is shown with links to open or stop it. One
// failed poll keeps the last known sweep; two in a row hide it.
async function pollActive() {
  clearTimeout(activeTimer);
  if (document.hidden) return; // resumed by visibilitychange
  try {
    const sweep = await getCurrentSweep();
    state.activeMisses = 0;
    setActive(sweep?.state === "running" ? sweep : null);
  } catch {
    state.activeMisses += 1;
    if (state.activeMisses >= 2) setActive(null);
  }
  syncRunSweep();
  clearTimeout(activeTimer);
  activeTimer = setTimeout(pollActive, ACTIVE_POLL_MS);
}

// Show `sweep` (null: none) above the form. The banner is announced when a
// sweep appears and when it goes, not on each progress update. A sweep that
// ends refreshes the recent list.
function setActive(sweep) {
  const was = state.activeSweep;
  state.activeSweep = sweep;
  el.activeSweep.hidden = !sweep;
  if (!sweep) {
    showError(el.activeSweepError, "");
    if (was) {
      el.activeSweepStatus.textContent = "";
      refreshRecent();
    }
    return;
  }
  el.activeSweepId.textContent = sweep.id;
  el.activeSweepProgress.textContent = `${sweep.ok + sweep.failed + sweep.stopped}/${sweep.total}`;
  el.activeSweepOpen.href = `./sweep.html?id=${encodeURIComponent(sweep.id)}`;
  if (!was) el.activeSweepStatus.textContent = "A sweep is running";
}

async function stopActive() {
  el.activeSweepStop.disabled = true;
  el.activeSweepStop.textContent = "Stopping…";
  try {
    await stopSweep();
    showError(el.activeSweepError, "");
  } catch (err) {
    showError(el.activeSweepError, `stop: ${err.message || err}`);
  }
  el.activeSweepStop.disabled = false;
  el.activeSweepStop.textContent = "Stop it";
  pollActive();
}

// ---- recent sweeps ----

const startedText = (at) => {
  const when = new Date(at);
  return Number.isNaN(when.getTime()) ? at : when.toLocaleString();
};

// Every sweep under the sweeps dir, newest first.
async function refreshRecent() {
  try {
    const sweeps = await getSweeps();
    showError(el.recentError, "");
    el.recentEmpty.hidden = sweeps.length > 0;
    el.recentList.replaceChildren(...sweeps.map(recentItem));
  } catch (err) {
    showError(el.recentError, `Cannot list the sweeps: ${err.message || err}`);
  }
}

function recentItem(sweep) {
  return h("li", { class: "saved-item" },
    h("div", { class: "saved-text" },
      h("a", { class: "saved-name", href: `./sweep.html?id=${encodeURIComponent(sweep.id)}`, text: sweep.id }),
      h("span", { class: "small", text: `${sweep.scenario} · ${sweep.state}` }),
      h("span", { class: "muted small", text: `${sweep.ok} ok · ${sweep.failed} failed of ${sweep.total}` }),
      h("span", { class: "muted small", text: startedText(sweep.started_at) })));
}

// ---- wiring ----

el.form.addEventListener("change", changed); // any edit: the scenario, controllers, parameters, settings
el.form.addEventListener("input", changed); // typing re-plans after the debounce, not only on blur
el.form.addEventListener("submit", (e) => e.preventDefault());
el.checkTrims.addEventListener("click", () => {
  const request = buildRequest();
  if (!request) {
    showError(el.planError, whyNot());
    return;
  }
  send(request, true);
});
el.runSweep.addEventListener("click", runSweep);
el.activeSweepStop.addEventListener("click", stopActive);
document.addEventListener("visibilitychange", () => { if (!document.hidden) pollActive(); });
el.saveForm.addEventListener("submit", (e) => {
  e.preventDefault();
  save(false);
});
el.saveName.addEventListener("input", () => {
  el.saveReplace.hidden = true; // replacing applies to the name that got the 409 only
  showError(el.saveError, "");
});
el.saveReplace.addEventListener("click", () => save(true));

// Unsaved changes: the request differs from the one last saved or opened (or,
// for a fresh pick, from its first plan).
window.addEventListener("beforeunload", (e) => {
  if (state.leaving || state.savedJson === null || snapshot() === state.savedJson) return;
  e.preventDefault();
  e.returnValue = "";
});

// Back/forward cache: a page restored from it keeps the script's state, so a
// finished start (or a leave) must not carry over.
window.addEventListener("pageshow", (e) => {
  if (!e.persisted) return;
  state.leaving = false;
  state.starting = false;
  syncRunSweep();
});

// ---- boot ----

async function boot() {
  refreshSaved();
  refreshRecent();
  pollActive();
  try {
    // The forms exist before the picker loads, so no pick can arrive without them.
    const controllers = await getControllers();
    form = new ControllerForm({ host: el.fleet, controllers: Array.isArray(controllers) ? controllers : [] });
    variation = new VariationForm({ host: el.variation, onChange: changed });
    await picker.load();
  } catch (e) {
    showError(el.loadError, `Cannot load the studio catalogs: ${e.message || e}`);
    return;
  }
  changed();
  const params = new URLSearchParams(location.search);
  const want = params.get("sweep");
  const scenario = params.get("scenario");
  if (want) openSweep(want);
  else if (scenario !== null) picker.pick(scenario);
}

boot();
