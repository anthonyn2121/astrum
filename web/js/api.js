// The studio's JSON API (`/api`, crates/astrum-web/src/api.rs), on the host
// that served the page. Every reply is JSON; a failure is `{"error": "…"}` with
// an HTTP status, raised here as an ApiError carrying both.

export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request(method, path, body) {
  const init = { method, headers: {} };
  if (body !== undefined) {
    // Non-GET routes require a JSON body type (the server's CSRF guard).
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(body);
  }
  const resp = await fetch(`/api${path}`, init);
  let data = null;
  try {
    data = await resp.json();
  } catch {
    throw new ApiError(resp.status, `${method} /api${path}: not a JSON reply (HTTP ${resp.status})`);
  }
  if (!resp.ok) {
    const why = data && typeof data.error === "string" ? data.error : `HTTP ${resp.status}`;
    throw new ApiError(resp.status, why);
  }
  return data;
}

export const apiGet = (path) => request("GET", path);
export const apiPost = (path, body = {}) => request("POST", path, body);
export const apiPut = (path, body = {}) => request("PUT", path, body);
export const apiDelete = (path) => request("DELETE", path, {});

// Catalogs (read-only).
export const getEnvironments = () => apiGet("/environments");
export const getModels = () => apiGet("/models");
export const getControllers = () => apiGet("/controllers").then((c) => c.entries);
export const getScenarios = () => apiGet("/scenarios");
export const getScenario = (name) => apiGet(`/scenarios/${encodeURIComponent(name)}`);
// A race scenario's course (race.js parseCourse reads the reply) for course seed
// `seed` and, when given, `gates` gates in place of the scenario's own count.
export const getScenarioCourse = (name, seed, gates = null) =>
  apiGet(`/scenarios/${encodeURIComponent(name)}/course?seed=${encodeURIComponent(seed)}`
    + (gates == null ? "" : `&gates=${encodeURIComponent(gates)}`));
// Scenario writes and checks (crates/astrum-web/src/scenarios.rs).
export const saveScenario = (name, body) => apiPut("/scenarios/" + encodeURIComponent(name), body);
export const checkScenario = (scenario) => apiPost("/scenario-check", { scenario });

// The bridge: which sim the telemetry stream carries, and that sim's run control.
export const getBridge = () => apiGet("/bridge");
export const controlSim = (action) => apiPost("/bridge/control", action);

// Live interactive runs (one at a time; crates/astrum-web/src/runs.rs).
export const startRun = (body) => apiPost("/runs", body);
export const getCurrentRun = () => apiGet("/runs/current").then((r) => r.run);
export const stopRun = () => apiDelete("/runs/current");
export const getRunLog = (since) => apiGet(`/runs/current/log?since=${encodeURIComponent(since)}`);
export const getRunReplay = (id) => apiGet(`/runs/${encodeURIComponent(id)}/replay`);
export const getRunMetrics = (id) => apiGet(`/runs/${encodeURIComponent(id)}/metrics`);

// Sweeps (crates/astrum-web/src/sweeps.rs): plan one without flying it, and
// save / list / read the sweep files `astrum sweep` runs.
export const planSweep = (request) => apiPost("/sweeps/plan", request);
export const getSavedSweeps = () => apiGet("/sweeps/saved");
export const getSavedSweep = (name) => apiGet(`/sweeps/saved/${encodeURIComponent(name)}`);
export const saveSweep = (name, body) => apiPut("/sweeps/saved/" + encodeURIComponent(name), body);

// Sweeps flown from the browser (crates/astrum-web/src/sweepruns.rs): one at a
// time; the sweep page reads each one, and the viewer reads a run's bag.
export const startSweep = (request) => apiPost("/sweeps", request);
export const getCurrentSweep = () => apiGet("/sweeps/current").then((r) => r.sweep);
export const stopSweep = () => apiDelete("/sweeps/current");
export const getSweeps = () => apiGet("/sweeps");
export const getSweep = (id) => apiGet(`/sweeps/${encodeURIComponent(id)}`);
export const getSweepResults = (id) => apiGet(`/sweeps/${encodeURIComponent(id)}/results`);
export const getSweepReplay = (id, n) => apiGet(`/sweeps/${encodeURIComponent(id)}/runs/${encodeURIComponent(n)}/replay`);
// The flight tracks of runs `from`…`from + count - 1` (1-based), thinned to 80 points each.
export const getSweepTracks = (id, from, count) =>
  apiGet(`/sweeps/${encodeURIComponent(id)}/tracks?from=${from}&count=${count}&points=80`);
