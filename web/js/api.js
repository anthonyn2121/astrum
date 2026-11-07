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
