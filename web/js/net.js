// Telemetry sources for the renderer.
//
//  * WebSocket — the `astrum-web` bridge streams fleet snapshots,
//    `{"type":"frames","frames":[FrameJson…]}`, run events,
//    `{"type":"events","events":[EventJson…]}`, from the live sim or
//    `astrum-bag-play`, each aircraft's JSBSim model from the scenario,
//    `{"type":"models","models":[{"id","model"}…]}` (on connect and on change),
//    and which run it carries, `{"type":"run","run":N,"endpoint"}` (on connect
//    and whenever the bridge is attached elsewhere: a new run starts there).
//    The store buffers frames and events so you can scrub back through what's
//    been received (events only from when this page connected).
//  * File — an `astrum-web-export` document (a whole bag baked to JSON), from a
//    file or a finished studio run (`GET /api/runs/<id>/replay`). Loaded
//    whole, giving a fully seekable timeline with no running services; its
//    `meta.models` names the models.

import { parseCourse } from "./race.js";

// The export documents this viewer reads (astrum_web::ExportDoc): version 3 has
// `events`; version 2 predates them (opened with none).
const EXPORT_KIND = "astrum-replay";
const EXPORT_VERSIONS = [2, 3];

// Longest model name accepted (JSBSim model directory names are short).
const MODEL_MAX_LEN = 64;

// Validate a `[{ id, model }]` list (untrusted) into a Map id -> model name.
export function parseModels(list) {
  const out = new Map();
  if (!Array.isArray(list)) return out;
  for (const m of list) {
    if (!m || !Number.isInteger(m.id) || m.id < 0 || m.id > 0xffffffff) continue;
    if (typeof m.model !== "string" || !m.model || m.model.length > MODEL_MAX_LEN) continue;
    out.set(m.id, m.model);
  }
  return out;
}

export class Net {
  constructor({ onFrames, onEvents, onModels, onRun, onStatus }) {
    this.onFrames = onFrames;
    this.onEvents = onEvents || (() => {});
    this.onModels = onModels || (() => {});
    this.onRun = onRun || (() => {});
    this.onStatus = onStatus || (() => {});
    this.ws = null;
  }

  // Connect (or reconnect) to a telemetry WebSocket. `url` defaults to the same
  // host that served the page, path /telemetry.
  connect(url) {
    this.disconnect();
    const target = url || defaultWsUrl();
    this.onStatus("connecting", target);
    let ws;
    try {
      ws = new WebSocket(target);
    } catch (e) {
      this.onStatus("error", String(e));
      return;
    }
    this.ws = ws;
    ws.onopen = () => this.onStatus("live", target);
    ws.onclose = () => { if (this.ws === ws) { this.ws = null; this.onStatus("closed", target); } };
    ws.onerror = () => this.onStatus("error", target);
    ws.onmessage = (ev) => {
      let msg;
      try {
        msg = JSON.parse(ev.data);
      } catch {
        return; // ignore a malformed message
      }
      if (!msg) return;
      if (msg.type === "frames" && Array.isArray(msg.frames)) this.onFrames(msg.frames);
      else if (msg.type === "events" && Array.isArray(msg.events)) this.onEvents(msg.events);
      else if (msg.type === "models") this.onModels(parseModels(msg.models));
      else if (msg.type === "run" && Number.isInteger(msg.run)) this.onRun(msg.run);
    };
  }

  disconnect() {
    if (this.ws) {
      const ws = this.ws;
      this.ws = null;
      ws.onclose = null; ws.onerror = null; ws.onmessage = null;
      try { ws.close(); } catch { /* already closing */ }
    }
  }

  get connected() {
    return this.ws && this.ws.readyState === WebSocket.OPEN;
  }
}

export function defaultWsUrl() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const host = location.host || "localhost:8080";
  return `${proto}//${host}/telemetry`;
}

// Parse an astrum-web-export document — a File/Blob of JSON, or the document
// already parsed (untrusted either way). Returns { frames, events, meta,
// models, course } (models: Map id -> JSBSim model, empty for exports without
// them; course: a race recording's `meta.course`, race.js parseCourse, else null) or
// throws.
export async function loadExportFile(source) {
  let doc = source;
  if (source instanceof Blob) {
    try {
      doc = JSON.parse(await source.text());
    } catch {
      throw new Error("not JSON (expected an astrum-web-export document)");
    }
  }
  if (!doc || doc.kind !== EXPORT_KIND || !Array.isArray(doc.frames)) {
    throw new Error("not an ASTRUM replay export (expected an astrum-web-export document)");
  }
  if (!EXPORT_VERSIONS.includes(doc.version)) {
    const newest = EXPORT_VERSIONS[EXPORT_VERSIONS.length - 1];
    throw new Error(`unsupported export version ${String(doc.version)} (this viewer reads versions ${EXPORT_VERSIONS.join(", ")}; re-export the bag for version ${newest})`);
  }
  const meta = doc.meta && typeof doc.meta === "object" ? doc.meta : null;
  return {
    frames: doc.frames,
    events: Array.isArray(doc.events) ? doc.events : [],
    meta,
    models: parseModels(meta?.models),
    course: parseCourse(meta?.course),
  };
}
