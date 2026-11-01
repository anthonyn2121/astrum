// Telemetry sources for the renderer.
//
//  * WebSocket — the `astrum-web` bridge streams fleet snapshots,
//    `{"type":"frames","frames":[FrameJson…]}`, and run events,
//    `{"type":"events","events":[EventJson…]}`, from the live sim or
//    `astrum-bag-play`. The store buffers them so you can scrub back through
//    what's been received (events only from when this page connected).
//  * File — an `astrum-web-export` document (a whole bag baked to JSON). Loaded
//    whole, giving a fully seekable timeline with no running services.

// The export documents this viewer reads (astrum_web::ExportDoc): version 3 has
// `events`; version 2 predates them (opened with none).
const EXPORT_KIND = "astrum-replay";
const EXPORT_VERSIONS = [2, 3];

export class Net {
  constructor({ onFrames, onEvents, onStatus }) {
    this.onFrames = onFrames;
    this.onEvents = onEvents || (() => {});
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

// Parse an astrum-web-export document. Returns { frames, events, meta } or throws.
export async function loadExportFile(file) {
  let doc;
  try {
    doc = JSON.parse(await file.text());
  } catch {
    throw new Error("not JSON (expected an astrum-web-export document)");
  }
  if (!doc || doc.kind !== EXPORT_KIND || !Array.isArray(doc.frames)) {
    throw new Error("not an ASTRUM replay export (expected an astrum-web-export document)");
  }
  if (!EXPORT_VERSIONS.includes(doc.version)) {
    const newest = EXPORT_VERSIONS[EXPORT_VERSIONS.length - 1];
    throw new Error(`unsupported export version ${String(doc.version)} (this viewer reads versions ${EXPORT_VERSIONS.join(", ")}; re-export the bag for version ${newest})`);
  }
  return {
    frames: doc.frames,
    events: Array.isArray(doc.events) ? doc.events : [],
    meta: doc.meta && typeof doc.meta === "object" ? doc.meta : null,
  };
}
