// Telemetry sources for the renderer.
//
//  * WebSocket — the `astrum-web` bridge streams fleet snapshots,
//    `{"type":"frames","frames":[FrameJson…]}`, from the live sim or
//    `astrum-bag-play`. The store buffers them so you can scrub back through
//    what's been received.
//  * File — an `astrum-web-export` document (a whole bag baked to JSON). Loaded
//    whole, giving a fully seekable timeline with no running services.

// The export document this viewer reads (astrum_web::ExportDoc).
const EXPORT_KIND = "astrum-replay";
const EXPORT_VERSION = 2;

export class Net {
  constructor({ onFrames, onStatus }) {
    this.onFrames = onFrames;
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
      if (msg && msg.type === "frames" && Array.isArray(msg.frames)) this.onFrames(msg.frames);
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

// Parse an astrum-web-export document. Returns { frames, meta } or throws.
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
  if (doc.version !== EXPORT_VERSION) {
    throw new Error(`unsupported export version ${String(doc.version)} (this viewer reads version ${EXPORT_VERSION}; re-export the bag)`);
  }
  return { frames: doc.frames, meta: doc.meta && typeof doc.meta === "object" ? doc.meta : null };
}
