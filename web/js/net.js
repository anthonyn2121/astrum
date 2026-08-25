// Telemetry sources for the renderer.
//
//  * WebSocket — the `astrum-web` bridge streams one JSON frame per sample from
//    the live sim or `astrum-bag-play`. Frames arrive continuously; the store
//    buffers them so you can still scrub back through what's been received.
//  * File — a `astrum-web-export` document (a whole bag baked to JSON). Loaded
//    whole, giving a fully seekable timeline with no running services.

export class Net {
  constructor({ onFrame, onStatus }) {
    this.onFrame = onFrame;
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
      try {
        this.onFrame(JSON.parse(ev.data));
      } catch { /* ignore a malformed frame */ }
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

// Parse a astrum-web-export document. Returns { frames, meta } or throws.
export async function loadExportFile(file) {
  const text = await file.text();
  const doc = JSON.parse(text);
  if (doc && doc.kind === "astrum-replay-bag" && Array.isArray(doc.frames)) {
    return { frames: doc.frames, meta: doc.meta || null };
  }
  // Tolerate a bare array of frames too.
  if (Array.isArray(doc)) return { frames: doc, meta: null };
  throw new Error("not a ASTRUM replay export (expected a astrum-web-export JSON document)");
}
