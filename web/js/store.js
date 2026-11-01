// Timeline store: the renderer's model of the fleet over time.
//
// Every telemetry frame (from the live WebSocket or a loaded export file) lands
// in a per-aircraft track ordered by sim time. The renderer asks for
// `sampleAt(id, t)` — an interpolated state at an arbitrary playhead time — so
// playback, scrubbing and slow-motion all read from the same history.
//
//  * Live tracks are fixed-capacity ring buffers: O(1) append, the oldest
//    sample is overwritten once a track is full.
//  * File tracks keep every frame of the opened export.
//
// Run events (hits, kills, …) sit in one list sorted by sim time beside the
// tracks: appended live (and trimmed once the rings have evicted the frames of
// their time), or loaded whole from a file.
//
// Frames and events are untrusted input: `parseFrame` / `parseEvent` validate
// the id, kind and clock and coerce every other field to a finite number /
// known string.

// Live history per aircraft: 20 min at the bridge's default 30 Hz.
const LIVE_CAPACITY = 36000;

// Continuous fields, linearly interpolated between samples.
const LERP_FIELDS = [
  "lat_deg", "lon_deg", "alt_hae_m",
  "enu_e", "enu_n", "enu_u",
  "vel_e_mps", "vel_n_mps", "vel_u_mps",
  "phi_rad", "theta_rad", "psi_rad",
  "p_rad_s", "q_rad_s", "r_rad_s",
  "tas_mps", "mach", "alpha_rad", "nz_g",
];
// Angles interpolated along the shortest arc (heading/roll cross ±π).
const ANGLE_FIELDS = new Set(["phi_rad", "theta_rad", "psi_rad"]);
// Discrete numeric fields, held from the earlier sample.
const HOLD_FIELDS = ["tick", "team", "hit_points", "hit_points_max", "ammo"];

const KINDS = new Set(["fixed-wing", "multirotor", "unknown"]);
const CONTROL_MODES = new Set(["uncommanded", "raw", "autopilot", "failsafe", "unknown"]);
const MAX_ID = 0xffffffff;

function wrapPi(a) {
  const twoPi = Math.PI * 2;
  return ((a + Math.PI) % twoPi + twoPi) % twoPi - Math.PI;
}

function angleLerp(a, b, alpha) {
  return a + wrapPi(b - a) * alpha;
}

function finite(v) {
  return typeof v === "number" && Number.isFinite(v) ? v : 0;
}

// Validate one astrum_web::FrameJson. Returns { id, kind, s } or null.
export function parseFrame(f) {
  if (!f || typeof f !== "object") return null;
  const { id, sim_time_s: t } = f;
  if (!Number.isInteger(id) || id < 0 || id > MAX_ID) return null;
  if (typeof t !== "number" || !Number.isFinite(t)) return null;
  const s = { t };
  for (const k of LERP_FIELDS) s[k] = finite(f[k]);
  for (const k of HOLD_FIELDS) s[k] = finite(f[k]);
  s.alive = f.alive !== false;
  s.control_mode = CONTROL_MODES.has(f.control_mode) ? f.control_mode : "unknown";
  return { id, kind: KINDS.has(f.kind) ? f.kind : "unknown", s };
}

const EVENT_KINDS = new Set([
  "trigger-pressed", "trigger-released", "hit", "kill", "mode-change",
  "controller-up", "controller-down", "run-ended",
]);
// Aircraft-id fields an event may carry (by kind).
const EVENT_IDS = ["aircraft_id", "shooter_id", "target_id", "victim_id"];
const EVENT_NUMBERS = ["ammo", "rounds_fired", "damage", "hit_points"];
const EVENT_NAMES = ["from", "to", "reason", "message"];

function validId(v) {
  return Number.isInteger(v) && v >= 0 && v <= MAX_ID;
}

// Validate one astrum_web::EventJson. Returns { t, tick, kind, …fields } or
// null; `pos_enu` becomes `pos` ([e, n, u] or null), `killer_id` stays null
// when unknown.
export function parseEvent(e) {
  if (!e || typeof e !== "object" || !EVENT_KINDS.has(e.kind)) return null;
  const { sim_time_s: t, tick } = e;
  if (typeof t !== "number" || !Number.isFinite(t)) return null;
  const ev = { t, tick: Number.isInteger(tick) ? tick : 0, kind: e.kind };
  for (const k of EVENT_IDS) if (validId(e[k])) ev[k] = e[k];
  for (const k of EVENT_NUMBERS) if (k in e) ev[k] = finite(e[k]);
  for (const k of EVENT_NAMES) if (typeof e[k] === "string") ev[k] = e[k];
  if (e.kind === "kill") ev.killer_id = validId(e.killer_id) ? e.killer_id : null;
  const p = e.pos_enu;
  ev.pos = Array.isArray(p) && p.length === 3 && p.every((v) => typeof v === "number" && Number.isFinite(v))
    ? [p[0], p[1], p[2]] : null;
  return ev;
}

// First index in sorted `events` whose time is > t (`strict`) or >= t.
function eventIndex(events, t, strict) {
  let lo = 0, hi = events.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (strict ? events[mid].t <= t : events[mid].t < t) lo = mid + 1; else hi = mid;
  }
  return lo;
}

// Fixed-capacity ring of samples, oldest first.
class RingTrack {
  constructor(capacity) {
    this.buf = new Array(capacity);
    this.capacity = capacity;
    this.start = 0;
    this.length = 0;
  }
  at(i) { return this.buf[(this.start + i) % this.capacity]; }
  first() { return this.length ? this.buf[this.start] : undefined; }
  last() { return this.length ? this.at(this.length - 1) : undefined; }
  setLast(s) { this.buf[(this.start + this.length - 1) % this.capacity] = s; }
  // Append; returns true when the oldest sample was overwritten.
  push(s) {
    if (this.length < this.capacity) {
      this.buf[(this.start + this.length) % this.capacity] = s;
      this.length++;
      return false;
    }
    this.buf[this.start] = s;
    this.start = (this.start + 1) % this.capacity;
    return true;
  }
}

// Every sample of an opened file, sorted by time.
class ListTrack {
  constructor(samples) { this.samples = samples; }
  get length() { return this.samples.length; }
  at(i) { return this.samples[i]; }
  first() { return this.samples[0]; }
  last() { return this.samples[this.samples.length - 1]; }
}

export class Store {
  constructor() {
    this.reset();
  }

  reset(mode = "idle") {
    this.tracks = new Map(); // id -> { id, kind, samples: RingTrack | ListTrack }
    this.mode = mode;        // "idle" | "live" | "file"
    this._ids = [];
    this._frameCount = 0;
    this._min = Infinity;
    this._max = -Infinity;
    this._minStale = false; // a ring evicted its oldest sample
    this.events = [];       // sorted by t (do not mutate)
  }

  get frameCount() { return this._frameCount; }

  // Live: append one snapshot of frames. Returns false — having stopped — if an
  // aircraft's clock went backwards (a new run started); the caller resets the
  // store and appends the snapshot again.
  appendLive(frames) {
    for (const f of frames) {
      const p = parseFrame(f);
      if (!p) continue;
      let track = this.tracks.get(p.id);
      if (!track) {
        track = this._addTrack(p.id, p.kind, new RingTrack(LIVE_CAPACITY));
      } else {
        track.kind = p.kind;
      }
      const ring = track.samples;
      const last = ring.last();
      if (last && p.s.t < last.t) return false;
      if (last && p.s.t === last.t) {
        ring.setLast(p.s); // same tick re-published: replace
      } else {
        if (ring.push(p.s)) this._minStale = true;
        else if (ring.length === 1 && p.s.t < this._min) this._min = p.s.t;
        if (p.s.t > this._max) this._max = p.s.t;
      }
      this._frameCount++;
    }
    return true;
  }

  // Live: append a batch of events (the sim sends them in order). Returns
  // false — having stopped — if one is earlier than the newest buffered event
  // (a new run started); the caller resets the store and appends again.
  appendEvents(events) {
    for (const e of events) {
      const ev = parseEvent(e);
      if (!ev) continue;
      const list = this.events;
      const newest = list[list.length - 1];
      if (newest && ev.t < newest.t) return false;
      list.push(ev);
    }
    return true;
  }

  // File: load every frame and event of an export document (any order).
  loadFile(frames, events = []) {
    this.reset("file");
    const byId = new Map(); // id -> { kind, samples: [] }
    for (const f of frames) {
      const p = parseFrame(f);
      if (!p) continue;
      let acc = byId.get(p.id);
      if (!acc) { acc = { kind: p.kind, samples: [] }; byId.set(p.id, acc); }
      acc.kind = p.kind;
      acc.samples.push(p.s);
      this._frameCount++;
    }
    for (const [id, acc] of byId) {
      // Stable sort; then keep the last of any samples sharing a time.
      acc.samples.sort((a, b) => a.t - b.t);
      const out = [];
      for (const s of acc.samples) {
        if (out.length && out[out.length - 1].t === s.t) out[out.length - 1] = s;
        else out.push(s);
      }
      this._addTrack(id, acc.kind, new ListTrack(out));
      if (out[0].t < this._min) this._min = out[0].t;
      if (out[out.length - 1].t > this._max) this._max = out[out.length - 1].t;
    }
    // Stable: events sharing a time keep the document's order.
    this.events = events.map(parseEvent).filter(Boolean).sort((a, b) => a.t - b.t);
  }

  _addTrack(id, kind, samples) {
    const track = { id, kind, samples };
    this.tracks.set(id, track);
    this._ids.push(id);
    this._ids.sort((a, b) => a - b);
    return track;
  }

  // Sorted list of aircraft ids currently known (do not mutate).
  ids() { return this._ids; }

  track(id) { return this.tracks.get(id); }

  // Global sim-time span across all tracks. The max is kept incrementally; the
  // min is only recomputed (over tracks, not samples) after a ring eviction.
  bounds() {
    if (this._minStale) {
      let min = Infinity;
      for (const tr of this.tracks.values()) {
        const first = tr.samples.first();
        if (first && first.t < min) min = first.t;
      }
      this._min = min;
      this._minStale = false;
      // Events older than every buffered frame went with them.
      if (Number.isFinite(min)) {
        const keep = eventIndex(this.events, min, false);
        if (keep > 0) this.events = this.events.slice(keep);
      }
    }
    const { _min: min, _max: max } = this;
    if (!Number.isFinite(min)) return { min: 0, max: 0, span: 0, empty: true };
    return { min, max, span: Math.max(0, max - min), empty: false };
  }

  // Events with t0 < t <= t1, in time order (a half-open window, so stepping
  // the playhead forward window by window yields each event once).
  eventsBetween(t0, t1) {
    if (!(t1 > t0)) return [];
    return this.events.slice(eventIndex(this.events, t0, true), eventIndex(this.events, t1, true));
  }

  // Interpolated state of aircraft `id` at playhead time `t`.
  // Returns { s, inRange } or null if the track has no samples. `inRange` is
  // false when `t` is outside the track's own recorded window (so the renderer
  // can dim an aircraft that hasn't spawned yet or has stopped reporting).
  sampleAt(id, t) {
    const track = this.tracks.get(id);
    if (!track || !track.samples.length) return null;
    const arr = track.samples;

    const first = arr.first();
    if (t <= first.t) return { s: first, inRange: first.t - t < 0.5 };
    const lastS = arr.last();
    if (t >= lastS.t) return { s: lastS, inRange: t - lastS.t < 0.5 };

    // Binary search for the first sample with time >= t.
    let lo = 0, hi = arr.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (arr.at(mid).t < t) lo = mid + 1; else hi = mid;
    }
    const b = arr.at(lo), a = arr.at(lo - 1);
    const span = b.t - a.t;
    const alpha = span > 0 ? (t - a.t) / span : 0;

    const s = { t, alive: a.alive, control_mode: a.control_mode };
    for (const k of LERP_FIELDS) {
      s[k] = ANGLE_FIELDS.has(k) ? angleLerp(a[k], b[k], alpha)
                                 : a[k] + (b[k] - a[k]) * alpha;
    }
    for (const k of HOLD_FIELDS) s[k] = a[k];
    return { s, inRange: true };
  }
}
