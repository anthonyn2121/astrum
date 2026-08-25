// Timeline store: the renderer's model of the fleet over time.
//
// Every telemetry frame (from the live WebSocket or a loaded export file) is
// appended to a per-aircraft track keyed on sim time. The renderer asks for
// `sampleAt(id, t)` — an interpolated state at an arbitrary playhead time — so
// playback, scrubbing and slow-motion all read from the same buffered history.
// Live mode appends to the tracks as frames arrive; file mode loads them whole.

const MAX_SAMPLES = 36000; // ~5 min at 120 Hz per aircraft, then ring-buffer

// Fields carried per sample (all numbers). `t` is sim_time_s, the master clock.
const NUM_FIELDS = [
  "lat_deg", "lon_deg", "alt_m",
  "enu_e", "enu_n", "enu_u",
  "vel_north_mps", "vel_east_mps", "vel_down_mps",
  "phi_rad", "theta_rad", "psi_rad",
  "airspeed_mps", "health",
];

// Angles interpolated with shortest-arc wrapping (heading crosses ±π).
const ANGLE_FIELDS = new Set(["phi_rad", "theta_rad", "psi_rad"]);

function wrapPi(a) {
  const twoPi = Math.PI * 2;
  return ((a + Math.PI) % twoPi + twoPi) % twoPi - Math.PI;
}

function angleLerp(a, b, alpha) {
  return a + wrapPi(b - a) * alpha;
}

export class Store {
  constructor() {
    this.tracks = new Map(); // id -> { id, kind, samples: [] }
    this.mode = "idle";      // "idle" | "live" | "file"
    this._frameCount = 0;
  }

  reset(mode = "idle") {
    this.tracks.clear();
    this.mode = mode;
    this._frameCount = 0;
  }

  get frameCount() { return this._frameCount; }

  // Append one frame (a plain object matching forge_web::FrameJson). Samples are
  // kept sorted by sim time.
  addFrame(f) {
    const id = f.id;
    let track = this.tracks.get(id);
    if (!track) {
      track = { id, kind: f.kind || "fixed-wing", samples: [] };
      this.tracks.set(id, track);
    } else if (f.kind) {
      track.kind = f.kind;
    }

    const s = { t: f.sim_time_s };
    for (const k of NUM_FIELDS) s[k] = f[k] ?? 0;

    const arr = track.samples;
    const last = arr[arr.length - 1];
    if (!last || s.t > last.t) {
      arr.push(s);
    } else if (s.t === last.t) {
      arr[arr.length - 1] = s; // same tick re-published: replace
    } else {
      // Out-of-order (rare): binary-insert to keep the track sorted.
      let lo = 0, hi = arr.length;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (arr[mid].t < s.t) lo = mid + 1; else hi = mid;
      }
      arr.splice(lo, 0, s);
    }

    if (arr.length > MAX_SAMPLES) arr.splice(0, arr.length - MAX_SAMPLES);
    this._frameCount++;
  }

  // Sorted list of aircraft ids currently known.
  ids() {
    return [...this.tracks.keys()].sort((a, b) => a - b);
  }

  track(id) { return this.tracks.get(id); }

  // Global sim-time span across all tracks.
  bounds() {
    let min = Infinity, max = -Infinity;
    for (const t of this.tracks.values()) {
      const a = t.samples;
      if (!a.length) continue;
      if (a[0].t < min) min = a[0].t;
      if (a[a.length - 1].t > max) max = a[a.length - 1].t;
    }
    if (!isFinite(min)) return { min: 0, max: 0, span: 0, empty: true };
    return { min, max, span: Math.max(0, max - min), empty: false };
  }

  // Interpolated state of aircraft `id` at playhead time `t`.
  // Returns { s, inRange } or null if the track has no samples. `inRange` is
  // false when `t` is outside the track's own recorded window (so the renderer
  // can dim/hide an aircraft that hasn't spawned yet or has stopped reporting).
  sampleAt(id, t) {
    const track = this.tracks.get(id);
    if (!track || !track.samples.length) return null;
    const arr = track.samples;

    if (t <= arr[0].t) return { s: arr[0], inRange: arr[0].t - t < 0.5 };
    const lastS = arr[arr.length - 1];
    if (t >= lastS.t) return { s: lastS, inRange: t - lastS.t < 0.5 };

    // Binary search for the first sample with time >= t.
    let lo = 0, hi = arr.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (arr[mid].t < t) lo = mid + 1; else hi = mid;
    }
    const b = arr[lo], a = arr[lo - 1];
    const span = b.t - a.t;
    const alpha = span > 0 ? (t - a.t) / span : 0;

    const s = { t };
    for (const k of NUM_FIELDS) {
      s[k] = ANGLE_FIELDS.has(k) ? angleLerp(a[k], b[k], alpha)
                                 : a[k] + (b[k] - a[k]) * alpha;
    }
    return { s, inRange: true };
  }
}
