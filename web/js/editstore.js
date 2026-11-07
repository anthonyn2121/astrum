// A scenario's initial conditions as the scene reads a timeline store: the
// adapter Scene.update draws an editor's fleet from. It offers exactly what
// the scene asks of store.js — ids(), track(id).kind and sampleAt(id, t) —
// with one still sample per aircraft whatever `t`: its ENU position about the
// scenario origin (geo.js, = crates/astrum-geo), attitude from the initial
// heading / climb / bank, and the velocity of its TAS along the flight path.
//
// Like telemetry, a sample's attitude is in the local-level frame at the
// aircraft (psi_rad = the scenario's true heading) and its velocity is in the
// grid (the tangent plane at the origin, turned against the local level by
// the meridian convergence, Datum.northBearing); the scene turns attitudes
// onto the grid itself, for telemetry and for these samples alike.

import { Datum, DEG2RAD } from "./geo.js";

const KINDS = new Set(["fixed-wing", "multirotor"]);

export class EditStore {
  constructor() {
    this._ids = [];
    this.tracks = new Map(); // id -> { id, kind }
    this._samples = new Map(); // id -> sample (store.js field names)
    this._datum = null;
  }

  // Rebuild from a scenario: `world` its `[world]`, `aircraft` its specs
  // (scenario JSON objects), `kindOf(spec)` → "fixed-wing" | "multirotor"
  // (anything else draws as "unknown"). Aircraft with a non-finite position
  // are left out — they cannot be placed.
  setScenario(world, aircraft, kindOf) {
    const d = this._datum;
    if (!d || !d.isAt(world.origin_lat_deg, world.origin_lon_deg, world.origin_alt_hae_m)) {
      this._datum = Datum.fromDegrees(world.origin_lat_deg, world.origin_lon_deg, world.origin_alt_hae_m);
    }
    this.tracks.clear();
    this._samples.clear();
    for (const a of aircraft) {
      const s = this._sample(a);
      if (!s) continue;
      const kind = kindOf(a);
      this.tracks.set(a.id, { id: a.id, kind: KINDS.has(kind) ? kind : "unknown" });
      this._samples.set(a.id, s);
    }
    this._ids = [...this.tracks.keys()].sort((x, y) => x - y);
  }

  _sample(a) {
    const nums = [a.lat_deg, a.lon_deg, a.alt_hae_m];
    if (!nums.every(Number.isFinite)) return null;
    const angle = (v) => (Number.isFinite(v) ? v * DEG2RAD : 0);
    const lat = a.lat_deg * DEG2RAD, lon = a.lon_deg * DEG2RAD;
    const p = this._datum.toEnu(lat, lon, a.alt_hae_m);
    const psi = angle(a.heading_deg);
    const track = psi + this._datum.northBearing(lat, lon); // the heading's grid bearing
    const gamma = angle(a.climb_deg), phi = angle(a.bank_deg);
    const tas = Number.isFinite(a.tas_mps) ? a.tas_mps : 0;
    const level = tas * Math.cos(gamma);
    return {
      t: 0, tick: 0, alive: true, control_mode: "uncommanded",
      lat_deg: a.lat_deg, lon_deg: a.lon_deg, alt_hae_m: a.alt_hae_m,
      enu_e: p.e, enu_n: p.n, enu_u: p.u,
      vel_e_mps: level * Math.sin(track), vel_n_mps: level * Math.cos(track), vel_u_mps: tas * Math.sin(gamma),
      phi_rad: phi, theta_rad: gamma, psi_rad: psi,
      p_rad_s: 0, q_rad_s: 0, r_rad_s: 0,
      tas_mps: tas, mach: 0, alpha_rad: 0, nz_g: 1,
      team: Number.isInteger(a.team) ? a.team : 0,
      hit_points: Number.isFinite(a.health) ? a.health : 100,
      hit_points_max: Number.isFinite(a.health) ? a.health : 100,
      ammo: Number.isInteger(a.ammo) ? a.ammo : 0,
    };
  }

  // Sorted aircraft ids (do not mutate).
  ids() { return this._ids; }

  track(id) { return this.tracks.get(id); }

  // The aircraft's one sample at any time, always in range; null if unknown.
  sampleAt(id) {
    const s = this._samples.get(id);
    return s ? { s, inRange: true } : null;
  }
}
