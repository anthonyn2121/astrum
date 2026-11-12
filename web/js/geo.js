// WGS84 geodetic ⇄ ECEF ⇄ local ENU, the same math as crates/astrum-geo, so
// a scenario's lat/lon/alt and the sim's ENU grid agree to the millimetre in
// the browser too. Angles are radians unless a name says `deg`; heights are
// metres above the ellipsoid (HAE).
//
// The scenario's ENU frame is the WGS84 tangent plane at its `[world]` origin
// (astrum-geo's LocalFrame): Datum. Its conversions write into a caller's
// `out` object, so the per-frame paths allocate nothing.

export const WGS84_A = 6378137;
const WGS84_F = 1 / 298.257223563;
const WGS84_E2 = WGS84_F * (2 - WGS84_F);

export const DEG2RAD = Math.PI / 180;
export const RAD2DEG = 180 / Math.PI;

// Scratch values (never retained).
const _ecef = { x: 0, y: 0, z: 0 };
const _ecef2 = { x: 0, y: 0, z: 0 };
const _enu = { e: 0, n: 0, u: 0 };
const _geo = { lat: 0, lon: 0, h: 0 };

function geodeticToEcef(lat, lon, h, out) {
  const sl = Math.sin(lat), cl = Math.cos(lat);
  const n = WGS84_A / Math.sqrt(1 - WGS84_E2 * sl * sl);
  out.x = (n + h) * cl * Math.cos(lon);
  out.y = (n + h) * cl * Math.sin(lon);
  out.z = (n * (1 - WGS84_E2) + h) * sl;
  return out;
}

// Iterates the latitude with a height formula that stays well-conditioned at
// the poles (the same scheme as astrum-geo).
function ecefToGeodetic(e, out) {
  const p = Math.hypot(e.x, e.y);
  let lat = Math.atan2(e.z, p * (1 - WGS84_E2));
  let h = 0;
  for (let i = 0; i < 10; i++) {
    const s = Math.sin(lat), c = Math.cos(lat);
    const n = WGS84_A / Math.sqrt(1 - WGS84_E2 * s * s);
    h = p * c + e.z * s - (WGS84_A * WGS84_A) / n;
    const next = Math.atan2(e.z, p * (1 - (WGS84_E2 * n) / (n + h)));
    const done = Math.abs(next - lat) < 1e-14;
    lat = next;
    if (done) break;
  }
  out.lat = lat;
  out.lon = Math.atan2(e.y, e.x);
  out.h = h;
  return out;
}

// The scenario's ENU frame: the WGS84 tangent plane at a geodetic origin
// (radians, metres HAE), as in astrum-geo's LocalFrame.
export class Datum {
  constructor(lat, lon, h) {
    this.lat = lat;
    this.lon = lon;
    this.h = h;
    this._o = geodeticToEcef(lat, lon, h, { x: 0, y: 0, z: 0 });
    const sl = Math.sin(lat), cl = Math.cos(lat), so = Math.sin(lon), co = Math.cos(lon);
    // Rows of the ECEF → ENU rotation: E, N, U.
    this._r = new Float64Array([-so, co, 0, -sl * co, -sl * so, cl, cl * co, cl * so, sl]);
  }

  // A scenario `[world]` origin: degrees and metres HAE.
  static fromDegrees(latDeg, lonDeg, h) {
    return new Datum(latDeg * DEG2RAD, lonDeg * DEG2RAD, h);
  }

  // Solve the origin from one sample with both a geodetic and an ENU position
  // (telemetry carries both, but not the origin). Moving the origin by δ
  // (local metres) moves the sample's ENU by ≈ −δ (up to a tilt of distance /
  // R), so stepping the origin against the residual converges to
  // sub-centimetre in a few iterations. Null for an inconsistent sample.
  static solve(s) {
    const lat = s.lat_deg * DEG2RAD, lon = s.lon_deg * DEG2RAD, h = s.alt_hae_m;
    let lat0 = lat - s.enu_n / WGS84_A;
    let lon0 = lon - s.enu_e / (WGS84_A * Math.cos(lat));
    let h0 = h - s.enu_u;
    for (let i = 0; i < 20; i++) {
      const d = new Datum(lat0, lon0, h0);
      d.toEnu(lat, lon, h, _enu);
      const re = s.enu_e - _enu.e, rn = s.enu_n - _enu.n, ru = s.enu_u - _enu.u;
      if (Math.hypot(re, rn, ru) < 0.005) return d;
      const sl = Math.sin(lat0);
      const w = 1 - WGS84_E2 * sl * sl;
      const nRad = WGS84_A / Math.sqrt(w);         // prime-vertical radius
      const mRad = (nRad * (1 - WGS84_E2)) / w;    // meridian radius
      lat0 -= rn / (mRad + h0);
      lon0 -= re / ((nRad + h0) * Math.cos(lat0));
      h0 -= ru;
    }
    return null;
  }

  // Whether this is the origin (lat°, lon°, h) — the datum a `[world]` names.
  isAt(latDeg, lonDeg, h) {
    return this.lat === latDeg * DEG2RAD && this.lon === lonDeg * DEG2RAD && this.h === h;
  }

  // Geodetic (radians, m HAE) → ENU metres into `out` { e, n, u }.
  toEnu(lat, lon, h, out = { e: 0, n: 0, u: 0 }) {
    const p = geodeticToEcef(lat, lon, h, _ecef);
    const dx = p.x - this._o.x, dy = p.y - this._o.y, dz = p.z - this._o.z;
    const r = this._r;
    out.e = r[0] * dx + r[1] * dy + r[2] * dz;
    out.n = r[3] * dx + r[4] * dy + r[5] * dz;
    out.u = r[6] * dx + r[7] * dy + r[8] * dz;
    return out;
  }

  // ENU metres → geodetic into `out` { lat, lon, h } (radians, m HAE).
  toGeodetic(e, n, u, out = { lat: 0, lon: 0, h: 0 }) {
    const r = this._r;
    _ecef2.x = r[0] * e + r[3] * n + r[6] * u + this._o.x;
    _ecef2.y = r[1] * e + r[4] * n + r[7] * u + this._o.y;
    _ecef2.z = r[2] * e + r[5] * n + r[8] * u + this._o.z;
    return ecefToGeodetic(_ecef2, out);
  }

  // Degree forms for scenario fields: (lat°, lon°, h) → { e, n, u }, and
  // (e, n, u) → { lat_deg, lon_deg, alt_hae_m }.
  enuOf(latDeg, lonDeg, h, out = { e: 0, n: 0, u: 0 }) {
    return this.toEnu(latDeg * DEG2RAD, lonDeg * DEG2RAD, h, out);
  }

  geodeticOf(e, n, u, out = { lat_deg: 0, lon_deg: 0, alt_hae_m: 0 }) {
    const g = this.toGeodetic(e, n, u, _geo);
    out.lat_deg = g.lat * RAD2DEG;
    out.lon_deg = g.lon * RAD2DEG;
    out.alt_hae_m = g.h;
    return out;
  }

  // The grid bearing (radians, clockwise from grid north = +n) of true north
  // at the point (lat, lon): the local-level frame there is turned by this
  // much against the grid (meridian convergence; ≈ −sin(lat) · Δlon, so west
  // of north east of the origin in the northern hemisphere). A true heading
  // ψ is grid bearing ψ + northBearing; astrum-geo's `ned_to_enu` rotates
  // (1, 0, 0) NED into the same direction.
  northBearing(lat, lon) {
    const sl = Math.sin(lat), cl = Math.cos(lat), so = Math.sin(lon), co = Math.cos(lon);
    // Local north as an ECEF vector (row N of that point's rotation), into the grid.
    const x = -sl * co, y = -sl * so, z = cl;
    const r = this._r;
    const e = r[0] * x + r[1] * y + r[2] * z;
    const n = r[3] * x + r[4] * y + r[5] * z;
    return Math.atan2(e, n);
  }
}
