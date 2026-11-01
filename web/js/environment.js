// The world around the fleet — background, fog, lights and the ENU grid — in
// one of two looks the parent picks from the fleet's models (setMode):
//
//   * "plain": the dark, map-less look every platform gets by default: a
//     near-black background, a short fog, cool hemisphere + key lights and
//     the grid;
//   * "sky": a daytime look for the F-16 scenarios: a shader sky dome with a
//     sun, a sun light, a long haze, two procedural cloud layers and a ground
//     map streamed in LOD rings under wherever the camera is looking.
//
// The plain objects (lights, grid, fog, background) always exist and are
// retuned per mode; everything the sky adds is built on entering it and freed
// on leaving, so a plain session holds no sky GPU resources.
//
// The map needs the scenario's geodetic origin, which telemetry doesn't carry.
// Every sample has both its geodetic and its ENU position, though, and the
// sim's ENU is the WGS84 tangent plane at the origin (crates/astrum-geo), so
// observe() solves the origin exactly from the first geolocated sample. Tile
// vertices then sit at their exact ENU position on the ellipsoid (h = 0): the
// map curves away with the Earth just as the aircraft positions do (~120 m
// below the tangent plane at 40 km out, ~1.8 km at 150 km).

import * as THREE from "three";

const MODES = new Set(["plain", "sky"]);
const DEG2RAD = Math.PI / 180;

// Today's dark look, restored exactly whenever the sky is left.
const PLAIN = {
  background: 0x0a0e14,
  fog: { color: 0x0a0e14, near: 3000, far: 14000 },
  hemi: { sky: 0xbfd3ff, ground: 0x0a0e14, intensity: 0.9 },
  sun: { color: 0xffffff, intensity: 1.4 },
  ambient: { color: 0x8899aa, intensity: 0.25 },
  gridOpacity: 0.28,
};
const PLAIN_SUN_POS = new THREE.Vector3(1, 2, 1.5);

// Daytime. The haze is one colour everywhere — the sky's horizon, the dome
// below it, the fog and the background — so the map and the far clouds melt
// into the horizon without a seam. The fog starts past 25 km so anything in a
// typical engagement (< 20 km) stays crisp.
const HAZE = 0xc6d5e4;
const SKY = {
  background: HAZE,
  fog: { color: HAZE, near: 25000, far: 140000 },
  hemi: { sky: 0xd6e6ff, ground: 0x6e6a58, intensity: 1.2 }, // sky light / earth bounce
  sun: { color: 0xfff1de, intensity: 2.4 },
  ambient: { color: 0xb8c8d8, intensity: 0.2 },
  gridOpacity: 0.1, // a scale cue over the imagery, not a floor
  zenith: 0x2b64c4,
  sunDisc: 0xfff4e0,
};

// The sun: elevation above the horizon and compass azimuth (0 = north,
// clockwise), as a unit world (three) vector: x = east, y = up, z = −north.
const SUN_ELEV_DEG = 38, SUN_AZ_DEG = 140;
const SUN_DIR = new THREE.Vector3(
  Math.cos(SUN_ELEV_DEG * DEG2RAD) * Math.sin(SUN_AZ_DEG * DEG2RAD),
  Math.sin(SUN_ELEV_DEG * DEG2RAD),
  -Math.cos(SUN_ELEV_DEG * DEG2RAD) * Math.cos(SUN_AZ_DEG * DEG2RAD),
);
const SUN_RADIUS_DEG = 0.7; // drawn disc (the real sun is 0.27°; this reads at a glance)
const DOME_RADIUS = 1000;   // m; drawn first without depth, so it only has to clear the near plane

// Clouds: two flat layers that follow the camera horizontally and sag with the
// Earth's curvature away from the ENU origin. Their noise is sampled in world
// XZ, so the pattern stays put as the layer moves with the camera, and drifts
// with the wind × sim time, so scrubbing replays the same sky.
const WIND_EN = [4.33, 2.5];  // m/s toward ENE (5 m/s from the WSW)
const EARTH_R = 6371000;     // m; a sphere is plenty for the clouds' sag
const CLOUD_SEGMENTS = 64;   // per side; keeps the sag smooth (< 0.3 m error)
const CUMULUS = {
  height: 1800,              // m above the origin's tangent plane
  half: 140000,              // m; half-width of the layer, faded out from 70 % of it
  scale: 2400,               // m per noise cell of the base octave
  octaves: 5,
  cover: [0.50, 0.61],       // fbm edge thresholds → ~30 % coverage, soft edges
  alpha: 0.92,
  litTop: 0xffffff, shadeTop: 0xb4bfcc, underLight: 0xc9d1db, underDark: 0x7d8796,
};
const CIRRUS = {
  height: 9000,
  half: 160000,
  streak: [14000, 1500],     // m per noise cell along / across the wind: long thin streaks
  octaves: 4,
  cover: [0.52, 0.70],
  alpha: 0.32,
  color: 0xf2f6fb,
};

// Ground map: Web-Mercator XYZ imagery from astrum-web, which proxies and
// disk-caches Esri World Imagery; a 404 means unavailable, and the haze (or a
// coarser tile) shows through instead.
const MAP = {
  url: (z, x, y) => `./tiles/${z}/${x}/${y}.jpg`,
  minZoom: 10, maxZoom: 16, // finest level, chosen so a texel ≈ a CSS pixel
  radius: 2,                // tiles each way per level: 5 × 5 blocks
  coarser: 7,               // at most this many levels below the finest…
  reach: 150000,            // …stopping at the first whose block reaches ±150 km
  segments: 8,              // grid cells per tile side, so tiles follow the curvature
  maxInFlight: 6,
  planMs: 250,              // re-plan at most 4 Hz
  cache: 300,               // loaded tiles kept (LRU); each is ~350 kB of GPU memory
};

// Draw order of the backdrop: dome, then map tiles by zoom (finer on top), all
// before anything with depth. Clouds draw after the opaque world and the
// trails / grid / drop lines (renderOrder 0), before the scene's aircraft
// ghosts (1.8, which restore aircraft behind a cloud), flashes (2) and labels.
const ORDER_DOME = -1000;
const ORDER_MAP = -900;     // + zoom
const ORDER_CLOUDS = 1;     // + 0.5 for whichever layer is nearer the camera

// WGS84 (crates/astrum-geo).
const WGS84_A = 6378137;
const WGS84_F = 1 / 298.257223563;
const WGS84_E2 = WGS84_F * (2 - WGS84_F);

// Scratch values for the per-frame paths (never retained).
const _ecef = { x: 0, y: 0, z: 0 };
const _ecef2 = { x: 0, y: 0, z: 0 };
const _enu = { e: 0, n: 0, u: 0 };
const _centre = { e: 0, n: 0, u: 0 };
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
class Datum {
  constructor(lat, lon, h) {
    this.lat = lat;
    this.lon = lon;
    this.h = h;
    this._o = geodeticToEcef(lat, lon, h, { x: 0, y: 0, z: 0 });
    const sl = Math.sin(lat), cl = Math.cos(lat), so = Math.sin(lon), co = Math.cos(lon);
    // Rows of the ECEF → ENU rotation: E, N, U.
    this._r = new Float64Array([-so, co, 0, -sl * co, -sl * so, cl, cl * co, cl * so, sl]);
  }

  // Solve the origin from one sample with both a geodetic and an ENU position.
  // Moving the origin by δ (local metres) moves the sample's ENU by ≈ −δ (up
  // to a tilt of distance / R), so stepping the origin against the residual
  // converges to sub-centimetre in a few iterations. Null for an inconsistent
  // sample.
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

  // Geodetic (radians, m HAE) → ENU metres into `out` { e, n, u }.
  toEnu(lat, lon, h, out) {
    const p = geodeticToEcef(lat, lon, h, _ecef);
    const dx = p.x - this._o.x, dy = p.y - this._o.y, dz = p.z - this._o.z;
    const r = this._r;
    out.e = r[0] * dx + r[1] * dy + r[2] * dz;
    out.n = r[3] * dx + r[4] * dy + r[5] * dz;
    out.u = r[6] * dx + r[7] * dy + r[8] * dz;
    return out;
  }

  // ENU metres → geodetic into `out` { lat, lon, h }.
  toGeodetic(e, n, u, out) {
    const r = this._r;
    _ecef2.x = r[0] * e + r[3] * n + r[6] * u + this._o.x;
    _ecef2.y = r[1] * e + r[4] * n + r[7] * u + this._o.y;
    _ecef2.z = r[2] * e + r[5] * n + r[8] * u + this._o.z;
    return ecefToGeodetic(_ecef2, out);
  }
}

// Web-Mercator tile coordinates (fractional) ⇄ geodetic radians; `n` is the
// tiles per axis (2 ** zoom).
function lonToTileX(lon, n) { return (lon / (2 * Math.PI) + 0.5) * n; }
function latToTileY(lat, n) { return ((1 - Math.log(Math.tan(lat) + 1 / Math.cos(lat)) / Math.PI) / 2) * n; }
function tileXToLon(x, n) { return (x / n - 0.5) * 2 * Math.PI; }
function tileYToLat(y, n) { return Math.atan(Math.sinh(Math.PI * (1 - (2 * y) / n))); }

// Block offsets within a level, nearest the centre first (loading order).
const RING = [];
for (let dy = -MAP.radius; dy <= MAP.radius; dy++) {
  for (let dx = -MAP.radius; dx <= MAP.radius; dx++) RING.push([dx, dy]);
}
RING.sort((a, b) => Math.max(Math.abs(a[0]), Math.abs(a[1])) - Math.max(Math.abs(b[0]), Math.abs(b[1]))
  || (a[0] ** 2 + a[1] ** 2) - (b[0] ** 2 + b[1] ** 2));

// UVs and triangles of a tile's grid, the same for every tile (each geometry
// wraps them in its own attributes, so disposing one leaves the others).
// Row 0 is the tile's north edge; textures keep flipY off (ImageBitmap), so
// v = 0 is the image's top row.
const TILE_UV = new Float32Array((MAP.segments + 1) ** 2 * 2);
const TILE_INDEX = new Uint16Array(MAP.segments ** 2 * 6);
{
  const s = MAP.segments, v = s + 1;
  let k = 0;
  for (let j = 0; j <= s; j++) {
    for (let i = 0; i <= s; i++) { TILE_UV[k++] = i / s; TILE_UV[k++] = j / s; }
  }
  k = 0;
  for (let j = 0; j < s; j++) {
    for (let i = 0; i < s; i++) {
      const a = j * v + i, b = a + 1, c = a + v, d = c + 1; // NW, NE, SW, SE
      // Counter-clockwise seen from above, so the face looks up.
      TILE_INDEX[k++] = a; TILE_INDEX[k++] = c; TILE_INDEX[k++] = b;
      TILE_INDEX[k++] = b; TILE_INDEX[k++] = c; TILE_INDEX[k++] = d;
    }
  }
}

// Streams imagery tiles in LOD rings centred on the ground point below the
// camera's focus: a 5 × 5 block at the finest zoom (texel ≈ pixel at the
// camera's distance), then ever coarser blocks out to ±150 km. Coarse levels
// load first so the ground is covered at once, finer ones paint on top.
class GroundMap {
  constructor(scene, datum, failed, maxAnisotropy) {
    this._scene = scene;
    this._datum = datum;
    this._failed = failed;     // keys that 404'd / errored this source (owned by Environment)
    this._aniso = maxAnisotropy;
    this._group = new THREE.Group();
    scene.add(this._group);
    this._tiles = new Map();   // key → { mesh, bitmap, ctrl }; insertion order = LRU
    this._want = [];           // the current plan, coarse first: { key, z, x, y }
    this._wanted = new Set();  // its keys
    this._ready = 0;
    this._inFlight = 0;
    this._planAt = -Infinity;
    this._z = -1; this._tx = -1; this._ty = -1; // the plan's finest zoom and focus tile
    this._disposed = false;
  }

  // Throttled: at most every planMs, find the focus tile and finest zoom, and
  // re-plan only if either changed.
  frame(camera, focus, viewH, now) {
    if (now - this._planAt < MAP.planMs) return;
    this._planAt = now;
    const d = this._datum;
    const g = d.toGeodetic(focus.x, -focus.z, focus.y, _geo);
    const ground = d.toEnu(g.lat, g.lon, 0, _centre);
    const cam = camera.position;
    const dist = Math.max(1, Math.hypot(cam.x - ground.e, cam.y - ground.u, cam.z + ground.n));
    const mPerPx = (dist * 2 * Math.tan((camera.fov * DEG2RAD) / 2)) / Math.max(1, viewH);
    const circ = 2 * Math.PI * WGS84_A * Math.cos(g.lat);
    const z = Math.min(MAP.maxZoom, Math.max(MAP.minZoom, Math.round(Math.log2(circ / (256 * mPerPx)))));
    const n = 2 ** z;
    const tx = Math.floor(lonToTileX(g.lon, n)), ty = Math.floor(latToTileY(g.lat, n));
    if (z === this._z && tx === this._tx && ty === this._ty) return;
    this._z = z; this._tx = tx; this._ty = ty;
    this._plan(g.lat, g.lon, z);
  }

  _plan(lat, lon, z0) {
    const levels = [];
    const circ = 2 * Math.PI * WGS84_A * Math.cos(lat);
    for (let z = z0; z >= 1; z--) {
      levels.push(z);
      if (levels.length > MAP.coarser || ((MAP.radius + 0.5) * circ) / 2 ** z >= MAP.reach) break;
    }
    const want = [];
    for (let i = levels.length - 1; i >= 0; i--) {
      const z = levels[i], n = 2 ** z;
      const cx = Math.floor(lonToTileX(lon, n)), cy = Math.floor(latToTileY(lat, n));
      for (const [dx, dy] of RING) {
        const y = cy + dy;
        if (y < 0 || y >= n) continue;
        const x = (((cx + dx) % n) + n) % n; // wrap across the antimeridian
        want.push({ key: `${z}/${x}/${y}`, z, x, y });
      }
    }
    const wanted = new Set(want.map((w) => w.key));
    this._want = want;
    this._wanted = wanted;

    // Hide what left the plan and cancel its loads; then mark the plan's tiles
    // most recently used, in plan order.
    for (const [key, t] of this._tiles) {
      if (wanted.has(key)) continue;
      if (t.mesh) t.mesh.visible = false;
      else { t.ctrl.abort(); this._tiles.delete(key); }
    }
    for (const w of want) {
      const t = this._tiles.get(w.key);
      if (!t) continue;
      this._tiles.delete(w.key);
      this._tiles.set(w.key, t);
      if (t.mesh) t.mesh.visible = true;
    }
    this._evict();
    this._pump();
  }

  // Start loads in plan order (coarse first) while slots are free.
  _pump() {
    for (const w of this._want) {
      if (this._inFlight >= MAP.maxInFlight) return;
      if (!this._tiles.has(w.key) && !this._failed.has(w.key)) this._load(w);
    }
  }

  // Drop the least recently used loaded tiles outside the plan past the cap.
  _evict() {
    for (const [key, t] of this._tiles) {
      if (this._ready <= MAP.cache) return;
      if (!t.mesh || this._wanted.has(key)) continue;
      this._dropTile(t);
      this._tiles.delete(key);
    }
  }

  async _load(w) {
    const t = { mesh: null, bitmap: null, ctrl: new AbortController() };
    this._tiles.set(w.key, t);
    this._inFlight++;
    let bitmap = null;
    try {
      const res = await fetch(MAP.url(w.z, w.x, w.y), { signal: t.ctrl.signal });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      bitmap = await createImageBitmap(await res.blob());
    } catch {
      // Cancelled loads were already forgotten by whoever cancelled them; a
      // real failure isn't retried until the next source.
      if (!t.ctrl.signal.aborted) {
        this._failed.add(w.key);
        this._tiles.delete(w.key);
      }
    }
    this._inFlight--;
    if (this._disposed) { bitmap?.close(); return; }
    if (bitmap && t.ctrl.signal.aborted) { bitmap.close(); bitmap = null; }
    if (bitmap) {
      t.bitmap = bitmap;
      t.mesh = this._buildTile(w, bitmap);
      t.mesh.visible = this._wanted.has(w.key);
      this._group.add(t.mesh);
      this._ready++;
      this._evict();
    }
    this._pump();
  }

  // A tile mesh: a segments² grid at the exact ENU of each vertex's (lat, lon,
  // h = 0), relative to the tile's centre (the mesh origin) so float32 keeps
  // centimetres. Drawn without depth, under everything, finer zooms on top.
  _buildTile({ z, x, y }, bitmap) {
    const n = 2 ** z, s = MAP.segments, d = this._datum;
    const c = d.toEnu(tileYToLat(y + 0.5, n), tileXToLon(x + 0.5, n), 0, _centre);
    const pos = new Float32Array((s + 1) ** 2 * 3);
    let k = 0;
    for (let j = 0; j <= s; j++) {
      const lat = tileYToLat(y + j / s, n);
      for (let i = 0; i <= s; i++) {
        const p = d.toEnu(lat, tileXToLon(x + i / s, n), 0, _enu);
        pos[k++] = p.e - c.e;
        pos[k++] = p.u - c.u;
        pos[k++] = c.n - p.n;
      }
    }
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geom.setAttribute("uv", new THREE.BufferAttribute(TILE_UV, 2));
    geom.setIndex(new THREE.BufferAttribute(TILE_INDEX, 1));
    geom.computeBoundingSphere();

    const tex = new THREE.Texture(bitmap);
    tex.flipY = false; // ignored for ImageBitmap uploads anyway; the UVs account for it
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.anisotropy = this._aniso;
    tex.needsUpdate = true;

    const mesh = new THREE.Mesh(geom, new THREE.MeshBasicMaterial({ map: tex, depthTest: false, depthWrite: false }));
    mesh.position.set(c.e, c.u, -c.n);
    mesh.renderOrder = ORDER_MAP + z;
    return mesh;
  }

  _dropTile(t) {
    this._group.remove(t.mesh);
    t.mesh.geometry.dispose();
    t.mesh.material.map.dispose();
    t.mesh.material.dispose();
    t.bitmap.close();
    this._ready--;
  }

  dispose() {
    this._disposed = true;
    for (const t of this._tiles.values()) {
      if (t.mesh) this._dropTile(t);
      else t.ctrl.abort();
    }
    this._tiles.clear();
    this._scene.remove(this._group);
  }
}

// Sky dome: zenith blue easing into the haze at the horizon, the haze below
// it (where the map, when there is one, is drawn over it), and a sun disc with
// a soft glow. Output goes through the renderer's colour-space conversion but
// no tone mapping, so the haze matches the fog exactly.
const DOME_VERT = /* glsl */ `
varying vec3 vDir;
void main() {
  vDir = position; // the dome is centred on the camera: position is the view ray
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;
const DOME_FRAG = /* glsl */ `
uniform vec3 uZenith, uHaze, uSunColor, uSunDir;
uniform vec2 uSunCos; // cos of the disc's outer / inner glow radius
varying vec3 vDir;
void main() {
  vec3 d = normalize(vDir);
  vec3 col = mix(uHaze, uZenith, pow(max(d.y, 0.0), 0.45));
  float c = max(dot(d, uSunDir), 0.0);
  col += uSunColor * (smoothstep(uSunCos.x, uSunCos.y, c) * 4.0 + pow(c, 600.0) * 0.8 + pow(c, 24.0) * 0.25);
  gl_FragColor = vec4(col, 1.0);
  #include <colorspace_fragment>
}`;

// Cloud layers share the vertex shader (camera-following plane sagging with
// the curvature away from the ENU origin) and an fbm of value noise; the
// hash is Dave Hoskins' hash12, which stays well-behaved for the large
// lattice coordinates world-space sampling produces (sin()-hashes band).
const CLOUD_VERT = /* glsl */ `
uniform float uHeight;
varying vec3 vWorld;
void main() {
  vec4 w = modelMatrix * vec4(position, 1.0);
  w.y = uHeight - dot(w.xz, w.xz) * INV_2R;
  vWorld = w.xyz;
  gl_Position = projectionMatrix * viewMatrix * w;
}`;
const NOISE = /* glsl */ `
float hash12(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}
float vnoise(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash12(i), hash12(i + vec2(1.0, 0.0)), u.x),
             mix(hash12(i + vec2(0.0, 1.0)), hash12(i + vec2(1.0, 1.0)), u.x), u.y);
}
// Each octave doubles and turns ~37°, so the lattices never line up.
float fbm(vec2 p) {
  float s = 0.0, a = 0.5;
  for (int k = 0; k < OCTAVES; k++) {
    s += a * vnoise(p);
    p = mat2(1.6, 1.2, -1.2, 1.6) * p;
    a *= 0.5;
  }
  return s;
}
uniform vec3 uHaze;
uniform vec2 uOffset, uCover;
uniform float uFogNear, uFogFar, uHalf, uAlpha;
varying vec3 vWorld;
// Haze by distance, matching the scene fog.
vec3 hazed(vec3 col, float dist) {
  return mix(col, uHaze, clamp((dist - uFogNear) / (uFogFar - uFogNear), 0.0, 1.0));
}
// Fade out toward the layer's edge so it never shows a rim.
float edge(vec3 view) {
  return 1.0 - smoothstep(0.7 * uHalf, uHalf, length(view.xz));
}`;
const CUMULUS_FRAG = /* glsl */ `
${NOISE}
uniform vec3 uLitTop, uShadeTop, uUnderLight, uUnderDark;
uniform vec2 uSunXZ;
uniform float uScale;
void main() {
  vec3 view = vWorld - cameraPosition;
  float dist = length(view);
  vec2 p = (vWorld.xz - uOffset) / uScale;
  float n = fbm(p);
  // Widen the soft edge far away, where the finest octaves would shimmer.
  float soft = 0.1 * smoothstep(8000.0, 60000.0, dist);
  float a = smoothstep(uCover.x - soft, uCover.y + soft, n);
  if (a < 0.004) discard;
  // Where the cloud thickens toward the sun, this side faces away from it.
  float lit = clamp(0.6 + (n - fbm(p + uSunXZ * 0.3)) * 6.0, 0.0, 1.0);
  vec3 col = cameraPosition.y > vWorld.y
    ? mix(uShadeTop, uLitTop, lit)
    : mix(uUnderLight, uUnderDark, smoothstep(uCover.y, uCover.y + 0.12, n)) * (0.9 + 0.1 * lit);
  gl_FragColor = vec4(hazed(col, dist), a * uAlpha * edge(view));
  #include <colorspace_fragment>
}`;
const CIRRUS_FRAG = /* glsl */ `
${NOISE}
uniform vec3 uColor;
uniform vec2 uAlong, uStreak;
void main() {
  vec3 view = vWorld - cameraPosition;
  float dist = length(view);
  vec2 w = vWorld.xz - uOffset;
  // Stretched along the wind into long thin streaks.
  vec2 q = vec2(dot(w, uAlong), dot(w, vec2(-uAlong.y, uAlong.x))) / uStreak;
  float soft = 0.1 * smoothstep(20000.0, 120000.0, dist);
  float a = smoothstep(uCover.x - soft, uCover.y + soft, fbm(q));
  if (a < 0.004) discard;
  gl_FragColor = vec4(hazed(uColor, dist), a * uAlpha * edge(view));
  #include <colorspace_fragment>
}`;

function cloudLayer(spec, frag, uniforms) {
  const geom = new THREE.PlaneGeometry(2 * spec.half, 2 * spec.half, CLOUD_SEGMENTS, CLOUD_SEGMENTS);
  geom.rotateX(-Math.PI / 2); // XY → XZ
  const mat = new THREE.ShaderMaterial({
    vertexShader: CLOUD_VERT,
    fragmentShader: frag,
    defines: { OCTAVES: spec.octaves, INV_2R: (1 / (2 * EARTH_R)).toExponential(9) },
    uniforms: {
      uHeight: { value: spec.height },
      uHaze: { value: new THREE.Color(HAZE) },
      uOffset: { value: new THREE.Vector2() },
      uCover: { value: new THREE.Vector2(spec.cover[0], spec.cover[1]) },
      uFogNear: { value: SKY.fog.near },
      uFogFar: { value: SKY.fog.far },
      uHalf: { value: spec.half },
      uAlpha: { value: spec.alpha },
      ...uniforms,
    },
    transparent: true,
    depthWrite: false, // depth-tested so aircraft on the far side are covered, but never occluding
    side: THREE.DoubleSide,
  });
  const mesh = new THREE.Mesh(geom, mat);
  mesh.frustumCulled = false; // always around the camera; the shader moves its vertices
  mesh.renderOrder = ORDER_CLOUDS;
  return mesh;
}

export class Environment {
  constructor(scene, { maxAnisotropy = 1 } = {}) {
    this._scene = scene;
    this._aniso = maxAnisotropy;
    this._mode = "plain";
    this._t = 0;
    this._datum = null;
    this._failed = new Set(); // tile keys not to request again this source
    this._sky = null;         // { dome, cumulus, cirrus } while in sky mode
    this._map = null;         // GroundMap while in sky mode with a datum

    // One background colour and one fog, retuned per mode (same fog type, so
    // a mode switch recompiles no material).
    scene.background = new THREE.Color(PLAIN.background);
    scene.fog = new THREE.Fog(PLAIN.fog.color, PLAIN.fog.near, PLAIN.fog.far);
    this._hemi = new THREE.HemisphereLight();
    this._sun = new THREE.DirectionalLight();
    this._ambient = new THREE.AmbientLight();
    scene.add(this._hemi, this._sun, this._ambient);

    const size = 12000, divisions = 60;
    this._grid = new THREE.GridHelper(size, divisions, 0x2a5a8a, 0x18324a);
    // Kept faint: it's a scale/heading overlay, not the floor.
    this._grid.material.transparent = true;
    this._grid.position.y = 0.2;
    scene.add(this._grid);

    // Faint origin disc. Grid lies in the X(East)–Z plane with North = -Z (see
    // meshes.js); the compass itself is drawn by the DOM from viewInfo().
    this._disc = new THREE.Mesh(
      new THREE.CircleGeometry(60, 48),
      new THREE.MeshBasicMaterial({ color: 0x22374d, transparent: true, opacity: 0.2 }),
    );
    this._disc.rotation.x = -Math.PI / 2;
    scene.add(this._disc);

    this._applyLook(PLAIN);
    this._sun.position.copy(PLAIN_SUN_POS);
  }

  get mode() {
    return this._mode;
  }

  // "plain" | "sky"; idempotent.
  setMode(mode) {
    if (!MODES.has(mode)) throw new Error(`unknown environment: ${mode}`);
    if (mode === this._mode) return;
    this._mode = mode;
    if (mode === "sky") {
      this._applyLook(SKY);
      this._sun.position.copy(SUN_DIR);
      this._buildSky();
    } else {
      this._disposeSky();
      this._applyLook(PLAIN);
      this._sun.position.copy(PLAIN_SUN_POS);
    }
  }

  get hasDatum() {
    return this._datum !== null;
  }

  // Solve the scenario origin from a sample, if it is geolocated (0, 0 is a
  // sim without geodetic output). Call until hasDatum.
  observe(s) {
    if (this._datum || !Number.isFinite(s.lat_deg) || (s.lat_deg === 0 && s.lon_deg === 0)) return;
    this._datum = Datum.solve(s);
  }

  // The playhead's sim time, which drives the cloud drift.
  setTime(t) {
    if (Number.isFinite(t)) this._t = t;
  }

  // A new source: its origin and its tiles may differ. Keeps the mode.
  reset() {
    this._map?.dispose();
    this._map = null;
    this._datum = null;
    this._failed.clear();
  }

  // Per render, after the camera is posed: keep the dome and clouds on the
  // camera, drift the clouds, and stream the map around `focus` (the orbit
  // target / tracked point). `viewH` is the view height in CSS px.
  frame(camera, focus, viewH) {
    const sky = this._sky;
    if (!sky) return;
    const cam = camera.position;
    sky.dome.position.copy(cam);
    // Drift with the wind: the pattern at x now is the one at x − wind·t at t = 0.
    const de = WIND_EN[0] * this._t, dn = WIND_EN[1] * this._t;
    sky.cumulus.position.set(cam.x, 0, cam.z);
    sky.cumulus.material.uniforms.uOffset.value.set(de, -dn);
    sky.cirrus.position.set(cam.x, 0, cam.z);
    sky.cirrus.material.uniforms.uOffset.value.set(de, -dn);
    // The layer farther from the camera's height draws first, so the nearer
    // one blends over it.
    const nearerCumulus = Math.abs(cam.y - CUMULUS.height) < Math.abs(cam.y - CIRRUS.height);
    sky.cumulus.renderOrder = ORDER_CLOUDS + (nearerCumulus ? 0.5 : 0);
    sky.cirrus.renderOrder = ORDER_CLOUDS + (nearerCumulus ? 0 : 0.5);

    if (!this._datum) return;
    if (!this._map) this._map = new GroundMap(this._scene, this._datum, this._failed, this._aniso);
    this._map.frame(camera, focus, viewH, performance.now());
  }

  _applyLook(look) {
    this._scene.background.setHex(look.background);
    const fog = this._scene.fog;
    fog.color.setHex(look.fog.color);
    fog.near = look.fog.near;
    fog.far = look.fog.far;
    this._hemi.color.setHex(look.hemi.sky);
    this._hemi.groundColor.setHex(look.hemi.ground);
    this._hemi.intensity = look.hemi.intensity;
    this._sun.color.setHex(look.sun.color);
    this._sun.intensity = look.sun.intensity;
    this._ambient.color.setHex(look.ambient.color);
    this._ambient.intensity = look.ambient.intensity;
    this._grid.material.opacity = look.gridOpacity;
  }

  _buildSky() {
    const dome = new THREE.Mesh(
      new THREE.SphereGeometry(DOME_RADIUS, 48, 24),
      new THREE.ShaderMaterial({
        vertexShader: DOME_VERT,
        fragmentShader: DOME_FRAG,
        uniforms: {
          uZenith: { value: new THREE.Color(SKY.zenith) },
          uHaze: { value: new THREE.Color(HAZE) },
          uSunColor: { value: new THREE.Color(SKY.sunDisc) },
          uSunDir: { value: SUN_DIR.clone() },
          uSunCos: { value: new THREE.Vector2(Math.cos(SUN_RADIUS_DEG * 1.15 * DEG2RAD), Math.cos(SUN_RADIUS_DEG * 0.85 * DEG2RAD)) },
        },
        side: THREE.BackSide,
        depthTest: false,
        depthWrite: false,
      }),
    );
    dome.frustumCulled = false;
    dome.renderOrder = ORDER_DOME;

    const sunXZ = new THREE.Vector2(SUN_DIR.x, SUN_DIR.z).normalize();
    const cumulus = cloudLayer(CUMULUS, CUMULUS_FRAG, {
      uScale: { value: CUMULUS.scale },
      uSunXZ: { value: sunXZ },
      uLitTop: { value: new THREE.Color(CUMULUS.litTop) },
      uShadeTop: { value: new THREE.Color(CUMULUS.shadeTop) },
      uUnderLight: { value: new THREE.Color(CUMULUS.underLight) },
      uUnderDark: { value: new THREE.Color(CUMULUS.underDark) },
    });
    const cirrus = cloudLayer(CIRRUS, CIRRUS_FRAG, {
      uColor: { value: new THREE.Color(CIRRUS.color) },
      uAlong: { value: new THREE.Vector2(WIND_EN[0], -WIND_EN[1]).normalize() }, // wind in three XZ
      uStreak: { value: new THREE.Vector2(CIRRUS.streak[0], CIRRUS.streak[1]) },
    });

    this._sky = { dome, cumulus, cirrus };
    this._scene.add(dome, cumulus, cirrus);
  }

  _disposeSky() {
    this._map?.dispose();
    this._map = null;
    if (!this._sky) return;
    for (const o of Object.values(this._sky)) {
      this._scene.remove(o);
      o.geometry.dispose();
      o.material.dispose();
    }
    this._sky = null;
  }

  dispose() {
    this._disposeSky();
    this._scene.remove(this._hemi, this._sun, this._ambient, this._grid, this._disc);
    this._sun.dispose();
    this._grid.geometry.dispose();
    this._grid.material.dispose();
    this._disc.geometry.dispose();
    this._disc.material.dispose();
  }
}
