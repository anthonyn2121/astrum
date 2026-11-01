// The 3-D view: scene, camera, the environment (environment.js: background,
// fog, lights, grid, and in "sky" mode the sky, clouds and streamed ground
// map), and one live object per aircraft driven from the store each frame.
// Also owns:
//
//   * sizing — the renderer tracks its container (a grid cell that docks
//     resize) through a ResizeObserver, not window resizes;
//   * the camera modes — free orbit, follow (orbit around the selection), and
//     the tracked chase / top / side views, which the scene drives itself each
//     frame; a drag in a tracked view hands the camera back to the user;
//   * click picking in screen space (select, or Shift+click to pin a target);
//   * the target overlay — line of sight, range label and reticle between the
//     selected aircraft and the pinned target — and the selection brackets;
//   * readability — aircraft keep a minimum on-screen size, labels a constant
//     one, drop lines tie each aircraft to the ground, and a constant-width
//     outline (a mask of the bodies on MASK_LAYER, dilated in one full-screen
//     pass) sets every silhouette off from sky, imagery or grid;
//   * colours — per environment and team (meshes.js aircraftColor), applied in
//     place when either changes, and exposed to the 2-D UI by colorOf();
//   * in the sky, a half-opacity "ghost" redraw of each body after the clouds,
//     so an aircraft behind a cloud still shows through at about half strength;
//   * motion trails and hit / kill flashes;
//   * viewInfo(), the camera heading and scale the DOM compass / scale bar use.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import {
  buildAircraft, enuToThree, attitudeToQuaternion, aircraftColor, tintAircraft, setTwoTone,
} from "./meshes.js";
import { Environment } from "./environment.js";

const TRAIL_MAX = 240;      // points per trail
const TRAIL_MIN_STEP = 1.0; // metres between recorded trail points

// Event flashes at an event's `pos_enu`: world size (m, the sprite's full width
// at peak) and lifetime (real seconds). Kills are the prominent one.
const FLASH = {
  hit: { size: 45, life: 0.45, color: 0xffc46a },
  kill: { size: 260, life: 1.4, color: 0xff6a3a },
};
const FLASH_MAX = 96; // live flashes at once; older ones are dropped first

const DEG2RAD = Math.PI / 180;

// On-screen sizes (CSS px). Up close an aircraft is drawn at `modelScale` × its
// true size (setModelScale); with distance it grows so the model's largest
// dimension never drops below MIN_AIRCRAFT_PX. Labels, the range tag, the
// reticle and the selection brackets are fixed-size sprites.
const MIN_AIRCRAFT_PX = 36;
const MODEL_SCALE_DEFAULT = 4, MODEL_SCALE_MIN = 1, MODEL_SCALE_MAX = 20;
const LABEL_PX = 20;        // name tag / range tag height
const NAME_ASPECT = 2;      // name tag canvas is 128 × 64 (meshes.js)
const RANGE_ASPECT = 4;     // range tag canvas is 256 × 64
const LABEL_GAP_PX = 4;     // name tag's bottom edge above the model or its frame
const RETICLE_PX = 40;      // minimum reticle size; grows to frame a big close target
const SELECT_PX = 44;       // minimum selection-bracket size; grows likewise
// Past this on-screen size the reticle and brackets hide: the model is then
// unmistakable, and a sprite stretched further draws its strokes as thick bars.
const FRAME_MAX_PX = 220;
const SELECT_EMISSIVE = 0.3; // selection glow (× the aircraft colour), low enough to keep the shading
const DASH_PX = 10, GAP_PX = 7; // line-of-sight dash pattern at the midpoint's depth

// Altitude cues: a faint vertical line to the ground and a flat ring at its foot.
const DROP_OPACITY = 0.35;
const FOOT_RADIUS = 6;      // m; grows with distance to stay FOOT_MIN_PX in radius
const FOOT_MIN_PX = 5;
const FOOT_Y = 0.4;         // above the ground under the aircraft: the grid (0.2) or the sky map's sea level
const DIM = 0.28;           // opacity factor for stale / destroyed aircraft

const TARGET_COLOR = 0xff5a4d; // --live: the pinned target / bandit

// Ghosts (sky only): each body redrawn at half opacity after the clouds. Where
// nothing covers the body this blends it over itself (no change); where a
// cloud did, it brings back about half of the aircraft.
const GHOST_OPACITY = 0.5;
const GHOST_ORDER = 1.8;    // after both cloud layers (1, 1.5), before flashes (2)

// Outline: the bodies are also on MASK_LAYER; each frame that layer is drawn
// flat white into an MSAA mask, which a full-screen pass dilates by
// OUTLINE_PX in 8 directions × 2 radii and paints (dilated − mask) in the
// outline colour. The mask ignores clouds, so outlines show through the deck.
const MASK_LAYER = 1;
const OUTLINE_PX = 2;       // CSS px (× pixel ratio)
const OUTLINE = {
  plain: { color: 0xdbe4ef, alpha: 0.8 },
  sky: { color: 0x0b1220, alpha: 0.9 },
};

// Picking: a click is a short, nearly still press; the nearest aircraft within
// PICK_PX of the pointer (projected centre) wins, so tiny distant ones stay
// clickable.
const CLICK_SLOP_PX = 5;
const CLICK_MAX_MS = 400;
const PICK_PX = 16;

// Camera modes. Tracked modes are driven by the scene, with OrbitControls off.
const CAMERA_MODES = new Set(["orbit", "follow", "chase", "top", "side"]);
const TRACKED = new Set(["chase", "top", "side"]);
const CHASE = { back: 140, rise: 32, ahead: 300 }; // m behind / above, look-ahead
const TOP_HEIGHT = 900;   // m above the selection (minimum when framing a pair)
const SIDE_DIST = 420;    // m abeam (minimum when framing a pair)
const FIT_MARGIN = 1.3;   // half-extent multiplier when framing a pair
const FIT_PAD = 40;       // m added to each half-extent (the models themselves)
const SLOW_MPS = 3;       // below this the flight path is noise: use the heading
// Smoothing time constants (real s): camera offsets from the tracked point, the
// glide of the tracked point after a selection / target change, and the low-pass
// on the chase (3-D) and side (horizontal) reference directions — slow for the
// side view so turns don't whip the camera around.
const TAU_OFFSET = 0.18;
const TAU_GLIDE = 0.45;
const TAU_CHASE_DIR = 0.35;
const TAU_SIDE_DIR = 1.5;
const ZOOM_PER_PX = 0.0015; // wheel delta → exp() distance multiplier
const ZOOM_MIN = 0.1, ZOOM_MAX = 40;

const UP = new THREE.Vector3(0, 1, 0);
const NORTH = new THREE.Vector3(0, 0, -1); // screen-up of the top view

// Scratch values for the per-frame paths (never retained).
const _p = new THREE.Vector3();
const _q = new THREE.Quaternion();
const _v = new THREE.Vector3();
const _w = new THREE.Vector3();
const _want = new THREE.Vector3();     // desired tracked point
const _wantCam = new THREE.Vector3();  // desired camera offset from it
const _wantLook = new THREE.Vector3(); // desired look-at offset from it
const _m4 = new THREE.Matrix4();
const _col = new THREE.Color();
const _clear = new THREE.Color();
const _size = new THREE.Vector2();

// Frame-rate-independent exponential smoothing: the blend toward the goal
// after `dt` seconds with time constant `tau`.
function blend(dt, tau) { return 1 - Math.exp(-dt / tau); }

// Unit flight-path direction of a sample in world (three) axes; the heading
// when it is too slow for the velocity to mean anything. `horizontal` drops the
// vertical component (the side view's track).
function pathDir(s, horizontal, out) {
  out.set(s.vel_e_mps, horizontal ? 0 : s.vel_u_mps, -s.vel_n_mps);
  const speed = out.length();
  if (speed < SLOW_MPS) return out.set(Math.sin(s.psi_rad), 0, -Math.cos(s.psi_rad));
  return out.divideScalar(speed);
}

// Low-pass a unit direction toward unit `want` (blend `k`), staying unit.
function smoothDir(dir, want, k) {
  dir.lerp(want, k);
  const len = dir.length();
  if (len < 1e-6) dir.copy(want); // exactly opposite: no meaningful midway
  else dir.divideScalar(len);
}

export class Scene {
  constructor(container, { onSelect, onTarget, onCameraMode } = {}) {
    this.container = container;
    this.onSelect = onSelect || (() => {});
    this.onTarget = onTarget || (() => {});
    this.onCameraMode = onCameraMode || (() => {});
    this.fleet = new Map(); // id -> { group, kind, label, drop, foot, trail, sample, frame }
    this.selectedId = null;
    this.targetId = null;
    this._frame = 0;         // update() counter; marks which aircraft were seen
    this._h = 1;             // container height (CSS px), kept by resize()
    this._flashes = [];      // { sprite, age, life, size }
    this._flashTex = null;   // shared radial glow, built on first use
    this._modelScale = MODEL_SCALE_DEFAULT;

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    // Block-level so the canvas adds no inline line box to the container.
    this.renderer.domElement.style.display = "block";
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this._env = new Environment(this.scene, { maxAnisotropy: this.renderer.capabilities.getMaxAnisotropy() });

    this.camera = new THREE.PerspectiveCamera(55, 1, 1, 500000);
    this.camera.position.set(320, 240, 420);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.maxPolarAngle = Math.PI * 0.495; // don't go under the ground
    this.controls.target.set(0, 60, 0);

    // Camera mode state. `_anchor` is the tracked point of a tracked mode (the
    // focus viewInfo() measures to), `_anchorErr` its remaining glide offset;
    // `_camOff` / `_lookOff` are the smoothed camera and look-at offsets from it.
    this._mode = "orbit";
    this._snap = true;       // next tracked frame jumps straight to its pose
    this._glide = false;     // next tracked frame glides from the current view
    this._tracking = false;  // the last frame had something to track
    this._trackSel = null;   // ids tracked last frame (changes start a glide)
    this._trackTgt = null;
    this._zoom = { chase: 1, top: 1, side: 1 };
    this._anchor = new THREE.Vector3();
    this._anchorErr = new THREE.Vector3();
    this._camOff = new THREE.Vector3();
    this._lookOff = new THREE.Vector3();
    this._chaseDir = new THREE.Vector3(0, 0, -1);
    this._sideDir = new THREE.Vector3(0, 0, -1);
    this._viewInfo = { headingRad: 0, metersPerPx: 0 };

    this._addCueResources();
    this._addTargetOverlay();
    this._addOutline();
    this._applyEnvironment();

    this._raycaster = new THREE.Raycaster();
    this._pointer = new THREE.Vector2();
    // The press being watched for a click (or, in a tracked mode, a drag).
    this._down = { active: false, id: 0, type: "mouse", button: 0, x: 0, y: 0, t: 0, shift: false, ctrl: false, meta: false };
    // Keep handler refs so dispose() can detach them on a renderer swap.
    this._onDown = (e) => this._onPointerDown(e);
    this._onMove = (e) => this._onPointerMove(e);
    this._onUp = (e) => this._onPointerUp(e);
    this._onCancel = () => { this._down.active = false; };
    this._onWheel = (e) => this._onWheelZoom(e);
    const el = this.renderer.domElement;
    el.addEventListener("pointerdown", this._onDown);
    el.addEventListener("pointermove", this._onMove);
    el.addEventListener("pointerup", this._onUp);
    el.addEventListener("pointercancel", this._onCancel);
    el.addEventListener("wheel", this._onWheel, { passive: false });

    // Docks resize the container without any window resize. The observer also
    // fires once on observe(), which sizes the renderer initially.
    this._resizeObserver = new ResizeObserver(() => this.resize());
    this._resizeObserver.observe(container);
    this.resize();
  }

  // Geometry shared by every aircraft's altitude cue: a unit drop line from the
  // object origin straight down (scaled per aircraft to its height) and a unit
  // flat ring (scaled to FOOT_RADIUS or more).
  _addCueResources() {
    this._dropGeom = new THREE.BufferGeometry();
    this._dropGeom.setAttribute("position", new THREE.BufferAttribute(new Float32Array([0, 0, 0, 0, -1, 0]), 3));
    this._footGeom = new THREE.RingGeometry(0.72, 1, 32);
    this._footGeom.rotateX(-Math.PI / 2); // XY -> flat XZ, facing up
  }

  // The selected → target overlay: a dashed line of sight drawn through
  // everything, a range tag at its midpoint and a reticle on the target.
  _addTargetOverlay() {
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(6), 3));
    geom.setAttribute("lineDistance", new THREE.BufferAttribute(new Float32Array(2), 1));
    this._losLine = new THREE.Line(geom, new THREE.LineDashedMaterial({
      color: TARGET_COLOR, dashSize: 1, gapSize: 1, transparent: true, opacity: 0.85,
      depthTest: false, depthWrite: false, fog: false,
    }));
    this._losLine.frustumCulled = false;
    this._losLine.renderOrder = 9;

    const canvas = document.createElement("canvas");
    canvas.width = 256;
    canvas.height = 64;
    this._rangeCtx = canvas.getContext("2d");
    this._rangeKey = NaN; // the value last drawn (see _setRangeText)
    const rangeTex = new THREE.CanvasTexture(canvas);
    rangeTex.colorSpace = THREE.SRGBColorSpace;
    this._rangeLabel = new THREE.Sprite(new THREE.SpriteMaterial({
      map: rangeTex, transparent: true, depthTest: false, depthWrite: false, sizeAttenuation: false, fog: false,
    }));
    this._rangeLabel.center.set(0.5, -0.25); // sit just above the line
    this._rangeLabel.renderOrder = 12;

    this._reticle = new THREE.Sprite(new THREE.SpriteMaterial({
      map: reticleTexture(), color: TARGET_COLOR, transparent: true,
      depthTest: false, depthWrite: false, sizeAttenuation: false, fog: false,
    }));
    this._reticle.renderOrder = 11;

    // Selection brackets: four corners around the selected aircraft, never over it.
    this._selectMarker = new THREE.Sprite(new THREE.SpriteMaterial({
      map: bracketTexture(), transparent: true,
      depthTest: false, depthWrite: false, sizeAttenuation: false, fog: false,
    }));
    this._selectMarker.renderOrder = 11;

    for (const o of [this._losLine, this._rangeLabel, this._reticle, this._selectMarker]) {
      o.visible = false;
      this.scene.add(o);
    }
  }

  // The outline resources: the body mask target (sized by resize()), its flat
  // white override material, and the full-screen composite.
  _addOutline() {
    this._maskRT = new THREE.WebGLRenderTarget(1, 1, { samples: 4, depthBuffer: false });
    this._maskMat = new THREE.MeshBasicMaterial({ color: 0xffffff, fog: false, side: THREE.DoubleSide });
    this._outlineMat = new THREE.ShaderMaterial({
      vertexShader: OUTLINE_VERT,
      fragmentShader: OUTLINE_FRAG,
      uniforms: {
        tMask: { value: this._maskRT.texture },
        uStep: { value: new THREE.Vector2() },
        uColor: { value: new THREE.Color() },
        uAlpha: { value: 1 },
      },
      transparent: true,
      depthTest: false,
      depthWrite: false,
    });
    this._outlineQuad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this._outlineMat);
    this._outlineQuad.frustumCulled = false;
    this._outlineScene = new THREE.Scene();
    this._outlineScene.add(this._outlineQuad);
    this._outlineCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  }

  // Drop every aircraft and event flash and free all of their GPU resources:
  // geometries, materials (shared ones once; ghosts included, as they are
  // children of the aircraft) and textures (the canvas label sprites, the
  // flash glow). Sprites share one three.js-owned quad geometry, which stays;
  // so do the scene-owned drop-line / foot-ring geometries, which the next
  // fleet reuses (dispose() frees them).
  clearFleet() {
    const geometries = new Set(), materials = new Set();
    const collect = (o) => {
      if (o.geometry && !o.isSprite) geometries.add(o.geometry);
      if (o.material) for (const m of [].concat(o.material)) materials.add(m);
    };
    for (const { group, label, drop, foot, trail } of this.fleet.values()) {
      for (const o of [group, label, drop, foot, trail.line]) this.scene.remove(o);
      group.traverse(collect);
      collect(label);
      collect(trail.line);
      materials.add(drop.material);
      materials.add(foot.material);
    }
    for (const g of geometries) g.dispose();
    for (const m of materials) { m.map?.dispose(); m.dispose(); }
    this.fleet.clear();
    this.selectedId = null;
    this.targetId = null;
    this._tracking = false;
    this._env.reset(); // a new source may sit at a different datum
    this._clearFlashes();
  }

  // Flash every hit / kill in `events` (store events with `pos`) at its
  // position: small and brief for a hit, large and longer for a kill.
  flash(events) {
    for (const ev of events) {
      const spec = FLASH[ev.kind];
      if (!spec || !ev.pos) continue;
      if (!this._flashTex) this._flashTex = glowTexture();
      const mat = new THREE.SpriteMaterial({
        map: this._flashTex, color: spec.color, transparent: true,
        blending: THREE.AdditiveBlending, depthWrite: false, fog: false,
      });
      const sprite = new THREE.Sprite(mat);
      enuToThree(ev.pos[0], ev.pos[1], ev.pos[2], sprite.position);
      sprite.renderOrder = 2;
      this.scene.add(sprite);
      this._flashes.push({ sprite, age: 0, life: spec.life, size: spec.size });
    }
    while (this._flashes.length > FLASH_MAX) this._dropFlash(this._flashes.shift());
  }

  // Age the flashes by `dt` real seconds: each swells and fades, then goes.
  _animateFlashes(dt) {
    const keep = [];
    for (const f of this._flashes) {
      f.age += dt;
      const k = f.age / f.life;
      if (k >= 1) { this._dropFlash(f); continue; }
      const s = f.size * (0.35 + 0.65 * Math.sqrt(k));
      f.sprite.scale.set(s, s, 1);
      f.sprite.material.opacity = (1 - k) * (1 - k);
      keep.push(f);
    }
    this._flashes = keep;
  }

  _dropFlash(f) {
    this.scene.remove(f.sprite);
    f.sprite.material.dispose();
  }

  _clearFlashes() {
    for (const f of this._flashes) this._dropFlash(f);
    this._flashes = [];
    this._flashTex?.dispose();
    this._flashTex = null;
  }

  // Forget every trail (after a seek, so no chord joins the old and new spot),
  // and cut the tracked camera straight to the new spot rather than gliding.
  resetTrails() {
    for (const { trail } of this.fleet.values()) {
      trail.count = 0;
      trail.last.set(NaN, NaN, NaN);
      trail.geom.setDrawRange(0, 0);
    }
    this._snap = true;
  }

  setSelected(id) {
    this.selectedId = id;
  }

  get environment() {
    return this._env.mode;
  }

  // "plain" (dark, no map) or "sky" (daytime sky, clouds and ground map);
  // idempotent, and kept across clearFleet(). Aircraft recolour on their next
  // update(); ghosts come and go with the sky right away.
  setEnvironment(mode) {
    if (mode === this._env.mode) return;
    this._env.setMode(mode);
    this._applyEnvironment();
    const sky = mode === "sky";
    for (const a of this.fleet.values()) {
      if (sky) this._addGhosts(a);
      else this._removeGhosts(a);
    }
  }

  // The per-environment state outside environment.js: the body two-tone
  // (module-wide in meshes.js, so set from the scene that is live) and the
  // outline colour.
  _applyEnvironment() {
    const sky = this._env.mode === "sky";
    setTwoTone(sky);
    const o = sky ? OUTLINE.sky : OUTLINE.plain;
    this._outlineMat.uniforms.uColor.value.setHex(o.color);
    this._outlineMat.uniforms.uAlpha.value = o.alpha;
  }

  get modelScale() {
    return this._modelScale;
  }

  // The close-range size exaggeration (× true size), clamped to 1–20; 1 is
  // true size up close. The MIN_AIRCRAFT_PX floor still applies at range.
  setModelScale(k) {
    const v = Number.isFinite(k) ? k : MODEL_SCALE_DEFAULT;
    this._modelScale = Math.min(MODEL_SCALE_MAX, Math.max(MODEL_SCALE_MIN, v));
  }

  // The aircraft's current colour as a CSS hex string ("#rrggbb"), or null if
  // it isn't in the fleet; the 2-D UI uses it so swatches match the 3-D view.
  colorOf(id) {
    return this.fleet.get(id)?.colorHex ?? null;
  }

  // Pin (or with null, unpin) the target the overlay and the padlock views use.
  setTarget(id) {
    this.targetId = id;
  }

  get cameraMode() {
    return this._mode;
  }

  // Switch camera mode ("orbit" | "follow" | "chase" | "top" | "side"). The
  // first frame of the new mode snaps to its pose. Leaving a tracked mode hands
  // OrbitControls the current view.
  setCameraMode(mode) {
    if (!CAMERA_MODES.has(mode)) throw new Error(`unknown camera mode: ${mode}`);
    if (mode === this._mode) return;
    if (TRACKED.has(this._mode)) this._handToControls();
    else if (TRACKED.has(mode)) this._anchor.copy(this.controls.target); // focus until something is tracked
    this._mode = mode;
    this.controls.enabled = !TRACKED.has(mode);
    this._snap = true;
  }

  // Aim OrbitControls at the point the camera is looking at, as far away as the
  // tracked point, so enabling them keeps the current view.
  _handToControls() {
    const d = Math.max(1, this.camera.position.distanceTo(this._anchor));
    _v.set(0, 0, -1).applyQuaternion(this.camera.quaternion);
    this.controls.target.copy(this.camera.position).addScaledVector(_v, d);
  }

  // A drag in a tracked mode: hand the camera to the user in follow mode, from
  // the current view, and replay the press to OrbitControls (which ignored it
  // while disabled) so this same drag already rotates / pans.
  _handoff(d) {
    this._handToControls();
    this._mode = "follow";
    this.controls.enabled = true;
    this._glide = true;
    this.onCameraMode("follow");
    this.renderer.domElement.dispatchEvent(new PointerEvent("pointerdown", {
      bubbles: true, cancelable: true, pointerId: d.id, pointerType: d.type, isPrimary: true,
      button: 0, buttons: 1, clientX: d.x, clientY: d.y,
      shiftKey: d.shift, ctrlKey: d.ctrl, metaKey: d.meta,
    }));
  }

  _ensure(id, kind, team) {
    let a = this.fleet.get(id);
    if (a) return a;
    const sky = this._env.mode === "sky";
    const group = buildAircraft(kind, id, aircraftColor(kind, id, team, sky, _col), MASK_LAYER);
    const color = group.userData.baseColor;
    this.scene.add(group);
    const label = group.userData.label;
    this.scene.add(label);

    const drop = new THREE.Line(this._dropGeom, new THREE.LineBasicMaterial({
      color, transparent: true, opacity: DROP_OPACITY,
    }));
    const foot = new THREE.Mesh(this._footGeom, new THREE.MeshBasicMaterial({
      color, transparent: true, opacity: DROP_OPACITY, depthWrite: false, side: THREE.DoubleSide,
    }));
    this.scene.add(drop);
    this.scene.add(foot);

    const positions = new Float32Array(TRAIL_MAX * 3);
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geom.setDrawRange(0, 0);
    const line = new THREE.Line(geom, new THREE.LineBasicMaterial({
      color, transparent: true, opacity: 0.55,
    }));
    line.frustumCulled = false;
    this.scene.add(line);

    a = {
      group, kind, label, drop, foot, sample: null, frame: 0,
      trail: { line, positions, geom, count: 0, last: new THREE.Vector3(NaN, NaN, NaN) },
      // The colour's inputs (recolour when either changes) and its CSS form.
      colorTeam: team, colorSky: sky, colorHex: `#${color.getHexString()}`,
      ghosts: null, ghostMats: null, // sky only (_addGhosts)
    };
    if (sky) this._addGhosts(a);
    this.fleet.set(id, a);
    return a;
  }

  // Recolour `a` if its team or the environment changed since it was last
  // coloured: the body (meshes.js tintAircraft: materials, label border; the
  // ghosts share the body's Color objects), trail, drop line and foot ring.
  _colorize(a, team) {
    const sky = this._env.mode === "sky";
    if (a.colorTeam === team && a.colorSky === sky) return;
    a.colorTeam = team;
    a.colorSky = sky;
    const color = aircraftColor(a.kind, a.group.userData.id, team, sky, _col);
    tintAircraft(a.group, color);
    a.trail.line.material.color.copy(color);
    a.drop.material.color.copy(color);
    a.foot.material.color.copy(color);
    a.colorHex = `#${color.getHexString()}`;
  }

  // Give each body mesh a ghost sibling: same geometry and transform, a
  // half-opacity clone of its material drawn after the clouds without writing
  // depth. LessEqual depth lets it land exactly on its own body. The ghost
  // materials share the body's colour / emissive objects, so recolouring and
  // the selection glow carry over; update() keeps their opacity in step.
  _addGhosts(a) {
    if (a.ghosts) return;
    const byBody = new Map();
    a.ghosts = [];
    a.ghostMats = [];
    for (const body of a.group.userData.bodies) {
      const src = body.material;
      let mat = byBody.get(src);
      if (!mat) {
        mat = src.clone();
        mat.onBeforeCompile = src.onBeforeCompile; // the shared two-tone patch (not copied by clone)
        mat.color = src.color;
        mat.emissive = src.emissive;
        mat.transparent = true;
        mat.depthWrite = false;
        mat.depthFunc = THREE.LessEqualDepth;
        mat.opacity = GHOST_OPACITY * src.opacity;
        mat.userData.src = src;
        byBody.set(src, mat);
        a.ghostMats.push(mat);
      }
      const ghost = new THREE.Mesh(body.geometry, mat);
      ghost.position.copy(body.position);
      ghost.quaternion.copy(body.quaternion);
      ghost.scale.copy(body.scale);
      ghost.renderOrder = GHOST_ORDER;
      body.parent.add(ghost);
      a.ghosts.push(ghost);
    }
  }

  // Drop the ghosts (leaving the sky): their materials; the geometry is the body's.
  _removeGhosts(a) {
    if (!a.ghosts) return;
    for (const g of a.ghosts) g.parent.remove(g);
    for (const m of a.ghostMats) m.dispose();
    a.ghosts = a.ghostMats = null;
  }

  _pushTrail(trail, p) {
    if (isFinite(trail.last.x) && trail.last.distanceTo(p) < TRAIL_MIN_STEP) return;
    const { positions } = trail;
    if (trail.count < TRAIL_MAX) {
      const o = trail.count * 3;
      positions[o] = p.x; positions[o + 1] = p.y; positions[o + 2] = p.z;
      trail.count++;
    } else {
      positions.copyWithin(0, 3);
      const o = (TRAIL_MAX - 1) * 3;
      positions[o] = p.x; positions[o + 1] = p.y; positions[o + 2] = p.z;
    }
    trail.geom.setDrawRange(0, trail.count);
    trail.geom.attributes.position.needsUpdate = true;
    trail.last.copy(p);
  }

  // Drive the whole fleet from the store at playhead time `t`, then move the
  // camera if a follow / tracked mode is on. `dt` is real seconds since the
  // last frame (flash ageing, rotor spin, camera smoothing).
  update(store, t, opts) {
    const {
      showTrails = true, showLabels = true, showDropLines = true,
      spinRotors = true, spinRate = 1, dt = 0,
    } = opts || {};
    this._animateFlashes(dt);
    this._env.setTime(t);
    const frame = ++this._frame;

    for (const id of store.ids()) {
      const track = store.track(id);
      const res = store.sampleAt(id, t);
      if (!res) continue;
      const { s, inRange } = res;
      const a = this._ensure(id, track.kind, s.team);
      a.frame = frame;
      a.sample = s;
      this._colorize(a, s.team);

      if (!this._env.hasDatum) this._env.observe(s);

      enuToThree(s.enu_e, s.enu_n, s.enu_u, _p);
      a.group.position.copy(_p);
      attitudeToQuaternion(s.phi_rad, s.theta_rad, s.psi_rad, _q);
      a.group.quaternion.copy(_q);
      a.group.visible = true;

      // Stale (outside this track's window) or destroyed → dim it.
      const opacity = inRange && s.alive ? 1.0 : DIM;
      for (const m of a.group.userData.mats) {
        m.transparent = opacity < 1;
        m.opacity = opacity;
      }
      if (a.ghostMats) for (const m of a.ghostMats) m.opacity = GHOST_OPACITY * m.userData.src.opacity;

      const selected = id === this.selectedId;
      for (const m of a.group.userData.mats) {
        m.emissive.copy(a.group.userData.baseColor).multiplyScalar(selected ? SELECT_EMISSIVE : 0.0);
      }

      a.label.visible = showLabels || selected;

      // Altitude cue: the unit drop line scaled down to the ground — y = 0 in
      // plain mode; in sky mode the sea-level surface under the aircraft, which
      // falls below the tangent plane with distance from the origin.
      a.drop.visible = a.foot.visible = showDropLines;
      if (showDropLines) {
        const groundY = this._env.mode === "sky" ? _p.y - s.alt_hae_m : 0;
        a.drop.position.copy(_p);
        a.drop.scale.y = _p.y - groundY;
        a.foot.position.set(_p.x, groundY + FOOT_Y, _p.z);
        a.drop.material.opacity = a.foot.material.opacity = DROP_OPACITY * opacity;
      }

      if (spinRotors && a.group.userData.rotors) {
        const axis = a.group.userData.spinAxis;
        for (const r of a.group.userData.rotors) r.rotation[axis] += dt * 40 * Math.max(0.25, spinRate);
      }

      if (showTrails && inRange) this._pushTrail(a.trail, _p);
      a.trail.line.visible = showTrails;
    }

    // Hide any aircraft the store no longer reports at this time.
    for (const a of this.fleet.values()) {
      if (a.frame !== frame) {
        a.group.visible = a.label.visible = a.drop.visible = a.foot.visible = false;
        a.trail.line.visible = false;
      }
    }

    this._trackCamera(dt);
  }

  // The fleet entry for `id` if it is on screen this frame, else null.
  _visible(id) {
    if (id == null) return null;
    const a = this.fleet.get(id);
    return a && a.group.visible ? a : null;
  }

  // Per-frame camera for follow and the tracked modes. With no visible
  // selection the camera holds still. Snaps on a mode's first frame, after a
  // seek, and when tracking resumes; glides when the selection / target change
  // or after a hand-off.
  _trackCamera(dt) {
    const mode = this._mode;
    if (mode === "orbit") return;
    const own = this._visible(this.selectedId);
    if (!own) { this._tracking = false; return; }
    let tgt = this._visible(this.targetId);
    if (tgt === own) tgt = null;
    const tgtId = tgt ? this.targetId : null;

    // A target change only moves the tracked point in top / side (the pair's
    // midpoint); follow and chase stay anchored on the selection.
    const tgtMoves = (mode === "top" || mode === "side") && tgtId !== this._trackTgt;
    const snap = this._snap || !this._tracking;
    const glide = !snap && (this._glide || this.selectedId !== this._trackSel || tgtMoves);
    this._snap = this._glide = false;
    this._tracking = true;
    this._trackSel = this.selectedId;
    this._trackTgt = tgtId;
    const p = own.group.position;

    if (mode === "follow") {
      // Orbit the selection, keeping the user's offset (camera − target): the
      // target follows the aircraft and the camera moves with it.
      const target = this.controls.target;
      this._glideErr(target, p, snap, glide, dt);
      _v.addVectors(p, this._anchorErr);
      _w.subVectors(_v, target);
      target.copy(_v);
      this.camera.position.add(_w);
      return;
    }

    const zoom = this._zoom[mode];
    const tanHalf = Math.tan(this.camera.fov * DEG2RAD / 2);
    const aspect = this.camera.aspect;
    let up = UP;

    if (mode === "chase") {
      // Behind and above along the smoothed flight path, looking ahead; with a
      // target, along the line of sight (padlock), looking at the target.
      _want.copy(p);
      if (tgt) _v.subVectors(tgt.group.position, p);
      const los = tgt ? _v.length() : 0;
      if (los > 1e-3) _v.divideScalar(los);
      else pathDir(own.sample, false, _v);
      smoothDir(this._chaseDir, _v, snap ? 1 : blend(dt, TAU_CHASE_DIR));
      _wantCam.copy(this._chaseDir).multiplyScalar(-CHASE.back * zoom).addScaledVector(UP, CHASE.rise * zoom);
      if (los > 1e-3) _wantLook.subVectors(tgt.group.position, p);
      else _wantLook.copy(this._chaseDir).multiplyScalar(CHASE.ahead * zoom);
    } else if (mode === "top") {
      // Straight down, north up; a pinned pair is centred and kept in frame.
      up = NORTH;
      let h = TOP_HEIGHT;
      if (tgt) {
        const b = tgt.group.position;
        _want.addVectors(p, b).multiplyScalar(0.5);
        const halfX = (Math.abs(p.x - b.x) / 2) * FIT_MARGIN + FIT_PAD;
        const halfZ = (Math.abs(p.z - b.z) / 2) * FIT_MARGIN + FIT_PAD;
        // Fit from the higher aircraft, which sits closer to the camera.
        const fit = Math.max(halfZ / tanHalf, halfX / (tanHalf * aspect)) + Math.abs(p.y - b.y) / 2;
        h = Math.max(h, fit);
      } else {
        _want.copy(p);
      }
      _wantCam.set(0, h * zoom, 0);
      _wantLook.set(0, 0, 0);
    } else {
      // Side: level, abeam the smoothed horizontal track (or, with a target,
      // perpendicular to the horizontal line of sight, framing both).
      let dist = SIDE_DIST;
      let haveLos = false;
      if (tgt) {
        const b = tgt.group.position;
        _v.set(b.x - p.x, 0, b.z - p.z);
        const lh = _v.length();
        if (lh > 1e-3) {
          haveLos = true;
          _v.divideScalar(lh);
          // The line of sight has no preferred sense: keep the camera on
          // whichever side it already is instead of swinging it across.
          if (_v.dot(this._sideDir) < 0) _v.negate();
          _want.addVectors(p, b).multiplyScalar(0.5);
          const halfL = (lh / 2) * FIT_MARGIN + FIT_PAD;
          const halfY = (Math.abs(p.y - b.y) / 2) * FIT_MARGIN + FIT_PAD;
          dist = Math.max(dist, halfL / (tanHalf * aspect), halfY / tanHalf);
        }
      }
      if (!haveLos) {
        pathDir(own.sample, true, _v);
        _want.copy(p);
      }
      smoothDir(this._sideDir, _v, snap ? 1 : blend(dt, TAU_SIDE_DIR));
      // Camera on the left of the track looking across it, so the motion runs
      // left → right on screen: forward = up × track, camera = −forward · d.
      const h = this._sideDir;
      _wantCam.set(-h.z, 0, h.x).multiplyScalar(dist * zoom);
      _wantLook.set(0, 0, 0);
    }

    this._glideErr(this._anchor, _want, snap, glide, dt);
    this._anchor.addVectors(_want, this._anchorErr);
    const k = snap ? 1 : blend(dt, TAU_OFFSET);
    this._camOff.lerp(_wantCam, k);
    this._lookOff.lerp(_wantLook, k);

    this.camera.position.addVectors(this._anchor, this._camOff);
    _v.addVectors(this._anchor, this._lookOff);
    _m4.lookAt(this.camera.position, _v, up);
    this.camera.quaternion.setFromRotationMatrix(_m4);
  }

  // Update `_anchorErr`, the offset of the tracked point from where it should
  // be: zero on a snap, `from − want` when a glide starts, then decaying.
  _glideErr(from, want, snap, glide, dt) {
    if (snap) this._anchorErr.set(0, 0, 0);
    else if (glide) this._anchorErr.subVectors(from, want);
    else this._anchorErr.multiplyScalar(1 - blend(dt, TAU_GLIDE));
  }

  // Frame the whole visible fleet (wherever it actually is — often ~km MSL, not
  // at the grid). Falls back to the origin when nothing is on screen yet.
  // Switches to free orbit, telling onCameraMode if that changed the mode.
  frameFleet() {
    const box = new THREE.Box3();
    let any = false;
    for (const { group } of this.fleet.values()) {
      if (group.visible) { box.expandByPoint(group.position); any = true; }
    }
    const center = new THREE.Vector3();
    let radius;
    if (any) {
      box.getCenter(center);
      radius = Math.max(box.getBoundingSphere(new THREE.Sphere()).radius, 150);
    } else {
      center.set(0, 60, 0);
      radius = 400;
    }
    const fov = THREE.MathUtils.degToRad(this.camera.fov);
    const dist = radius / Math.sin(fov / 2) + radius * 0.5;
    const dir = new THREE.Vector3(0.55, 0.5, 0.75).normalize();
    this.controls.target.copy(center);
    this.camera.position.copy(center).addScaledVector(dir, dist);
    if (this._mode !== "orbit") {
      this._mode = "orbit";
      this.controls.enabled = true;
      this.onCameraMode("orbit");
    }
    this.controls.update();
  }

  // World metres per CSS pixel at 1 m from the camera (vertical, at the centre).
  _pxScale() {
    return (2 * Math.tan(this.camera.fov * DEG2RAD / 2)) / this._h;
  }

  // Camera heading and scale for the DOM compass / scale bar. `headingRad` is
  // the compass bearing (0 = north, +π/2 = east) of the screen's up direction
  // on the ground; `metersPerPx` is the scale at the camera's focus point. The
  // returned object is reused every call: read it, don't keep it.
  viewInfo() {
    const q = this.camera.quaternion;
    _v.set(0, 0, -1).applyQuaternion(q); // forward
    if (_v.x * _v.x + _v.z * _v.z < 1e-8) _v.set(0, 1, 0).applyQuaternion(q); // looking straight down: use up
    let heading = Math.atan2(_v.x, -_v.z); // three: x = east, −z = north
    if (heading < 0) heading += 2 * Math.PI;
    const focus = this._focus();
    const info = this._viewInfo;
    info.headingRad = heading;
    info.metersPerPx = this.camera.position.distanceTo(focus) * this._pxScale();
    return info;
  }

  // The point the camera is about: the tracked point in a tracked mode, else
  // the OrbitControls target.
  _focus() {
    return TRACKED.has(this._mode) ? this._anchor : this.controls.target;
  }

  render() {
    if (this.controls.enabled) this.controls.update(); // tracked modes posed the camera in update()
    this._env.frame(this.camera, this._focus(), this._h);
    this._sizeForScreen();
    this.renderer.render(this.scene, this.camera);
    if (this.fleet.size) this._renderOutline();
  }

  // The outline: draw only MASK_LAYER (the bodies) flat white on black into
  // the mask, without background or fog, then composite the dilated edge over
  // the frame. Every renderer / scene / camera setting touched is restored.
  _renderOutline() {
    const r = this.renderer, scene = this.scene, cam = this.camera;
    const background = scene.background, fog = scene.fog, layers = cam.layers.mask;
    const target = r.getRenderTarget(), autoClear = r.autoClear;
    r.getClearColor(_clear);
    const clearAlpha = r.getClearAlpha();

    scene.background = null;
    scene.fog = null;
    scene.overrideMaterial = this._maskMat;
    cam.layers.set(MASK_LAYER);
    r.setRenderTarget(this._maskRT);
    r.setClearColor(0x000000, 1);
    r.autoClear = true;
    r.render(scene, cam);

    scene.overrideMaterial = null;
    scene.background = background;
    scene.fog = fog;
    cam.layers.mask = layers;
    r.setRenderTarget(target);
    r.setClearColor(_clear, clearAlpha);
    r.autoClear = false;
    r.render(this._outlineScene, this._outlineCam);
    r.autoClear = autoClear;
  }

  // Everything sized in screen terms, from the camera's final pose this frame:
  // aircraft size, name tags, drop-line foot rings, selection brackets, target
  // overlay.
  _sizeForScreen() {
    const px = this._pxScale();
    const cam = this.camera.position;
    const labelH = LABEL_PX * px;
    for (const [id, a] of this.fleet) {
      const g = a.group;
      if (!g.visible) continue;
      const dist = cam.distanceTo(g.position);
      const s = Math.max(this._modelScale, (MIN_AIRCRAFT_PX * px * dist) / g.userData.span);
      g.scale.setScalar(s);
      if (a.label.visible) {
        // Anchored at the aircraft and lifted in screen space just clear of the
        // model, or of the brackets / reticle framing it, so it never covers either.
        const ownPx = (g.userData.span * s) / (dist * px);
        let halfPx = ownPx / 2;
        if (id === this.selectedId && this._mode !== "chase") halfPx = Math.max(SELECT_PX, ownPx * 1.3) / 2;
        else if (id === this.targetId) halfPx = Math.max(RETICLE_PX, ownPx * 1.3) / 2;
        a.label.position.copy(g.position);
        a.label.center.y = -(halfPx + LABEL_GAP_PX) / LABEL_PX;
        a.label.scale.set(labelH * NAME_ASPECT, labelH, 1);
      }
      if (a.foot.visible) {
        a.foot.scale.setScalar(Math.max(FOOT_RADIUS, FOOT_MIN_PX * px * cam.distanceTo(a.foot.position)));
      }
    }
    this._updateSelectMarker(px);
    this._updateTargetOverlay(px);
  }

  // Brackets around the visible selection, at least SELECT_PX and wider than
  // a big close model; hidden in chase, which sits right behind it.
  _updateSelectMarker(px) {
    const own = this._mode === "chase" ? null : this._visible(this.selectedId);
    this._selectMarker.visible = !!own;
    if (!own) return;
    const g = own.group;
    const ownPx = (g.userData.span * g.scale.x) / (this.camera.position.distanceTo(g.position) * px);
    const sizePx = Math.max(SELECT_PX, ownPx * 1.3);
    if (sizePx > FRAME_MAX_PX) { this._selectMarker.visible = false; return; }
    this._selectMarker.position.copy(g.position);
    this._selectMarker.scale.set(sizePx * px, sizePx * px, 1);
  }

  // Line of sight, range tag and reticle while both the selection and the
  // target are on screen; hidden otherwise. Range is between rendered positions.
  _updateTargetOverlay(px) {
    const own = this._visible(this.selectedId);
    const tgt = this._visible(this.targetId);
    const show = !!own && !!tgt && own !== tgt;
    this._losLine.visible = this._rangeLabel.visible = this._reticle.visible = show;
    if (!show) return;

    const a = own.group.position, b = tgt.group.position;
    const range = a.distanceTo(b);
    const geom = this._losLine.geometry;
    const pos = geom.attributes.position;
    pos.setXYZ(0, a.x, a.y, a.z);
    pos.setXYZ(1, b.x, b.y, b.z);
    pos.needsUpdate = true;
    const dist = geom.attributes.lineDistance;
    dist.setX(1, range);
    dist.needsUpdate = true;

    _v.addVectors(a, b).multiplyScalar(0.5);
    const mPerPx = this.camera.position.distanceTo(_v) * px;
    const mat = this._losLine.material;
    mat.dashSize = DASH_PX * mPerPx;
    mat.gapSize = GAP_PX * mPerPx;

    this._rangeLabel.position.copy(_v);
    const labelH = LABEL_PX * px;
    this._rangeLabel.scale.set(labelH * RANGE_ASPECT, labelH, 1);
    this._setRangeText(range);

    // Frame the target: at least RETICLE_PX, wider than a big close model.
    const tgtPx = (tgt.group.userData.span * tgt.group.scale.x) / (this.camera.position.distanceTo(b) * px);
    const reticlePx = Math.max(RETICLE_PX, tgtPx * 1.3);
    this._reticle.visible = reticlePx <= FRAME_MAX_PX;
    this._reticle.position.copy(b);
    this._reticle.scale.set(reticlePx * px, reticlePx * px, 1);
  }

  // Redraw the range tag ("357 m", "1.24 km") only when the shown value changes.
  _setRangeText(range) {
    const key = range < 999.5 ? Math.round(range) : -Math.round(range / 10);
    if (key === this._rangeKey) return;
    this._rangeKey = key;
    const text = key >= 0 ? `${key} m` : `${(-key / 100).toFixed(2)} km`;
    const ctx = this._rangeCtx;
    ctx.clearRect(0, 0, 256, 64);
    ctx.fillStyle = "rgba(10,14,20,0.78)";
    ctx.beginPath();
    ctx.roundRect(40, 6, 176, 52, 12);
    ctx.fill();
    ctx.strokeStyle = "#ff5a4d";
    ctx.lineWidth = 3;
    ctx.stroke();
    ctx.fillStyle = "#dbe4ef";
    ctx.font = "bold 32px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, 128, 34);
    this._rangeLabel.material.map.needsUpdate = true;
  }

  // Track the container's size, and the outline mask with the drawing buffer;
  // skipped while it is 0 × 0 (e.g. hidden).
  resize() {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    if (!w || !h) return;
    this._h = h;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
    this.renderer.getDrawingBufferSize(_size);
    this._maskRT.setSize(_size.x, _size.y);
    const radius = OUTLINE_PX * this.renderer.getPixelRatio();
    this._outlineMat.uniforms.uStep.value.set(radius / _size.x, radius / _size.y);
  }

  // Tear down for a renderer swap: drop the fleet, detach listeners, release the
  // GL context and remove the canvas so the container is clean for the next scene.
  dispose() {
    this.clearFleet();
    this._resizeObserver.disconnect();
    const el = this.renderer.domElement;
    el.removeEventListener("pointerdown", this._onDown);
    el.removeEventListener("pointermove", this._onMove);
    el.removeEventListener("pointerup", this._onUp);
    el.removeEventListener("pointercancel", this._onCancel);
    el.removeEventListener("wheel", this._onWheel);
    this.controls.dispose();
    this._dropGeom.dispose();
    this._footGeom.dispose();
    this._env.dispose();
    this._losLine.geometry.dispose();
    for (const o of [this._losLine, this._rangeLabel, this._reticle, this._selectMarker]) {
      o.material.map?.dispose();
      o.material.dispose();
    }
    this._maskRT.dispose();
    this._maskMat.dispose();
    this._outlineQuad.geometry.dispose();
    this._outlineMat.dispose();
    this.renderer.dispose();
    if (el.parentNode) el.parentNode.removeChild(el);
  }

  _onPointerDown(e) {
    if (!e.isTrusted) return; // our own replay for OrbitControls (see _handoff)
    const d = this._down;
    // A second finger (pinch) is neither a click nor a hand-off drag.
    if (!e.isPrimary) { d.active = false; return; }
    d.active = true;
    d.id = e.pointerId; d.type = e.pointerType; d.button = e.button;
    d.x = e.clientX; d.y = e.clientY; d.t = e.timeStamp;
    d.shift = e.shiftKey; d.ctrl = e.ctrlKey; d.meta = e.metaKey;
    // OrbitControls is off in tracked modes, so capture the drag ourselves.
    if (TRACKED.has(this._mode) && e.button === 0) this.renderer.domElement.setPointerCapture(e.pointerId);
  }

  _onPointerMove(e) {
    const d = this._down;
    if (!d.active || e.pointerId !== d.id || d.button !== 0 || !TRACKED.has(this._mode)) return;
    if (Math.hypot(e.clientX - d.x, e.clientY - d.y) < CLICK_SLOP_PX) return;
    d.active = false;
    this._handoff(d);
  }

  // A click (short, nearly still press): select, or with Shift pin the target.
  // A click on empty space does nothing.
  _onPointerUp(e) {
    const d = this._down;
    if (!d.active || e.pointerId !== d.id) return;
    d.active = false;
    if (d.button !== 0) return;
    if (Math.hypot(e.clientX - d.x, e.clientY - d.y) >= CLICK_SLOP_PX) return;
    if (e.timeStamp - d.t >= CLICK_MAX_MS) return;
    const id = this._pick(e.clientX, e.clientY);
    if (id == null) return;
    if (e.shiftKey) this.onTarget(id);
    else this.onSelect(id);
  }

  // The aircraft under a client-space point: the nearest projected centre
  // within PICK_PX, else a raycast hit on a mesh (big close aircraft).
  _pick(clientX, clientY) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    const x = clientX - rect.left, y = clientY - rect.top;
    let best = null, bestD = PICK_PX;
    const groups = [];
    for (const [id, a] of this.fleet) {
      if (!a.group.visible) continue;
      groups.push(a.group);
      _v.copy(a.group.position).project(this.camera);
      if (_v.z < -1 || _v.z > 1) continue; // behind the camera / past the far plane
      const sx = ((_v.x + 1) / 2) * rect.width, sy = ((1 - _v.y) / 2) * rect.height;
      const dist = Math.hypot(sx - x, sy - y);
      if (dist <= bestD) { bestD = dist; best = id; }
    }
    if (best != null) return best;

    this._pointer.set((x / rect.width) * 2 - 1, -(y / rect.height) * 2 + 1);
    this._raycaster.setFromCamera(this._pointer, this.camera);
    const hits = this._raycaster.intersectObjects(groups, true);
    for (const hit of hits) {
      let o = hit.object;
      while (o && o.userData.id === undefined) o = o.parent;
      if (o) return o.userData.id;
    }
    return null;
  }

  // Wheel in a tracked mode scales that mode's camera distance (OrbitControls
  // dollies in orbit / follow).
  _onWheelZoom(e) {
    if (!TRACKED.has(this._mode)) return;
    e.preventDefault();
    let dy = e.deltaY;
    if (e.deltaMode === 1) dy *= 16;           // lines
    else if (e.deltaMode === 2) dy *= this._h; // pages
    const z = this._zoom[this._mode] * Math.exp(dy * ZOOM_PER_PX);
    this._zoom[this._mode] = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, z));
  }
}

// A soft white radial glow (tinted per flash by the sprite colour).
function glowTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 128;
  const ctx = canvas.getContext("2d");
  const g = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);
  g.addColorStop(0, "rgba(255,255,255,1)");
  g.addColorStop(0.2, "rgba(255,255,255,0.85)");
  g.addColorStop(0.5, "rgba(255,255,255,0.25)");
  g.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 128, 128);
  return new THREE.CanvasTexture(canvas);
}

// A white diamond outline with an open centre (tinted by the sprite colour).
function reticleTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 128;
  const ctx = canvas.getContext("2d");
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = 6;
  ctx.lineJoin = "miter";
  ctx.beginPath();
  ctx.moveTo(64, 6);
  ctx.lineTo(122, 64);
  ctx.lineTo(64, 122);
  ctx.lineTo(6, 64);
  ctx.closePath();
  ctx.stroke();
  return new THREE.CanvasTexture(canvas);
}

// Four white corner brackets with a thin dark edge, so they read on bright sky
// and on dark ground alike; the centre stays open over the aircraft.
function bracketTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 128;
  const ctx = canvas.getContext("2d");
  const inset = 10, arm = 32;
  const corners = () => {
    ctx.beginPath();
    for (const [x, y, sx, sy] of [[inset, inset, 1, 1], [128 - inset, inset, -1, 1], [inset, 128 - inset, 1, -1], [128 - inset, 128 - inset, -1, -1]]) {
      ctx.moveTo(x, y + sy * arm);
      ctx.lineTo(x, y);
      ctx.lineTo(x + sx * arm, y);
    }
  };
  ctx.lineCap = "square";
  ctx.lineJoin = "miter";
  corners();
  ctx.strokeStyle = "rgba(8,12,20,0.85)";
  ctx.lineWidth = 11;
  ctx.stroke();
  corners();
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = 5;
  ctx.stroke();
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

// The outline composite: a full-screen quad (clip-space positions, no camera
// transform) that takes the mask's maximum over 8 directions × 2 radii
// (`uStep` is the full radius in UV units) and paints what the dilation adds
// beyond the mask itself. The MSAA mask has soft edges, so the outline does.
const OUTLINE_VERT = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}`;
const OUTLINE_FRAG = /* glsl */ `
uniform sampler2D tMask;
uniform vec2 uStep;
uniform vec3 uColor;
uniform float uAlpha;
varying vec2 vUv;
void main() {
  float centre = texture2D(tMask, vUv).r;
  float near = 0.0;
  for (int i = 0; i < 8; i++) {
    float a = float(i) * 0.7853982;
    vec2 d = vec2(cos(a), sin(a)) * uStep;
    near = max(near, max(texture2D(tMask, vUv + d).r, texture2D(tMask, vUv + 0.5 * d).r));
  }
  float edge = near - centre;
  if (edge <= 0.004) discard;
  gl_FragColor = vec4(uColor, edge * uAlpha);
  #include <colorspace_fragment>
}`;
