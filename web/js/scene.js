// The 3-D view: scene, camera, the environment (environment.js: background,
// fog, lights, grid, and in "sky" mode the sky, clouds and streamed ground
// map), and one live object per aircraft driven from the store each frame.
// Also owns:
//
//   * sizing — the renderer tracks its container (a grid cell that docks
//     resize) through a ResizeObserver, not window resizes;
//   * the camera modes — free orbit, follow (orbit around the selection), the
//     tracked chase / top / side views, which the scene drives itself each
//     frame (a drag in a tracked view hands the camera back to the user), and
//     "map": a top-down orthographic camera, north up, that pans and zooms on
//     its own, independent of the selection;
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
//   * a race course (setCourse): one frame per gate — drawn at the model scale
//     about the centre of its opening, as the aircraft are, so a drone and the
//     gate it flies through stay in proportion — a number tag, a line through
//     the gate centres, and the pending gate highlighted (setPendingGate).
//     While a course is on screen multirotors are drawn at true size
//     (meshes.js F450_UNIT) instead of the sandbox's exaggerated quad mesh;
//     neither gates nor drone get a minimum on-screen size then, and the chase /
//     top / side cameras sit metres away, not hundreds;
//   * viewInfo(), the camera heading and scale the DOM compass / scale bar use;
//   * edit mode (setEditing), the scenario editor's interaction: click /
//     Shift+click / Shift+box selection, dragging aircraft on the horizontal
//     plane at their altitude, a heading handle and an altitude handle (or
//     Alt+drag), click-on-ground points, and translucent preview "ghost"
//     aircraft (setGhosts). The scene only reports what the pointer did
//     (ENU / heading values through callbacks) and draws what its store says;
//     the editor owns the scenario and feeds the scene back via editstore.js.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import {
  buildAircraft, buildGhostGeometry, enuToThree, attitudeToQuaternion, aircraftColor, tintAircraft, setTwoTone,
  buildGate, buildGateLabel, placeGate, setGateState, F450_UNIT,
} from "./meshes.js";
import { Environment } from "./environment.js";
import { DEG2RAD, RAD2DEG } from "./geo.js";

const TRAIL_MAX = 240;      // points per trail
const TRAIL_MIN_STEP = 1.0; // metres between recorded trail points

// Event flashes at an event's `pos_enu`: world size (m, the sprite's full width
// at peak) and lifetime (real seconds). Kills are the prominent one.
const FLASH = {
  hit: { size: 45, life: 0.45, color: 0xffc46a },
  kill: { size: 260, life: 1.4, color: 0xff6a3a },
};
const FLASH_MAX = 96; // live flashes at once; older ones are dropped first

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
  chart: { color: 0x8fd0ff, alpha: 0.85 }, // a light halo: glowing markers on the dark chart
};

// Picking: a click is a short, nearly still press; the nearest aircraft within
// PICK_PX of the pointer (projected centre) wins, so tiny distant ones stay
// clickable.
const CLICK_SLOP_PX = 5;
const CLICK_MAX_MS = 400;
const PICK_PX = 16;

// Camera modes. Tracked modes are driven by the scene, with OrbitControls off.
const CAMERA_MODES = new Set(["orbit", "follow", "chase", "top", "side", "map"]);
const TRACKED = new Set(["chase", "top", "side"]);
const CHASE = { back: 140, rise: 32, ahead: 300 }; // m behind / above, look-ahead
const TOP_HEIGHT = 900;   // m above the selection (minimum when framing a pair)
const SIDE_DIST = 420;    // m abeam (minimum when framing a pair)
// The same views while a race course is on screen: a 0.57 m drone and 1 m gates
// are seen from metres away, not hundreds.
const RACE_CHASE = { back: 5, rise: 1.5, ahead: 12 };
const RACE_TOP_HEIGHT = 25;
const RACE_SIDE_DIST = 8;
const RACE_NEAR = 0.1, RACE_FAR = 20000; // the perspective clip planes with a course (else 1, 500000)
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

// The map camera: orthographic, straight down from MAP_HEIGHT above the grid,
// screen-up = north; its scale is metres per CSS pixel, zoomed about the cursor.
const MAP_HEIGHT = 100000;
const MAP_MPP_DEFAULT = 4, MAP_MPP_MIN = 0.05, MAP_MPP_MAX = 400;
const MAP_FIT_MPP_MIN = 1.5; // framing never zooms past the imagery's finest detail (~zoom 16)

// Edit handles on the primary selection: a heading knob HEADING_KNOB_PX out
// along the nose, an altitude knob ALT_KNOB_PX out behind the tail (both CSS
// px from the aircraft's centre, KNOB_PX across), and the selection box.
const KNOB_PX = 22;
const HEADING_KNOB_PX = 70;
const ALT_KNOB_PX = 48;
const KNOB_HIT_PX = 15;
const HANDLE_COLOR = 0xffd166;
const ROTATE_SNAP_DEG = 5;  // Shift while rotating snaps the (true) heading to this
// Perspective drags on a horizontal plane: a pointer ray within 3° of the
// plane is ignored, and a hit may land at most DRAG_REACH × the grabbed
// aircraft's distance from the camera.
const GRAZE_SIN = Math.sin(3 * Math.PI / 180);
const DRAG_REACH = 4;
// Preview ghosts (setGhosts): one instanced, flat, translucent silhouette per
// item, at least PREVIEW_MIN_PX on screen.
const PREVIEW_OPACITY = 0.6;
const PREVIEW_MIN_PX = 28;
const PREVIEW_COLOR = 0x8fe3ff; // a light cyan that reads on imagery and on the dark grid

// Race course: the line through the gate centres is COURSE_LINE_COLOR.
const COURSE_LINE_COLOR = 0x8fb4e0;
const FOOT_COURSE_RADIUS = 0.4; // m: a drone's ground ring with a course on screen (0.57 m drone)

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
const _plane = new THREE.Plane();
const _hit = new THREE.Vector3();
const _s3 = new THREE.Vector3();
const _geo = { lat: 0, lon: 0, h: 0 };

// Frame-rate-independent exponential smoothing: the blend toward the goal
// after `dt` seconds with time constant `tau`.
function blend(dt, tau) { return 1 - Math.exp(-dt / tau); }

// Unit flight-path direction of a sample in world (three) axes; the heading
// (`psi`, the fleet entry's grid bearing) when it is too slow for the velocity
// to mean anything. `horizontal` drops the vertical component (the side
// view's track).
function pathDir(s, psi, horizontal, out) {
  out.set(s.vel_e_mps, horizontal ? 0 : s.vel_u_mps, -s.vel_n_mps);
  const speed = out.length();
  if (speed < SLOW_MPS) return out.set(Math.sin(psi), 0, -Math.cos(psi));
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
    this._w = 1;             // container width / height (CSS px), kept by resize()
    this._h = 1;
    this._flashes = [];      // { sprite, age, life, size }
    this._flashTex = null;   // shared radial glow, built on first use
    this._modelScale = MODEL_SCALE_DEFAULT;
    this._course = null;     // { gates: [{ index, group, label, center }], line }, see setCourse
    this._pending = 0;       // index of the gate to highlight (setPendingGate)

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    // Block-level so the canvas adds no inline line box to the container.
    this.renderer.domElement.style.display = "block";
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this._env = new Environment(this.scene, { maxAnisotropy: this.renderer.capabilities.getMaxAnisotropy() });

    // The active camera is `camera`: the perspective one (OrbitControls' and
    // every mode but "map"), or the map's orthographic one.
    this._persp = new THREE.PerspectiveCamera(55, 1, 1, 500000);
    this._persp.position.set(320, 240, 420);
    this.camera = this._persp;
    this._ortho = new THREE.OrthographicCamera(-1, 1, 1, -1, 1, MAP_HEIGHT * 3);
    this._ortho.rotation.set(-Math.PI / 2, 0, 0); // straight down, screen-up = north (−z)
    this._mapCenter = new THREE.Vector3(); // the map view's centre on the grid (y = 0)
    this._mapMpp = MAP_MPP_DEFAULT;        // its scale, m per CSS px

    this.controls = new OrbitControls(this._persp, this.renderer.domElement);
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
    this._addEditResources();
    this._applyEnvironment();

    this._raycaster = new THREE.Raycaster();
    this._pointer = new THREE.Vector2();
    // The press being watched for a click (or, in a tracked mode, a drag).
    this._down = { active: false, id: 0, type: "mouse", button: 0, x: 0, y: 0, t: 0, shift: false, ctrl: false, meta: false };
    // Edit mode (setEditing): the handlers, the selection, the press / drag in
    // progress and the preview ghosts.
    this._edit = null;
    this._editSel = [];
    this._editSelSet = new Set();
    this._drag = null;
    this._ghostLayers = new Map();          // layer key → [{ mesh, pos, quat, span }]
    this._ghostCam = new THREE.Matrix4();   // the camera pose the ghosts were last sized for
    this._ghostStamp = NaN;                 // …and the px scale × model scale
    // Keep handler refs so dispose() can detach them on a renderer swap.
    this._onDown = (e) => this._onPointerDown(e);
    this._onMove = (e) => this._onPointerMove(e);
    this._onUp = (e) => this._onPointerUp(e);
    this._onCancel = () => { this._down.active = false; };
    this._onWheel = (e) => this._onWheelZoom(e);
    // The map pans with any button; a right-drag must not open the menu.
    this._onMenu = (e) => { if (this._mode === "map") e.preventDefault(); };
    // Edit presses and map pans are watched in the capture phase, ahead of
    // OrbitControls, so a press they claim never reaches it.
    this._onCapDown = (e) => this._onCaptureDown(e);
    this._onCapMove = (e) => this._onCaptureMove(e);
    this._onCapUp = (e) => this._onCaptureUp(e);
    this._onCapCancel = () => this.cancelEdit();
    const el = this.renderer.domElement;
    el.addEventListener("pointerdown", this._onCapDown, { capture: true });
    el.addEventListener("pointermove", this._onCapMove, { capture: true });
    el.addEventListener("pointerup", this._onCapUp, { capture: true });
    el.addEventListener("pointercancel", this._onCapCancel, { capture: true });
    el.addEventListener("pointerdown", this._onDown);
    el.addEventListener("pointermove", this._onMove);
    el.addEventListener("pointerup", this._onUp);
    el.addEventListener("pointercancel", this._onCancel);
    el.addEventListener("wheel", this._onWheel, { passive: false });
    el.addEventListener("contextmenu", this._onMenu);

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

  // Drop every aircraft and event flash and free all of their GPU resources
  // (see _disposeAircraft) and the flash glow texture.
  clearFleet() {
    for (const a of this.fleet.values()) this._disposeAircraft(a);
    this.fleet.clear();
    this.selectedId = null;
    this.targetId = null;
    this._tracking = false;
    this._env.reset(); // a new source may sit at a different datum
    this._clearFlashes();
  }

  // A course is drawn at true size (see _sizeForScreen), so its camera sees
  // gates from well under a metre: the perspective clip planes follow it.
  _clipForCourse(on) {
    this._persp.near = on ? RACE_NEAR : 1;
    this._persp.far = on ? RACE_FAR : 500000;
    this._persp.updateProjectionMatrix();
  }

  // Show a race course — the validated course JSON (race.js parseCourse) — in
  // place of the current one, or none with null. Gate i is built from the
  // course's dimensions and stands where its centre / yaw say (meshes.js
  // describes the placement); the highlighted one follows setPendingGate.
  setCourse(course) {
    this._disposeCourse();
    this._clipForCourse(!!course);
    if (!course) return;
    const gates = course.gates.map((g) => {
      const group = buildGate(course);
      placeGate(group, g);
      const label = buildGateLabel(g.index);
      label.visible = false; // placed by _sizeForScreen
      this.scene.add(group, label);
      return { index: g.index, group, label, center: group.position.clone() };
    });
    const geom = new THREE.BufferGeometry().setFromPoints(gates.map((g) => g.center));
    const line = new THREE.Line(geom, new THREE.LineBasicMaterial({
      color: COURSE_LINE_COLOR, transparent: true, opacity: 0.6, fog: false,
    }));
    this.scene.add(line);
    this._course = { gates, line };
    this._paintGates();
  }

  // Highlight gate `index` as the next to fly; the gates before it are passed
  // (all of them, once it is past the last).
  setPendingGate(index) {
    if (index === this._pending) return;
    this._pending = index;
    this._paintGates();
  }

  _paintGates() {
    if (!this._course) return;
    for (const g of this._course.gates) {
      setGateState(g.group, g.index < this._pending ? "passed" : g.index === this._pending ? "pending" : "future", g.label);
    }
  }

  // Take the course out of the scene and free its geometries, materials and
  // number-tag textures.
  _disposeCourse() {
    const c = this._course;
    if (!c) return;
    this._course = null;
    for (const { group, label } of c.gates) {
      this.scene.remove(group, label);
      group.traverse((o) => o.geometry?.dispose());
      for (const m of group.userData.mats) m.dispose();
      label.material.map.dispose();
      label.material.dispose();
    }
    this.scene.remove(c.line);
    c.line.geometry.dispose();
    c.line.material.dispose();
  }

  // Take one aircraft out of the scene and free its geometries, materials
  // (shared ones once; cloud ghosts included, as they are children of the
  // group) and textures (the canvas label sprite). Sprites share one
  // three.js-owned quad geometry, which stays; so do the scene-owned
  // drop-line / foot-ring geometries (dispose() frees them).
  _disposeAircraft({ group, label, drop, foot, trail }) {
    const geometries = new Set(), materials = new Set();
    const collect = (o) => {
      if (o.geometry && !o.isSprite) geometries.add(o.geometry);
      if (o.material) for (const m of [].concat(o.material)) materials.add(m);
    };
    for (const o of [group, label, drop, foot, trail.line]) this.scene.remove(o);
    group.traverse(collect);
    collect(label);
    collect(trail.line);
    materials.add(drop.material);
    materials.add(foot.material);
    for (const g of geometries) g.dispose();
    for (const m of materials) { m.map?.dispose(); m.dispose(); }
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

  // "plain" (dark, no map), "sky" (daytime sky, clouds and ground map) or
  // "map" (the editor's daylight and ground map, no clouds); idempotent, and
  // kept across clearFleet(). Aircraft recolour on their next update(); cloud
  // ghosts come and go with the sky right away.
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

  // Whether the environment is a daylight one (sky or map): team colours,
  // two-tone bodies, the dark outline, drop lines to the curved sea level.
  get _daylight() {
    return this._env.mode !== "plain";
  }

  // Place the map's ENU origin from a scenario `[world]` (degrees, m HAE), for
  // a scene with no telemetry to solve it from (the editor). Kept until
  // clearFleet() or another setOrigin().
  setOrigin(latDeg, lonDeg, altHaeM) {
    this._env.setOrigin(latDeg, lonDeg, altHaeM);
  }

  // The editor map's look: "imagery" (the default: daylight dome and ground
  // map) or "chart" (a dark star chart with a zoom-aware grid; aircraft get a
  // light halo instead of the dark outline). Only map environment shows it.
  setMapLook(look) {
    if (look !== "imagery" && look !== "chart") throw new Error(`unknown map look: ${look}`);
    this._env.setChart(look === "chart");
    this._applyEnvironment();
  }

  // The per-environment state outside environment.js: the body two-tone
  // (module-wide in meshes.js, so set from the scene that is live) and the
  // outline colour.
  _applyEnvironment() {
    const day = this._daylight;
    setTwoTone(day);
    const o = this._env.chart ? OUTLINE.chart : day ? OUTLINE.sky : OUTLINE.plain;
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

  // Switch camera mode ("orbit" | "follow" | "chase" | "top" | "side" |
  // "map"). The first frame of the new mode snaps to its pose. Leaving a
  // tracked mode hands OrbitControls the current view; entering or leaving
  // the map carries the view's centre and scale across.
  setCameraMode(mode) {
    if (!CAMERA_MODES.has(mode)) throw new Error(`unknown camera mode: ${mode}`);
    if (mode === this._mode) return;
    this.cancelEdit();
    if (this._mode === "map") this._perspFromMap();
    if (TRACKED.has(this._mode)) this._handToControls();
    else if (TRACKED.has(mode)) this._anchor.copy(this.controls.target); // focus until something is tracked
    if (mode === "map") this._mapFromPersp();
    this._mode = mode;
    this.camera = mode === "map" ? this._ortho : this._persp;
    this.controls.enabled = !TRACKED.has(mode) && mode !== "map";
    this._snap = true;
  }

  // Centre the map on the orbit target at the scale the perspective view
  // has there.
  _mapFromPersp() {
    const t = this.controls.target;
    this._mapCenter.set(t.x, 0, t.z);
    const mpp = this._persp.position.distanceTo(t) * this._perspPxScale();
    this._mapMpp = THREE.MathUtils.clamp(mpp, MAP_MPP_MIN, MAP_MPP_MAX);
  }

  // Aim the perspective view at the map's centre, at the visible fleet's mean
  // height (the grid without one), from the south and above, about as wide as
  // the map was.
  _perspFromMap() {
    let y = 0, n = 0;
    for (const { group } of this.fleet.values()) if (group.visible) { y += group.position.y; n++; }
    const t = this.controls.target;
    t.set(this._mapCenter.x, n ? y / n : 0, this._mapCenter.z);
    const dist = Math.max(50, (this._mapMpp * this._h) / 2 / Math.tan((this._persp.fov * DEG2RAD) / 2));
    _v.set(0, 0.82, 0.57).normalize();
    this._persp.position.copy(t).addScaledVector(_v, dist);
    this._persp.lookAt(t);
  }

  // The map view's centre on the grid (ENU metres) — the orbit target outside
  // the map — and moving it there (the orbit camera moves along).
  viewCenter() {
    const c = this._mode === "map" ? this._mapCenter : this._focus();
    return { e: c.x, n: -c.z };
  }

  setViewCenter(e, n) {
    if (!Number.isFinite(e) || !Number.isFinite(n)) return;
    if (this._mode === "map") { this._mapCenter.set(e, 0, -n); return; }
    const t = this.controls.target;
    _v.set(e - t.x, 0, -n - t.z);
    t.add(_v);
    this._persp.position.add(_v);
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

  // The fleet entry for `id`, built on first sight — and rebuilt when its
  // vehicle kind changed (an editor switching an aircraft's model).
  _ensure(id, kind, team) {
    let a = this.fleet.get(id);
    if (a && a.kind === kind) return a;
    if (a) { this._disposeAircraft(a); this.fleet.delete(id); }
    const sky = this._daylight;
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
    if (this._env.mode === "sky") this._addGhosts(a);
    this.fleet.set(id, a);
    return a;
  }

  // Recolour `a` if its team or the environment changed since it was last
  // coloured: the body (meshes.js tintAircraft: materials, label border; the
  // ghosts share the body's Color objects), trail, drop line and foot ring.
  _colorize(a, team) {
    const sky = this._daylight;
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
      // The attitude is about the local level at the aircraft; its heading
      // turns onto the grid by the meridian convergence there.
      a.psi = s.psi_rad + this._northBearing(s.lat_deg, s.lon_deg);
      attitudeToQuaternion(s.phi_rad, s.theta_rad, a.psi, _q);
      a.group.quaternion.copy(_q);
      a.group.visible = true;

      // Stale (outside this track's window) or destroyed → dim it.
      const opacity = inRange && s.alive ? 1.0 : DIM;
      for (const m of a.group.userData.mats) {
        m.transparent = opacity < 1;
        m.opacity = opacity;
      }
      if (a.ghostMats) for (const m of a.ghostMats) m.opacity = GHOST_OPACITY * m.userData.src.opacity;

      const selected = id === this.selectedId || this._editSelSet.has(id);
      for (const m of a.group.userData.mats) {
        m.emissive.copy(a.group.userData.baseColor).multiplyScalar(selected ? SELECT_EMISSIVE : 0.0);
      }

      a.label.visible = showLabels || selected;

      // Altitude cue: the unit drop line scaled down to the ground — y = 0 in
      // plain mode; in daylight the sea-level surface under the aircraft, which
      // falls below the tangent plane with distance from the origin.
      a.drop.visible = a.foot.visible = showDropLines;
      if (showDropLines) {
        const groundY = this._daylight ? _p.y - s.alt_hae_m : 0;
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

    // Hide any aircraft the store no longer reports at this time; an editor's
    // store holds exactly the scenario, so in edit mode they are gone for good.
    for (const [id, a] of this.fleet) {
      if (a.frame === frame) continue;
      if (this._edit) {
        this._disposeAircraft(a);
        this.fleet.delete(id);
        continue;
      }
      a.group.visible = a.label.visible = a.drop.visible = a.foot.visible = false;
      a.trail.line.visible = false;
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
    if (mode === "orbit" || mode === "map") return;
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
      else pathDir(own.sample, own.psi, false, _v);
      smoothDir(this._chaseDir, _v, snap ? 1 : blend(dt, TAU_CHASE_DIR));
      const view = this._course ? RACE_CHASE : CHASE;
      _wantCam.copy(this._chaseDir).multiplyScalar(-view.back * zoom).addScaledVector(UP, view.rise * zoom);
      if (los > 1e-3) _wantLook.subVectors(tgt.group.position, p);
      else _wantLook.copy(this._chaseDir).multiplyScalar(view.ahead * zoom);
    } else if (mode === "top") {
      // Straight down, north up; a pinned pair is centred and kept in frame.
      up = NORTH;
      let h = this._course ? RACE_TOP_HEIGHT : TOP_HEIGHT;
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
      let dist = this._course ? RACE_SIDE_DIST : SIDE_DIST;
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
        pathDir(own.sample, own.psi, true, _v);
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
  // The map stays the map (centred and scaled to fit); any other mode
  // switches to free orbit, telling onCameraMode if that changed the mode.
  frameFleet() {
    const box = new THREE.Box3();
    let any = false;
    for (const { group } of this.fleet.values()) {
      if (group.visible) { box.expandByPoint(group.position); any = true; }
    }
    if (this._course) {
      for (const g of this._course.gates) box.expandByPoint(g.center);
      any = true;
    }
    const center = new THREE.Vector3();
    if (this._mode === "map") {
      let mpp = MAP_MPP_DEFAULT;
      if (any) {
        box.getCenter(center);
        const w = box.max.x - box.min.x + 2 * FIT_PAD, h = box.max.z - box.min.z + 2 * FIT_PAD;
        mpp = Math.max(w / this._w, h / this._h, MAP_FIT_MPP_MIN / FIT_MARGIN) * FIT_MARGIN;
      }
      this._mapCenter.set(center.x, 0, center.z);
      this._mapMpp = THREE.MathUtils.clamp(mpp, MAP_MPP_MIN, MAP_MPP_MAX);
      return;
    }
    let radius;
    if (any) {
      box.getCenter(center);
      // A race course is metres across: no 150 m floor, or it frames as a speck.
      radius = Math.max(box.getBoundingSphere(new THREE.Sphere()).radius, this._course ? 8 : 150);
    } else {
      center.set(0, 60, 0);
      radius = 400;
    }
    const fov = THREE.MathUtils.degToRad(this._persp.fov);
    const dist = radius / Math.sin(fov / 2) + radius * 0.5;
    const dir = new THREE.Vector3(0.55, 0.5, 0.75).normalize();
    this.controls.target.copy(center);
    this._persp.position.copy(center).addScaledVector(dir, dist);
    if (this._mode !== "orbit") {
      this._mode = "orbit";
      this.controls.enabled = true;
      this.onCameraMode("orbit");
    }
    this.controls.update();
  }

  // On-screen scale: world metres per CSS pixel are `_pxScale() × _depth(p)`
  // at point p. Perspective: the scale at 1 m from the camera (vertical, at
  // the centre) × the distance; the map: its one scale × 1. Sprites with
  // sizeAttenuation off take the same product as their world size.
  _pxScale() {
    return this._mode === "map" ? this._mapMpp : this._perspPxScale();
  }

  _perspPxScale() {
    return (2 * Math.tan(this._persp.fov * DEG2RAD / 2)) / this._h;
  }

  _depth(p) {
    return this._mode === "map" ? 1 : this.camera.position.distanceTo(p);
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
    info.metersPerPx = this._depth(focus) * this._pxScale();
    return info;
  }

  // The point the camera is about: the tracked point in a tracked mode, the
  // map's centre, else the OrbitControls target.
  _focus() {
    if (this._mode === "map") return this._mapCenter;
    return TRACKED.has(this._mode) ? this._anchor : this.controls.target;
  }

  // Pose the map camera: straight down over its centre, the frustum the
  // container's size × the scale.
  _layoutMap() {
    const o = this._ortho, hw = (this._w * this._mapMpp) / 2, hh = (this._h * this._mapMpp) / 2;
    o.left = -hw; o.right = hw; o.top = hh; o.bottom = -hh;
    o.position.set(this._mapCenter.x, MAP_HEIGHT, this._mapCenter.z);
    o.updateProjectionMatrix();
    o.updateMatrixWorld();
  }

  render() {
    if (this._mode === "map") this._layoutMap();
    else if (this.controls.enabled) this.controls.update(); // tracked modes posed the camera in update()
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
  // overlay, and in edit mode the handles and preview ghosts.
  _sizeForScreen() {
    const px = this._pxScale();
    const labelH = LABEL_PX * px;
    for (const [id, a] of this.fleet) {
      const g = a.group;
      if (!g.visible) continue;
      const dist = this._depth(g.position);
      // A drone is drawn at true size next to a course's gates.
      const unit = this._course && g.userData.kind === "multirotor" ? F450_UNIT : 1;
      // With a course on screen nothing is inflated past the Size setting: no
      // minimum on-screen size (1× is exactly true scale); the selection
      // brackets and the name tag keep a far drone findable.
      const s = this._course
        ? this._modelScale * unit
        : Math.max(this._modelScale, (MIN_AIRCRAFT_PX * px * dist) / g.userData.span);
      g.scale.setScalar(s);
      if (a.label.visible) {
        // Anchored at the aircraft and lifted in screen space just clear of the
        // model, or of the brackets / reticle framing it, so it never covers either.
        const ownPx = (g.userData.span * s) / (dist * px);
        let halfPx = ownPx / 2;
        const selected = id === this.selectedId || this._editSelSet.has(id);
        if (selected && this._mode !== "chase") halfPx = Math.max(SELECT_PX, ownPx * 1.3) / 2;
        else if (id === this.targetId) halfPx = Math.max(RETICLE_PX, ownPx * 1.3) / 2;
        a.label.position.copy(g.position);
        a.label.center.y = -(halfPx + LABEL_GAP_PX) / LABEL_PX;
        a.label.scale.set(labelH * NAME_ASPECT, labelH, 1);
      }
      if (a.foot.visible) {
        a.foot.scale.setScalar(this._course ? FOOT_COURSE_RADIUS
          : Math.max(FOOT_RADIUS, FOOT_MIN_PX * px * this._depth(a.foot.position)));
      }
    }
    if (this._course) this._sizeCourse(px, labelH);
    this._updateSelectMarker(px);
    this._updateTargetOverlay(px);
    this._updateHandles(px);
    this._sizeGhosts(px);
  }

  // Gate sizes and number tags: a gate is drawn at the model scale about the
  // centre of its opening — exactly true size at 1×, with no minimum on-screen
  // size — and its tag (a constant screen size) sits just above the frame's top
  // edge.
  _sizeCourse(px, labelH) {
    for (const { group, label } of this._course.gates) {
      const { top } = group.userData;
      const dist = this._depth(group.position);
      const s = this._modelScale;
      group.scale.setScalar(s);
      label.visible = true;
      label.position.copy(group.position);
      label.center.y = -((top * s) / (dist * px) + LABEL_GAP_PX) / LABEL_PX;
      label.scale.set(labelH * NAME_ASPECT, labelH, 1);
    }
  }

  // The bracket size (CSS px) around a built aircraft: at least SELECT_PX and
  // wider than a big close model.
  _bracketPx(g, px) {
    const ownPx = (g.userData.span * g.scale.x) / (this._depth(g.position) * px);
    return Math.max(SELECT_PX, ownPx * 1.3);
  }

  // Brackets around the visible selection, hidden in chase, which sits right
  // behind it; in edit mode around every selected aircraft instead.
  _updateSelectMarker(px) {
    if (this._edit) {
      this._selectMarker.visible = false;
      this._updateEditMarkers(px);
      return;
    }
    const own = this._mode === "chase" ? null : this._visible(this.selectedId);
    this._selectMarker.visible = !!own;
    if (!own) return;
    const g = own.group;
    const sizePx = this._bracketPx(g, px);
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
    const mPerPx = this._depth(_v) * px;
    const mat = this._losLine.material;
    mat.dashSize = DASH_PX * mPerPx;
    mat.gapSize = GAP_PX * mPerPx;

    this._rangeLabel.position.copy(_v);
    const labelH = LABEL_PX * px;
    this._rangeLabel.scale.set(labelH * RANGE_ASPECT, labelH, 1);
    this._setRangeText(range);

    // Frame the target: at least RETICLE_PX, wider than a big close model.
    const tgtPx = (tgt.group.userData.span * tgt.group.scale.x) / (this._depth(b) * px);
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
    this._w = w;
    this._h = h;
    this._persp.aspect = w / h;
    this._persp.updateProjectionMatrix();
    this.renderer.setSize(w, h);
    this.renderer.getDrawingBufferSize(_size);
    this._maskRT.setSize(_size.x, _size.y);
    const radius = OUTLINE_PX * this.renderer.getPixelRatio();
    this._outlineMat.uniforms.uStep.value.set(radius / _size.x, radius / _size.y);
  }

  // Tear down for a renderer swap: drop the fleet, detach listeners, release the
  // GL context and remove the canvas so the container is clean for the next scene.
  dispose() {
    this.cancelEdit();
    this.clearFleet();
    this._disposeCourse();
    for (const layer of [...this._ghostLayers.keys()]) this._dropGhostLayer(layer);
    this._resizeObserver.disconnect();
    const el = this.renderer.domElement;
    el.removeEventListener("pointerdown", this._onCapDown, { capture: true });
    el.removeEventListener("pointermove", this._onCapMove, { capture: true });
    el.removeEventListener("pointerup", this._onCapUp, { capture: true });
    el.removeEventListener("pointercancel", this._onCapCancel, { capture: true });
    el.removeEventListener("pointerdown", this._onDown);
    el.removeEventListener("pointermove", this._onMove);
    el.removeEventListener("pointerup", this._onUp);
    el.removeEventListener("pointercancel", this._onCancel);
    el.removeEventListener("wheel", this._onWheel);
    el.removeEventListener("contextmenu", this._onMenu);
    this.controls.dispose();
    for (const { geom } of this._previewGeoms.values()) geom.dispose();
    this._headingLine.geometry.dispose();
    for (const o of [this._headingLine, this._headingKnob, this._altKnob]) {
      o.material.map?.dispose();
      o.material.dispose();
    }
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
    if (!e.isTrusted || this._edit) return; // our own replay for OrbitControls (see _handoff); edit mode has its own
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
    if (this._edit) return;
    if (!d.active || e.pointerId !== d.id || d.button !== 0 || !TRACKED.has(this._mode)) return;
    if (Math.hypot(e.clientX - d.x, e.clientY - d.y) < CLICK_SLOP_PX) return;
    d.active = false;
    this._handoff(d);
  }

  // A click (short, nearly still press): select, or with Shift pin the target.
  // A click on empty space does nothing.
  _onPointerUp(e) {
    const d = this._down;
    if (this._edit) return;
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
  // dollies in orbit / follow); on the map it zooms about the cursor.
  _onWheelZoom(e) {
    if (this._mode === "map") {
      e.preventDefault();
      const rect = this.renderer.domElement.getBoundingClientRect();
      const sx = e.clientX - rect.left - this._w / 2, sy = e.clientY - rect.top - this._h / 2;
      const before = this._mapMpp;
      const after = THREE.MathUtils.clamp(before * Math.exp(wheelPx(e, this._h) * ZOOM_PER_PX), MAP_MPP_MIN, MAP_MPP_MAX);
      // Keep the ground point under the cursor where it is.
      this._mapCenter.x += sx * (before - after);
      this._mapCenter.z += sy * (before - after);
      this._mapMpp = after;
      return;
    }
    if (!TRACKED.has(this._mode)) return;
    e.preventDefault();
    const z = this._zoom[this._mode] * Math.exp(wheelPx(e, this._h) * ZOOM_PER_PX);
    this._zoom[this._mode] = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, z));
  }

  // ---- Edit mode ----------------------------------------------------------

  // The handles (heading knob + line, altitude knob), the pool of selection
  // brackets, and the preview ghosts' per-kind geometry cache.
  _addEditResources() {
    const knob = (tex) => {
      const s = new THREE.Sprite(new THREE.SpriteMaterial({
        map: tex, transparent: true, depthTest: false, depthWrite: false, sizeAttenuation: false, fog: false,
      }));
      s.renderOrder = 13;
      s.visible = false;
      this.scene.add(s);
      return s;
    };
    this._headingKnob = knob(knobTexture("rotate"));
    this._altKnob = knob(knobTexture("raise"));
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(6), 3));
    this._headingLine = new THREE.Line(geom, new THREE.LineBasicMaterial({
      color: HANDLE_COLOR, transparent: true, opacity: 0.9, depthTest: false, depthWrite: false, fog: false,
    }));
    this._headingLine.frustumCulled = false;
    this._headingLine.renderOrder = 12;
    this._headingLine.visible = false;
    this.scene.add(this._headingLine);
    // Canvas-px centres of the knobs as last drawn, for hit tests.
    this._knobs = { id: null, rotate: { x: NaN, y: NaN }, raise: { x: NaN, y: NaN } };
    this._editMarkers = [];         // bracket sprites sharing _selectMarker's material
    this._boxEl = null;             // the selection box (a fixed-position div) while dragging one
    this._previewGeoms = new Map(); // kind → { geom, span }
  }

  // Turn edit mode on with `handlers`, or off with null. The scene reports,
  // never changes, the scenario:
  //   onSelect(ids, mode) — a click or box asks for a new selection: "replace"
  //     (a click; [] on empty ground), "toggle" (Shift+click) or "add"
  //     (Shift+drag box, or Shift+drag of an unselected aircraft);
  //   onEdit({ type, phase, changes }) — a drag of the selection: type "move"
  //     (horizontal, on the plane at the grabbed aircraft's altitude), "raise"
  //     (the altitude knob, or Alt+drag: up on screen = up) or "rotate" (the
  //     heading knob); phase "start" | "move" | "end" | "cancel"
  //     (cancelEdit / pointercancel: the start values again); changes
  //     [{ id, e, n, u }] (ENU metres) for move / raise, [{ id, heading_deg }]
  //     for rotate — the TRUE heading (0–360, the samples' psi_rad
  //     convention), turned by the pointer's sweep on the grid; Shift snaps
  //     that true heading to 5°;
  //   onPoint({ e, n }) — a click on empty ground (the u = 0 plane).
  // Pressing an aircraft or a knob keeps the press from OrbitControls, and
  // the tracked-view hand-off never runs while editing. Editing turns the
  // environment to "map" (imagery for any fleet) and back to "plain" when off.
  setEditing(handlers) {
    this.cancelEdit();
    const noop = () => {};
    this._edit = handlers
      ? { onSelect: handlers.onSelect || noop, onEdit: handlers.onEdit || noop, onPoint: handlers.onPoint || noop }
      : null;
    if (!handlers) {
      this.setSelection([]);
      for (const m of this._editMarkers) m.visible = false;
    }
    this.setEnvironment(handlers ? "map" : "plain");
    this.renderer.domElement.style.cursor = "";
  }

  // The edit selection, in order; the last id is the primary, which carries
  // the handles (and is `selectedId`).
  setSelection(ids) {
    this._editSel = [...ids];
    this._editSelSet = new Set(this._editSel);
    this.selectedId = this._editSel.length ? this._editSel[this._editSel.length - 1] : null;
  }

  // Abort the press or drag in progress; a started edit drag reports
  // phase "cancel" with its start values.
  cancelEdit() {
    const d = this._drag;
    if (!d) return;
    this._drag = null;
    this._hideBox();
    const el = this.renderer.domElement;
    if (el.hasPointerCapture(d.pointer)) el.releasePointerCapture(d.pointer);
    if (!d.pending && d.start && this._edit) this._emit(d, "cancel");
  }

  _emit(d, phase) {
    const o = phase === "cancel" ? NO_OFFSET : d.offset;
    const changes = d.start.map((s) => (d.type === "rotate"
      ? { id: s.id, heading_deg: wrap360(s.heading + o.dh) }
      : { id: s.id, e: s.e + o.de, n: s.n + o.dn, u: s.u + o.du }));
    this._edit.onEdit({ type: d.type, phase, changes });
  }

  // Presses the scene claims, ahead of OrbitControls: in edit mode, on an
  // aircraft or knob (drag or click), Shift on empty space (box), any press
  // on the map (pan, or a ground click); outside edit mode only map pans,
  // which leave the press to the viewer's click handling too.
  _onCaptureDown(e) {
    if (!e.isTrusted || !e.isPrimary || this._drag) return;
    const map = this._mode === "map";
    if (!this._edit && !map) return;
    const el = this.renderer.domElement;
    const d = {
      type: "pan", pointer: e.pointerId, pending: true, id: null, start: null, shift: e.shiftKey,
      x0: e.clientX, y0: e.clientY, lx: e.clientX, ly: e.clientY, t0: e.timeStamp,
      offset: { de: 0, dn: 0, du: 0, dh: 0 },
    };
    const claim = () => {
      e.stopImmediatePropagation();
      e.preventDefault();
      el.setPointerCapture(e.pointerId);
      this._drag = d;
    };
    if (!this._edit) {
      el.setPointerCapture(e.pointerId);
      this._drag = d;
      return;
    }
    if (e.button !== 0) {
      if (map) claim(); // any other button pans the map; in orbit it is OrbitControls'
      return;
    }
    const rect = el.getBoundingClientRect();
    const knob = this._knobAt(e.clientX - rect.left, e.clientY - rect.top);
    const id = knob ? this._knobs.id : this._pick(e.clientX, e.clientY);
    if (id != null) {
      d.type = knob ?? (e.altKey ? "raise" : "move");
      d.id = id;
      claim();
    } else if (e.shiftKey) {
      d.type = "box";
      claim();
    } else {
      // Empty ground: a click, or a drag that pans the map / is OrbitControls' in orbit.
      d.type = "ground";
      if (map) claim();
      else this._drag = d;
    }
  }

  _onCaptureMove(e) {
    const d = this._drag;
    if (!d) {
      if (this._edit) this._hover(e);
      return;
    }
    if (e.pointerId !== d.pointer) return;
    if (d.pending) {
      if (Math.hypot(e.clientX - d.x0, e.clientY - d.y0) < CLICK_SLOP_PX) return;
      d.pending = false;
      if (d.type === "ground") {
        if (this._mode !== "map") { this._drag = null; return; } // OrbitControls has it
        d.type = "pan";
      }
      if (!this._beginDrag(d)) { this.cancelEdit(); return; }
    }
    this._dragTo(d, e);
  }

  _onCaptureUp(e) {
    const d = this._drag;
    if (!d || e.pointerId !== d.pointer) return;
    this._drag = null;
    const el = this.renderer.domElement;
    if (el.hasPointerCapture(e.pointerId)) el.releasePointerCapture(e.pointerId);
    el.style.cursor = "";
    if (!this._edit) return; // a map pan outside edit mode; the viewer's click handling follows
    if (d.pending) {
      this._click(d, e);
    } else if (d.type === "box") {
      this._hideBox();
      const ids = this._idsInBox(d.x0, d.y0, e.clientX, e.clientY);
      if (ids.length) this._edit.onSelect(ids, "add");
    } else if (d.start) {
      this._emit(d, "end");
    }
  }

  // A press that never moved: select the aircraft (Shift toggles it), or on
  // empty ground clear the selection and report the point. A knob press, a
  // Shift press on empty space or a long still press on the ground does nothing.
  _click(d, e) {
    if (d.type === "move" || d.type === "raise") {
      this._edit.onSelect([d.id], d.shift ? "toggle" : "replace");
      return;
    }
    if (d.type !== "ground" || e.timeStamp - d.t0 >= CLICK_MAX_MS) return;
    this._edit.onSelect([], "replace");
    if (this._rayToPlane(e.clientX, e.clientY, 0, _hit, MAP_HEIGHT)) this._edit.onPoint({ e: _hit.x, n: -_hit.z });
  }

  // The drag passed the click slop: fix what it moves and where from, and
  // report "start". Dragging an unselected aircraft selects it first (Shift:
  // adds it). False when there is nothing to drag (it left the view).
  _beginDrag(d) {
    if (d.type === "pan") return true;
    if (d.type === "box") { this._showBox(d.x0, d.y0, d.x0, d.y0); return true; }
    const a = this._visible(d.id);
    if (!a || !a.sample) return false;
    let ids = this._editSel;
    if (!this._editSelSet.has(d.id)) {
      ids = d.shift ? [...this._editSel, d.id] : [d.id];
      this._edit.onSelect([d.id], d.shift ? "add" : "replace");
    }
    d.start = [];
    for (const id of ids) {
      const s = this._visible(id)?.sample;
      if (s) d.start.push({ id, e: s.enu_e, n: s.enu_n, u: s.enu_u, heading: wrap360(s.psi_rad * RAD2DEG) });
    }
    const p = a.group.position;
    d.planeY = p.y;
    // A perspective drag never lands beyond a few times the aircraft's distance.
    d.maxDist = DRAG_REACH * this.camera.position.distanceTo(p);
    if (d.type === "move" || d.type === "rotate") {
      if (!this._rayToPlane(d.x0, d.y0, d.planeY, _hit, d.maxDist)) return false;
      d.hx = _hit.x;
      d.hz = _hit.z;
      d.cx = p.x;
      d.cz = p.z;
      d.b0 = Math.atan2(_hit.x - p.x, -(_hit.z - p.z)) * RAD2DEG; // pointer bearing about the aircraft
      d.h0 = wrap360(a.sample.psi_rad * RAD2DEG);
    } else {
      d.mpp = this._depth(p) * this._pxScale();
    }
    this.renderer.domElement.style.cursor = d.type === "raise" ? "ns-resize" : "grabbing";
    this._emit(d, "start");
    return true;
  }

  _dragTo(d, e) {
    const o = d.offset;
    if (d.type === "pan") {
      this._mapCenter.x -= (e.clientX - d.lx) * this._mapMpp;
      this._mapCenter.z -= (e.clientY - d.ly) * this._mapMpp;
      this.renderer.domElement.style.cursor = "grabbing";
    } else if (d.type === "box") {
      this._showBox(d.x0, d.y0, e.clientX, e.clientY);
    } else if (d.type === "raise") {
      o.du = (d.y0 - e.clientY) * d.mpp;
      this._emit(d, "move");
    } else if (this._rayToPlane(e.clientX, e.clientY, d.planeY, _hit, d.maxDist)) {
      if (d.type === "move") {
        o.de = _hit.x - d.hx;
        o.dn = -(_hit.z - d.hz);
      } else {
        let h = d.h0 + Math.atan2(_hit.x - d.cx, -(_hit.z - d.cz)) * RAD2DEG - d.b0;
        if (e.shiftKey) h = Math.round(h / ROTATE_SNAP_DEG) * ROTATE_SNAP_DEG;
        o.dh = h - d.h0;
      }
      this._emit(d, "move");
    }
    d.lx = e.clientX;
    d.ly = e.clientY;
  }

  // Cursor feedback while hovering in edit mode.
  _hover(e) {
    const el = this.renderer.domElement;
    const rect = el.getBoundingClientRect();
    const knob = this._knobAt(e.clientX - rect.left, e.clientY - rect.top);
    let cursor = this._mode === "map" ? "grab" : "";
    if (knob === "rotate") cursor = "crosshair";
    else if (knob === "raise") cursor = "ns-resize";
    else if (this._pick(e.clientX, e.clientY) != null) cursor = e.altKey ? "ns-resize" : "move";
    if (el.style.cursor !== cursor) el.style.cursor = cursor;
  }

  // The knob ("rotate" | "raise") under a canvas-px point, or null.
  _knobAt(x, y) {
    if (this._knobs.id == null) return null;
    for (const k of ["rotate", "raise"]) {
      const p = this._knobs[k];
      if (Math.hypot(p.x - x, p.y - y) <= KNOB_HIT_PX) return k;
    }
    return null;
  }

  // Where the pointer's ray meets the horizontal plane at three-y `y`, into
  // `out`. In perspective, null for a ray within GRAZE_SIN of parallel to the
  // plane or a hit farther than `maxDist` from the camera (a near-horizon
  // pointer would fling the point kilometres away).
  _rayToPlane(clientX, clientY, y, out, maxDist) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this._pointer.set(((clientX - rect.left) / rect.width) * 2 - 1, -((clientY - rect.top) / rect.height) * 2 + 1);
    this._raycaster.setFromCamera(this._pointer, this.camera);
    const ray = this._raycaster.ray;
    if (this._mode !== "map" && Math.abs(ray.direction.y) < GRAZE_SIN) return null;
    _plane.set(UP, -y);
    const hit = ray.intersectPlane(_plane, out);
    if (!hit) return null;
    if (this._mode !== "map" && hit.distanceTo(this.camera.position) > maxDist) return null;
    return hit;
  }

  // A world point's canvas-px position into `out` (a Vector3; z is NDC depth).
  _toScreen(p, out) {
    out.copy(p).project(this.camera);
    out.set(((out.x + 1) / 2) * this._w, ((1 - out.y) / 2) * this._h, out.z);
    return out;
  }

  _showBox(x0, y0, x1, y1) {
    if (!this._boxEl) {
      const b = document.createElement("div");
      b.className = "scene-select-box";
      Object.assign(b.style, {
        position: "fixed", pointerEvents: "none", zIndex: "20",
        border: "1px solid #4aa3ff", background: "rgba(74,163,255,0.12)",
      });
      document.body.appendChild(b);
      this._boxEl = b;
    }
    Object.assign(this._boxEl.style, {
      left: `${Math.min(x0, x1)}px`, top: `${Math.min(y0, y1)}px`,
      width: `${Math.abs(x1 - x0)}px`, height: `${Math.abs(y1 - y0)}px`,
    });
  }

  _hideBox() {
    this._boxEl?.remove();
    this._boxEl = null;
  }

  // Visible aircraft whose centre is inside a client-space box.
  _idsInBox(x0, y0, x1, y1) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    const l = Math.min(x0, x1) - rect.left, r = Math.max(x0, x1) - rect.left;
    const t = Math.min(y0, y1) - rect.top, b = Math.max(y0, y1) - rect.top;
    const ids = [];
    for (const [id, a] of this.fleet) {
      if (!a.group.visible) continue;
      const s = this._toScreen(a.group.position, _v);
      if (s.z >= -1 && s.z <= 1 && s.x >= l && s.x <= r && s.y >= t && s.y <= b) ids.push(id);
    }
    return ids;
  }

  // Brackets around every selected aircraft in edit mode (a pooled sprite each).
  _updateEditMarkers(px) {
    let k = 0;
    for (const id of this._editSel) {
      const a = this._visible(id);
      if (!a) continue;
      const sizePx = this._bracketPx(a.group, px);
      if (sizePx > FRAME_MAX_PX) continue;
      let m = this._editMarkers[k];
      if (!m) {
        m = new THREE.Sprite(this._selectMarker.material);
        m.renderOrder = 11;
        this.scene.add(m);
        this._editMarkers.push(m);
      }
      m.visible = true;
      m.position.copy(a.group.position);
      m.scale.set(sizePx * px, sizePx * px, 1);
      k++;
    }
    for (; k < this._editMarkers.length; k++) this._editMarkers[k].visible = false;
  }

  // The primary selection's handles: the heading knob HEADING_KNOB_PX out
  // along the nose (with a line to it), the altitude knob ALT_KNOB_PX out the
  // opposite way on screen, so the two never overlap. Hidden (and not
  // pickable) while the aircraft or the heading knob is behind the camera.
  _updateHandles(px) {
    const a = this._edit ? this._visible(this.selectedId) : null;
    let show = !!a?.sample;
    let ps, hs;
    if (show) {
      const p = a.group.position;
      _w.set(Math.sin(a.psi), 0, -Math.cos(a.psi)).multiplyScalar(HEADING_KNOB_PX * px * this._depth(p)).add(p);
      ps = this._toScreen(p, _s3);
      hs = this._toScreen(_w, _v);
      show = Math.abs(ps.z) <= 1 && Math.abs(hs.z) <= 1;
    }
    this._headingKnob.visible = this._altKnob.visible = this._headingLine.visible = show;
    this._knobs.id = show ? this.selectedId : null;
    if (!show) return;
    const p = a.group.position;
    const pos = this._headingLine.geometry.attributes.position;
    pos.setXYZ(0, p.x, p.y, p.z);
    pos.setXYZ(1, _w.x, _w.y, _w.z);
    pos.needsUpdate = true;
    const knobH = KNOB_PX * px;
    this._headingKnob.position.copy(_w);
    this._headingKnob.scale.set(knobH, knobH, 1);

    this._knobs.rotate.x = hs.x;
    this._knobs.rotate.y = hs.y;
    let dx = hs.x - ps.x, dy = hs.y - ps.y;
    const len = Math.hypot(dx, dy);
    if (len < 1) { dx = 0; dy = -1; } else { dx /= len; dy /= len; }
    const ox = -dx * ALT_KNOB_PX, oy = -dy * ALT_KNOB_PX; // canvas px, y down
    this._knobs.raise.x = ps.x + ox;
    this._knobs.raise.y = ps.y + oy;
    // Anchored at the aircraft, shifted in screen space by the sprite centre.
    this._altKnob.position.copy(p);
    this._altKnob.center.set(0.5 - ox / KNOB_PX, 0.5 + oy / KNOB_PX);
    this._altKnob.scale.set(knobH, knobH, 1);
  }

  // Grid bearing of true north at (lat°, lon°) about the scene's datum; 0
  // without one (a sim with no geodetic output draws headings as given).
  _northBearing(latDeg, lonDeg) {
    const d = this._env.datum;
    return d ? d.northBearing(latDeg * DEG2RAD, lonDeg * DEG2RAD) : 0;
  }

  // Translucent preview aircraft ("ghosts") in named layers, so independent
  // previews (a spawn preview, a sweep's spawn cloud) never wipe each other:
  // `items` [{ kind, e, n, u, heading_deg, climb_deg?, bank_deg?, color? }]
  // (ENU metres, heading a TRUE heading like the samples', colour a CSS
  // string or hex number) replace layer `layer`'s; null or [] clears it.
  // Each layer is one instanced flat silhouette per kind — cheap for
  // thousands — never picked, every ghost kept at least PREVIEW_MIN_PX on
  // screen at its own distance. Items with a non-finite position are skipped.
  setGhosts(layer, items) {
    this._dropGhostLayer(layer);
    const byKind = new Map();
    for (const it of items || []) {
      if (![it.e, it.n, it.u].every(Number.isFinite)) continue;
      const kind = it.kind === "multirotor" ? "multirotor" : "fixed-wing";
      if (!byKind.has(kind)) byKind.set(kind, []);
      byKind.get(kind).push(it);
    }
    if (!byKind.size) return;
    const datum = this._env.datum;
    const meshes = [];
    for (const [kind, list] of byKind) {
      let shape = this._previewGeoms.get(kind);
      if (!shape) { shape = buildGhostGeometry(kind); this._previewGeoms.set(kind, shape); }
      const mesh = new THREE.InstancedMesh(shape.geom, new THREE.MeshBasicMaterial({
        color: 0xffffff, transparent: true, opacity: PREVIEW_OPACITY,
        depthWrite: false, side: THREE.DoubleSide, fog: false,
      }), list.length);
      mesh.frustumCulled = false; // instances spread far beyond the shape's bounds
      mesh.renderOrder = 3;
      const pos = new Float32Array(list.length * 3), quat = new Float32Array(list.length * 4);
      list.forEach((it, i) => {
        enuToThree(it.e, it.n, it.u, _p).toArray(pos, i * 3);
        const deg = (v) => (Number.isFinite(v) ? v * DEG2RAD : 0);
        let psi = deg(it.heading_deg);
        if (datum) {
          const g = datum.toGeodetic(it.e, it.n, it.u, _geo);
          psi += datum.northBearing(g.lat, g.lon);
        }
        attitudeToQuaternion(deg(it.bank_deg), deg(it.climb_deg), psi, _q).toArray(quat, i * 4);
        mesh.setColorAt(i, _col.set(it.color ?? PREVIEW_COLOR));
      });
      mesh.instanceColor.needsUpdate = true;
      this.scene.add(mesh);
      meshes.push({ mesh, pos, quat, span: shape.span });
    }
    this._ghostLayers.set(layer, meshes);
    this._ghostStamp = NaN; // size the new layer on the next frame
  }

  // Size every ghost for this frame: in perspective each by its own distance,
  // on the map all alike. The instance matrices are rebuilt only when the
  // camera, the scale or a layer changed.
  _sizeGhosts(px) {
    if (!this._ghostLayers.size) return;
    const cam = this.camera;
    const stamp = px * this._modelScale;
    if (stamp === this._ghostStamp && this._ghostCam.equals(cam.matrixWorld)) return;
    this._ghostStamp = stamp;
    this._ghostCam.copy(cam.matrixWorld);
    for (const meshes of this._ghostLayers.values()) {
      for (const { mesh, pos, quat, span } of meshes) {
        for (let i = 0; i < mesh.count; i++) {
          _p.fromArray(pos, i * 3);
          _q.fromArray(quat, i * 4);
          _s3.setScalar(Math.max(this._modelScale, (PREVIEW_MIN_PX * px * this._depth(_p)) / span));
          mesh.setMatrixAt(i, _m4.compose(_p, _q, _s3));
        }
        mesh.instanceMatrix.needsUpdate = true;
      }
    }
  }

  // Drop one ghost layer (the per-kind geometry stays cached).
  _dropGhostLayer(layer) {
    const meshes = this._ghostLayers.get(layer);
    if (!meshes) return;
    for (const { mesh } of meshes) {
      this.scene.remove(mesh);
      mesh.material.dispose();
      mesh.dispose();
    }
    this._ghostLayers.delete(layer);
  }
}

const NO_OFFSET = { de: 0, dn: 0, du: 0, dh: 0 };

function wrap360(deg) {
  return ((deg % 360) + 360) % 360;
}

// A wheel event's vertical delta in CSS px.
function wheelPx(e, pageH) {
  if (e.deltaMode === 1) return e.deltaY * 16;      // lines
  if (e.deltaMode === 2) return e.deltaY * pageH;   // pages
  return e.deltaY;
}

// An edit knob: a dark disc with a handle-coloured ring and glyph — a
// curved arrow for "rotate", up / down chevrons for "raise".
function knobTexture(kind) {
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const ctx = c.getContext("2d");
  const color = `#${HANDLE_COLOR.toString(16).padStart(6, "0")}`;
  ctx.fillStyle = "rgba(10,14,20,0.85)";
  ctx.beginPath();
  ctx.arc(32, 32, 28, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = color;
  ctx.lineWidth = 5;
  ctx.stroke();
  ctx.fillStyle = color;
  ctx.lineWidth = 4;
  if (kind === "rotate") {
    ctx.beginPath();
    ctx.arc(32, 32, 13, -Math.PI * 0.9, Math.PI * 0.35);
    ctx.stroke();
    ctx.beginPath(); // arrowhead at the arc's end
    ctx.moveTo(44, 46); ctx.lineTo(32, 46); ctx.lineTo(42, 34);
    ctx.closePath();
    ctx.fill();
  } else {
    ctx.beginPath();
    ctx.moveTo(32, 12); ctx.lineTo(43, 26); ctx.lineTo(21, 26); ctx.closePath();
    ctx.moveTo(32, 52); ctx.lineTo(43, 38); ctx.lineTo(21, 38); ctx.closePath();
    ctx.fill();
  }
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
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
