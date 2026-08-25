// The 3-D view: scene, camera, lighting, the ENU ground grid, and one live
// object per aircraft driven from the store each frame. Also owns the orbit /
// follow cameras, motion trails, and click-to-select picking.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { buildAircraft, enuToThree, attitudeToQuaternion, MODEL_SCALE } from "./meshes.js";

const TRAIL_MAX = 240;      // points per trail
const TRAIL_MIN_STEP = 1.0; // metres between recorded trail points

export class Scene {
  constructor(container, { onSelect } = {}) {
    this.container = container;
    this.onSelect = onSelect || (() => {});
    this.fleet = new Map(); // id -> { group, trail, kind }
    this.selectedId = null;

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0a0e14);
    this.scene.fog = new THREE.Fog(0x0a0e14, 3000, 14000);

    this.camera = new THREE.PerspectiveCamera(55, container.clientWidth / container.clientHeight, 1, 500000);
    this.camera.position.set(320, 240, 420);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.maxPolarAngle = Math.PI * 0.495; // don't go under the ground
    this.controls.target.set(0, 60, 0);

    this._addLights();
    this._addGround();

    this._raycaster = new THREE.Raycaster();
    this._pointer = new THREE.Vector2();
    this.renderer.domElement.addEventListener("pointerdown", (e) => this._onPointerDown(e));
    window.addEventListener("resize", () => this.resize());
  }

  _addLights() {
    this.scene.add(new THREE.HemisphereLight(0xbfd3ff, 0x0a0e14, 0.9));
    const sun = new THREE.DirectionalLight(0xffffff, 1.4);
    sun.position.set(1, 2, 1.5);
    this.scene.add(sun);
    this.scene.add(new THREE.AmbientLight(0x8899aa, 0.25));
  }

  _addGround() {
    const size = 12000, divisions = 60;
    const grid = new THREE.GridHelper(size, divisions, 0x2a5a8a, 0x18324a);
    grid.material.opacity = 0.5;
    grid.material.transparent = true;
    this.scene.add(grid);

    // Faint origin disc + N/E/S/W markers. Grid lies in the X(East)–Z plane with
    // North = -Z (see meshes.js).
    const originMat = new THREE.MeshBasicMaterial({ color: 0x22374d, transparent: true, opacity: 0.35 });
    const disc = new THREE.Mesh(new THREE.CircleGeometry(60, 48), originMat);
    disc.rotation.x = -Math.PI / 2;
    this.scene.add(disc);

    const half = size / 2;
    this._addCompass("N", 0, -half); this._addCompass("S", 0, half);
    this._addCompass("E", half, 0);  this._addCompass("W", -half, 0);
  }

  _addCompass(text, x, z) {
    const canvas = document.createElement("canvas");
    canvas.width = 128; canvas.height = 128;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#3f7fb0";
    ctx.font = "bold 96px system-ui, sans-serif";
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(text, 64, 68);
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(canvas), transparent: true, opacity: 0.8 }));
    sprite.scale.set(300, 300, 1);
    sprite.position.set(x, 120, z);
    this.scene.add(sprite);
  }

  clearFleet() {
    for (const { group, trail } of this.fleet.values()) {
      this.scene.remove(group);
      this.scene.remove(trail.line);
      trail.geom.dispose();
    }
    this.fleet.clear();
    this.selectedId = null;
  }

  setSelected(id) {
    this.selectedId = id;
  }

  _ensure(id, kind) {
    let a = this.fleet.get(id);
    if (a) return a;
    const group = buildAircraft(kind, id);
    this.scene.add(group);

    const positions = new Float32Array(TRAIL_MAX * 3);
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geom.setDrawRange(0, 0);
    const line = new THREE.Line(geom, new THREE.LineBasicMaterial({
      color: group.userData.baseColor, transparent: true, opacity: 0.55,
    }));
    line.frustumCulled = false;
    this.scene.add(line);

    a = { group, kind, trail: { line, positions, geom, count: 0, last: new THREE.Vector3(NaN, NaN, NaN) } };
    this.fleet.set(id, a);
    return a;
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

  // Drive the whole fleet from the store at playhead time `t`.
  update(store, t, opts) {
    const { showTrails = true, showLabels = true, spinRotors = true, spinRate = 1, dt = 0 } = opts || {};
    const seen = new Set();
    const _p = new THREE.Vector3();
    const _q = new THREE.Quaternion();

    for (const id of store.ids()) {
      const track = store.track(id);
      const res = store.sampleAt(id, t);
      if (!res) continue;
      seen.add(id);
      const a = this._ensure(id, track.kind);
      const { s, inRange } = res;

      enuToThree(s.enu_e, s.enu_n, s.enu_u, _p);
      a.group.position.copy(_p);
      attitudeToQuaternion(s.phi_rad, s.theta_rad, s.psi_rad, _q);
      a.group.quaternion.copy(_q);
      a.group.visible = true;

      // Stale (outside this track's window) → ghost it.
      const opacity = inRange ? 1.0 : 0.28;
      for (const m of a.group.userData.mats) {
        m.transparent = opacity < 1;
        m.opacity = opacity;
      }

      const selected = id === this.selectedId;
      a.group.userData.halo.visible = selected;
      for (const m of a.group.userData.mats) {
        m.emissive.copy(a.group.userData.baseColor).multiplyScalar(selected ? 0.5 : 0.0);
      }

      a.group.userData.label.visible = showLabels || selected;

      if (spinRotors && a.group.userData.rotors) {
        const axis = a.group.userData.spinAxis;
        for (const r of a.group.userData.rotors) r.rotation[axis] += dt * 40 * Math.max(0.25, spinRate);
      }

      if (showTrails && inRange) this._pushTrail(a.trail, _p);
      a.trail.line.visible = showTrails;
    }

    // Hide any aircraft the store no longer reports at this time.
    for (const [id, a] of this.fleet) {
      if (!seen.has(id)) { a.group.visible = false; a.trail.line.visible = false; }
    }
  }

  // Chase the selected aircraft while preserving the current orbit offset.
  follow() {
    const a = this.selectedId != null ? this.fleet.get(this.selectedId) : null;
    if (!a || !a.group.visible) return;
    const offset = this.camera.position.clone().sub(this.controls.target);
    this.controls.target.copy(a.group.position);
    this.camera.position.copy(a.group.position).add(offset);
  }

  // Frame the whole visible fleet (wherever it actually is — often ~km MSL, not
  // at the grid). Falls back to the origin when nothing is on screen yet.
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
    this.controls.update();
  }

  render() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  resize() {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }

  _onPointerDown(e) {
    // Only treat quick taps (not orbit drags) as selection; OrbitControls owns drags.
    if (e.button !== 0) return;
    const rect = this.renderer.domElement.getBoundingClientRect();
    this._pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    this._pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    this._raycaster.setFromCamera(this._pointer, this.camera);
    // Inflate raycast tolerance for tiny distant meshes.
    this._raycaster.params.Line.threshold = 6;
    const groups = [...this.fleet.values()].map((a) => a.group).filter((g) => g.visible);
    const hits = this._raycaster.intersectObjects(groups, true);
    if (hits.length) {
      let o = hits[0].object;
      while (o && o.userData.id === undefined) o = o.parent;
      if (o) this.onSelect(o.userData.id);
    }
  }
}

export { MODEL_SCALE };
