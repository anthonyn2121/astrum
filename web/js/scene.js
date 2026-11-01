// The 3-D view: scene, camera, lighting, the ENU ground grid, and one live
// object per aircraft driven from the store each frame. Also owns the orbit /
// follow cameras, motion trails, and click-to-select picking.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { buildAircraft, enuToThree, attitudeToQuaternion, MODEL_SCALE } from "./meshes.js";

const TRAIL_MAX = 240;      // points per trail
const TRAIL_MIN_STEP = 1.0; // metres between recorded trail points

// Ground map: a slippy-map tile layer laid flat under the fleet, centred on the
// scenario datum. Tiles come from astrum-web at /tiles/{z}/{x}/{y}.jpg, which
// proxies + disk-caches an upstream tile server — so any geolocated datum gets a
// crisp aerial map, and an area seen once works offline afterwards. A tile that
// 404s (offline + uncached) is simply skipped, so the grid shows through.
const MAP = {
  url: (z, x, y) => `./tiles/${z}/${x}/${y}.jpg`,
  zoom: 14,    // ~7.5 m/px at mid-latitudes; one tile ≈ 1.9 km there
  radius: 3,   // tiles each way from centre → a (2·radius+1)² block ≈ 13.5 km wide
  y: -0.5,     // metres below the grid so the map never z-fights the gridlines
};
const M_PER_DEG_LAT = 111320; // metres per degree of latitude (spherical approx)
const DEG2RAD = Math.PI / 180;

// Web-Mercator (XYZ) tile <-> WGS84 helpers. `n` is the tiles-per-axis (2**zoom).
function lon2tileX(lon, n) { return ((lon + 180) / 360) * n; }
function lat2tileY(lat, n) {
  const r = lat * DEG2RAD;
  return ((1 - Math.log(Math.tan(r) + 1 / Math.cos(r)) / Math.PI) / 2) * n;
}
function tileX2lon(x, n) { return (x / n) * 360 - 180; }
function tileY2lat(y, n) { return Math.atan(Math.sinh(Math.PI * (1 - (2 * y) / n))) / DEG2RAD; }

export class Scene {
  constructor(container, { onSelect } = {}) {
    this.container = container;
    this.onSelect = onSelect || (() => {});
    this.fleet = new Map(); // id -> { group, trail, kind }
    this.selectedId = null;

    // Ground map tile layer (built lazily once the fleet's datum is known).
    this._mapGroup = null;
    this._mapTried = false;

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
    // Keep handler refs so dispose() can detach them on a renderer swap.
    this._onPointer = (e) => this._onPointerDown(e);
    this._onResize = () => this.resize();
    this.renderer.domElement.addEventListener("pointerdown", this._onPointer);
    window.addEventListener("resize", this._onResize);
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
    // Kept faint: it's a scale/heading overlay above the map raster, not the floor.
    grid.material.opacity = 0.28;
    grid.material.transparent = true;
    grid.position.y = 0.2; // sit just above the map plane to avoid z-fighting
    this.scene.add(grid);

    // Faint origin disc + N/E/S/W markers. Grid lies in the X(East)–Z plane with
    // North = -Z (see meshes.js).
    const originMat = new THREE.MeshBasicMaterial({ color: 0x22374d, transparent: true, opacity: 0.2 });
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

  // Once we've seen a geolocated sample, derive the scenario's ENU datum (the
  // lat/lon where enu = 0) and lay a block of map tiles flat under the fleet,
  // centred on it. Runs at most once per loaded source.
  //
  // Scoped to fixed-wing fleets: those fly a real geographic route where an
  // aerial map reads well. Multirotor (and any other kind) keep the plain ENU
  // grid, so the kind check comes before we mark the map as tried — a mixed
  // fleet still builds the map for its fixed-wing members.
  _tryBuildMap(s, kind) {
    if (kind !== "fixed-wing") return;
    if (!Number.isFinite(s.lat_deg) || (s.lat_deg === 0 && s.lon_deg === 0)) return;
    this._mapTried = true;

    // Datum: the lat/lon where enu = 0 (any sample yields the same point).
    const lat0 = s.lat_deg - s.enu_n / M_PER_DEG_LAT;
    const mPerDegLon = M_PER_DEG_LAT * Math.cos(lat0 * DEG2RAD);
    const lon0 = s.lon_deg - s.enu_e / mPerDegLon;

    const { zoom, radius } = MAP;
    const n = 2 ** zoom;
    const cx = Math.floor(lon2tileX(lon0, n));
    const cy = Math.floor(lat2tileY(lat0, n));

    const group = new THREE.Group();
    group.renderOrder = -1; // paint under the grid, trails and aircraft
    this.scene.add(group);
    this._mapGroup = group;

    const maxAniso = this.renderer.capabilities.getMaxAnisotropy();
    const loader = new THREE.TextureLoader();

    for (let dy = -radius; dy <= radius; dy++) {
      for (let dx = -radius; dx <= radius; dx++) {
        const tx = cx + dx, ty = cy + dy;
        if (tx < 0 || ty < 0 || tx >= n || ty >= n) continue;

        // Tile geographic bounds → ENU metres relative to the datum. Over a few
        // km the local scale is effectively constant, so this stays rectilinear.
        const west = tileX2lon(tx, n), east = tileX2lon(tx + 1, n);
        const north = tileY2lat(ty, n), south = tileY2lat(ty + 1, n);
        const eW = (west - lon0) * mPerDegLon, eE = (east - lon0) * mPerDegLon;
        const nN = (north - lat0) * M_PER_DEG_LAT, nS = (south - lat0) * M_PER_DEG_LAT;
        const width = eE - eW, depth = nN - nS;
        const eCentre = (eW + eE) / 2, nCentre = (nN + nS) / 2;

        // Texture loads async; default UVs + flipY put the tile's north edge at
        // -Z, west at -X — matching the ENU grid (see meshes.js enuToThree). A
        // 404 (offline + uncached) just leaves a gap where the grid shows through.
        loader.load(
          MAP.url(zoom, tx, ty),
          (tex) => {
            if (this._mapGroup !== group) { tex.dispose(); return; } // superseded
            tex.colorSpace = THREE.SRGBColorSpace;
            tex.anisotropy = maxAniso;
            tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
            const mesh = new THREE.Mesh(
              new THREE.PlaneGeometry(width, depth),
              new THREE.MeshBasicMaterial({ map: tex }),
            );
            mesh.rotation.x = -Math.PI / 2;        // XY plane -> flat XZ (normal +Y)
            mesh.position.set(eCentre, MAP.y, -nCentre);
            group.add(mesh);
          },
          undefined,
          () => {},
        );
      }
    }
  }

  _clearMap() {
    if (this._mapGroup) {
      this.scene.remove(this._mapGroup);
      this._mapGroup.traverse((o) => {
        if (o.isMesh) {
          o.geometry.dispose();
          o.material.map?.dispose();
          o.material.dispose();
        }
      });
    }
    this._mapGroup = null;
    this._mapTried = false;
  }

  // Drop every aircraft and free all of its GPU resources: geometries,
  // materials (shared ones once) and textures (the canvas label sprites).
  // Sprites share one three.js-owned quad geometry, which stays.
  clearFleet() {
    const geometries = new Set(), materials = new Set();
    const collect = (o) => {
      if (o.geometry && !o.isSprite) geometries.add(o.geometry);
      if (o.material) for (const m of [].concat(o.material)) materials.add(m);
    };
    for (const { group, trail } of this.fleet.values()) {
      this.scene.remove(group);
      this.scene.remove(trail.line);
      group.traverse(collect);
      collect(trail.line);
    }
    for (const g of geometries) g.dispose();
    for (const m of materials) { m.map?.dispose(); m.dispose(); }
    this.fleet.clear();
    this.selectedId = null;
    this._clearMap(); // a new source may sit at a different datum
  }

  // Forget every trail (after a seek, so no chord joins the old and new spot).
  resetTrails() {
    for (const { trail } of this.fleet.values()) {
      trail.count = 0;
      trail.last.set(NaN, NaN, NaN);
      trail.geom.setDrawRange(0, 0);
    }
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

      if (!this._mapTried) this._tryBuildMap(s, track.kind);

      enuToThree(s.enu_e, s.enu_n, s.enu_u, _p);
      a.group.position.copy(_p);
      attitudeToQuaternion(s.phi_rad, s.theta_rad, s.psi_rad, _q);
      a.group.quaternion.copy(_q);
      a.group.visible = true;

      // Stale (outside this track's window) or destroyed → ghost it.
      const opacity = inRange && s.alive ? 1.0 : 0.28;
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

  // Tear down for a renderer swap: drop the fleet, detach listeners, release the
  // GL context and remove the canvas so the container is clean for the next scene.
  dispose() {
    this.clearFleet();
    window.removeEventListener("resize", this._onResize);
    this.renderer.domElement.removeEventListener("pointerdown", this._onPointer);
    this.renderer.dispose();
    const canvas = this.renderer.domElement;
    if (canvas.parentNode) canvas.parentNode.removeChild(canvas);
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
