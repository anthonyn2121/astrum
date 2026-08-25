// Procedural aircraft meshes + the coordinate/attitude math that places them.
//
// No external model assets: an F-16-style jet (slender fuselage, chin intake,
// bubble canopy, cropped-delta wing, swept tail) and an F450-like quad (X-frame
// + 4 rotors) are built from primitives, tinted by vehicle class with a per-id
// hue so individuals in a fleet are distinguishable. The mesh is chosen by
// VehicleKind (fixed-wing vs multirotor), not the specific JSBSim model.
//
// Frames:
//   * World is Three.js (Y up). We map the sim's ENU grid as
//       three.x = East,  three.y = Up,  three.z = -North.
//   * Meshes are authored nose = +Z, up = +Y, starboard = +X.
//   * attitudeToQuaternion() converts aerospace body Euler angles (roll φ,
//     pitch θ, yaw ψ, 3-2-1 about NED) into the world orientation, verified so
//     ψ=0 faces North, +ψ yaws toward East, +θ pitches nose up, +φ drops the
//     right wing.

import * as THREE from "three";

// Visual exaggeration so ~10 m aircraft are legible against a km-scale grid.
export const MODEL_SCALE = 4;

const _m = new THREE.Matrix4();
const _x = new THREE.Vector3();
const _y = new THREE.Vector3();
const _z = new THREE.Vector3();

// ENU (metres) -> Three.js world position.
export function enuToThree(e, n, u, out = new THREE.Vector3()) {
  return out.set(e, u, -n);
}

// Body Euler (rad) -> world orientation quaternion.
export function attitudeToQuaternion(phi, theta, psi, out = new THREE.Quaternion()) {
  const cph = Math.cos(phi), sph = Math.sin(phi);
  const cth = Math.cos(theta), sth = Math.sin(theta);
  const cps = Math.cos(psi), sps = Math.sin(psi);

  // Body-to-NED DCM columns, as (N, E, D) components.
  const fN = cth * cps,               fE = cth * sps,               fD = -sth;             // nose (+X body)
  const sN = sph * sth * cps - cph * sps, sE = sph * sth * sps + cph * cps, sD = sph * cth; // starboard (+Y)
  const dN = cph * sth * cps + sph * sps, dE = cph * sth * sps - sph * cps, dD = cph * cth; // down (+Z)

  // Map each NED body axis (n,e,d) into world (e, -d, -n); local up = -down.
  // The physical triple (starboard, up, nose) is LEFT-handed (right x up = -nose),
  // but a mesh's local (+X,+Y,+Z) is right-handed, so makeBasis needs a proper
  // (det +1) rotation. We map local +X to PORT (-starboard) to flip the handedness;
  // both meshes are mirror-symmetric across their X=0 plane, so this is invisible
  // while making nose (+Z) and up (+Y) — the axes that matter — exactly correct.
  _z.set(fE, -fD, -fN);   // nose -> mesh local +Z
  _x.set(-sE, sD, sN);    // port (= -starboard) -> mesh local +X
  _y.set(-dE, dD, dN);    // up   -> mesh local +Y
  _m.makeBasis(_x, _y, _z);
  return out.setFromRotationMatrix(_m);
}

// Deterministic, distinct-per-aircraft colour within a per-class hue band.
export function aircraftColor(kind, id) {
  const baseHue = kind === "multirotor" ? 32 : 205; // amber vs. steel blue
  const jitter = ((id * 47) % 70) - 35;             // ±35° spread across the fleet
  const c = new THREE.Color();
  c.setHSL((((baseHue + jitter) % 360) + 360) % 360 / 360, 0.62, 0.58);
  return c;
}

function labelSprite(text, color) {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 64;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "rgba(10,14,20,0.72)";
  roundRect(ctx, 4, 4, 248, 56, 12);
  ctx.fill();
  ctx.strokeStyle = `#${color.getHexString()}`;
  ctx.lineWidth = 3;
  roundRect(ctx, 4, 4, 248, 56, 12);
  ctx.stroke();
  ctx.fillStyle = "#e8eef6";
  ctx.font = "bold 34px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, 128, 34);

  const tex = new THREE.CanvasTexture(canvas);
  tex.anisotropy = 4;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true }));
  sprite.scale.set(12, 3, 1);
  sprite.renderOrder = 10;
  return sprite;
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function bodyMaterial(color) {
  return new THREE.MeshStandardMaterial({
    color,
    metalness: 0.35,
    roughness: 0.55,
    emissive: color.clone().multiplyScalar(0.0),
    emissiveIntensity: 1.0,
  });
}

function box(w, h, d, mat, x = 0, y = 0, z = 0) {
  const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
  m.position.set(x, y, z);
  return m;
}

// A translucent, additive rotor/prop disc (axis along the mesh's local +Y).
function disc(radius, mat) {
  const m = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, 0.05, 24), mat);
  return m;
}

function buildJet(color, mats) {
  const g = new THREE.Group();
  const body = bodyMaterial(color); mats.push(body);
  const accent = bodyMaterial(color.clone().offsetHSL(0, -0.08, -0.24)); mats.push(accent);

  // Slender fuselage along Z (nose +Z) with a long pointed radome up front.
  g.add(box(0.85, 0.9, 6.4, body, 0, 0, -0.3));
  const nose = new THREE.Mesh(new THREE.ConeGeometry(0.42, 2.6, 16), accent);
  nose.rotation.x = Math.PI / 2; // apex -> +Z
  nose.position.set(0, -0.05, 4.1);
  g.add(nose);

  // Signature chin intake tucked under the forward fuselage.
  g.add(box(0.72, 0.5, 1.8, accent, 0, -0.58, 1.5));

  // Bubble canopy just ahead of the wing (dark tinted glass).
  const glass = new THREE.MeshStandardMaterial({ color: 0x0b1420, metalness: 0.3, roughness: 0.08 });
  mats.push(glass);
  const canopy = new THREE.Mesh(new THREE.SphereGeometry(0.4, 16, 12), glass);
  canopy.scale.set(0.95, 0.8, 1.9);
  canopy.position.set(0, 0.52, 1.35);
  g.add(canopy);

  // Cropped-delta wing (one flat plate spanning X). Authored in the shape's XY
  // plane then laid flat into XZ (shape +y -> world +Z = nose); double-sided so
  // it reads from below.
  const wingMat = bodyMaterial(color); wingMat.side = THREE.DoubleSide; mats.push(wingMat);
  const wing = new THREE.Shape();
  wing.moveTo(0, 1.7);      // root leading edge
  wing.lineTo(3.1, -0.5);   // starboard leading tip
  wing.lineTo(3.1, -1.4);   // starboard trailing tip
  wing.lineTo(0, -2.4);     // root trailing edge
  wing.lineTo(-3.1, -1.4);  // port trailing tip
  wing.lineTo(-3.1, -0.5);  // port leading tip
  wing.closePath();
  const wingMesh = new THREE.Mesh(new THREE.ShapeGeometry(wing), wingMat);
  wingMesh.rotation.x = Math.PI / 2;
  wingMesh.position.set(0, -0.12, -0.3);
  g.add(wingMesh);

  // Horizontal stabilators at the tail.
  g.add(box(3.4, 0.12, 1.0, body, 0, 0, -3.05));

  // Swept single vertical fin: shape XY -> world ZY (shape +x -> world -Z, aft).
  const finMat = bodyMaterial(color.clone().offsetHSL(0, 0, -0.05));
  finMat.side = THREE.DoubleSide; mats.push(finMat);
  const fin = new THREE.Shape();
  fin.moveTo(0, 0);       // base leading
  fin.lineTo(1.9, 0);     // base trailing
  fin.lineTo(2.05, 1.8);  // tip trailing (swept aft)
  fin.lineTo(1.25, 1.8);  // tip leading
  fin.closePath();
  const finMesh = new THREE.Mesh(new THREE.ShapeGeometry(fin), finMat);
  finMesh.rotation.y = Math.PI / 2;
  finMesh.position.set(0, 0.42, -1.4);
  g.add(finMesh);

  // Exhaust nozzle + a static afterburner glow at the tail.
  const nozzle = new THREE.Mesh(new THREE.CylinderGeometry(0.34, 0.4, 0.7, 16), accent);
  nozzle.rotation.x = Math.PI / 2;
  nozzle.position.set(0, 0, -3.7);
  g.add(nozzle);
  const burnerMat = new THREE.MeshBasicMaterial({ color: 0xff8a3d, transparent: true, opacity: 0.5, side: THREE.DoubleSide });
  const burner = disc(0.32, burnerMat);
  burner.rotation.x = Math.PI / 2; // face -Z
  burner.position.set(0, 0, -4.05);
  g.add(burner);

  g.userData.rotors = []; // a jet has no spinning prop/rotor discs
  return g;
}

function buildMultirotor(color, mats) {
  const g = new THREE.Group();
  const body = bodyMaterial(color); mats.push(body);
  const armMat = bodyMaterial(color.clone().offsetHSL(0, -0.05, -0.32)); mats.push(armMat);
  const front = bodyMaterial(new THREE.Color(0xff5a4d)); mats.push(front); // heading marker

  g.add(box(1.4, 0.5, 1.4, body)); // hub

  // Two crossed arms rotated 45° -> an X-frame; rotor discs at the four ends.
  const armLen = 4.6;
  const arms = new THREE.Group();
  arms.add(box(0.22, 0.16, armLen, armMat));
  arms.add(box(armLen, 0.16, 0.22, armMat));
  arms.rotation.y = Math.PI / 4;
  g.add(arms);

  const d = (armLen / 2) * Math.SQRT1_2; // arm-end offset after the 45° turn
  const rotorMat = new THREE.MeshBasicMaterial({ color: 0xbfe0ff, transparent: true, opacity: 0.3, side: THREE.DoubleSide });
  const rotors = [];
  for (const [sx, sz] of [[1, 1], [1, -1], [-1, 1], [-1, -1]]) {
    const r = disc(1.15, rotorMat.clone());
    r.position.set(sx * d, 0.28, sz * d);
    g.add(r);
    rotors.push(r);
    // Bright caps on the two front arms (+Z) so heading reads at a glance.
    if (sz > 0) {
      const cap = new THREE.Mesh(new THREE.SphereGeometry(0.34, 12, 8), front);
      cap.position.set(sx * d, 0.28, sz * d);
      g.add(cap);
    }
  }
  g.userData.rotors = rotors;
  g.userData.spinAxis = "y";
  return g;
}

// Build one aircraft object. Returns a THREE.Group carrying, in userData:
//   { id, kind, mats:[], rotors:[], spinAxis, halo, label }
export function buildAircraft(kind, id) {
  const color = aircraftColor(kind, id);
  const mats = [];
  const group = kind === "multirotor" ? buildMultirotor(color, mats) : buildJet(color, mats);
  group.scale.setScalar(MODEL_SCALE);

  // Selection halo (hidden until selected).
  const halo = new THREE.Mesh(
    new THREE.IcosahedronGeometry(3.6, 1),
    new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: true, transparent: true, opacity: 0.5 }),
  );
  halo.visible = false;
  group.add(halo);

  const label = labelSprite(String(id), color);
  label.position.set(0, 3.2, 0);
  group.add(label);

  group.userData.id = id;
  group.userData.kind = kind;
  group.userData.mats = mats;
  group.userData.baseColor = color;
  group.userData.halo = halo;
  group.userData.label = label;
  return group;
}
