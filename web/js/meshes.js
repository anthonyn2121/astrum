// Procedural aircraft meshes, their colours, and the coordinate/attitude math
// that places them.
//
// No external model assets: an F-16-style jet (slender fuselage, chin intake,
// bubble canopy, cropped-delta wing, swept tail) and an F450-like quad (X-frame
// + 4 rotors) are built from primitives. The mesh is chosen by VehicleKind
// (fixed-wing vs multirotor), not the specific JSBSim model.
//
// Colours (aircraftColor): in the plain environment, a per-class hue band with
// a per-id hue; in the sky environment, saturated team hues that stand out
// against sky and imagery, with per-id jitter so teammates differ. The scene
// recolours an aircraft in place (tintAircraft) when its team or the
// environment changes. In the sky the body is also two-tone (setTwoTone):
// a darker belly reads against bright sky from below, the full-colour top
// against the ground from above. That is one onBeforeCompile shared by every
// body material, keyed on the normal's y in aircraft axes, so all aircraft
// share one program; geometry rotations are baked in so mesh-local normals
// are aircraft-axis normals.
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

// The two-tone belly factor in sky mode (diffuse × this on downward faces).
const BELLY = 0.55;
const twoTone = { value: 1 }; // shared uniform: 1 = off (plain), BELLY = on (sky)

// Turn the sky two-tone on or off for every body material at once.
export function setTwoTone(on) {
  twoTone.value = on ? BELLY : 1;
}

// Sky-mode team hues (degrees): blue, red, green, amber; others are hashed.
const TEAM_HUES = [215, 4, 140, 38];

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

// The colour of aircraft `id` into `out`. Plain (`sky` false): a distinct
// per-id hue within a per-class band; an unknown vehicle class (a newer sim)
// renders as a desaturated jet. Sky: the team's hue, ±12° and lightness
// 0.46–0.56 per id; a missing team counts as team 0.
export function aircraftColor(kind, id, team, sky, out = new THREE.Color()) {
  if (!sky) {
    const baseHue = kind === "multirotor" ? 32 : 205; // amber vs. steel blue
    const jitter = ((id * 47) % 70) - 35;             // ±35° spread across the fleet
    return out.setHSL((((baseHue + jitter) % 360) + 360) % 360 / 360, kind === "unknown" ? 0.08 : 0.62, 0.58);
  }
  const t = Number.isInteger(team) ? team : 0;
  const hue = t >= 0 && t < TEAM_HUES.length ? TEAM_HUES[t] : t * 137.508 + 75; // golden-angle hash
  const jitter = (((id * 47) % 25) + 25) % 25 - 12;
  const light = 0.46 + ((((id * 29) % 11) + 11) % 11) / 100;
  // HSL in sRGB (not three's linear working space), so these are the
  // saturated mid-tones the numbers say; the plain colours keep today's look.
  return out.setHSL((((hue + jitter) % 360) + 360) % 360 / 360, 0.85, light, THREE.SRGBColorSpace);
}

// Recolour a built aircraft to `color`: its body materials (each keeping its
// HSL offset from the base; fixed-colour parts stay), the base colour and the
// label's border.
export function tintAircraft(group, color) {
  const ud = group.userData;
  ud.baseColor.copy(color);
  for (const m of ud.mats) {
    const tone = m.userData.tone;
    if (!tone) continue;
    m.color.copy(color);
    if (tone[0] || tone[1] || tone[2]) m.color.offsetHSL(tone[0], tone[1], tone[2]);
  }
  drawLabel(ud.label, color);
  ud.label.material.map.needsUpdate = true;
}

// A constant-screen-size name tag. With `sizeAttenuation: false` three.js
// multiplies the sprite's scale by its view depth, so a fixed scale is a fixed
// on-screen size; the scene sets that scale each frame from the container
// height, and lifts the tag in screen space (`center.y`) just clear of the
// aircraft and any frame around it, whichever way the camera looks. Fog is off
// so far-off tags stay legible. The canvas is 2:1 (the scene's NAME_ASPECT):
// wide enough for a four-digit id at full size; longer ids are squeezed to fit.
function labelSprite(text, color) {
  const canvas = document.createElement("canvas");
  canvas.width = 128;
  canvas.height = 64;
  const tex = new THREE.CanvasTexture(canvas);
  tex.anisotropy = 4;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
    map: tex, depthTest: false, transparent: true, sizeAttenuation: false, fog: false,
  }));
  sprite.center.set(0.5, -1); // until the scene places it
  sprite.renderOrder = 10;
  sprite.userData.text = text;
  drawLabel(sprite, color);
  return sprite;
}

// (Re)draw a name tag's canvas: dark plate, border in the aircraft colour.
function drawLabel(sprite, color) {
  const ctx = sprite.material.map.image.getContext("2d");
  ctx.clearRect(0, 0, 128, 64);
  ctx.fillStyle = "rgba(10,14,20,0.78)";
  roundRect(ctx, 4, 4, 120, 56, 12);
  ctx.fill();
  ctx.strokeStyle = `#${color.getHexString()}`;
  ctx.lineWidth = 4;
  roundRect(ctx, 4, 4, 120, 56, 12);
  ctx.stroke();
  ctx.fillStyle = "#e8eef6";
  ctx.font = "bold 40px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(sprite.userData.text, 64, 35, 104);
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

// The shared two-tone patch: the normal's y in aircraft axes (flipped on back
// faces, so a double-sided plate's underside counts as belly) darkens the
// diffuse colour toward BELLY. One function for every body material, so its
// source — three's program cache key — is the same and they share a program.
function twoToneCompile(shader) {
  shader.uniforms.uBelly = twoTone;
  shader.vertexShader = shader.vertexShader
    .replace("#include <common>", "#include <common>\nvarying float vBodyUp;")
    .replace("#include <beginnormal_vertex>", "#include <beginnormal_vertex>\n\tvBodyUp = objectNormal.y;");
  shader.fragmentShader = shader.fragmentShader
    .replace("#include <common>", "#include <common>\nuniform float uBelly;\nvarying float vBodyUp;")
    .replace("#include <color_fragment>", "#include <color_fragment>\n"
      + "\tdiffuseColor.rgb *= mix(uBelly, 1.0, smoothstep(-0.3, 0.3, gl_FrontFacing ? vBodyUp : -vBodyUp));");
}

// A body material in the aircraft colour offset by `tone` (HSL deltas), or —
// with `tone` null — a fixed colour that recolouring leaves alone.
function bodyMaterial(color, tone = [0, 0, 0]) {
  const m = new THREE.MeshStandardMaterial({
    color,
    metalness: 0.35,
    roughness: 0.55,
    emissive: 0x000000,
    emissiveIntensity: 1.0,
  });
  if (tone && (tone[0] || tone[1] || tone[2])) m.color.offsetHSL(tone[0], tone[1], tone[2]);
  m.userData.tone = tone;
  m.onBeforeCompile = twoToneCompile;
  return m;
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
  const accent = bodyMaterial(color, [0, -0.08, -0.24]); mats.push(accent);

  // Slender fuselage along Z (nose +Z) with a long pointed radome up front.
  g.add(box(0.85, 0.9, 6.4, body, 0, 0, -0.3));
  const nose = new THREE.Mesh(new THREE.ConeGeometry(0.42, 2.6, 16).rotateX(Math.PI / 2), accent); // apex -> +Z
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
  const wingMesh = new THREE.Mesh(new THREE.ShapeGeometry(wing).rotateX(Math.PI / 2), wingMat);
  wingMesh.position.set(0, -0.12, -0.3);
  g.add(wingMesh);

  // Horizontal stabilators at the tail.
  g.add(box(3.4, 0.12, 1.0, body, 0, 0, -3.05));

  // Swept single vertical fin: shape XY -> world ZY (shape +x -> world -Z, aft).
  const finMat = bodyMaterial(color, [0, 0, -0.05]);
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
  const nozzle = new THREE.Mesh(new THREE.CylinderGeometry(0.34, 0.4, 0.7, 16).rotateX(Math.PI / 2), accent);
  nozzle.position.set(0, 0, -3.7);
  g.add(nozzle);
  const burnerMat = new THREE.MeshBasicMaterial({ color: 0xff8a3d, transparent: true, opacity: 0.5, side: THREE.DoubleSide });
  const burner = disc(0.32, burnerMat);
  burner.rotation.x = Math.PI / 2; // face -Z
  burner.position.set(0, 0, -4.05);
  g.add(burner);

  g.userData.rotors = []; // a jet has no spinning prop/rotor discs
  g.userData.span = 9.5;  // nose-to-nozzle length, the largest dimension (model units)
  return g;
}

function buildMultirotor(color, mats) {
  const g = new THREE.Group();
  const body = bodyMaterial(color); mats.push(body);
  const armMat = bodyMaterial(color, [0, -0.05, -0.32]); mats.push(armMat);
  const front = bodyMaterial(new THREE.Color(0xff5a4d), null); mats.push(front); // heading marker

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
  g.userData.span = 5.6;  // rotor tip to rotor tip across the X-frame (model units)
  return g;
}

// Build one aircraft object in `color`. Returns a THREE.Group carrying, in
// userData:
//   { id, kind, mats:[], bodies:[], rotors:[], spinAxis, span, baseColor, label }
// `bodies` are the solid meshes (everything drawn with a `mats` material, not
// the translucent rotor / afterburner discs); they are also on `maskLayer`,
// which the scene's outline pass draws. `span` is the model's largest
// dimension in model units (before the scene's scale), which the scene uses to
// keep each aircraft a minimum size on screen. `label` is NOT parented to the
// group: it keeps a constant on-screen size, so it must not inherit the
// group's (distance-dependent) scale. The caller adds it to the scene and
// positions it above the aircraft each frame.
export function buildAircraft(kind, id, color, maskLayer) {
  const base = color.clone();
  const mats = [];
  const group = kind === "multirotor" ? buildMultirotor(base, mats) : buildJet(base, mats);
  const bodies = [];
  group.traverse((o) => {
    if (!o.isMesh || !mats.includes(o.material)) return;
    o.layers.enable(maskLayer);
    bodies.push(o);
  });

  group.userData.id = id;
  group.userData.kind = kind;
  group.userData.mats = mats;
  group.userData.bodies = bodies;
  group.userData.baseColor = base;
  group.userData.label = labelSprite(String(id), base);
  return group;
}
