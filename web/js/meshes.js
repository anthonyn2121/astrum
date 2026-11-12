// Procedural aircraft meshes, their colours, and the coordinate/attitude math
// that places them.
//
// No external model assets: an F-16-style jet (slender fuselage, chin intake,
// bubble canopy, cropped-delta wing, swept tail) and an F450-like quad (X-frame
// + 4 rotors) are built from primitives. The mesh is chosen by VehicleKind
// (fixed-wing vs multirotor), not the specific JSBSim model. Each kind also
// has a flat few-triangle silhouette (buildGhostGeometry) for the scene's
// instanced preview ghosts.
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

// A flat, few-triangle silhouette of the `kind`'s mesh for the scene's
// instanced preview ghosts, in the same model units and axes (nose +Z, up
// +Y) as buildAircraft: the jet's planform plus its fin, or the quad's hub,
// nose marker and rotor discs. Returns { geom, span } (span as in userData).
export function buildGhostGeometry(kind) {
  if (kind === "multirotor") {
    const shapes = [];
    const hub = new THREE.Shape();
    hub.moveTo(-0.7, -0.7); hub.lineTo(0.7, -0.7); hub.lineTo(0.7, 0.7); hub.lineTo(-0.7, 0.7); hub.closePath();
    const nose = new THREE.Shape();
    nose.moveTo(0, 1.6); nose.lineTo(-0.45, 0.85); nose.lineTo(0.45, 0.85); nose.closePath();
    shapes.push(hub, nose);
    const d = 2.3 * Math.SQRT1_2;
    for (const [sx, sy] of [[1, 1], [1, -1], [-1, 1], [-1, -1]]) {
      const r = new THREE.Shape();
      r.absarc(sx * d, sy * d, 1.15, 0, Math.PI * 2, false);
      shapes.push(r);
    }
    return { geom: new THREE.ShapeGeometry(shapes, 8).rotateX(Math.PI / 2), span: 5.6 };
  }
  // Planform (shape +y → +Z, the nose), one outline from the nose round the
  // starboard side and back up the port side.
  const half = [
    [0.45, 3.2], [0.5, 1.6], [3.1, -0.6], [3.1, -1.5], [0.5, -2.4],
    [1.7, -2.6], [1.7, -3.5], [0.4, -3.6], [0.35, -4.05],
  ];
  const plan = new THREE.Shape();
  plan.moveTo(0, 5.4);
  for (const [x, y] of half) plan.lineTo(x, y);
  for (let i = half.length - 1; i >= 0; i--) plan.lineTo(-half[i][0], half[i][1]);
  plan.closePath();
  const flat = new THREE.ShapeGeometry(plan).rotateX(Math.PI / 2).toNonIndexed();
  const p = flat.attributes.position.array;
  const fin = [0, 0.4, -1.4, 0, 2.2, -3.3, 0, 0.4, -3.4]; // the swept fin, one triangle
  const pos = new Float32Array(p.length + fin.length);
  pos.set(p);
  pos.set(fin, p.length);
  flat.dispose();
  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  return { geom, span: 9.5 };
}

// ---- Race gates ----
//
// A gate is a square clear opening in a frame of two uprights and a top bar
// (the course message's `opening_m` / `bar_m` / `top_bar_m`: the bar stock is
// `bar_m` square, the uprights flank the opening and are as tall as it is, and
// the top bar spans the opening plus both uprights). The sim places it by the
// CENTRE OF THE OPENING (`east_m`, `north_m`, `up_m`, ENU metres) and `yaw_rad`,
// the heading of the gate's normal — the flight direction through it — with 0 =
// North and + toward East, the same convention as an aircraft's heading.
//
// Placement. The sim's gate frame is (right, up, normal) with right = starboard
// of the flight direction, i.e. in ENU (psi = yaw_rad):
//     normal = ( sin psi,  cos psi, 0)      right = (cos psi, -sin psi, 0)
// Through the grid mapping (three.x = E, three.y = U, three.z = -N) those are
//     normal = ( sin psi, 0, -cos psi)      right = (cos psi,  0,  sin psi)
// (right, up, normal) is LEFT-handed, so — exactly as attitudeToQuaternion does
// for an aircraft — the mesh's local +X is the gate's PORT side (-right), which
// makes (port, up, normal) a proper rotation: Ry(theta) takes local +Z to
// (sin theta, 0, cos theta) and local +X to (cos theta, 0, -sin theta), so
// theta = pi - psi. The frame is mirror-symmetric across x = 0, so using port
// for local +X changes nothing visible. The gate stands upright: no bank, no tilt.

// Gate colours by progress: ahead of the drone, the one it must fly next, and
// behind it. `fill` is the opacity of the pane drawn in the opening.
const GATE_LOOK = {
  future: { color: 0x8fb4e0, emissive: 0.3, fill: 0.05, opacity: 1 },
  pending: { color: 0xffc23d, emissive: 0.65, fill: 0.2, opacity: 1 },
  passed: { color: 0x4fd18b, emissive: 0.25, fill: 0, opacity: 0.55 },
};

const _gateUp = new THREE.Vector3(0, 1, 0);

// The orientation of a gate whose normal has heading `yawRad` (see above).
export function gateQuaternion(yawRad, out = new THREE.Quaternion()) {
  return out.setFromAxisAngle(_gateUp, Math.PI - yawRad);
}

// Put a gate group (from buildGate) where `gate` — a course message gate
// { east_m, north_m, up_m, yaw_rad } — stands.
export function placeGate(group, gate) {
  enuToThree(gate.east_m, gate.north_m, gate.up_m, group.position);
  gateQuaternion(gate.yaw_rad, group.quaternion);
}

// One gate's frame, built about the centre of its opening in the gate's local
// axes (+X port, +Y up, +Z along the normal). Returns a THREE.Group whose
// userData carries { span, top, bars, pane, mats }: `span` is the frame's
// largest dimension (its diagonal) and `top` how far the frame's top edge is
// above the centre of the opening, in metres at scale 1, which the scene uses
// to keep a far gate visible and to lift its number tag. Colour it with setGateState.
export function buildGate({ opening_m: opening, bar_m: bar, top_bar_m: topBar }) {
  const g = new THREE.Group();
  const mat = new THREE.MeshStandardMaterial({ metalness: 0.2, roughness: 0.6, transparent: true });
  const half = opening / 2, t = bar / 2;
  const bars = [
    box(bar, opening, bar, mat, -(half + t), 0, 0), // port upright
    box(bar, opening, bar, mat, half + t, 0, 0),    // starboard upright
    box(topBar, bar, bar, mat, 0, half + t, 0),     // top bar, on both uprights
  ];
  g.add(...bars);
  const paneMat = new THREE.MeshBasicMaterial({
    transparent: true, depthWrite: false, side: THREE.DoubleSide, fog: false,
  });
  const pane = new THREE.Mesh(new THREE.PlaneGeometry(opening, opening), paneMat);
  g.add(pane);
  g.userData = {
    span: Math.hypot(topBar, opening + bar), top: half + bar,
    bars, pane, mats: [mat, paneMat],
  };
  setGateState(g, "future");
  return g;
}

// Colour a gate (and, given, its number tag) for its progress: "future",
// "pending" or "passed".
export function setGateState(group, state, label = null) {
  const look = GATE_LOOK[state];
  const [mat, paneMat] = group.userData.mats;
  mat.color.setHex(look.color);
  mat.emissive.setHex(look.color).multiplyScalar(look.emissive);
  mat.opacity = look.opacity;
  paneMat.color.setHex(look.color);
  paneMat.opacity = look.fill;
  group.userData.pane.visible = look.fill > 0;
  if (label) {
    drawLabel(label, new THREE.Color(look.color));
    label.material.map.needsUpdate = true;
  }
}

// A gate's number tag: the course message's `index` + 1, as aircraft tags are
// drawn (a constant on-screen size the scene sets each frame).
export function buildGateLabel(index) {
  return labelSprite(String(index + 1), new THREE.Color(GATE_LOOK.future.color));
}

// The scale that draws the quad mesh at its true size: model units × this =
// metres. buildMultirotor's motors sit 4.6 / 2 units out on the arms and carry
// 1.15-unit rotor discs, so the mesh reaches 3.45 units from the hub; the F450
// reaches ARM_M + PROP_RADIUS_M = 0.1651 + 0.1194 m (crates/astrum-race
// airframe.rs HALF_SPAN_M). The scene applies it to multirotors when a race
// course is on screen, so the drone and a ~1 m gate are in proportion.
export const F450_UNIT = (0.1651 + 9.4 * 0.0254 / 2) / (4.6 / 2 + 1.15);
