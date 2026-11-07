// The sandbox: a scenario editor on the scene's map. Loads `?scenario=<name>`
// (or starts from the empty environment's defaults), shows the fleet at its
// initial conditions over imagery, and edits them — drag / rotate / raise on
// the map, an inspector with every AircraftSpec field, spawn patterns,
// duplicate / delete, undo / redo, the world origin — while a live view shows
// the resulting scenario JSON.
//
// The scenario is kept in astrum_sim::scenario::Scenario's serde shape
// (`{world, sim, aircraft: [...]}`, optional fields absent when unset,
// numbers as numbers), so it can be sent back to the server unchanged. The
// scene never changes it: its edit callbacks report ENU / heading values,
// which land here, and the scene redraws from an EditStore built from the
// scenario. Checks that mirror Scenario::validate run on every change and show
// inline (fields, fleet list, the problems list over the JSON).

import { Scene } from "./scene.js";
import { EditStore } from "./editstore.js";
import { Datum, DEG2RAD } from "./geo.js";
import { getEnvironments, getModels, getScenario } from "./api.js";

const DEFAULT_ENVIRONMENT = "empty";
const HISTORY_MAX = 200;            // undo steps kept
const DUPLICATE_OFFSET_M = 100;     // a copy sits this far right of its original
const MAX_ID = 0xffffffff;          // AircraftSpec.id is a u32
const MAX_TEAM = 255;               // u8
const VIEW_KEEP_M = 100000;         // an origin move keeps the view on its ground within this range
const MAX_AMMO = 65535;             // u16
const MAX_SPAWN = 200;
const MIN_SPEED = 0.01;             // astrum_net::MIN_SPEED (sim.speed: 0 or ≥ this)
// A spawned aircraft's TAS by model, and by kind for other models: a fast
// cruise for the jets, hover (0) for multirotors.
const SPAWN_TAS = { f16: 150, c172x: 55 };
const SPAWN_TAS_KIND = { "fixed-wing": 100, multirotor: 0 };
// AircraftSpec's fields in serde order; `skip` says when serde leaves one out.
const AIRCRAFT_FIELDS = [
  ["id"], ["model"], ["kind", "unset"], ["autopilot_profile", "unset"],
  ["lat_deg"], ["lon_deg"], ["alt_hae_m"], ["heading_deg"], ["tas_mps"],
  ["climb_deg", "zero"], ["bank_deg", "zero"], ["trim", "unset"], ["lockstep", "false"],
  ["control_rate_hz", "unset"], ["team"], ["health"], ["ammo", "unset"],
];
const AIRCRAFT_DEFAULTS = { team: 0, health: 100 };

const $ = (id) => document.getElementById(id);

const state = {
  scenario: null,    // { world, sim, aircraft }
  selection: [],     // aircraft ids; the last is the primary (inspector, handles)
  models: [],        // /api/models
  env: null,         // the environment (/api/environments entry)
  undo: [],
  redo: [],
  dragBase: null,    // snapshot at an edit drag's start
  entry: new Map(),  // "<id>:<field>" | "world:<field>" → a refused entry's message
  pickingOrigin: false,
  previewSpawn: false,
  spawnSeed: 1,      // the random pattern's seed; advances after each spawn
};

const store = new EditStore();
const scene = new Scene($("view"));
scene.setCameraMode("map");
scene.setEditing({ onSelect, onEdit, onPoint });
scene.setModelScale(1); // true size up close: spacing reads right; the on-screen floor keeps far ones visible

// ---- Scenario helpers ------------------------------------------------------

const modelEntry = (model) => state.models.find((m) => m.model === model) ?? null;
const kindOf = (a) => a.kind ?? modelEntry(a.model)?.kind ?? "unknown";
const byId = (id) => state.scenario.aircraft.find((a) => a.id === id) ?? null;
const datum = () => {
  const w = state.scenario.world;
  return Datum.fromDegrees(w.origin_lat_deg, w.origin_lon_deg, w.origin_alt_hae_m);
};
const round = (v, dp) => Number(v.toFixed(dp));
const wrap360 = (deg) => ((deg % 360) + 360) % 360;
const isNum = (v) => typeof v === "number" && Number.isFinite(v);

// One aircraft in serde order, optional fields dropped when unset (and
// climb / bank 0, lockstep false), team / health always present.
function serdeAircraft(a) {
  const out = {};
  for (const [k, skip] of AIRCRAFT_FIELDS) {
    const v = a[k] ?? AIRCRAFT_DEFAULTS[k];
    if (v === undefined || v === null) continue;
    if (skip === "zero" && v === 0) continue;
    if (skip === "false" && v === false) continue;
    out[k] = v;
  }
  return out;
}

function scenarioDoc() {
  const { world, sim, aircraft } = state.scenario;
  return {
    world: { origin_lat_deg: world.origin_lat_deg, origin_lon_deg: world.origin_lon_deg, origin_alt_hae_m: world.origin_alt_hae_m },
    sim: { ...sim },
    aircraft: aircraft.map(serdeAircraft),
  };
}

function snapshot() {
  return JSON.stringify({ scenario: state.scenario, selection: state.selection });
}

function restore(snap) {
  const { scenario, selection } = JSON.parse(snap);
  state.scenario = scenario;
  state.selection = selection.filter((id) => scenario.aircraft.some((a) => a.id === id));
}

function pushHistory(snap) {
  state.undo.push(snap);
  if (state.undo.length > HISTORY_MAX) state.undo.shift();
  state.redo = [];
}

// Apply one undoable change: `fn` mutates state.scenario / state.selection.
function commit(fn) {
  const before = snapshot();
  fn();
  if (snapshot() !== before) pushHistory(before);
  refresh();
}

function undo() {
  if (!state.undo.length) return;
  scene.cancelEdit();
  state.redo.push(snapshot());
  travel(state.undo.pop());
}

function redo() {
  if (!state.redo.length) return;
  scene.cancelEdit();
  state.undo.push(snapshot());
  travel(state.redo.pop());
}

// Restore a history snapshot; if that moves the origin, the view stays on
// its ground as it does for the move itself.
function travel(snap) {
  const w0 = { ...state.scenario.world }, before = datum();
  restore(snap);
  state.entry.clear();
  const w = state.scenario.world;
  if (["origin_lat_deg", "origin_lon_deg", "origin_alt_hae_m"].some((k) => w[k] !== w0[k])) keepViewOnGround(before, datum());
  refresh();
}

// The `count` lowest unused ids from 1 up (fewer when the u32 range runs out).
function freshIds(count) {
  const used = new Set(state.scenario.aircraft.map((a) => a.id));
  const ids = [];
  for (let id = 1; ids.length < count && id <= MAX_ID; id++) {
    if (!used.has(id)) ids.push(id);
  }
  return ids;
}

// ---- Validation (mirrors Scenario::validate where cheap) -------------------

// [{ id?, field, msg }]: `id` an aircraft's, absent for world / sim / fleet.
function validate(scn) {
  const out = [];
  const add = (id, field, msg) => out.push({ id, field, msg: id === undefined ? msg : `aircraft ${id}: ${msg}` });
  const w = scn.world;
  for (const k of ["origin_lat_deg", "origin_lon_deg", "origin_alt_hae_m"]) {
    if (!isNum(w[k])) add(undefined, `world.${k}`, `world.${k} must be finite, got ${w[k]}`);
  }
  if (isNum(w.origin_lat_deg) && Math.abs(w.origin_lat_deg) > 90) {
    add(undefined, "world.origin_lat_deg", `world.origin_lat_deg out of range: ${w.origin_lat_deg}`);
  }
  const s = scn.sim;
  if (!(isNum(s.rate_hz) && s.rate_hz > 0)) add(undefined, "sim.rate_hz", `sim.rate_hz must be > 0, got ${s.rate_hz}`);
  if (!(isNum(s.max_time_s) && s.max_time_s >= 0)) add(undefined, "sim.max_time_s", `sim.max_time_s must be >= 0, got ${s.max_time_s}`);
  if (!(isNum(s.speed) && (s.speed === 0 || s.speed >= MIN_SPEED))) {
    add(undefined, "sim.speed", `sim.speed must be 0 (as fast as possible) or >= ${MIN_SPEED}, got ${s.speed}`);
  }
  if (!(isNum(s.command_lease_s) && s.command_lease_s >= 0)) add(undefined, "sim.command_lease_s", `sim.command_lease_s must be >= 0, got ${s.command_lease_s}`);
  if (!(isNum(s.lockstep_timeout_s) && s.lockstep_timeout_s > 0)) add(undefined, "sim.lockstep_timeout_s", `sim.lockstep_timeout_s must be > 0, got ${s.lockstep_timeout_s}`);
  // Integer fields serde would refuse to read otherwise (u64 / usize).
  if (!(Number.isSafeInteger(s.seed) && s.seed >= 0)) add(undefined, "sim.seed", `sim.seed must be an integer >= 0, got ${s.seed}`);
  if (!(Number.isSafeInteger(s.threads) && s.threads >= 0)) add(undefined, "sim.threads", `sim.threads must be an integer >= 0 (0 = every core), got ${s.threads}`);
  if (!scn.aircraft.length) add(undefined, "aircraft", "scenario has no aircraft");

  const seen = new Set();
  for (const a of scn.aircraft) {
    const id = a.id;
    if (!(Number.isInteger(id) && id >= 0 && id <= MAX_ID)) add(id, "id", `id must be an integer 0–${MAX_ID}`);
    if (seen.has(id)) add(id, "id", `duplicate aircraft id ${id}`);
    seen.add(id);
    for (const k of ["lat_deg", "lon_deg", "alt_hae_m", "heading_deg", "tas_mps", "climb_deg", "bank_deg"]) {
      const v = a[k] ?? 0;
      if (!isNum(v)) add(id, k, `${k} must be finite, got ${v}`);
    }
    if (isNum(a.lat_deg) && Math.abs(a.lat_deg) > 90) add(id, "lat_deg", `lat_deg out of range: ${a.lat_deg}`);
    for (const k of ["climb_deg", "bank_deg"]) {
      const v = a[k] ?? 0;
      if (isNum(v) && Math.abs(v) >= 90) add(id, k, `${k} must be within (-90, 90), got ${v}`);
    }
    if (isNum(a.tas_mps) && a.tas_mps < 0) add(id, "tas_mps", `tas_mps must be >= 0, got ${a.tas_mps}`);
    // health is an f32 on the server: a value past its range is infinite there.
    const health = a.health ?? 100;
    if (!(isNum(health) && Number.isFinite(Math.fround(health)) && health > 0)) add(id, "health", `health must be > 0, got ${health}`);
    const team = a.team ?? 0;
    if (!(Number.isInteger(team) && team >= 0 && team <= MAX_TEAM)) add(id, "team", `team must be an integer 0–${MAX_TEAM}`);
    if (a.lockstep && !s.lockstep) add(id, "lockstep", "lockstep = true requires sim.lockstep = true");
    const known = modelEntry(a.model);
    const kind = a.kind ?? known?.kind;
    if (!kind) add(id, "kind", `model '${a.model}' has no built-in profile; set kind = "fixed-wing" | "multirotor"`);
    else if (kind === "multirotor" && a.autopilot_profile != null) add(id, "autopilot_profile", "autopilot_profile applies to fixed-wing aircraft only");
    else if (kind === "fixed-wing" && a.autopilot_profile == null && !(known?.kind === "fixed-wing" && known.autopilot_profile)) {
      add(id, "autopilot_profile", `fixed-wing model '${a.model}' has no built-in autopilot gains; set autopilot_profile = "trainer" | "fighter"`);
    }
    if (a.ammo != null) {
      if (kind === "multirotor") add(id, "ammo", "ammo applies to armed fixed-wing aircraft only");
      else if (!(Number.isInteger(a.ammo) && a.ammo >= 0 && a.ammo <= MAX_AMMO)) add(id, "ammo", `ammo must be an integer 0–${MAX_AMMO}`);
    }
    if (a.control_rate_hz != null) {
      const c = a.control_rate_hz, ratio = s.rate_hz / c;
      if (!(isNum(c) && c > 0 && ratio >= 1) || Math.abs(ratio - Math.round(ratio)) > 1e-9 * ratio) {
        add(id, "control_rate_hz", `control_rate_hz must divide sim.rate_hz (${s.rate_hz}) evenly, got ${c}`);
      } else if (s.command_lease_s > 0 && s.command_lease_s < 2 / c) {
        add(id, "control_rate_hz", `control_rate_hz ${c} needs sim.command_lease_s >= ${2 / c} (two control periods), got ${s.command_lease_s}`);
      }
    }
  }
  if (s.lockstep && !scn.aircraft.some((a) => a.lockstep)) {
    add(undefined, "sim.lockstep", "sim.lockstep = true but no aircraft has lockstep = true");
  }
  return out;
}

// ---- Scene callbacks -------------------------------------------------------

function onSelect(ids, mode) {
  const sel = state.selection;
  if (mode === "replace") state.selection = [...ids];
  else if (mode === "toggle") {
    for (const id of ids) {
      const i = sel.indexOf(id);
      if (i >= 0) sel.splice(i, 1);
      else sel.push(id);
    }
  } else {
    for (const id of ids) if (!sel.includes(id)) sel.push(id);
  }
  state.entry.clear();
  refresh();
}

function onEdit({ type, phase, changes }) {
  if (phase === "start") { state.dragBase = snapshot(); return; }
  if (!state.dragBase) return;
  if (phase === "cancel") {
    restore(state.dragBase);
    state.dragBase = null;
    refresh();
    return;
  }
  const d = datum();
  for (const c of changes) {
    const a = byId(c.id);
    if (!a) continue;
    if (type === "rotate") {
      a.heading_deg = round(wrap360(c.heading_deg), 1); // a true heading (snapped by the scene on Shift)
    } else {
      const g = d.geodeticOf(c.e, c.n, c.u);
      if (type === "move") {
        a.lat_deg = round(g.lat_deg, 7);
        a.lon_deg = round(g.lon_deg, 7);
      } else {
        a.alt_hae_m = round(g.alt_hae_m, 1);
      }
    }
  }
  if (phase === "end") {
    if (snapshot() !== state.dragBase) pushHistory(state.dragBase);
    state.dragBase = null;
  }
  refresh();
}

function onPoint({ e, n }) {
  if (!state.pickingOrigin) return;
  const g = datum().geodeticOf(e, n, 0);
  setPicking(false);
  const w = state.scenario.world;
  moveOrigin(round(g.lat_deg, 7), round(g.lon_deg, 7), w.origin_alt_hae_m);
}

// ---- World origin ----------------------------------------------------------

// Move the ENU origin (an undoable change). The fleet keeps its lat / lon
// unless "move the fleet with the origin" is on, when it keeps its ENU
// offsets instead. The map view stays on the same ground — unless that is
// now beyond the map's reach from the origin, when it centres on the origin.
function moveOrigin(lat, lon, alt) {
  const w = state.scenario.world;
  if (lat === w.origin_lat_deg && lon === w.origin_lon_deg && alt === w.origin_alt_hae_m) return;
  const before = datum();
  const after = Datum.fromDegrees(lat, lon, alt);
  const withFleet = $("moveFleet").checked;
  commit(() => {
    if (withFleet) {
      for (const a of state.scenario.aircraft) {
        if (![a.lat_deg, a.lon_deg, a.alt_hae_m].every(isNum)) continue;
        const p = before.enuOf(a.lat_deg, a.lon_deg, a.alt_hae_m);
        const g = after.geodeticOf(p.e, p.n, p.u);
        a.lat_deg = round(g.lat_deg, 7);
        a.lon_deg = round(g.lon_deg, 7);
        a.alt_hae_m = round(g.alt_hae_m, 2);
      }
    }
    Object.assign(state.scenario.world, { origin_lat_deg: lat, origin_lon_deg: lon, origin_alt_hae_m: alt });
  });
  if (!withFleet) keepViewOnGround(before, after);
}

// Re-centre the view on the ground it showed in datum `before`, now in
// `after` — or on the origin when that ground is beyond the map's reach.
function keepViewOnGround(before, after) {
  const c = scene.viewCenter();
  const g = before.geodeticOf(c.e, c.n, 0);
  const p = after.enuOf(g.lat_deg, g.lon_deg, g.alt_hae_m);
  if (Math.hypot(p.e, p.n) <= VIEW_KEEP_M) scene.setViewCenter(p.e, p.n);
  else scene.setViewCenter(0, 0);
}

function setPicking(on) {
  state.pickingOrigin = on;
  $("pickOriginBtn").setAttribute("aria-pressed", String(on));
  $("pickBanner").hidden = !on;
}

// ---- Spawn / duplicate / delete --------------------------------------------

// The spawn form's aircraft (centred on the view), or { error }.
function spawnPlan() {
  const model = $("spawnModel").value;
  const count = Number($("spawnCount").value);
  const spacing = Number($("spawnSpacing").value);
  const alt = Number($("spawnAlt").value);
  const heading = Number($("spawnHeading").value);
  const tas = Number($("spawnTas").value);
  const team = Number($("spawnTeam").value);
  const pattern = $("spawnPattern").value;
  if (!model) return { error: "choose a model" };
  if (!(Number.isInteger(count) && count >= 1 && count <= MAX_SPAWN)) return { error: `count must be an integer 1–${MAX_SPAWN}` };
  for (const [name, input] of [["spacing", "spawnSpacing"], ["altitude", "spawnAlt"], ["heading", "spawnHeading"], ["TAS", "spawnTas"]]) {
    if ($(input).value.trim() === "" || !isNum(Number($(input).value))) return { error: `${name} must be a number` };
  }
  if (spacing < 0) return { error: "spacing must be >= 0" };
  if (tas < 0) return { error: "TAS must be >= 0" };
  if (!(Number.isInteger(team) && team >= 0 && team <= MAX_TEAM)) return { error: `team must be an integer 0–${MAX_TEAM}` };
  const ids = freshIds(count);
  if (ids.length < count) return { error: "no free aircraft ids left" };

  // Offsets (forward, right) in the formation's own frame, metres. Random
  // draws come from a seeded generator, so the preview is what spawns.
  const offsets = [];
  const cols = Math.ceil(Math.sqrt(count));
  const rand = mulberry32(state.spawnSeed);
  for (let i = 0; i < count; i++) {
    if (pattern === "abreast") offsets.push([0, (i - (count - 1) / 2) * spacing]);
    else if (pattern === "trail") offsets.push([-i * spacing, 0]);
    else if (pattern === "grid") {
      const row = Math.floor(i / cols), col = i % cols;
      offsets.push([-row * spacing, (col - (cols - 1) / 2) * spacing]);
    } else {
      offsets.push([(rand() - 0.5) * spacing, (rand() - 0.5) * spacing]);
    }
  }
  const d = datum(), c = scene.viewCenter();
  const h = heading * DEG2RAD, sh = Math.sin(h), ch = Math.cos(h);
  const aircraft = offsets.map(([f, r], i) => {
    const g = d.geodeticOf(c.e + f * sh + r * ch, c.n + f * ch - r * sh, 0);
    return {
      id: ids[i], model, lat_deg: round(g.lat_deg, 7), lon_deg: round(g.lon_deg, 7),
      alt_hae_m: alt, heading_deg: heading, tas_mps: tas, team, health: 100,
    };
  });
  return { aircraft };
}

function spawn() {
  const plan = spawnPlan();
  showSpawnError(plan.error);
  if (plan.error) return;
  commit(() => {
    state.scenario.aircraft.push(...plan.aircraft);
    state.selection = plan.aircraft.map((a) => a.id);
  });
  state.spawnSeed++; // the next random spawn draws afresh
  updatePreview();
}

// A small seeded PRNG (mulberry32): uniform [0, 1).
function mulberry32(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function showSpawnError(msg) {
  $("spawnError").hidden = !msg;
  $("spawnError").textContent = msg ?? "";
}

// The spawn preview: ghosts where the form would put its aircraft, while the
// pointer or focus is on the form. Its own ghost layer, so other previews stay.
function updatePreview() {
  if (!state.previewSpawn) { scene.setGhosts("spawn", null); return; }
  const plan = spawnPlan();
  showSpawnError(plan.error);
  if (plan.error) { scene.setGhosts("spawn", null); return; }
  const d = datum(), kind = modelEntry($("spawnModel").value)?.kind ?? "fixed-wing";
  scene.setGhosts("spawn", plan.aircraft.map((a) => {
    const p = d.enuOf(a.lat_deg, a.lon_deg, a.alt_hae_m);
    return { kind, e: p.e, n: p.n, u: p.u, heading_deg: a.heading_deg };
  }));
}

function showFleetStatus(msg) {
  $("fleetStatus").hidden = !msg;
  $("fleetStatus").textContent = msg ?? "";
}

function duplicate() {
  const src = state.selection.map(byId).filter(Boolean);
  if (!src.length) return;
  const ids = freshIds(src.length);
  if (ids.length < src.length) {
    showFleetStatus(`Cannot duplicate ${src.length} aircraft: only ${ids.length} aircraft ids are free.`);
    return;
  }
  showFleetStatus(null);
  const d = datum();
  commit(() => {
    const copies = src.map((a, i) => {
      const copy = { ...structuredClone(a), id: ids[i] };
      if ([a.lat_deg, a.lon_deg, a.alt_hae_m, a.heading_deg].every(isNum)) {
        const p = d.enuOf(a.lat_deg, a.lon_deg, a.alt_hae_m);
        const h = a.heading_deg * DEG2RAD;
        const g = d.geodeticOf(p.e + DUPLICATE_OFFSET_M * Math.cos(h), p.n - DUPLICATE_OFFSET_M * Math.sin(h), p.u);
        copy.lat_deg = round(g.lat_deg, 7);
        copy.lon_deg = round(g.lon_deg, 7);
      }
      return copy;
    });
    state.scenario.aircraft.push(...copies);
    state.selection = copies.map((a) => a.id);
  });
}

function removeSelected() {
  if (!state.selection.length) return;
  const gone = new Set(state.selection);
  showFleetStatus(null);
  commit(() => {
    state.scenario.aircraft = state.scenario.aircraft.filter((a) => !gone.has(a.id));
    state.selection = [];
  });
}

// ---- Inspector, world, sim -------------------------------------------------

const insInputs = [...document.querySelectorAll("#insForm [data-field]")];
const worldInputs = [...document.querySelectorAll("#worldForm [data-world]")];
const simInputs = [...document.querySelectorAll("#simForm [data-sim]")];
// Fields that are unset (left out of the JSON) when their control is blank.
const OPTIONAL = new Set(["kind", "autopilot_profile", "trim", "control_rate_hz", "ammo"]);
const INTEGERS = new Set(["id", "team", "ammo", "seed", "threads"]);
const SELECTS = new Set(["model", "kind", "autopilot_profile", "trim"]);

function fieldValue(a, field, d) {
  if (field.startsWith("enu_")) {
    if (![a.lat_deg, a.lon_deg, a.alt_hae_m].every(isNum)) return "";
    const p = d.enuOf(a.lat_deg, a.lon_deg, a.alt_hae_m);
    return String(round(p[field.slice(4)], 1));
  }
  const v = a[field] ?? AIRCRAFT_DEFAULTS[field] ?? (field === "climb_deg" || field === "bank_deg" ? 0 : null);
  return v === null || v === undefined ? "" : String(v);
}

// A control's typed value for `field`, or { error } (the entry is refused).
// Optional fields read a blank control as unset (undefined); a checkbox is
// true or unset (serde's default false).
function parseField(field, input) {
  if (input.type === "checkbox") return input.checked ? true : undefined;
  const raw = input.value;
  if (field === "trim") return raw === "" ? undefined : raw === "true";
  if (SELECTS.has(field)) return raw === "" ? undefined : raw;
  if (raw.trim() === "") {
    if (OPTIONAL.has(field)) return undefined;
    return { error: `${field} must be a number` };
  }
  const v = Number(raw);
  if (!Number.isFinite(v)) return { error: `${field} must be a finite number` };
  if (INTEGERS.has(field) && !Number.isSafeInteger(v)) return { error: `${field} must be an integer` };
  return v;
}

// Fields that only apply to fixed-wing aircraft go when the class turns multirotor.
function dropFixedWingFields(a) {
  if (kindOf(a) !== "multirotor") return;
  delete a.autopilot_profile;
  delete a.ammo;
}

function onInspectorChange(input) {
  const a = byId(state.selection.at(-1));
  if (!a) return;
  const field = input.dataset.field;
  const key = `${a.id}:${field}`;
  const v = parseField(field, input);
  if (v && typeof v === "object") {
    state.entry.set(key, `aircraft ${a.id}: ${v.error}`);
    renderIssues();
    return;
  }
  if (field === "id" && v !== a.id && byId(v)) {
    state.entry.set(key, `aircraft ${a.id}: id ${v} is taken`);
    renderIssues();
    return;
  }
  state.entry.delete(key);
  const d = datum();
  commit(() => {
    if (field === "id") {
      state.selection = state.selection.map((id) => (id === a.id ? v : id));
      a.id = v;
    } else if (field.startsWith("enu_")) {
      const p = d.enuOf(a.lat_deg, a.lon_deg, a.alt_hae_m);
      p[field.slice(4)] = v;
      const g = d.geodeticOf(p.e, p.n, p.u);
      a.lat_deg = round(g.lat_deg, 7);
      a.lon_deg = round(g.lon_deg, 7);
      if (field === "enu_u") a.alt_hae_m = round(g.alt_hae_m, 2);
    } else if (v === undefined) {
      delete a[field];
    } else {
      a[field] = v;
    }
    if (field === "model" || field === "kind") dropFixedWingFields(a);
  });
}

function onWorldChange(input) {
  const field = input.dataset.world;
  const key = `world:${field}`;
  const v = parseField(field, input);
  if (v && typeof v === "object") {
    state.entry.set(key, `world.${field}: ${v.error.replace(`${field} `, "")}`);
    renderIssues();
    return;
  }
  if (field === "origin_lat_deg" && Math.abs(v) > 90) {
    state.entry.set(key, `world.origin_lat_deg out of range: ${v}`);
    renderIssues();
    return;
  }
  state.entry.delete(key);
  const w = { ...state.scenario.world, [field]: v };
  moveOrigin(w.origin_lat_deg, w.origin_lon_deg, w.origin_alt_hae_m);
}

function onSimChange(input) {
  const field = input.dataset.sim;
  const key = `sim:${field}`;
  const v = field === "lockstep" ? input.checked : parseField(field, input);
  if (v && typeof v === "object") {
    state.entry.set(key, `sim.${field}: ${v.error.replace(`${field} `, "")}`);
    renderIssues();
    return;
  }
  state.entry.delete(key);
  commit(() => { state.scenario.sim[field] = v; });
}

function fillModelSelect(select, current) {
  const names = state.env?.models?.length ? state.env.models : state.models.map((m) => m.model);
  const options = [...new Set([...names, ...(current && !names.includes(current) ? [current] : [])])];
  if (select.options.length === options.length && [...select.options].every((o, i) => o.value === options[i])) return;
  select.replaceChildren(...options.map((m) => {
    const o = document.createElement("option");
    o.value = m;
    o.textContent = modelEntry(m) ? `${m} (${modelEntry(m).kind})` : m;
    return o;
  }));
}

// Issue messages by field: validate()'s whose `match` holds, then the
// refused entries under `prefix` ("<id>:", "world:", "sim:").
function badFields(issues, match, prefix, strip) {
  const bad = new Map();
  for (const i of issues) if (match(i)) bad.set(strip(i.field), i.msg);
  for (const [key, msg] of state.entry) if (key.startsWith(prefix)) bad.set(key.slice(prefix.length), msg);
  return bad;
}

function showErrors(el, bad) {
  const msgs = [...new Set(bad.values())];
  el.hidden = !msgs.length;
  el.textContent = msgs.join("\n");
}

function renderInspector(issues) {
  const sel = state.selection;
  const a = byId(sel.at(-1));
  $("insEmpty").hidden = !!a;
  $("insForm").hidden = !a;
  if (!a) return;
  $("insMulti").hidden = sel.length < 2;
  $("insMulti").textContent = sel.length < 2 ? "" : `${sel.length} aircraft selected — editing ${a.id}, the last picked`;
  const d = datum();
  const form = $("insForm");
  fillModelSelect(form.querySelector('[data-field="model"]'), a.model);
  // "Default" says what the model gives when the field is unset.
  const known = modelEntry(a.model);
  form.querySelector('[data-field="kind"] option[value=""]').textContent = `Default (${known?.kind ?? "none"})`;
  form.querySelector('[data-field="autopilot_profile"] option[value=""]').textContent =
    `Default (${known?.autopilot_profile ?? "none"})`;
  const bad = badFields(issues, (i) => i.id === a.id, `${a.id}:`, (f) => f);
  for (const input of insInputs) {
    const field = input.dataset.field;
    if (input.type === "checkbox") input.checked = !!a[field];
    else if (document.activeElement !== input) {
      input.value = field === "trim" ? (a.trim === undefined ? "" : String(a.trim)) : fieldValue(a, field, d);
    }
    input.setAttribute("aria-invalid", String(bad.has(field)));
  }
  const fixedWing = kindOf(a) === "fixed-wing";
  form.querySelector("[data-armed]").hidden = !fixedWing;
  form.querySelector("[data-fixed-wing]").hidden = !fixedWing && a.autopilot_profile == null;
  showErrors($("insErrors"), bad);
}

function renderWorld(issues) {
  const w = state.scenario.world;
  const bad = badFields(issues, (i) => i.field.startsWith("world."), "world:", (f) => f.slice(6));
  for (const input of worldInputs) {
    const field = input.dataset.world;
    if (document.activeElement !== input) input.value = String(w[field]);
    input.setAttribute("aria-invalid", String(bad.has(field)));
  }
  const locs = state.env?.locations ?? [];
  const sel = $("worldLocation");
  $("locationRow").hidden = !locs.length;
  const match = locs.findIndex((l) => l.origin_lat_deg === w.origin_lat_deg && l.origin_lon_deg === w.origin_lon_deg
    && l.origin_alt_hae_m === w.origin_alt_hae_m);
  if (sel.options.length !== locs.length + 1) {
    const custom = document.createElement("option");
    custom.value = "";
    custom.textContent = "Custom";
    sel.replaceChildren(custom, ...locs.map((l, i) => {
      const o = document.createElement("option");
      o.value = String(i);
      o.textContent = l.name;
      return o;
    }));
  }
  sel.value = match >= 0 ? String(match) : "";
  showErrors($("worldErrors"), bad);
}

function renderSim(issues) {
  const s = state.scenario.sim;
  const bad = badFields(issues, (i) => i.field.startsWith("sim."), "sim:", (f) => f.slice(4));
  for (const input of simInputs) {
    const field = input.dataset.sim;
    if (input.type === "checkbox") input.checked = !!s[field];
    else if (document.activeElement !== input) input.value = String(s[field]);
    input.setAttribute("aria-invalid", String(bad.has(field)));
  }
  showErrors($("simErrors"), bad);
}

// ---- Fleet list, JSON, issues ----------------------------------------------

function renderFleet(issues) {
  const list = $("fleetList");
  const bad = new Set(issues.filter((i) => i.id !== undefined).map((i) => i.id));
  const items = state.scenario.aircraft.map((a) => {
    const li = document.createElement("li");
    li.setAttribute("role", "option");
    li.tabIndex = 0;
    li.dataset.id = String(a.id);
    li.setAttribute("aria-selected", String(state.selection.includes(a.id)));
    const sw = document.createElement("span");
    sw.className = "swatch";
    sw.style.background = scene.colorOf(a.id) ?? "";
    const id = document.createElement("span");
    id.className = "sb-id";
    id.textContent = String(a.id);
    const meta = document.createElement("span");
    meta.className = "sb-meta";
    meta.textContent = `${a.model} · team ${a.team ?? 0}`;
    li.append(sw, id, meta);
    if (bad.has(a.id)) {
      const warn = document.createElement("span");
      warn.className = "sb-bad";
      warn.textContent = "!";
      warn.title = "has problems (see Scenario JSON)";
      li.append(warn);
    }
    return li;
  });
  list.setAttribute("role", "listbox");
  list.setAttribute("aria-multiselectable", "true");
  list.replaceChildren(...items);
  $("fleetEmpty").hidden = items.length > 0;
  $("fleetCount").textContent = String(items.length);
  $("dupBtn").disabled = $("delBtn").disabled = !state.selection.length;
}

function renderIssues(issues = validate(state.scenario)) {
  const all = [...issues.map((i) => i.msg), ...state.entry.values()];
  const ul = $("issues");
  if (!all.length) {
    const li = document.createElement("li");
    li.className = "sb-ok";
    li.textContent = "No problems found.";
    ul.replaceChildren(li);
  } else {
    ul.replaceChildren(...[...new Set(all)].map((m) => {
      const li = document.createElement("li");
      li.textContent = m;
      return li;
    }));
  }
  renderInspector(issues);
  renderWorld(issues);
  renderSim(issues);
  return issues;
}

// Everything derived from the scenario: the scene's store and origin, the
// panels, the JSON, the history buttons.
function refresh() {
  const w = state.scenario.world;
  if ([w.origin_lat_deg, w.origin_lon_deg, w.origin_alt_hae_m].every(isNum) && Math.abs(w.origin_lat_deg) <= 90) {
    scene.setOrigin(w.origin_lat_deg, w.origin_lon_deg, w.origin_alt_hae_m);
    store.setScenario(w, state.scenario.aircraft, kindOf);
  }
  state.selection = state.selection.filter((id) => byId(id));
  scene.setSelection(state.selection);
  const issues = renderIssues();
  renderFleet(issues);
  $("scenarioJson").textContent = JSON.stringify(scenarioDoc(), null, 2);
  $("undoBtn").disabled = !state.undo.length;
  $("redoBtn").disabled = !state.redo.length;
  updatePreview();
}

// ---- Copy ------------------------------------------------------------------

async function copyJson() {
  const text = $("scenarioJson").textContent;
  let ok = false;
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      ok = true;
    }
  } catch { /* fall back below */ }
  if (!ok) {
    // Plain-http pages have no async clipboard: copy through a selection.
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.setAttribute("readonly", "");
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.append(ta);
    ta.select();
    ok = document.execCommand("copy");
    ta.remove();
  }
  $("copyStatus").textContent = ok ? "Copied" : "Copy failed — select the text instead";
  setTimeout(() => { $("copyStatus").textContent = ""; }, 2500);
}

// ---- View ------------------------------------------------------------------

function setView(mode) {
  scene.setCameraMode(mode);
  $("viewMapBtn").setAttribute("aria-pressed", String(mode === "map"));
  $("viewOrbitBtn").setAttribute("aria-pressed", String(mode === "orbit"));
}

// The scale bar: a round length 60–150 px long at the view's scale.
function renderScale() {
  const mpp = scene.viewInfo().metersPerPx;
  if (!(mpp > 0)) return;
  const target = 100 * mpp;
  const p = 10 ** Math.floor(Math.log10(target));
  const len = [1, 2, 5, 10].map((k) => k * p).find((v) => v / mpp >= 60) ?? 10 * p;
  $("scaleBar").style.width = `${Math.round(len / mpp)}px`;
  $("scaleLabel").textContent = len >= 1000 ? `${len / 1000} km` : `${len} m`;
}

let last = performance.now();
function frame(now) {
  const dt = Math.min(0.1, (now - last) / 1000);
  last = now;
  scene.update(store, 0, { showTrails: false, spinRotors: false, dt });
  scene.render();
  renderScale();
  syncSwatches();
  requestAnimationFrame(frame);
}

// Fleet-list swatches take the scene's colours, which exist once an aircraft
// has been drawn (a new one only after the frame that builds it).
function syncSwatches() {
  for (const sw of $("fleetList").querySelectorAll(".swatch")) {
    const color = scene.colorOf(Number(sw.parentElement.dataset.id));
    if (color && sw.dataset.color !== color) {
      sw.dataset.color = color;
      sw.style.background = color;
    }
  }
}

// ---- Wiring ----------------------------------------------------------------

function wire() {
  $("undoBtn").addEventListener("click", undo);
  $("redoBtn").addEventListener("click", redo);
  $("viewMapBtn").addEventListener("click", () => setView("map"));
  $("viewOrbitBtn").addEventListener("click", () => setView("orbit"));
  $("fitBtn").addEventListener("click", () => scene.frameFleet());
  $("dupBtn").addEventListener("click", duplicate);
  $("delBtn").addEventListener("click", removeSelected);
  $("copyBtn").addEventListener("click", copyJson);
  $("pickOriginBtn").addEventListener("click", () => setPicking(!state.pickingOrigin));
  $("worldLocation").addEventListener("change", (e) => {
    const l = state.env?.locations?.[Number(e.target.value)];
    if (e.target.value !== "" && l) moveOrigin(l.origin_lat_deg, l.origin_lon_deg, l.origin_alt_hae_m);
  });
  for (const input of insInputs) input.addEventListener("change", () => onInspectorChange(input));
  for (const input of worldInputs) input.addEventListener("change", () => onWorldChange(input));
  for (const input of simInputs) input.addEventListener("change", () => onSimChange(input));
  for (const f of ["insForm", "worldForm", "simForm"]) $(f).addEventListener("submit", (e) => e.preventDefault());

  // A press on the map commits the field being typed in first (its change
  // event), so a drag never races a stale inspector value.
  $("view").addEventListener("pointerdown", () => {
    const el = document.activeElement;
    if (el && el.closest?.("form")) el.blur();
  }, { capture: true });

  const list = $("fleetList");
  const pickRow = (e) => {
    const li = e.target.closest("li[data-id]");
    if (!li) return;
    const id = Number(li.dataset.id);
    onSelect([id], e.shiftKey ? "toggle" : "replace");
    list.querySelector(`li[data-id="${id}"]`)?.focus();
  };
  list.addEventListener("click", pickRow);
  list.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pickRow(e); }
  });

  const form = $("spawnForm");
  form.addEventListener("submit", (e) => { e.preventDefault(); spawn(); });
  form.addEventListener("input", updatePreview);
  $("spawnModel").addEventListener("change", () => {
    const m = modelEntry($("spawnModel").value);
    $("spawnTas").value = String(SPAWN_TAS[m?.model] ?? SPAWN_TAS_KIND[m?.kind] ?? 0);
    updatePreview();
  });
  $("spawnPattern").addEventListener("change", () => {
    $("spawnSpacingLabel").textContent = $("spawnPattern").value === "random" ? "Box side (m)" : "Spacing (m)";
  });
  const section = form.closest("section");
  const preview = (on) => { state.previewSpawn = on || section.contains(document.activeElement); updatePreview(); };
  section.addEventListener("pointerenter", () => preview(true));
  section.addEventListener("pointerleave", () => preview(false));
  section.addEventListener("focusin", () => preview(true));
  section.addEventListener("focusout", () => setTimeout(() => preview(section.matches(":hover")), 0));

  document.addEventListener("keydown", (e) => {
    const typing = e.target.closest?.("input, select, textarea");
    const mod = e.ctrlKey || e.metaKey;
    if (e.key === "Escape") {
      if (state.pickingOrigin) setPicking(false);
      else if (!typing) { scene.cancelEdit(); onSelect([], "replace"); }
      return;
    }
    if (typing) return;
    const k = e.key.toLowerCase();
    if (mod && k === "z") { e.preventDefault(); if (e.shiftKey) redo(); else undo(); }
    else if (mod && k === "y") { e.preventDefault(); redo(); }
    else if (mod && k === "d") { e.preventDefault(); duplicate(); }
    else if (mod && k === "a") { e.preventDefault(); onSelect(state.scenario.aircraft.map((a) => a.id), "replace"); }
    else if (e.key === "Delete" || e.key === "Backspace") { e.preventDefault(); removeSelected(); }
    else if (!mod && k === "m") setView("map");
    else if (!mod && k === "o") setView("orbit");
    else if (!mod && k === "f") scene.frameFleet();
  });
}

// ---- Load ------------------------------------------------------------------

function showLoadError(msg) {
  $("loadError").hidden = false;
  $("loadError").textContent = msg;
}

function newScenario(env) {
  const loc = env?.locations?.[0];
  return {
    world: loc
      ? { origin_lat_deg: loc.origin_lat_deg, origin_lon_deg: loc.origin_lon_deg, origin_alt_hae_m: loc.origin_alt_hae_m }
      : { origin_lat_deg: 0, origin_lon_deg: 0, origin_alt_hae_m: 0 },
    sim: { ...(env?.sim ?? { rate_hz: 120, max_time_s: 40, speed: 1, seed: 0, threads: 1, command_lease_s: 0.5, lockstep: false, lockstep_timeout_s: 30 }) },
    aircraft: [],
  };
}

async function load() {
  wire();
  const name = new URLSearchParams(location.search).get("scenario");
  let envs = [];
  try {
    [state.models, envs] = await Promise.all([getModels(), getEnvironments()]);
  } catch (err) {
    showLoadError(`Catalogs unavailable: ${err.message}`);
  }
  state.env = envs.find((e) => e.id === DEFAULT_ENVIRONMENT) ?? envs[0] ?? null;
  if (name) {
    try {
      const doc = await getScenario(name);
      state.scenario = structuredClone(doc.scenario);
      state.env = envs.find((e) => e.id === doc.environment) ?? state.env;
      $("sourceLabel").textContent = doc.path;
      document.title = `ASTRUM · Sandbox · ${doc.name}`;
    } catch (err) {
      showLoadError(`Scenario "${name}": ${err.message} — starting a new scenario instead.`);
    }
  }
  if (!state.scenario) state.scenario = newScenario(state.env);
  fillModelSelect($("spawnModel"), null);
  $("spawnModel").dispatchEvent(new Event("change"));
  refresh();
  scene.update(store, 0, { showTrails: false, spinRotors: false });
  scene.frameFleet();
  requestAnimationFrame(frame);
}

load();
