// Entry point: wire the timeline store, the 3-D scene, the telemetry sources and
// the HUD (shell, fleet list, inspector, event log, plots) together, then run
// the animation / playback loop.
//
// main.js is the single source of truth for playback and selection state. The
// store buffers telemetry keyed on sim time; the scene is a pure function of
// (store, playhead) each frame; the HUD modules only emit gestures and display
// what we push to them.

import { Store } from "./store.js";
import { Scene } from "./scene.js";
import { Net, defaultWsUrl, loadExportFile } from "./net.js";
import { UI } from "./ui.js";
import { FleetList } from "./fleet.js";
import { Inspector } from "./inspector.js";
import { EventLog, EVENT_GROUPS } from "./eventlog.js";
import { Plots, CHANNELS } from "./plots.js";
import { relativeGeometry } from "./geometry.js";
import { loadPrefs, savePrefs } from "./prefs.js";

// A playhead move longer than this (sim s) in one frame is a seek: the events
// it skips over are not flashed.
const FLASH_SEEK_MAX_S = 3;
// The fleet rows' status (HP, mode, alive) refreshes at this period (ms);
// selection changes show at once.
const FLEET_REFRESH_MS = 100;
const DEFAULT_CHANNELS = ["alt", "tas", "nz"];

const prefs = loadPrefs();
const bool = (v, dflt) => (typeof v === "boolean" ? v : dflt);
const strings = (v, allowed) =>
  (Array.isArray(v) ? v.filter((k) => typeof k === "string" && allowed.has(k)) : null);

const store = new Store();

const state = {
  mode: "idle",      // "idle" | "live" | "file"
  connected: false,  // the live socket is open
  status: "idle",    // last source status (see UI STATUS_TEXT)
  wsUrl: "",
  fileError: "",
  playing: false,
  speed: 1,
  playhead: 0,       // sim time (s)
  followLive: false, // in live mode, stick to the newest sample
  selectedId: null,
  targetId: null,    // pinned target; never the selection
  trails: bool(prefs.trails, true),
  labels: bool(prefs.labels, true),
  dropLines: bool(prefs.dropLines, true),
  modelScale: Number.isInteger(prefs.modelScale) && prefs.modelScale >= 1 && prefs.modelScale <= 20
    ? prefs.modelScale : 4,
  needsFrame: false, // frame + auto-select once the first aircraft appears
  prevPlayhead: NaN, // last frame's playhead; events crossed since get flashed
  fleetDirty: true,
  models: new Map(), // aircraft id -> JSBSim model (scenario), when known
};

// ---- colours ----

// The scene owns each aircraft's colour (it follows the team and the
// environment), so the swatches, plots and kill feed match the 3-D view.
const UNKNOWN_COLOR = "#7d8da0";
function colorOf(id) {
  return scene.colorOf(id) ?? UNKNOWN_COLOR;
}

// ---- modules ----

const scene = new Scene(document.getElementById("view"), {
  onSelect: (id) => selectAircraft(id),
  onTarget: (id) => toggleTarget(id),
  onCameraMode: (mode) => ui.setCameraMode(mode),
});

const ui = new UI({
  onPlayToggle: () => {
    if (store.bounds().empty) return;
    state.playing = !state.playing;
    if (state.playing && atLiveHead()) state.followLive = liveAvailable();
    if (!state.playing) state.followLive = false;
  },
  onScrub: (frac) => {
    const b = store.bounds();
    seekTo(b.min + frac * b.span);
  },
  onStep: (dt) => seekTo(state.playhead + dt),
  onSeekEdge: (edge) => {
    const b = store.bounds();
    if (edge === "end" && liveAvailable()) goLive();
    else seekTo(edge === "start" ? b.min : b.max);
  },
  onLive: () => goLive(),
  onSpeed: (v) => { state.speed = v; },
  onConnect: (url) => connect(url || defaultWsUrl()),
  onRetry: () => connect(state.wsUrl || defaultWsUrl()),
  onFile: (file) => openFile(file),
  onCamera: (mode) => { scene.setCameraMode(mode); ui.setCameraMode(scene.cameraMode); },
  onFit: () => { scene.frameFleet(); ui.setCameraMode(scene.cameraMode); },
  onToggle: (key, val) => {
    state[key] = val;
    savePrefs({ [key]: val });
  },
  onModelScale: (k) => {
    state.modelScale = k;
    scene.setModelScale(k);
    savePrefs({ modelScale: k });
  },
  onSelectStep: (dir) => {
    const order = fleet.order();
    if (!order.length) return;
    const i = order.indexOf(state.selectedId);
    const next = i < 0 ? (dir > 0 ? 0 : order.length - 1) : (i + dir + order.length) % order.length;
    selectAircraft(order[next]);
  },
  // T steps the target through the other aircraft, then "none".
  onCycleTarget: (dir) => {
    const others = fleet.order().filter((id) => id !== state.selectedId);
    if (!others.length) return;
    const seq = [null, ...others];
    const i = seq.indexOf(state.targetId);
    setTarget(seq[(Math.max(0, i) + dir + seq.length) % seq.length]);
  },
  onLayout: () => plots.setVisible(ui.plotsVisible),
}, prefs);

const fleet = new FleetList({
  list: document.getElementById("fleetList"),
  count: document.getElementById("fleetCount"),
  empty: document.getElementById("fleetEmpty"),
  searchWrap: document.getElementById("fleetSearchWrap"),
  search: document.getElementById("fleetSearch"),
  onSelect: (id) => selectAircraft(id),
  onTarget: (id) => toggleTarget(id),
});

const inspector = new Inspector({
  onClearTarget: () => setTarget(null),
  onSwap: () => { if (state.targetId != null) selectAircraft(state.targetId); },
});

const eventLog = new EventLog({
  list: document.getElementById("evList"),
  empty: document.getElementById("evEmpty"),
  count: document.getElementById("evCount"),
  filters: document.getElementById("evFilters"),
  mineBtn: document.getElementById("evMineBtn"),
  kinds: strings(prefs.eventKinds, new Set(EVENT_GROUPS.map((g) => g.key))),
  onSeek: (t) => seekTo(t),
  onKindsChange: (kinds) => savePrefs({ eventKinds: kinds }),
});

const plots = new Plots({
  toolbar: document.getElementById("plotsToolbar"),
  body: document.getElementById("plotsBody"),
  channels: strings(prefs.channels, new Set(CHANNELS.map((c) => c.key))) ?? DEFAULT_CHANNELS,
  onSeek: (t) => seekTo(t),
  onChannelsChange: (keys) => savePrefs({ channels: keys }),
});
plots.setVisible(ui.plotsVisible);

ui.setToggles(state);
ui.setModelScale(state.modelScale);
scene.setModelScale(state.modelScale);
ui.setCameraMode(scene.cameraMode);

const net = new Net({
  onFrames: (frames) => {
    if (!store.appendLive(frames)) {
      // An aircraft's clock went backwards: a new run started upstream.
      startLiveRun();
      store.appendLive(frames);
    }
    if (state.followLive) state.playhead = store.bounds().max;
  },
  onEvents: (events) => {
    let from = store.events.length;
    if (!store.appendEvents(events)) {
      // An event went back in time: a new run started upstream.
      startLiveRun();
      store.appendEvents(events);
      from = 0;
    }
    // Following the head, an event at or before the last rendered playhead will
    // never be crossed by it: announce it now. Later ones are announced when the
    // next frame crosses them (announcing both ways would show them twice).
    if (state.followLive) {
      const seen = Number.isFinite(state.prevPlayhead) ? state.prevPlayhead : state.playhead;
      announce(store.events.slice(from).filter((e) => e.t <= seen), true);
    }
  },
  onModels: (models) => { state.models = models; },
  onStatus: (st, label) => {
    state.status = st;
    ui.setStatus(st, label);
    state.connected = st === "live";
    if (st === "closed" || st === "error") {
      // Keep what was buffered, paused, for scrubbing.
      state.followLive = false;
      state.playing = false;
    }
  },
});

// ---- source switching ----

function resetSelection() {
  state.selectedId = null;
  state.targetId = null;
  scene.setSelected(null);
  scene.setTarget(null);
  state.needsFrame = true;
  state.prevPlayhead = NaN;
  state.fleetDirty = true;
  ui.clearKillFeed();
}

// Empty the timeline and the scene for a fresh live run.
function startLiveRun() {
  store.reset("live");
  scene.clearFleet();
  state.playhead = 0;
  resetSelection();
}

function connect(url) {
  startLiveRun();
  state.models = new Map(); // the bridge sends the scenario's models on connect
  state.mode = "live";
  state.wsUrl = url;
  state.fileError = "";
  state.followLive = true;
  state.playing = true;
  ui.setWsUrl(url);
  net.connect(url);
}

async function openFile(file) {
  net.disconnect();
  state.connected = false;
  try {
    const { meta, frames, events, models } = await loadExportFile(file);
    store.loadFile(frames, events);
    state.models = models;
    scene.clearFleet();
    const b = store.bounds();
    state.mode = "file";
    state.fileError = "";
    state.followLive = false;
    state.playing = true;
    state.playhead = b.min;
    resetSelection();
    state.status = "file";
    ui.setStatus("file", `${file.name} · ${store.frameCount} frames`
      + ` · ${store.ids().length} aircraft`
      + (store.events.length ? ` · ${store.events.length} events` : "")
      + (meta?.rejected ? ` · ${Number(meta.rejected)} rejected` : ""));
  } catch (e) {
    state.fileError = String(e.message || e);
    state.status = "error";
    ui.setStatus("error", state.fileError);
  }
}

// ---- selection & target ----

function selectAircraft(id) {
  if (id === state.selectedId) return;
  const prev = state.selectedId;
  state.selectedId = id;
  scene.setSelected(id);
  // Selecting the target swaps the pair, keeping the engagement in view.
  if (id === state.targetId) setTarget(prev);
  state.fleetDirty = true;
}

function setTarget(id) {
  if (id === state.selectedId) id = null;
  state.targetId = id;
  scene.setTarget(id);
  state.fleetDirty = true;
}

function toggleTarget(id) {
  if (id === state.selectedId) return;
  setTarget(id === state.targetId ? null : id);
}

// First frame of a source: select the first aircraft and, in a fight, pin the
// nearest aircraft of another team as its target.
function autoSelect() {
  const ids = store.ids();
  if (state.selectedId == null) selectAircraft(ids[0]);
  if (state.targetId != null) return;
  const me = store.sampleAt(state.selectedId, state.playhead)?.s;
  if (!me) return;
  let best = null, bestRange = Infinity;
  for (const id of ids) {
    const s = store.sampleAt(id, state.playhead)?.s;
    if (id === state.selectedId || !s || s.team === me.team) continue;
    const { range } = relativeGeometry(me, s);
    if (range < bestRange) { best = id; bestRange = range; }
  }
  if (best != null) setTarget(best);
}

// ---- playback ----

function liveAvailable() {
  return state.mode === "live" && state.connected;
}

function atLiveHead() {
  const b = store.bounds();
  return b.empty || state.playhead >= b.max - 0.25;
}

function seekTo(t) {
  const b = store.bounds();
  if (b.empty) return;
  state.playhead = Math.max(b.min, Math.min(b.max, t));
  state.followLive = false;
  scene.resetTrails(); // a seek: don't draw a chord to the new position
}

function goLive() {
  if (!liveAvailable()) return;
  if (!state.followLive) scene.resetTrails(); // jumping to the head is a seek
  state.followLive = true;
  state.playing = true;
}

// ---- events: scene flashes + the presentation kill feed ----

function announce(events, forward) {
  if (!events.length) return;
  scene.flash(events);
  if (!forward) return;
  for (const e of events) {
    if (e.kind !== "kill") continue;
    ui.pushKill({
      killer: e.killer_id, victim: e.victim_id ?? "?",
      killerColor: e.killer_id != null ? colorOf(e.killer_id) : null,
      victimColor: e.victim_id != null ? colorOf(e.victim_id) : null,
    });
  }
}

// Announce the events the playhead crossed since the last frame (either way,
// so a scrub flashes too; only forward play feeds the kill feed); a long jump
// is a seek and announces nothing.
function announceCrossedEvents() {
  const prev = state.prevPlayhead, cur = state.playhead;
  state.prevPlayhead = cur;
  if (!Number.isFinite(prev) || Math.abs(cur - prev) > FLASH_SEEK_MAX_S) return;
  announce(store.eventsBetween(Math.min(prev, cur), Math.max(prev, cur)), cur > prev);
}

// ---- timeline markers (rebuilt only when events/bounds change) ----

const MARKER_KINDS = new Set(["hit", "kill"]);
let markerKey = null;

function markerTitle(e) {
  const at = `at ${e.t.toFixed(1)} s`;
  if (e.kind === "kill") return `kill: ${e.killer_id ?? "?"} → ${e.victim_id ?? "?"} ${at}`;
  return `hit: ${e.shooter_id ?? "?"} → ${e.target_id ?? "?"} ${at} (${e.damage} dmg, ${e.hit_points} hp left)`;
}

function updateMarkers(b) {
  const events = store.events;
  const key = [events, events.length, b.min, b.max];
  if (markerKey && key.every((v, i) => v === markerKey[i])) return;
  markerKey = key;
  const markers = [];
  for (const e of events) {
    if (!MARKER_KINDS.has(e.kind)) continue;
    markers.push({ frac: b.span > 0 ? (e.t - b.min) / b.span : 0, kind: e.kind, title: markerTitle(e) });
  }
  ui.setMarkers(markers);
}

// ---- environment: daytime sky + streamed ground map, only for F-16 fleets ----

// JSBSim F-16 models (f16 and its variants, e.g. f16kp10, f16nopid).
const F16_MODEL = /^f16/i;
let envModels = null, envIdCount = -1;

// Re-choose when the models or the set of aircraft change.
function updateEnvironment() {
  const ids = store.ids();
  if (state.models === envModels && ids.length === envIdCount) return;
  envModels = state.models;
  envIdCount = ids.length;
  const sky = ids.some((id) => F16_MODEL.test(state.models.get(id) ?? ""));
  scene.setEnvironment(sky ? "sky" : "plain");
  ui.setMapCredit(sky);
}

// ---- HUD ----

function emptyText() {
  if (state.mode === "file" || state.fileError) return state.fileError || "The recording has no telemetry.";
  const url = state.wsUrl || defaultWsUrl();
  if (state.status === "connecting") return `Connecting to ${url}…`;
  if (state.connected) return `Connected to ${url}. Waiting for the first frame.`;
  return `No live stream at ${url}. Start one with astrum run <scenario> --watch, or open a recording.`;
}

let lastFleetAt = -Infinity;

function updateFleet(now) {
  if (!state.fleetDirty && now - lastFleetAt < FLEET_REFRESH_MS) return;
  state.fleetDirty = false;
  lastFleetAt = now;
  const items = [];
  for (const id of store.ids()) {
    const res = store.sampleAt(id, state.playhead);
    if (!res) continue;
    const { s, inRange } = res;
    items.push({
      id, kind: store.track(id).kind, color: colorOf(id), team: s.team, alive: s.alive, inRange,
      hp: s.hit_points, hpMax: s.hit_points_max, mode: s.control_mode,
    });
  }
  fleet.update(items, state.selectedId, state.targetId);
}

function aircraftAt(id) {
  if (id == null) return null;
  const res = store.sampleAt(id, state.playhead);
  if (!res) return null;
  return { id, kind: store.track(id).kind, color: colorOf(id), s: res.s, inRange: res.inRange };
}

// ---- main loop ----

let last = performance.now();
let fpsEma = 0;

function tick(now) {
  const dt = Math.min(0.1, (now - last) / 1000);
  last = now;

  const b = store.bounds();
  ui.setEmpty(b.empty, emptyText(), state.mode !== "file" && !state.connected && state.status !== "connecting");

  if (!b.empty) {
    if (state.followLive) {
      state.playhead = b.max;
    } else if (state.playing) {
      state.playhead += dt * state.speed;
      if (state.playhead >= b.max) {
        state.playhead = b.max;
        if (liveAvailable()) state.followLive = true;
        else state.playing = false; // stop at the end of a file / a closed stream
      }
    }
    if (state.playhead < b.min) state.playhead = b.min;
    announceCrossedEvents();
  }

  updateEnvironment();
  scene.update(store, state.playhead, {
    showTrails: state.trails,
    showLabels: state.labels,
    showDropLines: state.dropLines,
    spinRotors: state.playing || state.followLive,
    spinRate: state.speed,
    dt,
  });

  // Once the first aircraft is on screen, select it and frame the fleet where
  // it actually is (aircraft fly at their HAE altitude, not on the grid).
  if (state.needsFrame && store.ids().length) {
    autoSelect();
    const mode = scene.cameraMode;
    if (mode === "orbit" || mode === "follow") {
      scene.frameFleet();
      if (mode === "follow") scene.setCameraMode("follow");
      ui.setCameraMode(scene.cameraMode);
    }
    state.needsFrame = false;
  }

  scene.render();
  ui.setView(scene.viewInfo());

  // HUD
  const frac = b.span > 0 ? (state.playhead - b.min) / b.span : 0;
  ui.setTime(state.playhead - b.min, b.span, frac);
  ui.setTransport({
    hasData: !b.empty,
    playing: state.playing || state.followLive,
    liveAvailable: liveAvailable(),
    liveActive: state.followLive,
  });
  updateMarkers(b);
  updateFleet(now);
  inspector.update(aircraftAt(state.selectedId), aircraftAt(state.targetId));
  eventLog.update(store.events, {
    playhead: state.playhead, t0: b.min, selectedId: state.selectedId,
    playing: state.playing || state.followLive,
  });
  plots.update(store, {
    playhead: state.playhead, bounds: b,
    selectedId: state.selectedId, targetId: state.targetId, colorOf,
  });
  ui.tickKillFeed(now);

  const inst = dt > 0 ? 1 / dt : 0;
  fpsEma = fpsEma ? fpsEma * 0.9 + inst * 0.1 : inst;
  ui.setFps(fpsEma);

  requestAnimationFrame(tick);
}

// ---- boot ----

const initialUrl = defaultWsUrl();
ui.setWsUrl(initialUrl);
connect(initialUrl); // auto-attach to the local bridge; the user can open a file instead
requestAnimationFrame(tick);
