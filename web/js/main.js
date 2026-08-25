// Entry point: wire the timeline store, the 3-D scene, the telemetry sources and
// the HUD together, then run the animation / playback loop.
//
// main.js is the single source of truth for playback state. The store buffers
// telemetry keyed on sim time; the scene is a pure function of (store, playhead)
// each frame; the UI only emits gestures and displays what we push to it.

import { Store } from "./store.js";
import { Scene } from "./scene.js";
import { Net, defaultWsUrl, loadExportFile } from "./net.js";
import { UI } from "./ui.js";
import { aircraftColor } from "./meshes.js";

const RAD2DEG = 180 / Math.PI;

const store = new Store();

const state = {
  mode: "idle",     // "idle" | "live" | "file"
  playing: false,
  speed: 1,
  playhead: 0,      // sim time (s)
  followLive: false, // in live mode, stick to the newest sample
  selectedId: null,
  follow: false,    // chase camera on the selection
  trails: true,
  labels: true,
  needsFrame: false, // frame + auto-select once the first aircraft appears
};

const scene = new Scene(document.getElementById("view"), {
  onSelect: (id) => selectAircraft(id),
});

// The idle splash logo; hidden as soon as any telemetry is buffered.
const splashEl = document.getElementById("splash");

const ui = new UI({
  onPlayToggle: () => {
    state.playing = !state.playing;
    if (state.playing && atLiveHead()) state.followLive = state.mode === "live";
    if (!state.playing) state.followLive = false;
    ui.setPlaying(state.playing);
    ui.setLiveActive(state.followLive);
  },
  onScrub: (frac) => {
    const b = store.bounds();
    state.playhead = b.min + frac * b.span;
    state.followLive = false;
    ui.setLiveActive(false);
  },
  onLive: () => {
    if (state.mode !== "live") return;
    state.followLive = true;
    state.playing = true;
    ui.setPlaying(true);
    ui.setLiveActive(true);
  },
  onSpeed: (v) => { state.speed = v; },
  onConnect: (url) => connect(url || defaultWsUrl()),
  onFile: (file) => openFile(file),
  onSelect: (id) => selectAircraft(id),
  onFit: () => scene.frameFleet(),
  onToggle: (key, val) => {
    if (key === "follow") state.follow = val;
    else if (key === "trails") state.trails = val;
    else if (key === "labels") state.labels = val;
  },
});

const net = new Net({
  onFrame: (f) => {
    store.addFrame(f);
    if (state.followLive) state.playhead = store.bounds().max;
  },
  onStatus: (st, label) => {
    ui.setStatus(st, label);
    if (st === "live") { state.mode = "live"; ui.setMode("live"); }
  },
});

// ---- source switching ----

function connect(url) {
  store.reset("live");
  scene.clearFleet();
  state.mode = "live";
  state.followLive = true;
  state.playing = true;
  state.playhead = 0;
  state.selectedId = null;
  scene.setSelected(null);
  state.needsFrame = true;
  ui.setMode("live");
  ui.setPlaying(true);
  ui.setLiveActive(true);
  ui.setWsUrl(url);
  net.connect(url);
}

async function openFile(file) {
  net.disconnect();
  try {
    const { frames, meta } = await loadExportFile(file);
    store.reset("file");
    scene.clearFleet();
    for (const f of frames) store.addFrame(f);
    const b = store.bounds();
    state.mode = "file";
    state.followLive = false;
    state.playing = true;
    state.playhead = b.min;
    state.selectedId = null;
    scene.setSelected(null);
    state.needsFrame = true;
    ui.setMode("file");
    ui.setPlaying(true);
    ui.setLiveActive(false);
    ui.setStatus("file", `${file.name} · ${frames.length} frames`
      + (meta?.aircraft ? ` · ${meta.aircraft.length} aircraft` : ""));
  } catch (e) {
    ui.setStatus("error", String(e.message || e));
  }
}

function selectAircraft(id) {
  state.selectedId = id;
  scene.setSelected(id);
}

function atLiveHead() {
  const b = store.bounds();
  return b.empty || state.playhead >= b.max - 0.25;
}

// ---- readout for the selected aircraft ----

function updateReadout() {
  const id = state.selectedId;
  if (id == null) { ui.setReadout(null); return; }
  const track = store.track(id);
  const res = track ? store.sampleAt(id, state.playhead) : null;
  if (!res) { ui.setReadout(null); return; }
  const s = res.s;
  const gs = Math.hypot(s.vel_north_mps, s.vel_east_mps);
  let hdg = (s.psi_rad * RAD2DEG) % 360; if (hdg < 0) hdg += 360;
  ui.setReadout({
    id, kind: track.kind, t: s.t,
    lat_deg: s.lat_deg, lon_deg: s.lon_deg, alt_m: s.alt_m,
    enu_e: s.enu_e, enu_n: s.enu_n, enu_u: s.enu_u,
    heading_deg: hdg, pitch_deg: s.theta_rad * RAD2DEG, roll_deg: s.phi_rad * RAD2DEG,
    airspeed_mps: s.airspeed_mps, ground_speed_mps: gs,
    vertical_speed_mps: -s.vel_down_mps,
  });
}

// ---- fleet list (rebuilt by UI only when it actually changes) ----

function updateFleetList() {
  const items = store.ids().map((id) => {
    const kind = store.track(id).kind;
    return { id, kind, color: `#${aircraftColor(kind, id).getHexString()}` };
  });
  ui.setFleet(items, state.selectedId);
}

// ---- main loop ----

let last = performance.now();
let fpsEma = 0;

function tick(now) {
  const dt = Math.min(0.1, (now - last) / 1000);
  last = now;

  const b = store.bounds();
  splashEl.classList.toggle("hidden", !b.empty);

  if (!b.empty) {
    if (state.followLive) {
      state.playhead = b.max;
    } else if (state.playing) {
      state.playhead += dt * state.speed;
      if (state.playhead >= b.max) {
        state.playhead = b.max;
        if (state.mode === "live") { state.followLive = true; ui.setLiveActive(true); }
        else { state.playing = false; ui.setPlaying(false); } // stop at end of file
      }
    }
    if (state.playhead < b.min) state.playhead = b.min;
  }

  scene.update(store, state.playhead, {
    showTrails: state.trails,
    showLabels: state.labels,
    spinRotors: state.playing || state.followLive,
    spinRate: state.speed,
    dt,
  });

  // Once the first aircraft is on screen, select it and frame the fleet where it
  // actually is (aircraft sit at their MSL altitude, not on the grid).
  if (state.needsFrame && !b.empty) {
    const ids = store.ids();
    if (ids.length) {
      if (state.selectedId == null) selectAircraft(ids[0]);
      scene.frameFleet();
      state.needsFrame = false;
    }
  }

  if (state.follow) scene.follow();
  scene.render();

  // HUD
  const frac = b.span > 0 ? (state.playhead - b.min) / b.span : 0;
  ui.setTime(state.playhead - b.min, b.span, frac);
  updateFleetList();
  updateReadout();

  const inst = dt > 0 ? 1 / dt : 0;
  fpsEma = fpsEma ? fpsEma * 0.9 + inst * 0.1 : inst;
  ui.setFps(fpsEma);

  requestAnimationFrame(tick);
}

// ---- boot ----

const initialUrl = defaultWsUrl();
ui.setWsUrl(initialUrl);
ui.setStatus("idle", "connecting to live stream…");
ui.setMode("idle");
ui.setPlaying(false);
connect(initialUrl); // auto-attach to the local bridge; user can open a file instead
requestAnimationFrame(tick);
