// The HUD controller: owns every DOM element in index.html, turns user gestures
// (play, scrub, speed, connect, open-file, select, toggles, drag-drop) into
// callbacks, and exposes setters the main loop calls to reflect playback state.
//
// It holds no timeline or scene state of its own — main.js is the single source
// of truth and pushes values down through the set*() methods.

const $ = (id) => document.getElementById(id);

// Seconds -> "M:SS.s".
function fmtTime(sec) {
  if (!isFinite(sec) || sec < 0) sec = 0;
  const m = Math.floor(sec / 60);
  const s = sec - m * 60;
  return `${m}:${s.toFixed(1).padStart(4, "0")}`;
}

const KIND_GLYPH = { "fixed-wing": "✈", multirotor: "✜" };
const STATUS_TEXT = {
  idle: "idle", connecting: "connecting…", live: "live",
  file: "file", closed: "disconnected", error: "error",
};

export class UI {
  constructor(handlers = {}) {
    this.h = handlers;
    this.selectedId = null;
    this._fleetSig = "";

    this.el = {
      statusPill: $("statusPill"), sourceLabel: $("sourceLabel"),
      wsUrl: $("wsUrl"), connectBtn: $("connectBtn"),
      openFileBtn: $("openFileBtn"), fileInput: $("fileInput"),
      fleetCount: $("fleetCount"), fleetList: $("fleetList"),
      readout: $("readout"),
      playBtn: $("playBtn"), liveBtn: $("liveBtn"),
      timeCur: $("timeCur"), scrub: $("scrub"), timeTot: $("timeTot"),
      speedSel: $("speedSel"), fps: $("fps"),
      tglFollow: $("tglFollow"), tglTrails: $("tglTrails"),
      tglLabels: $("tglLabels"), fitBtn: $("fitBtn"),
      dropHint: $("dropHint"),
    };
    // Cache the readout value spans (ro-id, ro-lat, …) once.
    this.ro = {};
    for (const k of ["id", "kind", "mode", "time", "lat", "lon", "alt", "e", "n", "u",
      "hdg", "pitch", "roll", "tas", "gs", "vs", "mach", "aoa", "nz", "team", "hp", "ammo"]) {
      this.ro[k] = $(`ro-${k}`);
    }

    this._wire();
  }

  _wire() {
    const h = this.h;
    this.el.playBtn.addEventListener("click", () => h.onPlayToggle?.());
    this.el.liveBtn.addEventListener("click", () => h.onLive?.());
    this.el.fitBtn.addEventListener("click", () => h.onFit?.());

    this.el.scrub.addEventListener("input", () => {
      h.onScrub?.(Number(this.el.scrub.value) / 1000);
    });
    this.el.speedSel.addEventListener("change", () => {
      h.onSpeed?.(Number(this.el.speedSel.value));
    });

    this.el.connectBtn.addEventListener("click", () => {
      h.onConnect?.(this.el.wsUrl.value.trim());
    });
    this.el.wsUrl.addEventListener("keydown", (e) => {
      if (e.key === "Enter") h.onConnect?.(this.el.wsUrl.value.trim());
    });

    this.el.openFileBtn.addEventListener("click", () => this.el.fileInput.click());
    this.el.fileInput.addEventListener("change", () => {
      const f = this.el.fileInput.files?.[0];
      if (f) h.onFile?.(f);
      this.el.fileInput.value = ""; // allow re-opening the same file
    });

    const tgl = (elName, key) =>
      this.el[elName].addEventListener("change", () =>
        h.onToggle?.(key, this.el[elName].checked));
    tgl("tglFollow", "follow");
    tgl("tglTrails", "trails");
    tgl("tglLabels", "labels");

    // Space toggles play/pause unless typing in a field.
    window.addEventListener("keydown", (e) => {
      if (e.code === "Space" && !/^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)) {
        e.preventDefault();
        h.onPlayToggle?.();
      }
    });

    // Drag-and-drop an export JSON anywhere on the window.
    let depth = 0;
    window.addEventListener("dragenter", (e) => { e.preventDefault(); if (depth++ === 0) this.el.dropHint.classList.add("show"); });
    window.addEventListener("dragover", (e) => e.preventDefault());
    window.addEventListener("dragleave", (e) => { e.preventDefault(); if (--depth <= 0) { depth = 0; this.el.dropHint.classList.remove("show"); } });
    window.addEventListener("drop", (e) => {
      e.preventDefault();
      depth = 0;
      this.el.dropHint.classList.remove("show");
      const f = e.dataTransfer?.files?.[0];
      if (f) h.onFile?.(f);
    });
  }

  // ---- setters driven by main.js ----

  setStatus(state, label) {
    const pill = this.el.statusPill;
    pill.className = `pill ${state}`;
    pill.textContent = STATUS_TEXT[state] || state;
    if (label != null) this.el.sourceLabel.textContent = label;
  }

  setMode(mode) {
    // The LIVE button only makes sense on a live stream.
    this.el.liveBtn.disabled = mode !== "live";
  }

  setPlaying(playing) {
    this.el.playBtn.textContent = playing ? "❚❚" : "▶";
    this.el.playBtn.title = playing ? "Pause (space)" : "Play (space)";
  }

  setLiveActive(active) {
    this.el.liveBtn.classList.toggle("active", !!active);
  }

  setSpeed(speed) {
    this.el.speedSel.value = String(speed);
  }

  // cur/tot are sim-time seconds; frac in [0,1] positions the scrubber.
  setTime(cur, tot, frac) {
    this.el.timeCur.textContent = fmtTime(cur);
    this.el.timeTot.textContent = fmtTime(tot);
    // Don't fight the user mid-drag; setting .value here won't fire "input".
    this.el.scrub.value = String(Math.round(Math.max(0, Math.min(1, frac)) * 1000));
  }

  setFps(fps) {
    this.el.fps.textContent = isFinite(fps) ? `${Math.round(fps)} fps` : "—";
  }

  // items: [{ id, kind, color }]. Rebuilds only when the fleet/selection changes.
  // Frame data only ever reaches the DOM through textContent.
  setFleet(items, selectedId) {
    this.selectedId = selectedId;
    this.el.fleetCount.textContent = String(items.length);
    const sig = items.map((i) => `${i.id}:${i.kind}`).join(",") + `|${selectedId}`;
    if (sig === this._fleetSig) return;
    this._fleetSig = sig;

    const list = this.el.fleetList;
    list.textContent = "";
    const span = (cls, text) => {
      const el = document.createElement("span");
      el.className = cls;
      if (text != null) el.textContent = text;
      return el;
    };
    for (const it of items) {
      const li = document.createElement("li");
      if (it.id === selectedId) li.className = "sel";
      const swatch = span("swatch");
      swatch.style.background = it.color;
      li.append(swatch, span("fid", String(it.id)),
        span("glyph", KIND_GLYPH[it.kind] || "•"), span("fmeta", it.kind));
      li.addEventListener("click", () => this.h.onSelect?.(it.id));
      list.appendChild(li);
    }
  }

  // data: null clears/hides the panel; otherwise fields fill the readout.
  setReadout(data) {
    if (!data) { this.el.readout.classList.add("hidden"); return; }
    this.el.readout.classList.remove("hidden");
    const r = this.ro;
    r.id.textContent = String(data.id);
    r.kind.textContent = data.kind;
    r.mode.textContent = data.control_mode;
    r.mode.className = `mode-tag ${data.control_mode}`;
    r.time.textContent = data.t.toFixed(2);
    r.lat.textContent = data.lat_deg.toFixed(6);
    r.lon.textContent = data.lon_deg.toFixed(6);
    r.alt.textContent = data.alt_hae_m.toFixed(1);
    r.e.textContent = data.enu_e.toFixed(1);
    r.n.textContent = data.enu_n.toFixed(1);
    r.u.textContent = data.enu_u.toFixed(1);
    r.hdg.textContent = data.heading_deg.toFixed(1);
    r.pitch.textContent = data.pitch_deg.toFixed(1);
    r.roll.textContent = data.roll_deg.toFixed(1);
    r.tas.textContent = data.tas_mps.toFixed(1);
    r.gs.textContent = data.ground_speed_mps.toFixed(1);
    r.vs.textContent = data.vertical_speed_mps.toFixed(1);
    r.mach.textContent = data.mach.toFixed(2);
    r.aoa.textContent = data.alpha_deg.toFixed(1);
    r.nz.textContent = data.nz_g.toFixed(2);
    r.team.textContent = String(data.team);
    r.hp.textContent = data.alive
      ? `${data.hit_points.toFixed(0)} / ${data.hit_points_max.toFixed(0)}`
      : "destroyed";
    r.ammo.textContent = String(data.ammo);
  }

  setWsUrl(url) {
    if (!this.el.wsUrl.value) this.el.wsUrl.value = url;
  }
}
