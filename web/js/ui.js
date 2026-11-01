// The HUD shell: owns the top bar (status, source and help menus, dock and
// presentation toggles), the transport, the scene overlays (camera toolbar,
// compass, scale bar, kill feed, fps), the empty state, drag-and-drop, the
// plot dock's resize handle and the keyboard shortcuts. It turns gestures into
// handler callbacks and exposes setters the main loop calls each frame.
//
// It holds no timeline or scene state — main.js is the single source of truth
// and pushes values down. The fleet list, inspector, event log and plots are
// their own modules. Setters write to the DOM only when a value changes.

import { fmtTime } from "./format.js";
import { savePrefs } from "./prefs.js";

const $ = (id) => document.getElementById(id);

const STATUS_TEXT = {
  idle: "idle", connecting: "connecting…", live: "live",
  file: "file", closed: "disconnected", error: "error",
};
const CAM_KEYS = { Digit1: "orbit", Digit2: "follow", Digit3: "chase", Digit4: "top", Digit5: "side" };
export const SPEEDS = [0.25, 0.5, 1, 2, 4, 8];
const KILL_FEED_S = 6;      // real seconds a kill stays in the feed
const KILL_FEED_MAX = 5;
const PLOTS_MIN_H = 110;
const PRESENT_HINT_MS = 2500;

function setText(node, text) {
  if (node.textContent !== text) node.textContent = text;
}

export class UI {
  constructor(handlers, prefs) {
    this.h = handlers;
    this.el = {
      app: $("app"),
      statusPill: $("statusPill"), sourceLabel: $("sourceLabel"),
      dockFleetBtn: $("dockFleetBtn"), dockInspectorBtn: $("dockInspectorBtn"), dockPlotsBtn: $("dockPlotsBtn"),
      presentBtn: $("presentBtn"), presentHint: $("presentHint"),
      sourceBtn: $("sourceBtn"), sourceMenu: $("sourceMenu"),
      helpBtn: $("helpBtn"), helpMenu: $("helpMenu"),
      wsUrl: $("wsUrl"), connectBtn: $("connectBtn"),
      openFileBtn: $("openFileBtn"), fileInput: $("fileInput"),
      splash: $("splash"), emptyStatus: $("emptyStatus"), retryBtn: $("retryBtn"), emptyOpenBtn: $("emptyOpenBtn"),
      sceneToolbar: $("sceneToolbar"), fitBtn: $("fitBtn"),
      tglTrails: $("tglTrails"), tglLabels: $("tglLabels"), tglDrop: $("tglDrop"),
      sizeRange: $("sizeRange"), sizeValue: $("sizeValue"),
      compassRose: $("compassRose"), scaleBar: $("scaleBar"), scaleLabel: $("scaleLabel"),
      killFeed: $("killFeed"), fps: $("fps"), mapAttribution: $("mapAttribution"),
      plotsResize: $("plotsResize"),
      playBtn: $("playBtn"), liveBtn: $("liveBtn"),
      timeCur: $("timeCur"), scrub: $("scrub"), timeTot: $("timeTot"),
      markers: $("markers"), speedSel: $("speedSel"),
      dropHint: $("dropHint"),
    };
    this.camButtons = [...this.el.sceneToolbar.querySelectorAll("[data-cam]")];
    const d = prefs.docks || {};
    this.docks = { fleet: d.fleet !== false, inspector: d.inspector !== false, plots: d.plots !== false };
    this.present = false;
    this._transport = {};
    this._view = { rot: "", scaleW: "", scaleText: "" };
    this._kills = []; // { li, at }
    this._hintTimer = 0;

    this._wire();
    this._applyDocks(false);
    if (Number.isFinite(prefs.plotsH)) this._setPlotsHeight(prefs.plotsH);
  }

  _wire() {
    const h = this.h, el = this.el;
    el.playBtn.addEventListener("click", () => h.onPlayToggle());
    el.liveBtn.addEventListener("click", () => h.onLive());
    el.scrub.addEventListener("input", () => h.onScrub(Number(el.scrub.value) / 1000));
    el.speedSel.addEventListener("change", () => h.onSpeed(Number(el.speedSel.value)));

    // Source menu.
    const connect = () => { h.onConnect(el.wsUrl.value.trim()); this._closeMenus(); };
    el.connectBtn.addEventListener("click", connect);
    el.wsUrl.addEventListener("keydown", (e) => { if (e.key === "Enter") connect(); });
    const openFile = () => { this._closeMenus(); el.fileInput.click(); };
    el.openFileBtn.addEventListener("click", openFile);
    el.emptyOpenBtn.addEventListener("click", openFile);
    el.retryBtn.addEventListener("click", () => h.onRetry());
    el.fileInput.addEventListener("change", () => {
      const f = el.fileInput.files?.[0];
      if (f) h.onFile(f);
      el.fileInput.value = ""; // allow re-opening the same file
    });
    this._menu(el.sourceBtn, el.sourceMenu, el.wsUrl);
    this._menu(el.helpBtn, el.helpMenu, null);
    document.addEventListener("pointerdown", (e) => {
      if (!e.target.closest(".menu-wrap")) this._closeMenus();
    });

    // Docks and presentation.
    const dock = (btn, key) => btn.addEventListener("click", () => {
      this.docks[key] = !this.docks[key];
      this._applyDocks();
      savePrefs({ docks: this.docks });
    });
    dock(el.dockFleetBtn, "fleet");
    dock(el.dockInspectorBtn, "inspector");
    dock(el.dockPlotsBtn, "plots");
    el.presentBtn.addEventListener("click", () => this.setPresent(!this.present));

    // Scene toolbar.
    for (const b of this.camButtons) b.addEventListener("click", () => h.onCamera(b.dataset.cam));
    el.fitBtn.addEventListener("click", () => h.onFit());
    const tgl = (btn, key) => btn.addEventListener("click", () => {
      const on = btn.getAttribute("aria-pressed") !== "true";
      btn.setAttribute("aria-pressed", String(on));
      h.onToggle(key, on);
    });
    tgl(el.tglTrails, "trails");
    tgl(el.tglLabels, "labels");
    tgl(el.tglDrop, "dropLines");
    el.sizeRange.addEventListener("input", () => {
      const k = Number(el.sizeRange.value);
      el.sizeValue.textContent = `${k}×`;
      h.onModelScale(k);
    });

    this._wireResize();
    this._wireKeys();
    this._wireDrop();
  }

  _menu(btn, menu, focus) {
    btn.addEventListener("click", () => {
      const open = menu.hidden;
      this._closeMenus();
      if (!open) return;
      menu.hidden = false;
      btn.setAttribute("aria-expanded", "true");
      focus?.focus();
    });
  }

  // Close every open menu; true if one was open.
  _closeMenus() {
    let closed = false;
    for (const [btn, menu] of [[this.el.sourceBtn, this.el.sourceMenu], [this.el.helpBtn, this.el.helpMenu]]) {
      if (!menu.hidden) { menu.hidden = true; btn.setAttribute("aria-expanded", "false"); closed = true; }
    }
    return closed;
  }

  _applyDocks(notify = true) {
    const { app, dockFleetBtn, dockInspectorBtn, dockPlotsBtn } = this.el;
    app.classList.toggle("no-fleet", !this.docks.fleet);
    app.classList.toggle("no-inspector", !this.docks.inspector);
    app.classList.toggle("no-plots", !this.docks.plots);
    dockFleetBtn.setAttribute("aria-pressed", String(this.docks.fleet));
    dockInspectorBtn.setAttribute("aria-pressed", String(this.docks.inspector));
    dockPlotsBtn.setAttribute("aria-pressed", String(this.docks.plots));
    if (notify) this.h.onLayout();
  }

  // Plots shown = its dock is on and we're not presenting.
  get plotsVisible() { return this.docks.plots && !this.present; }

  setPresent(on) {
    if (this.present === on) return;
    this.present = on;
    this._closeMenus();
    document.body.classList.toggle("present", on);
    this.el.presentBtn.setAttribute("aria-pressed", String(on));
    clearTimeout(this._hintTimer);
    this.el.presentHint.classList.toggle("show", on);
    if (on) this._hintTimer = setTimeout(() => this.el.presentHint.classList.remove("show"), PRESENT_HINT_MS);
    this.h.onLayout();
  }

  _setPlotsHeight(px) {
    const max = Math.max(PLOTS_MIN_H, window.innerHeight * 0.6);
    const hgt = Math.round(Math.max(PLOTS_MIN_H, Math.min(max, px)));
    this.el.app.style.setProperty("--plots-size", `${hgt}px`);
    return hgt;
  }

  // Drag the plot dock's top edge to resize it.
  _wireResize() {
    const handle = this.el.plotsResize;
    let startY = 0, startH = 0, h = 0;
    const move = (e) => { h = this._setPlotsHeight(startH + (startY - e.clientY)); };
    const up = () => {
      handle.classList.remove("drag");
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      savePrefs({ plotsH: h });
    };
    handle.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      startY = e.clientY;
      startH = h = $("plots").getBoundingClientRect().height;
      handle.classList.add("drag");
      window.addEventListener("pointermove", move);
      window.addEventListener("pointerup", up);
    });
  }

  _wireKeys() {
    const h = this.h;
    window.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        if (!this._closeMenus() && this.present) this.setPresent(false);
        return;
      }
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (/^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)) return;
      // The plot channel menu navigates with arrows and Space itself.
      if (e.target.closest?.("#plotsMenu")) return;
      const inList = e.target.closest?.(".ev-list");
      let handled = true;
      switch (e.code) {
        case "Space": h.onPlayToggle(); break;
        case "ArrowLeft": h.onStep(e.shiftKey ? -10 : -1); break;
        case "ArrowRight": h.onStep(e.shiftKey ? 10 : 1); break;
        case "ArrowUp": if (inList) handled = false; else h.onSelectStep(-1); break;
        case "ArrowDown": if (inList) handled = false; else h.onSelectStep(1); break;
        case "Home": h.onSeekEdge("start"); break;
        case "End": h.onSeekEdge("end"); break;
        case "KeyL": h.onLive(); break;
        case "KeyF": h.onFit(); break;
        case "KeyT": h.onCycleTarget(e.shiftKey ? -1 : 1); break;
        case "KeyP": this.setPresent(!this.present); break;
        case "Minus": this._speedStep(-1); break;
        case "Equal": this._speedStep(1); break;
        default:
          if (CAM_KEYS[e.code]) h.onCamera(CAM_KEYS[e.code]);
          else if (e.key === "?") this.el.helpBtn.click();
          else handled = false;
      }
      if (handled) e.preventDefault();
    });
  }

  _speedStep(dir) {
    const cur = Number(this.el.speedSel.value);
    let i = SPEEDS.indexOf(cur);
    if (i < 0) i = SPEEDS.indexOf(1);
    const next = SPEEDS[Math.max(0, Math.min(SPEEDS.length - 1, i + dir))];
    this.el.speedSel.value = String(next);
    this.h.onSpeed(next);
  }

  // Drag-and-drop an export JSON anywhere on the window.
  _wireDrop() {
    const hint = this.el.dropHint;
    let depth = 0;
    window.addEventListener("dragenter", (e) => { e.preventDefault(); if (depth++ === 0) hint.classList.add("show"); });
    window.addEventListener("dragover", (e) => e.preventDefault());
    window.addEventListener("dragleave", (e) => { e.preventDefault(); if (--depth <= 0) { depth = 0; hint.classList.remove("show"); } });
    window.addEventListener("drop", (e) => {
      e.preventDefault();
      depth = 0;
      hint.classList.remove("show");
      const f = e.dataTransfer?.files?.[0];
      if (f) this.h.onFile(f);
    });
  }

  // ---- setters driven by main.js ----

  setStatus(state, label) {
    const pill = this.el.statusPill;
    pill.className = `pill ${state}`;
    pill.textContent = STATUS_TEXT[state] || state;
    if (label != null) {
      this.el.sourceLabel.textContent = label;
      this.el.sourceLabel.title = label;
    }
  }

  setWsUrl(url) {
    if (!this.el.wsUrl.value) this.el.wsUrl.value = url;
  }

  // The empty state over the stage: shown until telemetry is buffered.
  setEmpty(visible, text, canRetry) {
    this.el.splash.classList.toggle("hidden", !visible);
    if (!visible) return;
    setText(this.el.emptyStatus, text);
    this.el.retryBtn.hidden = !canRetry;
  }

  // { hasData, playing, liveAvailable, liveActive }
  setTransport(t) {
    const last = this._transport, el = this.el;
    if (last.hasData !== t.hasData) {
      el.playBtn.disabled = el.scrub.disabled = !t.hasData;
    }
    const playing = t.playing && t.hasData;
    if (last.playing !== playing) {
      el.playBtn.classList.toggle("playing", playing);
      el.playBtn.setAttribute("aria-label", playing ? "Pause" : "Play");
      el.playBtn.title = playing ? "Pause (Space)" : "Play (Space)";
    }
    if (last.liveAvailable !== t.liveAvailable) el.liveBtn.disabled = !t.liveAvailable;
    const active = t.liveActive && t.liveAvailable;
    if (last.liveActive !== active) el.liveBtn.classList.toggle("active", active);
    this._transport = { hasData: t.hasData, playing, liveAvailable: t.liveAvailable, liveActive: active };
  }

  setSpeed(speed) {
    this.el.speedSel.value = String(speed);
  }

  // cur/tot are seconds from the timeline start; frac in [0,1] positions the scrubber.
  setTime(cur, tot, frac) {
    setText(this.el.timeCur, fmtTime(cur));
    setText(this.el.timeTot, fmtTime(tot));
    // Setting .value here doesn't fire "input", so it never fights a drag.
    const v = String(Math.round(Math.max(0, Math.min(1, frac)) * 1000));
    if (this.el.scrub.value !== v) this.el.scrub.value = v;
  }

  // markers: [{ frac, kind: "hit" | "kill", title }] — event ticks over the
  // scrubber, frac in [0,1] along it. Reuses the marker elements; clicking one
  // seeks there. The title reaches the DOM only through the title attribute.
  setMarkers(markers) {
    const layer = this.el.markers;
    while (layer.children.length > markers.length) layer.lastChild.remove();
    while (layer.children.length < markers.length) {
      const m = document.createElement("span");
      m.addEventListener("click", () => this.h.onScrub(Number(m.dataset.frac)));
      layer.appendChild(m);
    }
    markers.forEach((m, i) => {
      const node = layer.children[i];
      const frac = Math.max(0, Math.min(1, m.frac));
      const left = `${(frac * 100).toFixed(3)}%`;
      const cls = `marker ${m.kind}`;
      if (node.style.left !== left) node.style.left = left;
      if (node.className !== cls) node.className = cls;
      if (node.title !== m.title) node.title = m.title;
      node.dataset.frac = String(frac);
    });
  }

  setCameraMode(mode) {
    for (const b of this.camButtons) b.setAttribute("aria-pressed", String(b.dataset.cam === mode));
  }

  setToggles({ trails, labels, dropLines }) {
    this.el.tglTrails.setAttribute("aria-pressed", String(trails));
    this.el.tglLabels.setAttribute("aria-pressed", String(labels));
    this.el.tglDrop.setAttribute("aria-pressed", String(dropLines));
  }

  setModelScale(k) {
    this.el.sizeRange.value = String(k);
    this.el.sizeValue.textContent = `${k}×`;
  }

  // Compass rose and scale bar from the scene's view: headingRad is the bearing
  // the top of the screen faces; metersPerPx the world scale at the focus.
  setView({ headingRad, metersPerPx }) {
    const v = this._view, el = this.el;
    const rot = Number.isFinite(headingRad) ? `rotate(${(-headingRad * 180 / Math.PI).toFixed(1)})` : "";
    if (v.rot !== rot) { v.rot = rot; el.compassRose.setAttribute("transform", rot); }

    let w = "", text = "";
    if (Number.isFinite(metersPerPx) && metersPerPx > 0) {
      // Largest 1/2/5 × 10^n metres that fits in ~120 px.
      const target = 120 * metersPerPx;
      const p = 10 ** Math.floor(Math.log10(target));
      const len = [5, 2, 1].map((m) => m * p).find((l) => l <= target) ?? p;
      w = `${Math.round(len / metersPerPx)}px`;
      text = len >= 1000 ? `${len / 1000} km` : `${len} m`;
    }
    if (v.scaleW !== w) {
      v.scaleW = w;
      el.scaleBar.hidden = !w;
      el.scaleBar.firstElementChild.style.width = w;
    }
    if (v.scaleText !== text) { v.scaleText = text; el.scaleLabel.textContent = text; }
  }

  // A kill for the presentation-mode feed; colours are CSS colours.
  pushKill({ killer, victim, killerColor, victimColor }) {
    const li = document.createElement("li");
    const id = (text, color) => {
      const s = document.createElement("span");
      s.textContent = text;
      if (color) s.style.color = color;
      return s;
    };
    const x = document.createElement("span");
    x.className = "kf-x";
    x.textContent = "✕";
    if (killer != null) li.append(id(String(killer), killerColor), x, id(String(victim), victimColor));
    else li.append(x, id(`${victim} destroyed`, victimColor));
    this.el.killFeed.prepend(li);
    this._kills.unshift({ li, at: performance.now() });
    while (this._kills.length > KILL_FEED_MAX) this._kills.pop().li.remove();
  }

  // Age the kill feed (every frame).
  tickKillFeed(now) {
    while (this._kills.length) {
      const k = this._kills[this._kills.length - 1];
      const age = (now - k.at) / 1000;
      if (age < KILL_FEED_S) break;
      k.li.remove();
      this._kills.pop();
    }
    for (const k of this._kills) {
      if ((now - k.at) / 1000 > KILL_FEED_S - 0.8) k.li.classList.add("fade");
    }
  }

  clearKillFeed() {
    for (const k of this._kills) k.li.remove();
    this._kills = [];
  }

  setFps(fps) {
    setText(this.el.fps, Number.isFinite(fps) ? `${Math.round(fps)} fps` : "—");
  }

  // The imagery credit shows only while the streamed map is in use.
  setMapCredit(visible) {
    this.el.mapAttribution.hidden = !visible;
  }
}
