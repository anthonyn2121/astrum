// The live sim's control bar (#simBar): pause / resume / step N / pace, sent
// through the studio API to whichever sim the bridge is attached to
// (POST /api/bridge/control), and its status polled from GET /api/bridge
// (~2 Hz, only while the page is visible). It is the *sim's* control plane —
// separate from the local playback transport, which only replays what this
// page has received.
//
// The bar shows while `enabled()` holds (the viewer is showing the live
// stream) and the attached sim answers its status.

import { getBridge, controlSim } from "./api.js";
import { fmtNum } from "./format.js";

const POLL_MS = 500;
const ERROR_MS = 6000; // how long a refused request's reason stays up

const $ = (id) => document.getElementById(id);

function setText(node, text) {
  if (node.textContent !== text) node.textContent = text;
}

// A pace (× real time): "10×", "0.5×". As a target, 0 means unpaced ("max").
function fmtPace(x, target) {
  if (!Number.isFinite(x)) return "—";
  if (x === 0 && target) return "max";
  return `${x >= 10 ? Math.round(x) : fmtNum(x, 1)}×`;
}

export class SimControl {
  constructor({ enabled }) {
    this.enabled = enabled;
    this.el = {
      bar: $("simBar"), state: $("simState"), tick: $("simTick"), time: $("simTime"),
      pace: $("simPace"), waiting: $("simWaiting"), pause: $("simPauseBtn"),
      stepN: $("simStepN"), step: $("simStepBtn"), speed: $("simSpeed"), error: $("simError"),
    };
    this.speedButtons = [...this.el.speed.querySelectorAll("[data-speed]")];
    this.status = null;
    this.busy = false;     // a control request is in flight
    this.timer = 0;
    this.polling = false;  // a status request is in flight
    this.errorTimer = 0;
    this._wire();
  }

  _wire() {
    const el = this.el;
    el.pause.addEventListener("click", () => {
      const paused = this.status?.state === "paused";
      this._send(paused ? { action: "resume" } : { action: "pause" });
    });
    el.step.addEventListener("click", () => this._step());
    el.stepN.addEventListener("keydown", (e) => { if (e.key === "Enter") this._step(); });
    for (const b of this.speedButtons) {
      b.addEventListener("click", () => this._send({ action: "speed", value: Number(b.dataset.speed) }));
    }
    document.addEventListener("visibilitychange", () => {
      if (!document.hidden) this._schedule(0);
    });
  }

  start() {
    this._schedule(0);
  }

  _schedule(ms) {
    clearTimeout(this.timer);
    this.timer = setTimeout(() => this._poll(), ms);
  }

  async _poll() {
    if (document.hidden || this.polling) return; // resumed by visibilitychange
    this.polling = true;
    let status = null;
    try {
      const bridge = await getBridge();
      status = bridge.connected && bridge.status && typeof bridge.status === "object" ? bridge.status : null;
    } catch {
      status = null; // the server went away: hide the bar until it answers again
    }
    this.polling = false;
    this._render(status);
    this._schedule(POLL_MS);
  }

  _step() {
    const n = Number(this.el.stepN.value);
    if (!Number.isInteger(n) || n < 1) {
      this._error("step: enter a whole number of ticks (1 or more)");
      return;
    }
    this._send({ action: "step", ticks: n });
  }

  async _send(action) {
    if (this.busy) return;
    this.busy = true;
    this._syncButtons();
    try {
      await controlSim(action);
      this._error("");
    } catch (e) {
      this._error(`${action.action}: ${e.message || e}`);
    }
    this.busy = false;
    this._schedule(0); // show the effect at once
  }

  _error(msg) {
    clearTimeout(this.errorTimer);
    setText(this.el.error, msg);
    this.el.error.title = msg;
    if (msg) this.errorTimer = setTimeout(() => setText(this.el.error, ""), ERROR_MS);
  }

  _render(status) {
    const show = !!status && this.enabled();
    this.status = status;
    this.el.bar.hidden = !show;
    if (!show) return;
    const el = this.el;
    const st = typeof status.state === "string" ? status.state : "?";
    setText(el.state, st);
    el.state.className = `sim-state ${/^(running|paused|waiting)$/.test(st) ? st : ""}`;
    setText(el.tick, Number.isFinite(status.tick) ? String(status.tick) : "—");
    setText(el.time, Number.isFinite(status.sim_time_s) ? `${fmtNum(status.sim_time_s, 1)} s` : "—");
    setText(el.pace, `${fmtPace(status.achieved_speed, false)} / ${fmtPace(status.speed, true)}`);
    const waiting = Array.isArray(status.lockstep_waiting) ? status.lockstep_waiting.filter(Number.isInteger) : [];
    el.waiting.hidden = !(st === "waiting" && waiting.length);
    setText(el.waiting, `waiting on ${waiting.join(", ")}`);
    for (const b of this.speedButtons) {
      b.setAttribute("aria-pressed", String(Number(b.dataset.speed) === status.speed));
    }
    this._syncButtons();
  }

  _syncButtons() {
    const el = this.el;
    const paused = this.status?.state === "paused";
    setText(el.pause, paused ? "Resume" : "Pause");
    el.pause.title = paused ? "Resume the sim" : "Pause the sim";
    el.pause.disabled = this.busy;
    el.step.disabled = this.busy;
    for (const b of this.speedButtons) b.disabled = this.busy;
  }
}
