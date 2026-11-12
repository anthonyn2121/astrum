// The studio run's panel (#runPanel) on view.html?run=<id>: the run's state
// from GET /api/runs/current (~1 Hz while the page is visible and the run is
// not finished), a Stop button (DELETE /api/runs/current), its log
// (GET /api/runs/current/log?since=…, appended as it arrives) and, once the
// run has finished, its outcome, key metrics (GET /api/runs/<id>/metrics) and
// "Open full replay" (GET /api/runs/<id>/replay, handed to `onReplay`). A race
// run's course (the run view's `course`) is handed to `onCourse` — validated
// (race.js), null for any other run — every time it is read.
//
// A run that is not the server's current one (an earlier run's link) shows
// only its metrics and replay. Everything the server sends — ids, reasons,
// log lines — is untrusted and reaches the DOM as text only.

import { getCurrentRun, stopRun, getRunLog, getRunMetrics, getRunReplay, ApiError } from "./api.js";
import { fmtNum, fmtRange } from "./format.js";
import { parseCourse } from "./race.js";

const RUN_POLL_MS = 1000;
const LOG_DOM_MAX = 2000; // lines kept on the page (the server keeps its own ring)

const $ = (id) => document.getElementById(id);

function setText(node, text) {
  if (node.textContent !== text) node.textContent = text;
}

const num = (v) => (typeof v === "number" && Number.isFinite(v) ? v : null);

// The status pill's look for a run state / outcome.
function pillClass(view) {
  if (!view) return "idle";
  if (view.state === "starting") return "connecting";
  if (view.state === "running") return "live";
  const kind = view.outcome?.status?.kind;
  if (kind === "ok") return "file";
  if (kind === "failed") return "error";
  return "idle";
}

function outcomeText(outcome) {
  const kind = outcome?.status?.kind;
  const reason = typeof outcome?.status?.reason === "string" ? outcome.status.reason : "";
  const head = kind === "ok" ? "Finished" : kind === "stopped" ? "Stopped" : kind === "failed" ? "Failed" : "Ended";
  return reason ? `${head}: ${reason}` : head;
}

function paceText(outcome) {
  const sim = num(outcome?.sim_time_s), wall = num(outcome?.wall_s);
  if (sim == null || wall == null) return wall == null ? "" : `${fmtNum(wall, 1)} s wall`;
  const pace = wall > 0 ? ` (${fmtNum(sim / wall, 1)}× real time)` : "";
  return `${fmtNum(sim, 1)} s sim in ${fmtNum(wall, 1)} s wall${pace}`;
}

// [label, text] rows of the key metrics (astrum_bag::metrics::RunMetrics).
function metricRows(m) {
  const rows = [];
  const add = (label, text) => { if (text != null) rows.push([label, text]); };
  const unit = (v, digits, u) => (num(v) == null ? null : `${fmtNum(v, digits)} ${u}`);
  if (typeof m.end_reason === "string") {
    const msg = typeof m.end_message === "string" && m.end_message ? ` — ${m.end_message}` : "";
    add("end", `${m.end_reason}${msg}`);
  }
  add("sim time", unit(m.sim_time_s, 1, "s"));
  if (num(m.min_separation_m) != null) {
    const pair = Array.isArray(m.min_separation_pair) ? ` (${m.min_separation_pair.map(String).join("–")})` : "";
    add("closest approach", `${fmtRange(m.min_separation_m)}${pair}`);
  }
  add("lowest altitude", unit(m.min_alt_hae_m, 0, "m HAE"));
  add("max bank", unit(m.max_bank_deg, 1, "°"));
  add("max load", unit(m.max_nz_g, 2, "g"));
  // A race run's score (astrum_bag::metrics, C5).
  if (Number.isInteger(m.gates_passed)) add("gates passed", String(m.gates_passed));
  if (Number.isInteger(m.gates_missed)) add("gates missed", String(m.gates_missed));
  add("race time", unit(m.race_time_s, 2, "s"));
  if (typeof m.race_outcome === "string") add("race outcome", m.race_outcome);
  for (const [key, label] of [
    ["failsafe_entries", "failsafe entries"], ["controller_losses", "controller losses"],
    ["command_refusals", "command refusals"], ["hits", "hits"], ["kills", "kills"],
  ]) {
    if (Number.isInteger(m[key])) add(label, String(m[key]));
  }
  return rows;
}

export class RunPanel {
  constructor({ runId, onReplay, onCourse }) {
    this.runId = runId;
    this.onReplay = onReplay;
    this.onCourse = onCourse;
    this.el = {
      panel: $("runPanel"), id: $("runId"), state: $("runState"), stop: $("runStopBtn"),
      collapse: $("runCollapseBtn"), body: $("runBody"), info: $("runInfo"), error: $("runError"),
      result: $("runResult"), outcome: $("runOutcome"), metrics: $("runMetrics"), replay: $("runReplayBtn"),
      moved: $("runMovedNote"),
      log: $("runLog"), logCount: $("runLogCount"), logLines: $("runLogLines"),
    };
    this.view = null;        // the run's RunView while it is the current run
    this.current = false;    // it is the server's current run
    this.done = false;       // finished (or not current): nothing left to poll
    this.resultShown = false;
    this.replayable = false; // finished with a recording to replay
    this.missing = false;    // no such run here, or it has no recording
    this.logNext = 0;        // the next log seq to ask for
    this.logTotal = 0;
    this.timer = 0;
    this.polling = false;
    this._wire();
  }

  _wire() {
    const el = this.el;
    el.stop.addEventListener("click", () => this._stop());
    el.replay.addEventListener("click", () => this._replay());
    el.collapse.addEventListener("click", () => {
      const open = el.body.hidden;
      el.body.hidden = !open;
      el.collapse.setAttribute("aria-expanded", String(open));
      el.collapse.textContent = open ? "▾" : "▸";
    });
    document.addEventListener("visibilitychange", () => {
      if (!document.hidden && !this.done) this._schedule(0);
    });
  }

  start() {
    this.el.panel.hidden = false;
    setText(this.el.id, this.runId);
    this.el.id.title = this.runId;
    this._schedule(0);
  }

  _schedule(ms) {
    clearTimeout(this.timer);
    this.timer = setTimeout(() => this._poll(), ms);
  }

  async _poll() {
    if (document.hidden || this.polling || this.done) return;
    this.polling = true;
    try {
      const run = await getCurrentRun();
      // A Stop may have finished the run meanwhile: its view is newer.
      if (this.done) return;
      this.current = !!run && run.id === this.runId;
      if (this.current) {
        this.view = run;
        this.onCourse(parseCourse(run.course));
        await this._pollLog();
        if (this.done) return;
        if (run.state === "finished") this.done = true;
      } else {
        // An earlier run (or none since the server restarted): what it left.
        this.view = null;
        this.done = true;
      }
      this._error("");
    } catch (e) {
      if (this.done) return;
      this._error(`run status: ${e.message || e}`);
    } finally {
      this.polling = false;
    }
    this._render();
    if (this.done) this._showResult();
    else this._schedule(RUN_POLL_MS);
  }

  async _pollLog() {
    const { lines, next } = await getRunLog(this.logNext);
    // A concurrent poll (Stop's last one) may have appended these already.
    const fresh = Array.isArray(lines) ? lines.filter((l) => Number.isInteger(l?.seq) && l.seq >= this.logNext) : [];
    if (fresh.length) this._appendLog(fresh);
    if (Number.isInteger(next)) this.logNext = Math.max(this.logNext, next);
  }

  _appendLog(lines) {
    const box = this.el.logLines;
    const atBottom = box.scrollTop + box.clientHeight >= box.scrollHeight - 4;
    const frag = document.createDocumentFragment();
    for (const l of lines) {
      if (!l || typeof l.line !== "string") continue;
      const row = document.createElement("div");
      row.className = l.stream === "stderr" ? "log-line stderr" : "log-line";
      const src = document.createElement("span");
      src.className = "log-src";
      src.textContent = typeof l.source === "string" ? l.source : "?";
      const text = document.createElement("span");
      text.textContent = l.line;
      row.append(src, text);
      frag.appendChild(row);
      this.logTotal += 1;
    }
    box.appendChild(frag);
    while (box.childElementCount > LOG_DOM_MAX) box.firstElementChild.remove();
    setText(this.el.logCount, String(this.logTotal));
    if (atBottom) box.scrollTop = box.scrollHeight;
  }

  _render() {
    const el = this.el, v = this.view;
    el.state.className = `pill ${pillClass(v)}`;
    setText(el.state, v ? v.state : this.current ? "—" : "earlier run");
    el.stop.hidden = !(v && v.state !== "finished");
    el.log.hidden = !v;
    if (!v) {
      setText(el.info, "Not the server's current run: its recording and metrics, if any.");
      return;
    }
    const parts = [v.scenario ? `scenario ${v.scenario}` : "inline scenario"];
    const ctl = Array.isArray(v.assignments) ? v.assignments : [];
    parts.push(ctl.length
      ? ctl.map((a) => `${a.aircraft}: ${a.controller}`).join(", ")
      : "no controllers (uncommanded)");
    if (v.lockstep) parts.push("lockstep");
    if (num(v.speed) != null) parts.push(v.speed === 0 ? "started at max pace" : `started at ${fmtNum(v.speed, v.speed < 1 ? 2 : 0)}×`);
    if (num(v.max_time_s) != null) parts.push(v.max_time_s > 0 ? `until ${fmtNum(v.max_time_s, 0)} s` : "until stopped");
    setText(el.info, parts.join(" · "));
  }

  async _showResult() {
    if (this.resultShown) return;
    this.resultShown = true;
    const el = this.el;
    el.result.hidden = false;
    setText(el.outcome, this.view?.outcome
      ? [outcomeText(this.view.outcome), paceText(this.view.outcome)].filter(Boolean).join(" · ")
      : "");
    // Only a run this server flew and finished moved the live stream on.
    el.moved.hidden = !this.current;
    try {
      const metrics = await getRunMetrics(this.runId);
      el.metrics.replaceChildren();
      for (const [label, text] of metricRows(metrics || {})) {
        const dt = document.createElement("dt");
        dt.textContent = label;
        const dd = document.createElement("dd");
        dd.textContent = text;
        el.metrics.append(dt, dd);
      }
      this.replayable = true;
    } catch (e) {
      // No recording (404), or no such run (400): nothing to replay either.
      if (e instanceof ApiError && (e.status === 404 || e.status === 400)) {
        el.replay.hidden = true;
        el.moved.hidden = true;
        this.missing = true;
      } else {
        this.replayable = true; // the replay may still load
      }
      this._error(`metrics: ${e.message || e}`);
    }
  }

  async _stop() {
    const btn = this.el.stop;
    btn.disabled = true;
    btn.textContent = "Stopping…";
    try {
      const view = await stopRun();
      // Final from here on: a poll in flight must not bring back "running".
      this.view = view;
      this.onCourse(parseCourse(view.course));
      this.current = true;
      this.done = true;
      await this._pollLog().catch(() => {});
      this._error("");
    } catch (e) {
      this._error(`stop: ${e.message || e}`);
      this._schedule(0); // it may have ended meanwhile
    }
    btn.disabled = false;
    btn.textContent = "Stop";
    this._render();
    if (this.done) this._showResult();
  }

  async _replay() {
    const btn = this.el.replay;
    btn.disabled = true;
    btn.textContent = "Loading…";
    try {
      const doc = await getRunReplay(this.runId);
      await this.onReplay(doc, `run ${this.runId}`);
      this._error("");
    } catch (e) {
      this._error(`replay: ${e.message || e}`);
    }
    btn.disabled = false;
    btn.textContent = "Open full replay";
  }

  _error(msg) {
    setText(this.el.error, msg);
    this.el.error.hidden = !msg;
  }
}
