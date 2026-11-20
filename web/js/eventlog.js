// The event log in the aircraft dock: every run event (hits, kills, triggers,
// mode changes, controller up/down, run end) as one clickable row that seeks
// the playhead there, filterable by kind and to the selected aircraft.
//
// The list is virtual — fixed-height rows, only the visible ones in the DOM —
// so a long live run with thousands of events stays cheap. The row at or just
// before the playhead is highlighted (later ones dimmed) and kept in view while
// playing, unless the user scrolled the list in the last few seconds.

import { fmtTime } from "./format.js";

const ROW_H = 24;
const OVERSCAN = 4;
const USER_SCROLL_HOLD_MS = 3000;

// Filter chips: label, the event kinds each covers, and the default.
export const EVENT_GROUPS = [
  { key: "hit", label: "Hits", kinds: ["hit"], on: true },
  { key: "kill", label: "Kills", kinds: ["kill"], on: true },
  { key: "trigger", label: "Triggers", kinds: ["trigger-pressed", "trigger-released"], on: false },
  { key: "mode", label: "Mode", kinds: ["mode-change"], on: true },
  { key: "ctrl", label: "Controller", kinds: ["controller-up", "controller-down"], on: true },
  { key: "run", label: "Run", kinds: ["run-ended"], on: true },
  { key: "race", label: "Race", kinds: ["gate-passed", "gate-missed", "crashed"], on: true },
];
const GROUP_OF = new Map(EVENT_GROUPS.flatMap((g) => g.kinds.map((k) => [k, g.key])));
const BADGE = { hit: "HIT", kill: "KILL", trigger: "TRIG", mode: "MODE", ctrl: "CTRL", run: "END", race: "GATE" };
const ID_FIELDS = ["aircraft_id", "shooter_id", "target_id", "victim_id", "killer_id"];

const n0 = (v) => (Number.isFinite(v) ? v.toFixed(0) : "?");

// Gates are numbered from 1 on screen, as their tags in the scene are.
const gateNo = (e) => (Number.isInteger(e.gate) ? e.gate + 1 : "?");
const offsetText = (e) => (Number.isFinite(e.offset_x_m) && Number.isFinite(e.offset_y_m)
  ? ` · ${(e.offset_x_m * 100).toFixed(0)} cm right, ${(e.offset_y_m * 100).toFixed(0)} cm up` : "");

// One-line description of an event (plain text; set via textContent).
export function describeEvent(e) {
  switch (e.kind) {
    case "hit": return `${e.shooter_id ?? "?"} → ${e.target_id ?? "?"} · ${n0(e.damage)} dmg · ${n0(e.hit_points)} hp left`;
    case "kill": return e.killer_id == null ? `${e.victim_id ?? "?"} destroyed` : `${e.killer_id} killed ${e.victim_id ?? "?"}`;
    case "trigger-pressed": return `${e.aircraft_id ?? "?"} trigger down · ${n0(e.ammo)} rds`;
    case "trigger-released": return `${e.aircraft_id ?? "?"} trigger up · ${n0(e.rounds_fired)} fired`;
    case "mode-change": return `${e.aircraft_id ?? "?"} ${e.from ?? "?"} → ${e.to ?? "?"}${e.reason ? ` (${e.reason})` : ""}`;
    case "controller-up": return `${e.aircraft_id ?? "?"} controller connected`;
    case "controller-down": return `${e.aircraft_id ?? "?"} controller lost`;
    case "gate-passed": return `${e.aircraft_id ?? "?"} passed gate ${gateNo(e)}${offsetText(e)}`;
    case "gate-missed": return `${e.aircraft_id ?? "?"} missed gate ${gateNo(e)}`;
    case "crashed": return `${e.aircraft_id ?? "?"} crashed${e.outcome ? ` (${e.outcome})` : ""} at gate ${gateNo(e)}`;
    case "run-ended": return `run ended: ${e.reason ?? "?"}${e.message ? ` — ${e.message}` : ""}`;
    default: return e.kind;
  }
}

export class EventLog {
  constructor({ list, empty, count, filters, mineBtn, kinds, onSeek, onKindsChange }) {
    this.dom = { list, empty, count, mineBtn };
    this.onSeek = onSeek;
    this.enabled = new Set(kinds ?? EVENT_GROUPS.filter((g) => g.on).map((g) => g.key));
    this.mine = false;

    // Filtered view: indices into the store's event array.
    this._events = null;
    this._seen = 0;
    this._idx = [];
    this._selectedId = null;
    this._dirty = true;

    this._t0 = 0;
    this._current = -1;
    this._userScrollAt = -Infinity;
    this._rendered = { from: -1, to: -1, current: -2, t0: NaN };

    this._sizer = document.createElement("div");
    list.appendChild(this._sizer);
    this._pool = [];

    for (const g of EVENT_GROUPS) {
      const b = document.createElement("button");
      b.textContent = g.label;
      b.setAttribute("aria-pressed", String(this.enabled.has(g.key)));
      b.addEventListener("click", () => {
        if (!this.enabled.delete(g.key)) this.enabled.add(g.key);
        b.setAttribute("aria-pressed", String(this.enabled.has(g.key)));
        this._dirty = true;
        onKindsChange?.([...this.enabled]);
      });
      filters.appendChild(b);
    }
    mineBtn.addEventListener("click", () => {
      this.mine = !this.mine;
      mineBtn.setAttribute("aria-pressed", String(this.mine));
      this._dirty = true;
    });

    list.addEventListener("scroll", () => this._renderRows(true), { passive: true });
    // Only real user input marks a manual scroll (our own scrollTop writes don't).
    for (const ev of ["wheel", "touchmove", "keydown", "pointerdown"]) {
      list.addEventListener(ev, () => { this._userScrollAt = performance.now(); }, { passive: true });
    }
    list.addEventListener("click", (e) => {
      const row = e.target.closest(".ev-row");
      const ev = row && this._events?.[Number(row.dataset.idx)];
      if (ev) this.onSeek(ev.t);
    });
    this._resizeObs = new ResizeObserver(() => this._renderRows(true));
    this._resizeObs.observe(list);
  }

  // Every frame: the store's (sorted) event array, the playhead, the timeline
  // start (times are shown relative to it) and the selection.
  update(events, { playhead, t0, selectedId, playing }) {
    if (events !== this._events) { this._events = events; this._dirty = true; }
    if (this.mine && selectedId !== this._selectedId) this._dirty = true;
    this._selectedId = selectedId;

    if (this._dirty) {
      this._dirty = false;
      this._idx = [];
      this._seen = 0;
      this._rendered.from = -1; // force a re-render
    }
    if (events.length > this._seen) {
      for (let i = this._seen; i < events.length; i++) if (this._keep(events[i])) this._idx.push(i);
      this._seen = events.length;
      this._rendered.from = -1;
    }

    const n = this._idx.length;
    const { list, empty, count } = this.dom;
    const countText = String(n);
    if (count.textContent !== countText) count.textContent = countText;
    empty.hidden = n > 0;
    list.hidden = n === 0;
    const h = `${n * ROW_H}px`;
    if (this._sizer.style.height !== h) this._sizer.style.height = h;

    this._t0 = t0;
    this._current = this._indexAt(playhead);

    if (playing && this._current >= 0 && performance.now() - this._userScrollAt > USER_SCROLL_HOLD_MS) {
      const top = this._current * ROW_H;
      if (top < list.scrollTop || top + ROW_H > list.scrollTop + list.clientHeight) {
        list.scrollTop = Math.max(0, top - list.clientHeight / 2);
      }
    }
    this._renderRows(false);
  }

  _keep(e) {
    if (!this.enabled.has(GROUP_OF.get(e.kind))) return false;
    if (!this.mine || this._selectedId == null || e.kind === "run-ended") return true;
    return ID_FIELDS.some((k) => e[k] === this._selectedId);
  }

  // Last filtered row with t <= playhead, or -1.
  _indexAt(t) {
    const idx = this._idx, ev = this._events;
    let lo = 0, hi = idx.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (ev[idx[mid]].t <= t) lo = mid + 1; else hi = mid;
    }
    return lo - 1;
  }

  _renderRows(force) {
    const { list } = this.dom;
    const n = this._idx.length;
    const from = Math.max(0, Math.floor(list.scrollTop / ROW_H) - OVERSCAN);
    const to = Math.min(n, Math.ceil((list.scrollTop + list.clientHeight) / ROW_H) + OVERSCAN);
    const r = this._rendered;
    if (!force && r.from === from && r.to === to && r.current === this._current && r.t0 === this._t0) return;
    r.from = from; r.to = to; r.current = this._current; r.t0 = this._t0;

    while (this._pool.length < to - from) {
      const row = document.createElement("div");
      row.className = "ev-row";
      row.setAttribute("role", "listitem");
      const t = document.createElement("span"); t.className = "ev-t";
      const k = document.createElement("span"); k.className = "ev-kind";
      const x = document.createElement("span"); x.className = "ev-text";
      row.append(t, k, x);
      list.appendChild(row);
      this._pool.push({ row, t, k, x, idx: -1, i: -1, t0: NaN });
    }
    this._pool.forEach((p, j) => {
      const i = from + j;
      if (i >= to) { p.row.hidden = true; p.idx = -1; return; }
      p.row.hidden = false;
      const evIdx = this._idx[i];
      const e = this._events[evIdx];
      if (p.idx !== evIdx || p.i !== i || p.t0 !== this._t0) {
        p.idx = evIdx;
        p.i = i;
        p.t0 = this._t0;
        p.row.dataset.idx = String(evIdx);
        p.row.style.top = `${i * ROW_H}px`;
        p.t.textContent = fmtTime(e.t - this._t0);
        const group = GROUP_OF.get(e.kind);
        p.k.textContent = BADGE[group] || "";
        p.k.className = `ev-kind ${group}`;
        const text = describeEvent(e);
        p.x.textContent = text;
        p.row.title = `${text} at ${e.t.toFixed(2)} s`;
      }
      const cls = `ev-row${i === this._current ? " current" : i > this._current ? " future" : ""}`;
      if (p.row.className !== cls) p.row.className = cls;
    });
  }
}
