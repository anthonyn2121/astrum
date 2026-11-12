// The fleet dock: every aircraft, grouped by team once there is more than one,
// one status row each — colour, id, vehicle class, control mode, hit points
// (or destroyed / no data) — with a pin button that makes it the target.
//
// Rows are keyed by id and patched in place; the list structure is rebuilt
// only when the set of aircraft, their teams, the filter or a folded group
// changes. Frame data reaches the DOM only through textContent / styles.

import { MODE_SHORT, teamColor } from "./format.js";

const KIND_GLYPH = { "fixed-wing": "✈", multirotor: "✜" };
// The filter box appears once the fleet is this big (or while it holds text).
const SEARCH_MIN = 12;

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

function setText(node, text) {
  if (node.textContent !== text) node.textContent = text;
}

export class FleetList {
  constructor({ list, count, empty, searchWrap, search, onSelect, onTarget }) {
    this.dom = { list, count, empty, searchWrap, search };
    this.rows = new Map();      // id -> row record (element + last-written state)
    this.heads = new Map();     // team -> { el, count }
    this.collapsed = new Set(); // folded teams
    this.filter = "";
    this._sig = null;
    this._order = [];

    search.addEventListener("input", () => {
      this.filter = search.value.trim().toLowerCase();
      this._sig = null;
    });
    list.addEventListener("click", (e) => {
      const pin = e.target.closest(".pin");
      const row = e.target.closest(".frow");
      if (pin && row) { onTarget(Number(row.dataset.id)); return; }
      if (row) { onSelect(Number(row.dataset.id)); return; }
      const head = e.target.closest(".team-head");
      if (head) {
        const team = Number(head.dataset.team);
        if (!this.collapsed.delete(team)) this.collapsed.add(team);
        this._sig = null;
      }
    });
  }

  // Visible ids in display order (for ↑/↓ and target cycling).
  order() { return this._order; }

  // items: [{ id, kind, color, team, alive, inRange, hp, hpMax, mode }] in id order.
  update(items, selectedId, targetId) {
    const { list, count, empty, searchWrap } = this.dom;
    setText(count, String(items.length));
    empty.hidden = items.length > 0;
    searchWrap.hidden = items.length < SEARCH_MIN && !this.filter;

    const shown = this.filter ? items.filter((it) => this._matches(it)) : items;
    const teams = [...new Set(items.map((it) => it.team))].sort((a, b) => a - b);
    const grouped = teams.length > 1;

    const sig = `${this.filter}|${grouped}|${[...this.collapsed].join(",")}|`
      + shown.map((it) => `${it.id}:${it.kind}:${it.team}`).join(",");
    if (sig !== this._sig) {
      this._sig = sig;
      this._rebuild(shown, teams, grouped);
    }

    for (const it of shown) {
      const row = this.rows.get(it.id);
      if (row?.el.isConnected) this._patch(row, it, selectedId, targetId);
    }
    if (grouped) {
      for (const team of teams) {
        const head = this.heads.get(team);
        if (!head) continue;
        const members = items.filter((it) => it.team === team);
        setText(head.count, `${members.filter((it) => it.alive).length}/${members.length}`);
      }
    }
  }

  // Every whitespace-separated term must appear; "team 2" is one term, so it
  // matches the team and not every id containing a 2.
  _matches(it) {
    const hay = `${it.id} team:${it.team} ${it.kind} ${it.mode} ${MODE_SHORT[it.mode] || ""}`.toLowerCase();
    return this.filter.replace(/\bteam\s+/g, "team:").split(/\s+/).every((tok) => hay.includes(tok));
  }

  _rebuild(shown, teams, grouped) {
    const { list } = this.dom;
    list.textContent = "";
    this.heads.clear();
    this._order = [];
    const keep = new Set(shown.map((it) => it.id));
    for (const id of this.rows.keys()) if (!keep.has(id)) this.rows.delete(id);

    for (const team of grouped ? teams : [null]) {
      const members = team === null ? shown : shown.filter((it) => it.team === team);
      if (grouped) {
        if (!members.length) continue;
        const folded = this.collapsed.has(team);
        const head = el("button", "team-head");
        head.dataset.team = String(team);
        head.setAttribute("aria-expanded", String(!folded));
        const dot = el("span", "team-dot");
        dot.style.background = teamColor(team);
        const cnt = el("span", "team-count");
        head.append(el("span", "chev", "▾"), dot, el("span", null, `Team ${team}`), cnt);
        list.appendChild(head);
        this.heads.set(team, { el: head, count: cnt });
        if (folded) continue;
      }
      for (const it of members) {
        list.appendChild(this._row(it).el);
        this._order.push(it.id);
      }
    }
  }

  _row(it) {
    let row = this.rows.get(it.id);
    if (row && row.kind === it.kind) return row;
    const e = el("div", "frow");
    e.dataset.id = String(it.id);
    e.setAttribute("role", "option");
    const swatch = el("span", "swatch");
    const mode = el("span", "mode-chip");
    const hp = el("span", "hp");
    const hpFill = el("span", "hp-fill");
    hp.appendChild(hpFill);
    const status = el("span", "status");
    const cell = el("span", "fcell");
    cell.append(hp, status);
    const pin = el("button", "pin", "◎");
    pin.title = "Pin as target (T cycles)";
    pin.setAttribute("aria-label", `Pin aircraft ${it.id} as target`);
    e.append(swatch, el("span", "fid", String(it.id)),
      el("span", "glyph", KIND_GLYPH[it.kind] || "•"), mode, cell, pin);
    row = { el: e, kind: it.kind, swatch, mode, hp, hpFill, status, pin, last: {} };
    this.rows.set(it.id, row);
    return row;
  }

  _patch(row, it, selectedId, targetId) {
    const sel = it.id === selectedId, tgt = it.id === targetId;
    const dead = !it.alive, stale = !it.inRange;
    const cls = `frow${sel ? " sel" : ""}${tgt ? " tgt" : ""}${dead ? " dead" : ""}${stale && !dead ? " stale" : ""}`;
    if (row.el.className !== cls) {
      row.el.className = cls;
      row.el.setAttribute("aria-selected", String(sel));
      row.pin.setAttribute("aria-pressed", String(tgt));
      row.pin.title = tgt ? "Unpin target" : "Pin as target (T cycles)";
    }

    const last = row.last;
    // The colour follows the team and the environment (sky mode recolours).
    if (last.color !== it.color) { last.color = it.color; row.swatch.style.background = it.color; }
    if (last.mode !== it.mode) {
      last.mode = it.mode;
      row.mode.textContent = MODE_SHORT[it.mode] || "?";
      row.mode.className = `mode-chip ${it.mode}`;
      row.mode.title = `control mode: ${it.mode}`;
    }

    // One cell: the HP bar for a live combat aircraft, else a short status.
    const showHp = !dead && !stale && it.hpMax > 0;
    const status = dead ? "destroyed" : stale ? "no data" : showHp ? "" : it.kind;
    if (last.showHp !== showHp) { last.showHp = showHp; row.hp.hidden = !showHp; row.status.hidden = showHp; }
    if (!showHp) setText(row.status, status);
    else {
      const frac = Math.max(0, Math.min(1, it.hp / it.hpMax));
      const w = `${(frac * 100).toFixed(1)}%`;
      const fillCls = `hp-fill${frac < 0.3 ? " low" : frac < 0.6 ? " mid" : ""}`;
      if (last.w !== w) { last.w = w; row.hpFill.style.width = w; row.hp.title = `${it.hp.toFixed(0)} / ${it.hpMax.toFixed(0)} hp`; }
      if (last.fillCls !== fillCls) { last.fillCls = fillCls; row.hpFill.className = fillCls; }
    }
  }
}
