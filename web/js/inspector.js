// The aircraft dock's readout: the selected aircraft's flight state at the
// playhead (key values large, attitude and combat state below, position and
// time folded away) and, when a target is pinned, the relative geometry to it.
// Values reach the DOM only through textContent, and only when they change.

import { relativeGeometry } from "./geometry.js";
import { RAD2DEG, fmtNum, fmtRange, fmtSigned, headingDeg, teamColor } from "./format.js";

const $ = (id) => document.getElementById(id);

function setText(node, text) {
  if (node.textContent !== text) node.textContent = text;
}

const RO = ["alt", "vs", "tas", "mach", "nz", "aoa", "hdg", "pitch", "roll", "team", "hp", "hpFill",
  "ammo", "time", "lat", "lon", "altHae", "e", "n", "u", "gs"];
const TG = ["range", "closure", "ata", "aspect", "relalt"];

export class Inspector {
  constructor({ onClearTarget, onSwap }) {
    this.el = {
      swatch: $("insSwatch"), id: $("insId"), kind: $("insKind"), mode: $("insMode"), state: $("insState"),
      empty: $("insEmpty"), body: $("insBody"), combat: $("combatRow"),
      tgtId: $("tgtId"), tgtEmpty: $("tgtEmpty"), tgtBody: $("tgtBody"),
      tgtSwap: $("tgtSwapBtn"), tgtClear: $("tgtClearBtn"),
    };
    this.ro = Object.fromEntries(RO.map((k) => [k, $(`ro-${k}`)]));
    this.tg = Object.fromEntries(TG.map((k) => [k, $(`tg-${k}`)]));
    this.el.tgtClear.addEventListener("click", () => onClearTarget());
    this.el.tgtSwap.addEventListener("click", () => onSwap());
    this._last = { id: undefined, color: "", mode: "", state: "", tgtId: undefined };
  }

  // sel / tgt: { id, kind, color, s, inRange } or null.
  update(sel, tgt) {
    const e = this.el, last = this._last;
    if (!sel) {
      if (last.id !== null) {
        last.id = null;
        e.empty.hidden = false;
        e.body.hidden = true;
        e.kind.hidden = e.mode.hidden = e.state.hidden = true;
        setText(e.id, "—");
        e.swatch.style.background = "";
        last.color = last.mode = last.state = "";
        last.tgtId = undefined;
      }
      return;
    }
    const { s } = sel;
    if (last.id !== sel.id) {
      last.id = sel.id;
      e.empty.hidden = true;
      e.body.hidden = false;
      e.kind.hidden = e.mode.hidden = false;
      setText(e.id, String(sel.id));
      setText(e.kind, sel.kind);
    }
    if (last.color !== sel.color) { last.color = sel.color; e.swatch.style.background = sel.color; }
    if (last.mode !== s.control_mode) {
      last.mode = s.control_mode;
      e.mode.textContent = s.control_mode;
      e.mode.className = `tag ${s.control_mode}`;
    }
    const st = !s.alive ? "dead" : !sel.inRange ? "stale" : "";
    if (last.state !== st) {
      last.state = st;
      e.state.hidden = !st;
      e.state.className = `tag ${st}`;
      e.state.textContent = st === "dead" ? "destroyed" : "no data";
      e.body.classList.toggle("stale", st === "stale");
    }

    const r = this.ro;
    setText(r.alt, s.alt_hae_m.toFixed(0));
    setText(r.vs, fmtSigned(s.vel_u_mps, 1));
    setText(r.tas, s.tas_mps.toFixed(1));
    setText(r.mach, s.mach.toFixed(2));
    setText(r.nz, fmtNum(s.nz_g, 2));
    setText(r.aoa, fmtNum(s.alpha_rad * RAD2DEG, 1));
    setText(r.hdg, headingDeg(s.psi_rad).toFixed(1));
    setText(r.pitch, fmtNum(s.theta_rad * RAD2DEG, 1));
    setText(r.roll, fmtNum(s.phi_rad * RAD2DEG, 1));

    // Combat state only for aircraft that have any (formation / quad runs don't).
    const combat = s.hit_points_max > 0 || s.ammo > 0;
    e.combat.hidden = !combat;
    if (combat) {
      setText(r.team, `Team ${s.team}`);
      r.team.style.color = teamColor(s.team);
      const frac = s.hit_points_max > 0 ? Math.max(0, Math.min(1, s.hit_points / s.hit_points_max)) : 0;
      r.hpFill.style.width = `${(frac * 100).toFixed(1)}%`;
      r.hpFill.className = `hp-fill${frac < 0.3 ? " low" : frac < 0.6 ? " mid" : ""}`;
      setText(r.hp, s.alive ? `${s.hit_points.toFixed(0)} / ${s.hit_points_max.toFixed(0)}` : "destroyed");
      setText(r.ammo, String(s.ammo));
    }

    setText(r.time, s.t.toFixed(2));
    setText(r.lat, s.lat_deg.toFixed(6));
    setText(r.lon, s.lon_deg.toFixed(6));
    setText(r.altHae, s.alt_hae_m.toFixed(1));
    setText(r.e, fmtNum(s.enu_e, 1));
    setText(r.n, fmtNum(s.enu_n, 1));
    setText(r.u, fmtNum(s.enu_u, 1));
    setText(r.gs, Math.hypot(s.vel_e_mps, s.vel_n_mps).toFixed(1));

    this._target(sel, tgt);
  }

  _target(sel, tgt) {
    const e = this.el, last = this._last;
    const id = tgt ? tgt.id : null;
    if (last.tgtId !== id) {
      last.tgtId = id;
      e.tgtEmpty.hidden = !!tgt;
      e.tgtBody.hidden = !tgt;
      e.tgtSwap.hidden = e.tgtClear.hidden = !tgt;
      setText(e.tgtId, tgt ? String(tgt.id) : "");
      e.tgtId.style.color = tgt ? tgt.color : "";
    }
    if (!tgt) return;
    const g = relativeGeometry(sel.s, tgt.s);
    const t = this.tg;
    const deg = (rad) => (Number.isFinite(rad) ? `${(rad * RAD2DEG).toFixed(0)}°` : "—");
    setText(t.range, fmtRange(g.range));
    setText(t.closure, Number.isFinite(g.closure) ? `${fmtSigned(g.closure, 0)} m/s` : "—");
    setText(t.ata, deg(g.ata));
    setText(t.aspect, deg(g.aspect));
    setText(t.relalt, `${fmtSigned(g.relAlt, 0)} m`);
  }
}
