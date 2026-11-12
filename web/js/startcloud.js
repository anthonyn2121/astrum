// The starting positions of a sweep on a TopDown canvas. The base scenario's
// aircraft are arrows: muted, or in their series colour when the sweep moves
// them. Each planned run adds one star per moving aircraft at its start (a
// soft halo, the series-coloured disc and a bright core), so the spread of the
// draws and of the grid shows and dense starts glow; a star carries a heading
// tick when the run turns the aircraft. The legend counts each moving
// aircraft's starts; the tooltip shows a star's run values (keys and numbers)
// or a base arrow's pose.
//
// Positions are local east/north metres from the scenario's origin by a flat
// approximation (n = Δlat · 111132.92, e = Δlon · 111412.84 · cos(lat0)): a
// display aid, good to a few metres over tens of km, not geodesy. Names, keys
// and values reach the DOM as text only.

import { h } from "./dom.js";
import { TopDown } from "./topdown.js";

const M_PER_DEG_LAT = 111132.92;
const M_PER_DEG_LON_EQUATOR = 111412.84;
const SERIES = ["--accent", "--ok", "--warn", "--live", "--series-5", "--series-6"];
const TIP_GAP_PX = 14;
const HEADING_EPS_DEG = 0.01;
const STAR_HALO_PX = 5.5; // a start's soft halo
const STAR_BODY_PX = 2.4; // its disc, in the series colour
const STAR_CORE_PX = 1.1; // its core, the series colour toward --star
const STAR_HALO_ALPHA = 0.2;
const STAR_CORE_MIX = 0.7; // how far the core is toward --star

// East and north metres of (lat, lon) from the origin (lat0, lon0).
function enuOf(lat, lon, lat0, lon0) {
  return {
    e: (lon - lon0) * M_PER_DEG_LON_EQUATOR * Math.cos((lat0 * Math.PI) / 180),
    n: (lat - lat0) * M_PER_DEG_LAT,
  };
}

// A run value's key in short form: `aircraft.7.offset.forward_m` → `#7 forward_m`.
const shortKey = (key) => key.replace(/^aircraft\.(\d+)\.(offset\.)?/, "#$1 ");

// `color` (#rgb or #rrggbb) as [r, g, b], or null for any other form.
function rgbOf(color) {
  const m = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(color);
  if (!m) return null;
  const hex = m[1].length === 3 ? [...m[1]].map((c) => c + c).join("") : m[1];
  return [0, 2, 4].map((i) => parseInt(hex.slice(i, i + 2), 16));
}

// A start's three colours from its series colour: the halo (translucent), the
// disc and the core (toward `star`). A colour that is not hex keeps one colour.
function starOf(color, star) {
  const c = rgbOf(color);
  const s = rgbOf(star) ?? [255, 255, 255];
  if (!c) return { halo: color, body: color, core: color };
  const core = c.map((v, i) => Math.round(v + (s[i] - v) * STAR_CORE_MIX));
  return { halo: `rgba(${c.join(", ")}, ${STAR_HALO_ALPHA})`, body: color, core: `rgb(${core.join(", ")})` };
}

// The colours of this page, read once per call: the muted grey, the series
// colour of an aircraft id (by id, so an aircraft keeps its colour as others
// change) and its start's star colours.
function palette() {
  const css = getComputedStyle(document.documentElement);
  const get = (name) => css.getPropertyValue(name).trim();
  const series = SERIES.map(get);
  const stars = series.map((c) => starOf(c, get("--star")));
  return { muted: get("--muted"), colorOf: (id) => series[id % series.length], starOf: (id) => stars[id % stars.length] };
}

export class StartCloud {
  #canvas;
  #legend;
  #tip;
  #view;
  #doc = null; // the picked scenario doc
  #plan = null; // the plan on screen for #doc, or null
  #movingKey = null; // the moving set the view was last framed for (null: none yet)
  #tipItem = null; // the item the tooltip shows

  constructor({ canvas, legend, tip, fitBtn }) {
    this.#canvas = canvas;
    this.#legend = legend;
    this.#tip = tip;
    this.#view = new TopDown(canvas, { onHover: (item, pos) => this.#hover(item, pos) });
    fitBtn.addEventListener("click", () => this.#view.fit());
  }

  // Show the scenario `doc` (null: none) with the plan on screen (null: none,
  // or out of date). The view is framed when the scenario changes and when the
  // set of moving aircraft changes, not on every plan, so a zoom survives.
  set(doc, plan) {
    const changed = doc !== this.#doc;
    this.#doc = doc;
    this.#plan = doc ? plan : null;
    if (changed) this.#movingKey = null;
    const movingKey = this.#plan ? this.#plan.moving.join(",") : null;
    const reframe = changed || (movingKey !== null && movingKey !== this.#movingKey);
    if (movingKey !== null) this.#movingKey = movingKey;
    const colors = palette();
    this.#view.setLayers(this.#layers(colors));
    if (reframe) this.#view.fit();
    this.#renderLegend(colors);
    this.#tipItem = null;
    this.#tip.hidden = true;
  }

  #layers({ muted, colorOf, starOf }) {
    const doc = this.#doc;
    if (!doc) return [];
    const { origin_lat_deg: lat0, origin_lon_deg: lon0 } = doc.scenario.world;
    const base = new Map(doc.scenario.aircraft.map((a) => [a.id, a]));
    const moving = new Set(this.#plan?.moving ?? []);
    const arrows = doc.scenario.aircraft.map((a) => ({
      ...enuOf(a.lat_deg, a.lon_deg, lat0, lon0),
      heading_deg: a.heading_deg,
      color: moving.has(a.id) ? colorOf(a.id) : muted,
      label: `#${a.id}`,
      data: { kind: "base", id: a.id, model: a.model, heading_deg: a.heading_deg, alt_hae_m: a.alt_hae_m },
    }));
    // Each start is three items at one place, one per layer (halo, disc, core),
    // all carrying its data, so a hover on any of them shows the run.
    const halos = [];
    const bodies = [];
    const cores = [];
    for (const run of this.#plan?.initial ?? []) {
      for (const s of run.aircraft) {
        const b = base.get(s.id);
        // A tick only where the run turns the aircraft away from its base heading.
        const turned = b && Math.abs(((s.heading_deg - b.heading_deg + 540) % 360) - 180) > HEADING_EPS_DEG;
        const at = enuOf(s.lat_deg, s.lon_deg, lat0, lon0);
        const star = starOf(s.id);
        const data = { kind: "start", run: run.run, id: s.id };
        halos.push({ ...at, color: star.halo, radius: STAR_HALO_PX, data });
        bodies.push({ ...at, heading_deg: turned ? s.heading_deg : undefined, color: star.body, radius: STAR_BODY_PX, data });
        cores.push({ ...at, color: star.core, radius: STAR_CORE_PX, data });
      }
    }
    return [
      { kind: "arrows", items: arrows },
      { kind: "dots", items: halos },
      { kind: "dots", items: bodies },
      { kind: "dots", items: cores },
    ];
  }

  #renderLegend({ colorOf }) {
    const plan = this.#plan;
    if (!plan) {
      this.#legend.replaceChildren();
      return;
    }
    if (plan.moving.length === 0) {
      this.#legend.replaceChildren(h("span", { class: "muted", text: "No aircraft vary yet — add a random draw or a grid." }));
      return;
    }
    this.#legend.replaceChildren(...plan.moving.map((id) => {
      const starts = plan.initial.filter((run) => run.aircraft.some((a) => a.id === id)).length;
      const base = this.#doc.scenario.aircraft.find((a) => a.id === id);
      const name = base ? `aircraft ${id} (${base.model})` : `aircraft ${id}`;
      const swatch = h("span", { class: "swatch" });
      swatch.style.background = colorOf(id);
      swatch.style.boxShadow = `0 0 6px ${colorOf(id)}`;
      return h("span", { class: "start-legend-row" }, swatch, `${name}: ${starts} ${starts === 1 ? "start" : "starts"}`);
    }));
  }

  #hover(item, pos) {
    if (!item) {
      this.#tip.hidden = true;
      return;
    }
    if (item !== this.#tipItem) {
      this.#tipItem = item;
      this.#tip.replaceChildren(...this.#tipLines(item.data));
    }
    this.#tip.hidden = false;
    this.#placeTip(pos);
  }

  #tipLines(data) {
    if (data.kind === "base") {
      return [h("div", {
        text: `aircraft ${data.id} (${data.model}) · heading ${Math.round(data.heading_deg)}° · ${Math.round(data.alt_hae_m)} m`,
      })];
    }
    const lines = [h("div", { class: "start-tip-head", text: `run ${data.run} · aircraft ${data.id}` })];
    const values = this.#plan?.rows[data.run - 1]?.values ?? {};
    for (const [key, v] of Object.entries(values)) {
      const value = typeof v === "number" ? v.toFixed(1) : String(v);
      lines.push(h("div", { text: `${shortKey(key)}  ${value}` }));
    }
    return lines;
  }

  // Beside the pointer, flipped to stay inside the canvas.
  #placeTip({ x, y }) {
    const tip = this.#tip;
    let left = x + TIP_GAP_PX;
    let top = y + TIP_GAP_PX;
    if (left + tip.offsetWidth > this.#canvas.clientWidth) left = x - TIP_GAP_PX - tip.offsetWidth;
    if (top + tip.offsetHeight > this.#canvas.clientHeight) top = y - TIP_GAP_PX - tip.offsetHeight;
    tip.style.left = `${Math.max(0, left)}px`;
    tip.style.top = `${Math.max(0, top)}px`;
  }
}
