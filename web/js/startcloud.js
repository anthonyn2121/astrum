// The starting positions of a sweep on a TopDown canvas. The base scenario's
// aircraft are arrows: muted, or in their series colour when the sweep moves
// them. Each planned run adds one dot per moving aircraft at its start, so the
// spread of the draws and of the grid shows; a dot carries a heading tick when
// the run turns the aircraft. The legend counts each moving aircraft's starts;
// the tooltip shows a dot's run values (keys and numbers) or a base arrow's pose.
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

// East and north metres of (lat, lon) from the origin (lat0, lon0).
function enuOf(lat, lon, lat0, lon0) {
  return {
    e: (lon - lon0) * M_PER_DEG_LON_EQUATOR * Math.cos((lat0 * Math.PI) / 180),
    n: (lat - lat0) * M_PER_DEG_LAT,
  };
}

// A run value's key in short form: `aircraft.7.offset.forward_m` → `#7 forward_m`.
const shortKey = (key) => key.replace(/^aircraft\.(\d+)\.(offset\.)?/, "#$1 ");

// The colours of this page, read once per call: the muted grey and the series
// colour of an aircraft id (by id, so an aircraft keeps its colour as others change).
function palette() {
  const css = getComputedStyle(document.documentElement);
  const get = (name) => css.getPropertyValue(name).trim();
  const series = SERIES.map(get);
  return { muted: get("--muted"), colorOf: (id) => series[id % series.length] };
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

  #layers({ muted, colorOf }) {
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
    const dots = [];
    for (const run of this.#plan?.initial ?? []) {
      for (const s of run.aircraft) {
        const b = base.get(s.id);
        // A tick only where the run turns the aircraft away from its base heading.
        const turned = b && Math.abs(((s.heading_deg - b.heading_deg + 540) % 360) - 180) > HEADING_EPS_DEG;
        dots.push({
          ...enuOf(s.lat_deg, s.lon_deg, lat0, lon0),
          heading_deg: turned ? s.heading_deg : undefined,
          color: colorOf(s.id),
          data: { kind: "start", run: run.run, id: s.id },
        });
      }
    }
    return [{ kind: "arrows", items: arrows }, { kind: "dots", items: dots }];
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
