// The variation of a sweep: its random draws (`perturb`), its grid (`vary`) and
// the draws' settings (samples, sample_seed, min_separation_m, min_alt_hae_m),
// over each aircraft's starting position and attitude. An offset is in the
// aircraft's own frame: forward along its heading, right of it, up; heading,
// climb and bank are turned about it (crates/astrum-sim/src/perturb.rs).
//
// The rows are a model: an edit updates it in place, so no row is rebuilt while
// typing and focus stays put. Rows are rendered again only when one is added or
// removed, or when a scenario or a saved sweep is set. Catalog and saved text
// reach the DOM as text only.
//
// `onChange()` runs after every edit. `fields()` gives the request's variation
// fields, or null while a field is invalid (every problem is marked inline).

import { h, showError } from "./dom.js";

// The six offset axes: the request's key and the label with its unit.
const AXES = [
  { key: "forward_m", label: "Forward (m)" },
  { key: "right_m", label: "Right (m)" },
  { key: "up_m", label: "Up (m)" },
  { key: "heading_deg", label: "Heading (°)" },
  { key: "climb_deg", label: "Climb (°)" },
  { key: "bank_deg", label: "Bank (°)" },
];
// A draw's distribution on an axis: none, or the request's spread.
const SPREADS = [
  { value: "none", label: "none" },
  { value: "uniform", label: "± uniform" },
  { value: "normal", label: "σ normal" },
];
const DEFAULT_SAMPLES = 10;
const MAX_VALUES = 50;

// Show `msg` on `node` (empty clears it) and flag `input`; true when no problem.
function mark(node, input, msg) {
  showError(node, msg);
  input?.setAttribute("aria-invalid", String(Boolean(msg)));
  return !msg;
}

// The whole number `text` holds, when it is at least `min`; else undefined.
function whole(text, min) {
  const t = text.trim();
  const x = Number(t);
  return t !== "" && Number.isInteger(x) && x >= min ? x : undefined;
}

// An optional number of at least 0: null when empty, NaN when refused.
function optionalNumber(text) {
  const t = text.trim();
  if (t === "") return null;
  const x = Number(t);
  return Number.isFinite(x) && x >= 0 ? x : NaN;
}

// The numbers in a grid row's Values text (commas or spaces), or why not.
function parseValues(text) {
  const tokens = text.split(/[\s,]+/).filter((t) => t !== "");
  if (tokens.length === 0) return { error: "Enter at least one value." };
  if (tokens.length > MAX_VALUES) return { error: `Enter at most ${MAX_VALUES} values.` };
  const numbers = tokens.map(Number);
  const bad = tokens.find((_, i) => !Number.isFinite(numbers[i]));
  return bad === undefined ? { numbers } : { error: `"${bad}" is not a number.` };
}

// The model of a request's spread on one axis.
function spreadOf(spread) {
  if (spread && "uniform" in spread) return { dist: "uniform", amount: String(spread.uniform) };
  if (spread && "normal" in spread) return { dist: "normal", amount: String(spread.normal) };
  return { dist: "none", amount: "" };
}

// The aircraft and axis of a `aircraft.<id>.offset.<axis>` key, or null.
function parseVaryKey(key) {
  const m = /^aircraft\.(\d+)\.offset\.(\w+)$/.exec(key);
  return m && AXES.some((a) => a.key === m[2]) ? { id: Number(m[1]), axis: m[2] } : null;
}

// A labelled number input with its error line (hidden until a problem).
function numeric(id, label, attrs, hint) {
  const input = h("input", { id, type: "number", step: "any", inputmode: "decimal", ...attrs });
  const error = h("span", { class: "field-error", role: "alert", hidden: true });
  const node = h("div", { class: "field" },
    h("label", { class: "field-label", for: id, text: label }),
    input,
    hint ? h("span", { class: "muted small", text: hint }) : null,
    error);
  return { input, error, node };
}

export class VariationForm {
  #onChange;
  #fleet = [];   // the scenario's aircraft, [{id, model}] by id
  #draws = [];   // [{key, aircraft: Set<id>, rigid, axes: [{dist, amount}] in AXES order, ui}]
  #grid = [];    // [{key, aircraft, axis, values: the text as typed, ui}]
  #keys = 0;     // row keys (DOM ids)
  #empty;        // shown while no scenario is picked
  #notice;       // rows a scenario or a saved sweep could not keep
  #drawList;
  #addDraw;
  #shared;       // the draws' settings, shown with the draws
  #samples;
  #seed;
  #minSep;
  #minAlt;
  #gridList;
  #addRow;

  constructor({ host, onChange }) {
    this.#onChange = onChange;
    this.#empty = h("p", { class: "muted", text: "Pick a scenario first." });
    this.#notice = h("p", { class: "form-error", role: "alert", hidden: true });

    this.#drawList = h("div", { class: "var-list" });
    this.#addDraw = h("button", { type: "button", text: "Add random draw" });
    this.#addDraw.addEventListener("click", () => {
      this.#draws.push({
        key: ++this.#keys, aircraft: new Set(), rigid: false,
        axes: AXES.map(() => ({ dist: "none", amount: "" })),
      });
      this.#render();
      this.#changed();
    });

    const samples = numeric("varSamples", "Samples", { min: "1", step: "1", inputmode: "numeric", value: String(DEFAULT_SAMPLES) },
      "each sample is one set of draws, the same at every grid point");
    const seed = numeric("varSeed", "Seed", { min: "0", step: "1", inputmode: "numeric", value: "0" });
    const minSep = numeric("varMinSep", "Min separation (m)", { min: "0", placeholder: "none" },
      "a draw that brings two aircraft closer is redrawn");
    const minAlt = numeric("varMinAlt", "Min altitude (m HAE)", { min: "0", placeholder: "none" },
      "a draw that brings an aircraft lower is redrawn");
    for (const f of [samples, seed, minSep, minAlt]) f.input.addEventListener("input", () => this.#changed());
    this.#samples = samples.input;
    this.#seed = seed.input;
    this.#minSep = minSep.input;
    this.#minAlt = minAlt.input;
    this.#shared = { error: { samples: samples.error, seed: seed.error, minSep: minSep.error, minAlt: minAlt.error } };
    this.#shared.node = h("div", { class: "run-options var-shared" },
      samples.node, seed.node, minSep.node, minAlt.node);

    this.#gridList = h("div", { class: "var-list" });
    this.#addRow = h("button", { type: "button", text: "Add grid row" });
    this.#addRow.addEventListener("click", () => {
      this.#grid.push({ key: ++this.#keys, aircraft: this.#fleet[0]?.id ?? null, axis: AXES[0].key, values: "" });
      this.#render();
      this.#changed();
    });

    host.replaceChildren(
      this.#notice,
      this.#empty,
      h("section", { class: "var-part", "aria-labelledby": "varDrawsHead" },
        h("h3", { id: "varDrawsHead", text: "Random draws" }),
        h("p", { class: "muted small var-help", text: "A draw moves or turns each listed aircraft by a random offset in its own frame, and every grid point gets the same draws (common random numbers)." }),
        this.#drawList,
        this.#addDraw,
        this.#shared.node),
      h("section", { class: "var-part", "aria-labelledby": "varGridHead" },
        h("h3", { id: "varGridHead", text: "Grid" }),
        h("p", { class: "muted small var-help", text: "The grid is the product of its rows: every combination of their values is a grid point." }),
        this.#gridList,
        this.#addRow));
    this.#render();
  }

  // Set the scenario's aircraft (`doc` is its GET /api/scenarios/<name> reply;
  // null clears them). Draws and rows naming an aircraft it lacks are removed.
  setScenario(doc) {
    this.#fleet = doc
      ? [...(doc.fleet ?? [])].map((a) => ({ id: a.id, model: a.model })).sort((a, b) => a.id - b.id)
      : [];
    this.#note(doc ? this.#prune() : 0);
    this.#render();
  }

  // The request's variation fields ({perturb, vary, samples, sample_seed,
  // min_separation_m, min_alt_hae_m}, empty ones left out), or null while a
  // field is invalid. Every problem is marked inline.
  fields() {
    let ok = true;
    const perturb = [];
    for (const g of this.#draws) {
      const ids = [...g.aircraft].sort((a, b) => a - b);
      ok = mark(g.ui.aircraftError, null, ids.length ? "" : "Pick at least one aircraft.") && ok;
      const spec = {};
      let anyAxis = false;
      AXES.forEach((axis, i) => {
        const a = g.axes[i];
        const ui = g.ui.axes[i];
        if (a.dist === "none") {
          ok = mark(ui.error, ui.amount, "") && ok;
          return;
        }
        anyAxis = true;
        const x = a.amount.trim() === "" ? NaN : Number(a.amount);
        const good = Number.isFinite(x) && x > 0;
        ok = mark(ui.error, ui.amount, good ? "" : "Enter a number above 0.") && ok;
        if (good) spec[axis.key] = { [a.dist]: x };
      });
      ok = mark(g.ui.axesError, null, anyAxis ? "" : "Pick a distribution for at least one axis.") && ok;
      if (ids.length && anyAxis) perturb.push({ aircraft: ids, ...(g.rigid ? { rigid: true } : {}), ...spec });
    }

    const hasDraws = this.#draws.length > 0;
    const samples = whole(this.#samples.value, 1);
    const seed = whole(this.#seed.value, 0);
    const minSep = optionalNumber(this.#minSep.value);
    const minAlt = optionalNumber(this.#minAlt.value);
    const err = this.#shared.error;
    ok = mark(err.samples, this.#samples, hasDraws && samples === undefined ? "Enter a whole number, at least 1." : "") && ok;
    ok = mark(err.seed, this.#seed, hasDraws && seed === undefined ? "Enter a whole number, at least 0." : "") && ok;
    ok = mark(err.minSep, this.#minSep, hasDraws && Number.isNaN(minSep) ? "Enter a number of at least 0, or leave it empty." : "") && ok;
    ok = mark(err.minAlt, this.#minAlt, hasDraws && Number.isNaN(minAlt) ? "Enter a number of at least 0, or leave it empty." : "") && ok;

    const vary = {};
    const seen = new Set();
    for (const r of this.#grid) {
      const pair = `${r.aircraft}/${r.axis}`;
      const dup = seen.has(pair);
      seen.add(pair);
      const { numbers, error } = parseValues(r.values);
      ok = mark(r.ui.axisError, r.ui.axis, dup ? "This aircraft and axis already have a grid row." : "") && ok;
      ok = mark(r.ui.valuesError, r.ui.values, error ?? "") && ok;
      if (!dup && !error) vary[`aircraft.${r.aircraft}.offset.${r.axis}`] = numbers;
    }

    if (!ok) return null;
    const out = {};
    if (perturb.length) {
      out.perturb = perturb;
      out.samples = samples;
      if (seed) out.sample_seed = seed;
      if (minSep !== null) out.min_separation_m = minSep;
      if (minAlt !== null) out.min_alt_hae_m = minAlt;
    }
    if (Object.keys(vary).length) out.vary = vary;
    return out;
  }

  // Set the form from a request (a saved sweep's): its draws, grid rows and
  // settings. Rows naming an aircraft the scenario lacks are removed, and say so.
  setFields(request) {
    this.#draws = (request.perturb ?? []).map((p) => ({
      key: ++this.#keys,
      aircraft: new Set(p.aircraft),
      rigid: p.rigid === true,
      axes: AXES.map(({ key }) => spreadOf(p[key])),
    }));
    this.#grid = [];
    let unplaced = 0;
    for (const [key, values] of Object.entries(request.vary ?? {})) {
      const pos = parseVaryKey(key);
      if (pos) this.#grid.push({ key: ++this.#keys, aircraft: pos.id, axis: pos.axis, values: values.join(", ") });
      else unplaced += 1;
    }
    const dropped = this.#prune();
    this.#samples.value = String(request.samples || DEFAULT_SAMPLES);
    this.#seed.value = String(request.sample_seed ?? 0);
    this.#minSep.value = request.min_separation_m == null ? "" : String(request.min_separation_m);
    this.#minAlt.value = request.min_alt_hae_m == null ? "" : String(request.min_alt_hae_m);
    this.#note(dropped + unplaced);
    this.#render();
  }

  // Remove the draws and rows that name an aircraft the scenario lacks; the
  // number removed.
  #prune() {
    const has = (id) => this.#fleet.some((a) => a.id === id);
    const before = this.#draws.length + this.#grid.length;
    this.#draws = this.#draws.filter((g) => [...g.aircraft].every(has));
    this.#grid = this.#grid.filter((r) => has(r.aircraft));
    return before - this.#draws.length - this.#grid.length;
  }

  #note(n) {
    const one = n === 1;
    showError(this.#notice, n === 0 ? "" : `${n} variation ${one ? "row" : "rows"} could not be kept for this scenario and ${one ? "was" : "were"} removed.`);
  }

  #changed() {
    this.#onChange();
  }

  #render() {
    this.#empty.hidden = this.#fleet.length > 0;
    this.#addDraw.disabled = this.#fleet.length === 0;
    this.#addRow.disabled = this.#fleet.length === 0;
    this.#shared.node.hidden = this.#draws.length === 0;
    this.#drawList.replaceChildren(...this.#draws.map((g, i) => this.#drawCard(g, i + 1)));
    this.#gridList.replaceChildren(...this.#grid.map((r, i) => this.#rowCard(r, i + 1)));
  }

  #drawCard(g, n) {
    const id = `vd${g.key}`;
    const boxes = this.#fleet.map((a) => {
      const box = h("input", { type: "checkbox", id: `${id}-a${a.id}`, checked: g.aircraft.has(a.id) });
      box.addEventListener("change", () => {
        if (box.checked) g.aircraft.add(a.id);
        else g.aircraft.delete(a.id);
        this.#changed();
      });
      return h("label", { class: "var-check" }, box, `#${a.id} ${a.model}`);
    });

    const rigid = h("input", { type: "checkbox", id: `${id}-rigid`, checked: g.rigid });
    rigid.addEventListener("change", () => {
      g.rigid = rigid.checked;
      this.#changed();
    });

    const axes = AXES.map((axis, i) => {
      const a = g.axes[i];
      const sid = `${id}-${axis.key}`;
      const select = h("select", { id: sid }, ...SPREADS.map((s) => h("option", { value: s.value, text: s.label })));
      select.value = a.dist;
      const amount = h("input", { type: "number", min: "0", step: "any", inputmode: "decimal", "aria-label": `${axis.label} amount` });
      amount.value = a.amount;
      amount.hidden = a.dist === "none";
      const error = h("span", { class: "field-error", role: "alert", hidden: true });
      select.addEventListener("change", () => {
        a.dist = select.value;
        amount.hidden = a.dist === "none";
        this.#changed();
      });
      amount.addEventListener("input", () => {
        a.amount = amount.value;
        this.#changed();
      });
      return {
        error, amount,
        node: h("div", { class: "var-axis" },
          h("label", { class: "field-label", for: sid, text: axis.label }),
          select, amount, error),
      };
    });

    const aircraftError = h("span", { class: "field-error", role: "alert", hidden: true });
    const axesError = h("span", { class: "field-error", role: "alert", hidden: true });
    g.ui = { aircraftError, axesError, axes };

    const remove = h("button", { type: "button", class: "mini", text: "Remove", "aria-label": `Remove draw ${n}` });
    remove.addEventListener("click", () => {
      this.#draws = this.#draws.filter((d) => d !== g);
      this.#render();
      this.#changed();
    });

    return h("div", { class: "var-card" },
      h("div", { class: "var-card-head" }, h("span", { class: "var-card-title", text: `Draw ${n}` }), remove),
      h("div", { class: "var-aircraft", role: "group", "aria-label": "Aircraft" },
        h("span", { class: "field-label", text: "Aircraft" }), ...boxes),
      aircraftError,
      h("label", { class: "var-check" }, rigid, "Move as one rigid group"),
      h("div", { class: "var-axes" }, ...axes.map((x) => x.node)),
      axesError);
  }

  #rowCard(r, n) {
    const id = `vg${r.key}`;
    const aircraft = h("select", { id: `${id}-aircraft` },
      ...this.#fleet.map((a) => h("option", { value: String(a.id), text: `#${a.id} ${a.model}` })));
    aircraft.value = String(r.aircraft);
    aircraft.addEventListener("change", () => {
      r.aircraft = Number(aircraft.value);
      this.#changed();
    });

    const axis = h("select", { id: `${id}-axis` }, ...AXES.map((a) => h("option", { value: a.key, text: a.label })));
    axis.value = r.axis;
    axis.addEventListener("change", () => {
      r.axis = axis.value;
      this.#changed();
    });
    const axisError = h("span", { class: "field-error", role: "alert", hidden: true });

    const values = h("input", { id: `${id}-values`, type: "text", placeholder: "-200, 0, 200", autocomplete: "off", spellcheck: "false" });
    values.value = r.values;
    values.addEventListener("input", () => {
      r.values = values.value;
      this.#changed();
    });
    const valuesError = h("span", { class: "field-error", role: "alert", hidden: true });
    r.ui = { axis, axisError, values, valuesError };

    const remove = h("button", { type: "button", class: "mini", text: "Remove", "aria-label": `Remove grid row ${n}` });
    remove.addEventListener("click", () => {
      this.#grid = this.#grid.filter((x) => x !== r);
      this.#render();
      this.#changed();
    });

    return h("div", { class: "var-card" },
      h("div", { class: "var-card-head" }, h("span", { class: "var-card-title", text: `Grid row ${n}` }), remove),
      h("div", { class: "var-fields" },
        h("div", { class: "field" }, h("label", { class: "field-label", for: `${id}-aircraft`, text: "Aircraft" }), aircraft),
        h("div", { class: "field" }, h("label", { class: "field-label", for: `${id}-axis`, text: "Axis" }), axis, axisError),
        h("div", { class: "field var-values" },
          h("label", { class: "field-label", for: `${id}-values`, text: "Values" }),
          values,
          h("span", { class: "muted small", text: `comma or space separated, 1 to ${MAX_VALUES}` }),
          valuesError)));
  }
}
