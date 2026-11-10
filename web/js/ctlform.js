// The per-aircraft controller form: one row per aircraft of a scenario, each
// with a controller (a catalog controller that flies the aircraft's kind, or
// none) and that controller's parameters. The Interactive and Sweeps pages
// share it. Its checks mirror the server's (ControllerSpec::render) so most
// mistakes show before a request; the server's verdict is shown next to the
// parameter or aircraft it names. Catalog text is untrusted: it reaches the DOM
// as text only.
//
// Events: `controllerchange` is dispatched on the host when an aircraft's
// controller is picked.

import { h, showError } from "./dom.js";

const TARGET_UNITS = { heading_deg: "°", alt_hae_m: "m", tas_mps: "m/s" };

const inBounds = (p, x) => !(Number.isFinite(p.min) && x < p.min) && !(Number.isFinite(p.max) && x > p.max);

// Whose initial conditions a target param starts from: the aircraft picked in
// the controller's `aircraft` param (a formation follower holds its leader's
// altitude), else — none picked, or no such param — the controlled aircraft.
function targetSource(row) {
  for (const [, f] of row.fields) {
    if (f.spec.type !== "aircraft") continue;
    const id = Number(f.input.value);
    return f.input.value !== "" && Number.isInteger(id) ? id : row.aircraft.id;
  }
  return row.aircraft.id;
}

export class ControllerForm {
  #host;
  #controllers;
  #fleet = [];       // the scenario's aircraft (with kind), sorted by id
  #rows = new Map(); // aircraft id -> row
  #disabled = false;

  constructor({ host, controllers }) {
    this.#host = host;
    this.#controllers = controllers;
  }

  // Render a row per aircraft of a scenario doc (the reply of
  // GET /api/scenarios/<name>, its aircraft's kind filled in); null clears it.
  setScenario(doc) {
    this.#fleet = doc ? [...doc.scenario.aircraft].sort((a, b) => a.id - b.id) : [];
    this.#rows = new Map();
    this.#host.replaceChildren();
    for (const a of this.#fleet) this.#addRow(a);
    this.#applyDisabled();
  }

  // The request's assignments for the rows that have a controller, and whether
  // every parameter is valid. Every problem is marked inline.
  assignments() {
    let ok = true;
    const assignments = [];
    for (const row of this.#rows.values()) {
      showError(row.error, "");
      if (!row.controller) continue;
      const params = {};
      for (const p of row.controller.params || []) {
        if (!this.#check(row, p)) { ok = false; continue; }
        const { value } = this.#paramValue(row, p);
        if (value !== undefined) params[p.name] = value;
      }
      assignments.push({ aircraft: row.aircraft.id, controller: row.controller.id, params });
    }
    return { ok, assignments };
  }

  // Show the server's verdict next to what it names ("aircraft 2: controller
  // param `leader`: …"). Returns false when it names no row of this form.
  showServerError(msg) {
    const m = /^aircraft (\d+): (?:controller param `([^`]+)`)?/.exec(msg);
    const row = m ? this.#rows.get(Number(m[1])) : null;
    const field = row && m[2] ? row.fields.get(m[2]) : null;
    if (field) {
      showError(field.error, msg);
      field.input.setAttribute("aria-invalid", "true");
      field.input.focus();
      return true;
    }
    if (row) {
      showError(row.error, msg);
      row.select.focus();
      return true;
    }
    return false;
  }

  setDisabled(disabled) {
    this.#disabled = disabled;
    this.#applyDisabled();
  }

  #applyDisabled() {
    for (const control of this.#host.querySelectorAll("select, input, button")) control.disabled = this.#disabled;
  }

  #addRow(a) {
    const flies = this.#controllers.filter((c) => Array.isArray(c.kinds) && c.kinds.includes(a.kind));
    const select = h("select", { id: `ctl-${a.id}`, "aria-label": `Controller for aircraft ${a.id}` },
      h("option", { value: "", text: "none — uncommanded" }),
      ...flies.map((c) => h("option", { value: c.id, text: c.name, title: c.description })));
    const paramsBox = h("div", { class: "params" });
    const error = h("div", { class: "field-error", role: "alert", hidden: true });
    const row = { aircraft: a, select, paramsBox, error, fields: new Map(), controller: null };
    select.addEventListener("change", () => {
      row.controller = this.#controllers.find((c) => c.id === select.value) || null;
      this.#renderParams(row);
      showError(row.error, "");
      this.#host.dispatchEvent(new Event("controllerchange"));
    });
    const team = Number.isInteger(a.team) ? `team ${a.team}` : null;
    this.#host.append(h("div", { class: "fleet-row" },
      h("div", { class: "fleet-head" },
        h("span", { class: "fleet-id", text: `#${a.id}` }),
        h("span", { class: "fleet-model", text: a.model }),
        h("span", { class: "tag", text: a.kind }),
        team ? h("span", { class: "muted small", text: team }) : null,
        h("label", { class: "fleet-ctl" }, h("span", { class: "muted small", text: "Controller" }), select)),
      error,
      paramsBox));
    this.#rows.set(a.id, row);
  }

  // An aircraft's initial value of a parameter target (a scenario aircraft
  // field: heading_deg, alt_hae_m, tas_mps); headings wrapped to [0, 360).
  #initialOf(aircraftId, target) {
    const spec = this.#fleet.find((s) => s.id === aircraftId);
    if (!spec || !(target in TARGET_UNITS) || !Number.isFinite(spec[target])) return null;
    const v = spec[target];
    return target === "heading_deg" ? ((v % 360) + 360) % 360 : v;
  }

  // The catalog default of an `aircraft` param, if it is a valid choice for this
  // row (another aircraft of the fleet) — what the server would use.
  #validAircraftDefault(row, p) {
    const d = p.default;
    return Number.isInteger(d) && d !== row.aircraft.id && this.#fleet.some((o) => o.id === d) ? d : null;
  }

  // Fill each untouched target param from its source aircraft's initial value
  // (the catalog default when there is none or it is out of bounds).
  #prefillTargets(row) {
    const source = targetSource(row);
    for (const [, f] of row.fields) {
      const p = f.spec;
      if (!p.target || f.edited) continue;
      const from = p.target === "alt_hae_m" ? source : row.aircraft.id;
      const initial = this.#initialOf(from, p.target);
      const use = initial != null && inBounds(p, initial);
      f.input.value = use ? String(Math.round(initial * 100) / 100) : p.default != null ? String(p.default) : "";
      f.from.textContent = use ? `from #${from}'s start` : "";
      this.#check(row, p);
    }
  }

  #renderParams(row) {
    row.fields = new Map();
    row.paramsBox.replaceChildren();
    const c = row.controller;
    if (!c) return;
    row.paramsBox.append(h("p", { class: "muted small ctl-desc", text: c.description }));
    const grid = h("div", { class: "param-grid" });
    for (const p of c.params || []) {
      const id = `p-${row.aircraft.id}-${p.name}`;
      const label = h("label", { for: id, class: "field-label", text: p.unit ? `${p.label} (${p.unit})` : p.label });
      let input;
      if (p.type === "choice") {
        input = h("select", { id }, ...(p.choices || []).map((ch) => h("option", { value: ch, text: ch })));
        if (p.default != null) input.value = String(p.default);
      } else if (p.type === "aircraft") {
        const others = this.#fleet.filter((o) => o.id !== row.aircraft.id);
        input = h("select", { id },
          h("option", { value: "", text: "— pick an aircraft —" }),
          ...others.map((o) => h("option", { value: String(o.id), text: `#${o.id} · ${o.model}` })));
        const d = this.#validAircraftDefault(row, p);
        input.value = d != null ? String(d) : "";
      } else {
        input = h("input", {
          id, type: "number", inputmode: "decimal",
          step: p.type === "integer" ? "1" : "any",
          min: Number.isFinite(p.min) ? p.min : null,
          max: Number.isFinite(p.max) ? p.max : null,
        });
        if (p.default != null) input.value = String(p.default);
      }
      const error = h("span", { class: "field-error", hidden: true });
      const from = h("span", {});
      const field = h("div", { class: "field" }, label, input);
      const f = { spec: p, input, error, from, edited: false };
      if (p.target && p.type !== "aircraft" && p.type !== "choice") {
        const hint = h("span", { class: "muted small hint" }, from);
        if (p.default != null) {
          const reset = h("button", { type: "button", class: "mini", title: "Set it to the catalog default", text: "reset" });
          reset.addEventListener("click", () => {
            input.value = String(p.default);
            f.edited = true;
            from.textContent = "";
            this.#check(row, p);
          });
          hint.append(` default ${p.default} `, reset);
        }
        field.append(hint);
      }
      field.append(error);
      input.addEventListener("change", () => {
        f.edited = true;
        if (p.target) from.textContent = "";
        this.#check(row, p);
        if (p.type === "aircraft") this.#prefillTargets(row); // the leader's altitude follows it
      });
      row.fields.set(p.name, f);
      grid.append(field);
    }
    if (!(c.params || []).length) grid.append(h("p", { class: "muted small", text: "No parameters." }));
    row.paramsBox.append(grid);
    this.#prefillTargets(row);
    this.#applyDisabled();
  }

  // The value to send for one parameter: { value } (undefined = the catalog
  // default), or { error }.
  #paramValue(row, p) {
    const raw = row.fields.get(p.name).input.value.trim();
    const fleetIds = this.#fleet.map((a) => a.id);
    if (raw === "") {
      // An aircraft select left on (or set back to) "— pick an aircraft —"
      // asks for a pick, whatever the catalog default; elsewhere empty means
      // the catalog default.
      if (p.type === "aircraft") return { error: "pick an aircraft" };
      return p.default != null ? { value: undefined } : { error: "required" };
    }
    switch (p.type) {
      case "choice":
        return (p.choices || []).includes(raw) ? { value: raw } : { error: `must be one of ${(p.choices || []).join(", ")}` };
      case "aircraft": {
        const id = Number(raw);
        if (!Number.isInteger(id)) return { error: "pick an aircraft" };
        if (id === row.aircraft.id) return { error: "cannot be the controlled aircraft itself" };
        if (!fleetIds.includes(id)) return { error: `aircraft ${id} is not in the scenario` };
        return { value: id };
      }
      default: {
        const x = Number(raw);
        if (!Number.isFinite(x)) return { error: "must be a number" };
        if (p.type === "integer" && !Number.isInteger(x)) return { error: "must be a whole number" };
        const lo = Number.isFinite(p.min) ? p.min : -Infinity, hi = Number.isFinite(p.max) ? p.max : Infinity;
        if (x < lo || x > hi) {
          return { error: Number.isFinite(lo) && Number.isFinite(hi) ? `must be within [${lo}, ${hi}]`
            : Number.isFinite(lo) ? `must be at least ${lo}` : `must be at most ${hi}` };
        }
        return { value: x };
      }
    }
  }

  #check(row, p) {
    const f = row.fields.get(p.name);
    const { error } = this.#paramValue(row, p);
    showError(f.error, error || "");
    f.input.setAttribute("aria-invalid", String(!!error));
    return !error;
  }
}
