// The environment and scenario steps of a studio page: the environment radios,
// the scenario radios under the picked environment (only flyable scenarios can
// be picked), and the scenario's document once it is read. Scenario names,
// summaries and errors are catalog text: they reach the DOM as text only.
//
// `onChange(entry, doc, error)` reports every pick: `entry` is the picked
// ScenarioEntry (null when none), `doc` its ScenarioDoc once read (null while it
// is read, and when nothing is picked), `error` why its doc could not be read.

import { getEnvironments, getScenarios, getScenario } from "./api.js";
import { h, showError } from "./dom.js";

function fleetSummary(aircraft) {
  const counts = new Map();
  for (const a of aircraft) counts.set(a.model, (counts.get(a.model) || 0) + 1);
  const models = [...counts].map(([m, n]) => (n > 1 ? `${m} ×${n}` : m)).join(", ");
  return `${aircraft.length} aircraft${models ? `: ${models}` : ""}`;
}

export class ScenarioPicker {
  #els;
  #onChange;
  #environments = [];
  #scenarios = [];
  #envId = null;
  #entry = null; // the picked ScenarioEntry, if any

  constructor({ envList, scenarioList, scenarioEmpty, scenarioError, onChange }) {
    this.#els = { envList, scenarioList, scenarioEmpty, scenarioError };
    this.#onChange = onChange;
  }

  // Read the environments and scenarios and show them; nothing is picked.
  async load() {
    const [environments, scenarios] = await Promise.all([getEnvironments(), getScenarios()]);
    this.#environments = Array.isArray(environments) ? environments : [];
    this.#scenarios = Array.isArray(scenarios) ? scenarios : [];
    this.#envId = this.#environments[0]?.id ?? null;
    this.#renderEnvironments();
    this.#renderScenarios();
  }

  // Pick the scenario `name`, as a click would. Resolves to true once its doc
  // has been reported, false when it is not available (the reason is shown) or
  // another pick came first.
  pick(name) {
    const s = this.#scenarios.find((x) => x.name === name);
    if (!s || typeof s.error === "string") {
      const reason = s ? s.error : "no such scenario in the config directory";
      showError(this.#els.scenarioError, `Scenario "${name}" is not available: ${reason}`);
      return Promise.resolve(false);
    }
    this.#envId = s.environment;
    this.#renderEnvironments();
    const done = this.#select(s);
    this.#renderScenarios();
    return done;
  }

  #renderEnvironments() {
    this.#els.envList.replaceChildren();
    for (const env of this.#environments) {
      const input = h("input", { type: "radio", name: "env", value: env.id });
      input.checked = env.id === this.#envId;
      input.addEventListener("change", () => {
        this.#envId = env.id;
        this.#renderScenarios();
        this.#select(null);
      });
      this.#els.envList.append(h("label", { class: "choice" },
        input,
        h("span", { class: "choice-body" },
          h("span", { class: "choice-title", text: env.name }),
          h("span", { class: "choice-text muted", text: env.description }))));
    }
  }

  #renderScenarios() {
    const { scenarioList, scenarioEmpty } = this.#els;
    scenarioList.replaceChildren();
    const list = this.#scenarios.filter((s) => s.environment === this.#envId);
    scenarioEmpty.hidden = list.length > 0;
    for (const s of list) {
      const invalid = typeof s.error === "string";
      const input = h("input", { type: "radio", name: "scenario", value: s.name, disabled: invalid });
      input.checked = this.#entry?.name === s.name;
      input.addEventListener("change", () => this.#select(s));
      const body = h("span", { class: "choice-body" },
        h("span", { class: "choice-title", text: s.name }),
        s.summary ? h("span", { class: "choice-text", text: s.summary }) : null,
        invalid
          ? h("span", { class: "choice-text field-error", text: s.error })
          : h("span", { class: "choice-text muted", text: fleetSummary(s.aircraft) }));
      scenarioList.append(h("label", { class: invalid ? "choice disabled" : "choice", title: s.path }, input, body));
    }
  }

  // Make `entry` the pick (null clears it) and report it; a valid entry's doc
  // follows once read. Resolves to whether this pick's doc was reported.
  async #select(entry) {
    this.#entry = entry;
    if (entry) showError(this.#els.scenarioError, "");
    this.#onChange(entry, null, null);
    if (!entry) return false;
    try {
      const doc = await getScenario(entry.name);
      if (this.#entry !== entry) return false; // another one was picked meanwhile
      this.#onChange(entry, doc, null);
      return true;
    } catch (e) {
      if (this.#entry !== entry) return false;
      this.#onChange(entry, null, e.message || String(e));
      return false;
    }
  }
}
