// The All runs panel of the sweep page (sweep.html): every run's flight at once
// on a TopDown canvas. Each aircraft's path is coloured by its run's status (ok,
// failed, stopped); the selected run's paths are drawn over the rest, in the
// accent colour, and an X marks where an aircraft died. With "Positions at t"
// each aircraft has a dot at the sim time t of the slider: its track's position
// there, interpolated between samples and held at the ends.
//
// The tracks come from GET /api/sweeps/<id>/tracks, one request at a time, for
// the finished runs not asked for yet (up to CHUNK in a row). While the sweep
// runs, the page's polls say which runs finished and a request goes out at most
// every REFETCH_MS; a finished sweep is asked at once. A run is asked for once.
// Text from the server reaches the DOM as textContent only.

import { getSweepTracks } from "./api.js";
import { $, h, showError } from "./dom.js";
import { TopDown } from "./topdown.js";

const CHUNK = 100; // runs per request
const MAX_RUNS = 2000; // the first runs drawn
const REFETCH_MS = 3000; // while the sweep runs, at most one request this often
const FINISHED = new Set(["ok", "failed", "stopped"]);
const PATH_ALPHA = 0.35;
const SELECTED_WIDTH_PX = 3;
const POSITION_RADIUS_PX = 5;
const TIP_GAP_PX = 14;

const el = {
  canvas: $("tracksCanvas"), tip: $("tracksTip"), empty: $("tracksEmpty"), note: $("tracksNote"),
  error: $("tracksError"), slider: $("timeSlider"), label: $("timeLabel"), play: $("playBtn"),
  speed: $("speedSelect"), paths: $("pathsToggle"), positions: $("positionsToggle"),
};

// The page's colours for the canvas, which cannot read CSS variables.
function palette() {
  const css = getComputedStyle(document.documentElement);
  const get = (name) => css.getPropertyValue(name).trim();
  return { ok: get("--ok"), failed: get("--live"), stopped: get("--warn"), accent: get("--accent") };
}

// The [e, n] of a track at sim time `at`: linear between its samples, held at
// the first and the last one outside them.
function positionAt(track, at) {
  const { t, e, n } = track;
  const last = t.length - 1;
  if (at <= t[0]) return [e[0], n[0]];
  if (at >= t[last]) return [e[last], n[last]];
  let lo = 0;
  let hi = last;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (t[mid] <= at) lo = mid;
    else hi = mid;
  }
  const span = t[hi] - t[lo];
  const f = span > 0 ? (at - t[lo]) / span : 0;
  return [e[lo] + f * (e[hi] - e[lo]), n[lo] + f * (n[hi] - n[lo])];
}

export class SweepTracks {
  #onSelect;
  #view;
  #id = null; // the sweep the data belongs to
  #last = null; // the page's latest view of it
  #runs = new Map(); // run number -> its entry of the tracks reply
  #seen = new Set(); // run numbers asked for
  #selected = null;
  #framed = false;
  #pathLayers = [];
  #max = 0; // the slider's span: the largest sim time fetched
  #t = 0; // the sim time shown
  #busy = false;
  #lastFetch = 0;
  #timer = 0;
  #playing = false;
  #frame = 0;
  #frameAt = 0;

  constructor({ onSelect }) {
    this.#onSelect = onSelect;
    this.#view = new TopDown(el.canvas, {
      onHover: (item, pos) => this.#hover(item, pos),
      onPick: (item) => this.#onSelect(item.data.run),
    });
    el.slider.addEventListener("input", () => this.#seek(Number(el.slider.value)));
    el.play.addEventListener("click", () => (this.#playing ? this.#pause() : this.#play()));
    el.paths.addEventListener("change", () => this.#show());
    el.positions.addEventListener("change", () => this.#show());
    window.addEventListener("pagehide", () => this.#halt());
  }

  // The page's latest view of the sweep: fetch what it has finished, and draw.
  update(view) {
    if (view.id !== this.#id) this.#reset(view.id);
    this.#last = view;
    showError(el.note, view.total > MAX_RUNS ? `Showing the first ${MAX_RUNS} of ${view.total} runs.` : "");
    this.#schedule();
  }

  // Run `n` is selected on the page: its paths are drawn over the rest.
  setSelected(n) {
    this.#selected = n;
    this.#buildPaths();
    this.#show();
  }

  #reset(id) {
    this.#halt();
    this.#id = id;
    this.#runs.clear();
    this.#seen.clear();
    this.#framed = false;
    this.#t = 0;
    this.#changed();
  }

  // Stop what runs on a timer: the play and the wait for the next request.
  #halt() {
    this.#pause();
    clearTimeout(this.#timer);
    this.#timer = 0;
  }

  // The next run numbers to ask for: finished runs in a row, not asked for yet.
  #group() {
    const runs = this.#last?.runs ?? [];
    const wanted = (run) => FINISHED.has(run.status) && run.run <= MAX_RUNS && !this.#seen.has(run.run);
    const first = runs.findIndex(wanted);
    if (first < 0) return [];
    const group = [];
    for (let i = first; i < runs.length && group.length < CHUNK && wanted(runs[i]); i++) {
      group.push(runs[i].run);
    }
    return group;
  }

  // Ask for the next group when the sweep allows it: now, or after the wait.
  #schedule() {
    if (this.#busy || this.#timer || this.#group().length === 0) return;
    const wait = this.#last.state === "running" ? this.#lastFetch + REFETCH_MS - Date.now() : 0;
    if (wait > 0) {
      this.#timer = setTimeout(() => {
        this.#timer = 0;
        this.#schedule();
      }, wait);
    } else {
      this.#fetch();
    }
  }

  // Request the next group and merge the runs it returns. Asked-for runs stay
  // asked for even when they come back without tracks (no bag).
  async #fetch() {
    const group = this.#group();
    const id = this.#id;
    this.#busy = true;
    this.#lastFetch = Date.now();
    for (const n of group) this.#seen.add(n);
    try {
      const reply = await getSweepTracks(id, group[0], group.length);
      if (id === this.#id) {
        for (const run of reply.runs) this.#runs.set(run.run, run);
        showError(el.error, "");
        this.#changed();
      }
    } catch (err) {
      if (id === this.#id) showError(el.error, `Cannot read the tracks: ${err.message || err}`);
    } finally {
      this.#busy = false;
    }
    this.#schedule();
  }

  // Redraw from the runs fetched so far: the slider's span, the empty state, the
  // paths, and the framing on the first run.
  #changed() {
    let max = 0;
    for (const run of this.#runs.values()) max = Math.max(max, run.sim_time_s ?? 0);
    this.#max = max;
    el.slider.max = String(max);
    el.slider.disabled = max === 0;
    el.play.disabled = max === 0;
    el.empty.hidden = this.#runs.size > 0;
    this.#buildPaths();
    this.#seek(Math.min(this.#t, max));
    if (this.#runs.size > 0 && !this.#framed) {
      this.#view.fit();
      this.#framed = true;
    }
  }

  // The paths of the runs fetched: the others, then the selected run's on top.
  #buildPaths() {
    const colors = palette();
    const others = [];
    const selected = [];
    for (const run of [...this.#runs.values()].sort((a, b) => a.run - b.run)) {
      for (const a of run.aircraft) {
        const item = {
          pts: a.e.map((e, i) => [e, a.n[i]]),
          end: a.died_at_s == null ? undefined : "x",
          data: { run: run.run, aircraft: a.id, status: run.status },
        };
        if (run.run === this.#selected) {
          selected.push({ ...item, color: colors.accent, width: SELECTED_WIDTH_PX, alpha: 1 });
        } else {
          others.push({ ...item, color: colors[run.status], alpha: PATH_ALPHA });
        }
      }
    }
    this.#pathLayers = [{ kind: "paths", items: others }, { kind: "paths", items: selected }];
  }

  // One dot per aircraft of the runs fetched, at sim time t.
  #positions() {
    const colors = palette();
    const items = [];
    for (const run of this.#runs.values()) {
      for (const a of run.aircraft) {
        if (a.t.length === 0) continue;
        const [e, n] = positionAt(a, this.#t);
        items.push({
          e,
          n,
          color: colors[run.status],
          radius: POSITION_RADIUS_PX,
          data: { run: run.run, aircraft: a.id, status: run.status },
        });
      }
    }
    return items;
  }

  // The layers the switches ask for: the paths, then the positions on top.
  #show() {
    const layers = [];
    if (el.paths.checked) layers.push(...this.#pathLayers);
    if (el.positions.checked) layers.push({ kind: "dots", items: this.#positions() });
    this.#view.setLayers(layers);
  }

  // Show sim time `t` (within the span) and draw it.
  #seek(t) {
    this.#t = Math.min(Math.max(t, 0), this.#max);
    el.slider.value = String(this.#t);
    el.label.textContent = `t = ${this.#t.toFixed(1)} s`;
    this.#show();
  }

  // Play from the start again when the last press left it at the end.
  #play() {
    if (this.#t >= this.#max) this.#seek(0);
    this.#playing = true;
    el.play.textContent = "Pause";
    this.#frameAt = performance.now();
    this.#frame = requestAnimationFrame((now) => this.#tick(now));
  }

  // One frame of play: t advances by the wall time since the last frame, at the chosen speed.
  #tick(now) {
    const t = this.#t + ((now - this.#frameAt) / 1000) * Number(el.speed.value);
    this.#frameAt = now;
    if (t >= this.#max) {
      this.#seek(this.#max);
      this.#pause();
      return;
    }
    this.#seek(t);
    this.#frame = requestAnimationFrame((next) => this.#tick(next));
  }

  #pause() {
    this.#playing = false;
    cancelAnimationFrame(this.#frame);
    this.#frame = 0;
    el.play.textContent = "Play";
  }

  #hover(item, { x, y }) {
    if (!item) {
      el.tip.hidden = true;
      return;
    }
    const { run, aircraft, status } = item.data;
    const lines = [h("div", { class: "tracks-tip-head", text: `run ${run} · aircraft ${aircraft} · ${status}` })];
    if (el.positions.checked) lines.push(h("div", { text: `t = ${this.#t.toFixed(1)} s` }));
    el.tip.replaceChildren(...lines);
    el.tip.hidden = false;
    // Beside the pointer, flipped to stay inside the canvas.
    let left = x + TIP_GAP_PX;
    let top = y + TIP_GAP_PX;
    if (left + el.tip.offsetWidth > el.canvas.clientWidth) left = x - TIP_GAP_PX - el.tip.offsetWidth;
    if (top + el.tip.offsetHeight > el.canvas.clientHeight) top = y - TIP_GAP_PX - el.tip.offsetHeight;
    el.tip.style.left = `${Math.max(0, left)}px`;
    el.tip.style.top = `${Math.max(0, top)}px`;
  }
}
