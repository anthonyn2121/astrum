// Race helpers shared by the viewer and the setup pages: the course JSON of a
// race run (astrum_race::CourseView — what the API's `course` fields carry) and
// the drone's progress along it, read from the run's events.
//
// The course is untrusted input: `parseCourse` keeps only a well-formed one.

// The most gates one course may have, and the largest coordinate / dimension
// accepted (metres): far beyond any real course, they only bound what a
// malformed reply could make the scene build.
const MAX_GATES = 512;
const MAX_ABS_M = 1e6;

const num = (v) => typeof v === "number" && Number.isFinite(v);

// Validate a `course` reply: `{ seed, opening_m, bar_m, top_bar_m, length_m,
// gates: [{ index, east_m, north_m, up_m, yaw_rad }] }` (`length_m`, the course's
// total length, may be missing). Returns the same shape with the gates sorted
// by index and `length_m` always a number (the polyline through the gate centres
// when the reply has none), or null for anything else (including `null`,
// which a non-race scenario carries, and a course without gates).
export function parseCourse(raw) {
  if (!raw || typeof raw !== "object" || !Array.isArray(raw.gates)) return null;
  if (raw.gates.length === 0 || raw.gates.length > MAX_GATES) return null;
  const dims = [raw.opening_m, raw.bar_m, raw.top_bar_m];
  if (!dims.every((d) => num(d) && d > 0 && d < MAX_ABS_M)) return null;
  const gates = [];
  for (const g of raw.gates) {
    if (!g || !Number.isInteger(g.index) || g.index < 0) return null;
    const pos = [g.east_m, g.north_m, g.up_m];
    if (!pos.every((v) => num(v) && Math.abs(v) < MAX_ABS_M) || !num(g.yaw_rad)) return null;
    gates.push({ index: g.index, east_m: g.east_m, north_m: g.north_m, up_m: g.up_m, yaw_rad: g.yaw_rad });
  }
  gates.sort((a, b) => a.index - b.index);
  return {
    seed: num(raw.seed) ? raw.seed : null,
    opening_m: raw.opening_m, bar_m: raw.bar_m, top_bar_m: raw.top_bar_m,
    length_m: num(raw.length_m) && raw.length_m >= 0 ? raw.length_m : polylineLength(gates),
    gates,
  };
}

// The length of the path through the gate centres (m).
function polylineLength(gates) {
  let total = 0;
  for (let i = 1; i < gates.length; i++) {
    total += Math.hypot(gates[i].east_m - gates[i - 1].east_m, gates[i].north_m - gates[i - 1].north_m,
      gates[i].up_m - gates[i - 1].up_m);
  }
  return total;
}

// A string that is equal for equal courses: the viewer re-reads a run's course
// every poll and rebuilds the scene's gates only when it changed.
export function courseKey(course) {
  return course ? JSON.stringify([course.seed, course.length_m, course.opening_m, course.bar_m, course.top_bar_m, course.gates]) : "";
}

// The gate the drone has to fly next at sim time `t`: one past the highest gate
// it has passed (`gate-passed` events at or before `t`; a miss leaves the gate
// pending, so those events do not count). `events` is the store's list, sorted
// by time; `aircraftId` restricts to one aircraft's passes (any, when null).
// `gateCount` caps the result: a finished course has `gateCount` pending.
export function pendingGateAt(events, t, aircraftId, gateCount) {
  let pending = 0;
  for (const e of events) {
    if (e.t > t) break;
    if (e.kind !== "gate-passed" || !Number.isInteger(e.gate)) continue;
    if (aircraftId != null && e.aircraft_id !== aircraftId) continue;
    pending = Math.max(pending, e.gate + 1);
  }
  return Math.min(pending, gateCount);
}

// The race standings at sim time `t`, from the store's events (sorted by time):
// one row per aircraft in `ids`, in place order, and whether the run has ended.
// A row is { id, passed, total, finished, finishT, crashed, outcome, lastPassT,
// place }: `passed` counts distinct gates passed (`gate-passed`; a miss does not
// count), `finished` is every gate passed (`finishT`: the time of the last
// pass), `crashed` a `crashed` event (`outcome`: its RaceOutcome name).
// Places follow the sim's rule: the finishers by finish time, then the rest by
// gates passed (more first) and, on a tie, by the time of their last pass
// (earlier first), then by id. `winner` is the aircraft id in first place if it
// finished, else null; it is only meaningful once `ended` (a `run-ended` event
// at or before `t`).
export function raceStandings(events, t, ids, gateCount) {
  const rows = new Map(ids.map((id) => [id, {
    id, passed: new Set(), total: gateCount, finished: false, finishT: null,
    crashed: false, outcome: null, lastPassT: null, place: 0,
  }]));
  let ended = false;
  for (const e of events) {
    if (e.t > t) break;
    if (e.kind === "run-ended") { ended = true; continue; }
    const row = rows.get(e.aircraft_id);
    if (!row) continue;
    if (e.kind === "gate-passed" && Number.isInteger(e.gate)) {
      row.passed.add(e.gate);
      row.lastPassT = e.t;
      if (row.passed.size >= gateCount && !row.finished) { row.finished = true; row.finishT = e.t; }
    } else if (e.kind === "crashed") {
      row.crashed = true;
      row.outcome = e.outcome ?? null;
    }
  }
  const out = [...rows.values()].map((r) => ({ ...r, passed: r.passed.size }));
  out.sort((a, b) => {
    if (a.finished !== b.finished) return a.finished ? -1 : 1;
    if (a.finished) return a.finishT - b.finishT || a.id - b.id;
    return b.passed - a.passed || (a.lastPassT ?? Infinity) - (b.lastPassT ?? Infinity) || a.id - b.id;
  });
  out.forEach((r, i) => { r.place = i + 1; });
  return { rows: out, ended, winner: out.length && out[0].finished ? out[0].id : null };
}

// The course lengths the UI accepts (m), and the shortest leg a default envelope
// draws (m): a course of N gates has N - 1 legs of about length / (N - 1).
export const COURSE_LENGTH_MIN_M = 100;
export const COURSE_LENGTH_MAX_M = 5000;
export const LEG_MIN_M = 15;

// The time limit a longer course needs: the scenario's `maxS` scaled by
// length / `ownLength`, rounded up to 10 s, never below `maxS`. Null when it
// would be the scenario's own anyway (a course no longer than its own, or a
// scenario with no limit).
export function scaledMaxTime(maxS, ownLength, length) {
  if (!(maxS > 0) || !(ownLength > 0) || !(length > ownLength)) return null;
  const scaled = Math.max(maxS, Math.ceil((maxS * length / ownLength) / 10) * 10);
  return scaled > maxS ? scaled : null;
}
