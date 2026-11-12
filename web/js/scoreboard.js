// The race scoreboard (#scoreboard) and winner banner (#raceBanner) over the 3-D
// view: one row per aircraft in place order — colour swatch (the aircraft's colour
// in the scene), id, gates passed / total and its state (flying, finished with its
// time, crashed) — from race.js raceStandings at the playhead, live and in replay.
// A row selects its aircraft. Once the run has ended the banner names the winner.
// The DOM is rebuilt only when what it shows changes. Outcome names come from
// the server and reach the DOM as text only.

import { fmtNum } from "./format.js";

const ORDINAL = ["th", "st", "nd", "rd"];
function ordinal(n) {
  const v = n % 100;
  return `${n}${ORDINAL[(v - 20) % 10] || ORDINAL[v] || ORDINAL[0]}`;
}

function statusText(r) {
  if (r.finished) return `finished ${fmtNum(r.finishT, 2)} s`;
  if (r.crashed) return r.outcome ? `crashed (${r.outcome.replace(/^crashed-/, "")})` : "crashed";
  return "flying";
}

export class Scoreboard {
  constructor({ root, body, banner, onSelect }) {
    this.el = { root, body, banner };
    this.onSelect = onSelect;
    this.key = null;
  }

  // Show the standings (race.js raceStandings) for `selectedId`, or hide the
  // board (null: no course on screen). `colorOf(id)` is the aircraft's CSS colour.
  update(standings, { selectedId, colorOf }) {
    const { root, body, banner } = this.el;
    root.hidden = !standings || standings.rows.length === 0;
    if (root.hidden) { banner.hidden = true; this.key = null; return; }
    const { rows, ended, winner } = standings;
    const key = JSON.stringify([rows.map((r) => [r.id, r.passed, r.total, r.finishT, r.crashed, r.outcome, r.place, colorOf(r.id)]), ended, winner, selectedId]);
    if (key === this.key) return;
    this.key = key;
    body.replaceChildren(...rows.map((r) => {
      const tr = document.createElement("tr");
      tr.className = [r.id === selectedId ? "sel" : "", r.crashed ? "crashed" : "", r.finished ? "finished" : ""].join(" ").trim();
      tr.title = `Select aircraft ${r.id}`;
      tr.addEventListener("click", () => this.onSelect(r.id));
      const cells = [
        ["place", ordinal(r.place)],
        ["id", `#${r.id}`],
        ["gates", `${r.passed}/${r.total}`],
        ["state", statusText(r)],
      ];
      const swatch = document.createElement("td");
      const dot = document.createElement("span");
      dot.className = "sb-dot";
      dot.style.background = colorOf(r.id);
      swatch.append(dot);
      tr.append(swatch);
      for (const [cls, text] of cells) {
        const td = document.createElement("td");
        td.className = `sb-${cls}`;
        td.textContent = text;
        tr.append(td);
      }
      return tr;
    }));
    banner.hidden = !ended;
    if (ended) {
      const top = rows[0];
      banner.textContent = winner != null
        ? `Winner: aircraft ${winner} · ${fmtNum(top.finishT, 2)} s`
        : "No finisher";
      banner.classList.toggle("none", winner == null);
      if (winner != null) banner.style.setProperty("--winner", colorOf(winner));
    }
  }
}
