// The studio home page: the page cards are static; this announces a live
// studio run (GET /api/runs/current) and a running sweep (GET
// /api/sweeps/current), each with a link to watch it. Ids are server text: DOM
// text only.

import { getCurrentRun, getCurrentSweep } from "./api.js";
import { h } from "./dom.js";

const live = document.getElementById("liveRun");
const liveSweep = document.getElementById("liveSweep");

async function showRun() {
  let run = null;
  try {
    run = await getCurrentRun();
  } catch {
    return; // no API (e.g. the page served from elsewhere): nothing to announce
  }
  if (!run || run.state === "finished") {
    live.hidden = true;
    return;
  }
  live.replaceChildren("A run is live: ", h("strong", { text: run.id }), " ",
    h("a", { href: `./view.html?run=${encodeURIComponent(run.id)}`, text: "Watch it →" }));
  live.hidden = false;
}

async function showSweep() {
  let sweep = null;
  try {
    sweep = await getCurrentSweep();
  } catch {
    return;
  }
  if (!sweep || sweep.state !== "running") {
    liveSweep.hidden = true;
    return;
  }
  const done = sweep.ok + sweep.failed + sweep.stopped;
  liveSweep.replaceChildren("A sweep is running: ", h("strong", { text: sweep.id }),
    ` · ${done}/${sweep.total} — `,
    h("a", { href: `./sweep.html?id=${encodeURIComponent(sweep.id)}`, text: "Open it →" }));
  liveSweep.hidden = false;
}

showRun();
showSweep();
