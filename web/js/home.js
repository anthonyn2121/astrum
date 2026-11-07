// The studio home page: the page cards are static; this only announces a
// live studio run (GET /api/runs/current) with a link to watch it.

import { getCurrentRun } from "./api.js";

const live = document.getElementById("liveRun");

async function show() {
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
  const link = document.createElement("a");
  link.href = `./view.html?run=${encodeURIComponent(run.id)}`;
  link.textContent = "Watch it →";
  const id = document.createElement("strong");
  id.textContent = run.id;
  live.replaceChildren("A run is live: ", id, " ", link);
  live.hidden = false;
}

show();
