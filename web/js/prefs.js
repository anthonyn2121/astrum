// Viewer preferences (dock visibility and sizes, overlay toggles, plot
// channels, event filters) persisted in localStorage. Values read back are
// validated by their consumers; storage failures (private mode, quota) only
// cost persistence.

const KEY = "astrum.viewer.v1";

export function loadPrefs() {
  try {
    const v = JSON.parse(localStorage.getItem(KEY));
    return v && typeof v === "object" ? v : {};
  } catch {
    return {};
  }
}

export function savePrefs(patch) {
  try {
    localStorage.setItem(KEY, JSON.stringify({ ...loadPrefs(), ...patch }));
  } catch {
    // Storage unavailable: preferences just don't persist.
  }
}
