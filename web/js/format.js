// Number and time formatting shared by the HUD modules. Every value is a
// finite number by the time it gets here (the store coerces untrusted input).

export const RAD2DEG = 180 / Math.PI;

// Seconds -> "M:SS.s".
export function fmtTime(sec) {
  if (!Number.isFinite(sec) || sec < 0) sec = 0;
  const m = Math.floor(sec / 60);
  const s = sec - m * 60;
  return `${m}:${s.toFixed(1).padStart(4, "0")}`;
}

// Fixed decimals with an explicit sign ("+2.0", "−4.1"); a value that rounds
// to zero gets none.
export function fmtSigned(v, digits) {
  const t = Math.abs(v).toFixed(digits);
  if (Number(t) === 0) return t;
  return (v < 0 ? "−" : "+") + t;
}

// Fixed decimals with a typographic minus.
export function fmtNum(v, digits) {
  const t = v.toFixed(digits);
  return t.startsWith("-") && Number(t) !== 0 ? `−${t.slice(1)}` : t.replace(/^-/, "");
}

// Metres -> "357 m" / "1.24 km".
export function fmtRange(m) {
  if (!Number.isFinite(m)) return "—";
  return m < 1000 ? `${m.toFixed(0)} m` : `${(m / 1000).toFixed(2)} km`;
}

// Heading in degrees, wrapped to [0, 360).
export function headingDeg(psiRad) {
  const h = (psiRad * RAD2DEG) % 360;
  return h < 0 ? h + 360 : h;
}

// Short label for a control mode.
export const MODE_SHORT = {
  autopilot: "AP", raw: "RAW", failsafe: "FS", uncommanded: "—", unknown: "?",
};

// Team accent colours: a fixed set for the first few teams, then hashed hues.
const TEAM_COLORS = ["#4aa3ff", "#ff5a4d", "#46d38a", "#ffb547", "#c08bff", "#3fd0d4"];
export function teamColor(team) {
  return TEAM_COLORS[team] ?? `hsl(${(team * 67) % 360} 70% 62%)`;
}
