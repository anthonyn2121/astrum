// A top-down sketch of a race course for the setup pages: the gates seen from
// above (east right, north up) as short bars across the flight path, each with
// an arrowhead for the direction through it and its number, joined by a dashed
// line in course order. Drawn into an <svg> in plain user units (metres): the
// viewBox is the gates' extent plus a margin, so the sketch fits whatever the
// course's size; strokes keep their pixel width (vector-effect). Styles are
// css/interactive.css's `.course-svg` classes.
//
// Gate geometry (see meshes.js): gate normal = (sin yaw, cos yaw) east / north,
// the bar across it is along right = (cos yaw, -sin yaw).

const SVG_NS = "http://www.w3.org/2000/svg";

function svgEl(tag, attrs = {}, text = null) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  if (text != null) node.textContent = text;
  return node;
}

// A one-line summary of a course (validated, race.js parseCourse): the gate
// count and course length, seed, footprint and height range. Empty for no course.
export function courseSummary(course) {
  if (!course) return "";
  const { minE, maxE, minN, maxN } = extent(course);
  const ups = course.gates.map((g) => g.up_m);
  const fmt = (v) => v.toFixed(v < 10 ? 1 : 0);
  const seed = course.seed == null ? "" : ` · seed ${course.seed}`;
  return `${course.gates.length} gates, ${Math.round(course.length_m)} m${seed} · ${fmt(maxE - minE)} × ${fmt(maxN - minN)} m`
    + ` · ${fmt(Math.min(...ups))}–${fmt(Math.max(...ups))} m up`;
}

function extent(course) {
  let minE = Infinity, maxE = -Infinity, minN = Infinity, maxN = -Infinity;
  for (const g of course.gates) {
    minE = Math.min(minE, g.east_m); maxE = Math.max(maxE, g.east_m);
    minN = Math.min(minN, g.north_m); maxN = Math.max(maxN, g.north_m);
  }
  return { minE, maxE, minN, maxN };
}

// Replace `svg`'s contents with the sketch of `course` (validated), or empty it.
export function drawCoursePreview(svg, course) {
  svg.replaceChildren();
  if (!course) return;
  const { minE, maxE, minN, maxN } = extent(course);
  const size = Math.max(maxE - minE, maxN - minN, 10); // m: the longer side, at least a gate field
  const margin = size * 0.12;
  const bar = Math.max(course.top_bar_m, size * 0.08); // drawn gate width (m)
  const head = size * 0.02;                            // arrowhead half-size (m)
  const font = size * 0.045;                           // number tag height (m)
  // SVG y runs down, north up: y = -north.
  svg.setAttribute("viewBox", [
    minE - margin, -maxN - margin, maxE - minE + 2 * margin, maxN - minN + 2 * margin,
  ].map((v) => v.toFixed(3)).join(" "));

  svg.append(svgEl("polyline", {
    class: "path", "vector-effect": "non-scaling-stroke",
    points: course.gates.map((g) => `${g.east_m.toFixed(3)},${(-g.north_m).toFixed(3)}`).join(" "),
  }));
  course.gates.forEach((g, i) => {
    const s = Math.sin(g.yaw_rad), c = Math.cos(g.yaw_rad);
    const x = g.east_m, y = -g.north_m;
    const rx = c * bar / 2, ry = s * bar / 2; // right = (c, -s) in (e, n) → (c, +s) in (x, y)
    const nx = s, ny = -c;                    // normal = (s, c) in (e, n) → (s, -c) in (x, y)
    const cls = i === 0 ? "gate first" : "gate";
    svg.append(
      svgEl("line", { class: cls, "vector-effect": "non-scaling-stroke", x1: x - rx, y1: y - ry, x2: x + rx, y2: y + ry }),
      // The arrowhead sits half a head past the bar, pointing along the normal.
      svgEl("polygon", {
        class: "arrow",
        points: [
          [x + nx * head * 2, y + ny * head * 2],
          [x + nx * head * 0.4 + ny * head, y + ny * head * 0.4 - nx * head],
          [x + nx * head * 0.4 - ny * head, y + ny * head * 0.4 + nx * head],
        ].map(([px, py]) => `${px.toFixed(3)},${py.toFixed(3)}`).join(" "),
      }),
      // The number tag sits behind the gate (upstream), off to its left.
      svgEl("text", {
        class: "num", x: x - nx * head * 3 - rx * 1.25, y: y - ny * head * 3 - ry * 1.25,
        "font-size": font.toFixed(3), "text-anchor": "middle", "dominant-baseline": "middle",
      }, String(g.index + 1)),
    );
  });
}
