// Small DOM helpers shared by the studio pages. Text from files, servers and
// URLs reaches the DOM only through textContent (string children, `text`).

export const $ = (id) => document.getElementById(id);

// A DOM element; string children become text nodes.
export function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v === true ? "" : String(v));
  }
  for (const c of children) if (c != null) node.append(c);
  return node;
}

// Show `msg` in `node` (hidden when empty).
export function showError(node, msg) {
  const text = msg || "";
  if (node.textContent !== text) node.textContent = text;
  node.hidden = !text;
}
