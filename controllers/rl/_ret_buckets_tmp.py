"""Bucketed mean episode return vs collected steps, for several train logs side by side.

`ret(20)` in the train log is a rolling mean of the UNDISCOUNTED episode sum (`ep_return +=
reward`), so it is γ-independent and valid to compare across runs trained at different
discounts — which is the whole point here: curriculum-88 (γ stepped to 0.9985) and c88b
(stepped to 0.998) both collapsed monotonically, and c89 ramps to the same 0.998 target. Single
windows are noisy enough to read a collapse into (it happened once), so always bucket.

    uv run python _ret_buckets_tmp.py runs/curriculum-88/train_full.log ... --bucket 100000
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

# step   280,799 | ... | ret(20)    23.90 | kill   7% | ...
_LINE = re.compile(r"step\s+([\d,]+)\s*\|.*?ret\(\d+\)\s+(-?[\d.]+).*?kill\s+(\d+)%")


def parse(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps, rets, kills = [], [], []
    with path.open(errors="replace") as fh:
        for line in fh:
            m = _LINE.search(line)
            if m:
                steps.append(int(m.group(1).replace(",", "")))
                rets.append(float(m.group(2)))
                kills.append(float(m.group(3)))
    return np.array(steps), np.array(rets), np.array(kills)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--bucket", type=int, default=100_000)
    ap.add_argument("--max-step", type=int, default=0, help="0 = no cap; else clip all runs here")
    args = ap.parse_args()

    runs = {}
    for p in args.logs:
        s, r, k = parse(p)
        if s.size == 0:
            print(f"!! no train lines parsed from {p}")
            continue
        runs[p.parent.name or p.stem] = (s, r, k)

    if not runs:
        return

    cap = args.max_step or max(int(s.max()) for s, _, _ in runs.values())
    edges = np.arange(0, cap + args.bucket, args.bucket)
    width = max(len(n) for n in runs) + 2

    hdr = "".join(f"{int(e / 1000):>8}k" for e in edges[:-1])
    print(f"\nmean ret(20) per {args.bucket // 1000}k collected steps")
    print(f"{'run':<{width}}{hdr}")
    for name, (s, r, _) in runs.items():
        cells = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (s >= lo) & (s < hi)
            cells.append(f"{r[sel].mean():>9.1f}" if sel.any() else f"{'-':>9}")
        print(f"{name:<{width}}" + "".join(cells))

    print("\nmean exploration kill% (NOT greedy quality — judge by the out-of-band verdict)")
    print(f"{'run':<{width}}{hdr}")
    for name, (s, _, k) in runs.items():
        cells = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (s >= lo) & (s < hi)
            cells.append(f"{k[sel].mean():>9.1f}" if sel.any() else f"{'-':>9}")
        print(f"{name:<{width}}" + "".join(cells))
    print()


if __name__ == "__main__":
    main()
