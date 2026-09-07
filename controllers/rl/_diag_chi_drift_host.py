"""Host-side (stdlib-only) audit of what actually opens and closes a D1/D3 chi hold.

Three questions, all answered off the two replay CSVs already on disk:
  (1) how long is closure > 0 (D1 clause c) and closure > 120 m/s (D2 entry gate)?
  (2) with the commanded bank target HELD CONSTANT, how fast does chi drift? chi is
      phi_lift - phi_los, so a constant bank holds phi_lift and the LOS plane rotates out
      from under it. That drift rate, not the roll rate, sets the hold length.
  (3) what is the longest window in which |chi| stays inside a 30-deg-wide band, and inside
      the D1 band (>60 deg) / D3 band (>45 deg), with nz > 3?
"""

from __future__ import annotations

import csv

DT = 5.0 / 120.0


def load(path):
    with open(path) as fh:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(fh)]


def runs(mask):
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def report(path, label):
    rows = load(path)
    n = len(rows)
    t = [r["t_s"] for r in rows]
    clo = [r["closure_ms"] for r in rows]
    chi = [r["chi_deg"] for r in rows]
    nz = [r["nz_g"] for r in rows]
    cmd = [r["cmd_bank_deg"] for r in rows]
    phi = [r["phi_deg"] for r in rows]
    rng = [r["rng_m"] for r in rows]
    print(f"\n{'=' * 100}\n  {label}  rows={n}\n{'=' * 100}")

    for bar, name in ((0.0, "closure > 0  (D1 clause c: closing at hold ENTRY)"),
                      (120.0, "closure > 120 m/s (D2 entry gate)")):
        m = [c > bar for c in clo]
        rr = runs(m)
        tot = sum(1 for x in m if x)
        print(f"\n  {name}: {tot}/{n} ticks = {tot * DT:.3f} s of {n * DT:.3f} s, {len(rr)} window(s)")
        for i, j in rr:
            print(
                f"    steps {i:3d}-{j:3d}  t {t[i]:6.3f}-{t[j]:6.3f}  dur {t[j] - t[i] + DT:6.3f} s  "
                f"rng {rng[i]:6.0f}->{rng[j]:6.0f} m  closure {clo[i]:+7.0f}->{clo[j]:+7.0f} m/s"
            )

    # (2) chi drift with the command HELD
    print("\n  CHI DRIFT while the commanded bank target is CONSTANT (per segment)")
    print("    seg  steps      t (s)         cmd     phi(mean,sd)   chi start->end   d|chi|/dt  max|dchi/dt|(0.5s)")
    bnds = [0] + [k for k in range(1, n) if cmd[k] != cmd[k - 1]] + [n]
    for a, b in zip(bnds[:-1], bnds[1:]):
        # drop the roll-in transient: start once |phi - cmd| <= 10 deg
        s = a
        while s < b and abs(phi[s] - cmd[a]) > 10.0:
            s += 1
        if b - s < 6:
            print(f"    {a // 96:3d}  {a:3d}-{b - 1:3d}  {t[a]:6.3f}-{t[b - 1]:6.3f}  {cmd[a]:+8.1f}  "
                  f"(never settled within 10 deg)")
            continue
        ph = phi[s:b]
        mean = sum(ph) / len(ph)
        sd = (sum((x - mean) ** 2 for x in ph) / len(ph)) ** 0.5
        d = (chi[b - 1] - chi[s]) / (t[b - 1] - t[s])
        w = 12  # 0.5 s
        loc = max(
            (abs(chi[k + w] - chi[k]) / (w * DT) for k in range(s, b - w)), default=float("nan")
        )
        print(
            f"    {a // 96:3d}  {s:3d}-{b - 1:3d}  {t[s]:6.3f}-{t[b - 1]:6.3f}  {cmd[a]:+8.1f}  "
            f"{mean:+7.1f},{sd:4.1f}  {chi[s]:+7.1f}->{chi[b - 1]:+7.1f}  {d:+8.1f}   {loc:8.1f} deg/s"
        )

    # (3) longest window inside each chi band with nz > 3
    print("\n  LONGEST LOADED (nz>3) WINDOW inside each chi band")
    for lo, hi, name in ((60.0, 181.0, "D1  |chi| > 60"), (45.0, 181.0, "D3  |chi| > 45"),
                         (30.0, 60.0, "in-between 30..60")):
        m = [lo < abs(chi[k]) <= hi and nz[k] > 3.0 for k in range(n)]
        rr = runs(m)
        best = max(rr, key=lambda ij: ij[1] - ij[0], default=None)
        if best is None:
            print(f"    {name:20s}  no loaded window at all")
            continue
        i, j = best
        print(
            f"    {name:20s}  longest {t[j] - t[i] + DT:5.3f} s (steps {i}-{j}, t {t[i]:.3f}-{t[j]:.3f}), "
            f"{len(rr)} window(s), total {sum(1 for x in m if x) * DT:.3f} s"
        )


if __name__ == "__main__":
    base = "/home/anthony_nguyen/aca/astrum/controllers/rl/runs/d3b"
    report(f"{base}/replay_kill_seed62.csv", "KILL seed 62")
    report(f"{base}/replay_nokill_seed28.csv", "NOKILL seed 28 (full 24 s, all 6 segments)")
