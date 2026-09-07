"""Host-side (stdlib-only) roll-rate audit of the D3b replay dumps.

Measures, from the two per-step CSVs already on disk:
  * the achieved roll rate (dphi/dt differenced across rows, and the sim's own p column)
  * the settling time of each commanded bank-target STEP at a segment boundary
  * how long a 90 deg / 135 deg bank change actually takes
No numpy: pure csv + math so it runs on the host without the container.
"""

from __future__ import annotations

import csv
import sys

DT = 5.0 / 120.0  # ACTION_REPEAT / RATE_HZ = 41.6667 ms


def load(path: str) -> list[dict[str, float]]:
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        out.append({k: (float(v) if v not in ("", None) else float("nan")) for k, v in r.items()})
    return out


def unwrap(vals: list[float]) -> list[float]:
    """Unwrap a degree series so a -180/+180 crossing does not fake a 360 deg/step rate."""
    out = [vals[0]]
    off = 0.0
    for k in range(1, len(vals)):
        d = vals[k] - vals[k - 1]
        if d > 180.0:
            off -= 360.0
        elif d < -180.0:
            off += 360.0
        out.append(vals[k] + off)
    return out


def report(path: str, label: str) -> None:
    rows = load(path)
    n = len(rows)
    phi = unwrap([r["phi_deg"] for r in rows])
    t = [r["t_s"] for r in rows]
    p_col = [r["roll_rate_dps"] for r in rows]
    ail = [r["ail"] for r in rows]
    nz = [r["nz_g"] for r in rows]
    cmd = [r["cmd_bank_deg"] for r in rows]

    dphi = [(phi[k + 1] - phi[k]) / DT for k in range(n - 1)]

    print(f"\n{'=' * 96}\n  {label}   ({path})   rows={n}  dt={DT * 1000:.4f} ms\n{'=' * 96}")
    amax = max(range(n - 1), key=lambda k: abs(dphi[k]))
    print(
        f"  differenced dphi/dt   peak |.| = {abs(dphi[amax]):8.2f} deg/s  at step {amax:3d} "
        f"t={t[amax]:6.3f}s  (signed {dphi[amax]:+8.2f})"
    )
    pmax = max(range(n), key=lambda k: abs(p_col[k]))
    print(
        f"  sim roll_rate_dps     peak |.| = {abs(p_col[pmax]):8.2f} deg/s  at step {pmax:3d} "
        f"t={t[pmax]:6.3f}s  (signed {p_col[pmax]:+8.2f})   nz={nz[pmax]:.2f} ail={ail[pmax]:+.3f}"
    )
    # saturated-aileron subset: |ail| == 1.0 is the full roll-rate command
    sat = [k for k in range(n - 1) if abs(ail[k]) >= 0.999]
    if sat:
        r = sorted(abs(dphi[k]) for k in sat)
        print(
            f"  |ail|>=0.999 rows: {len(sat):3d}  |dphi/dt| median {r[len(r) // 2]:7.2f}  "
            f"p10 {r[int(0.1 * len(r))]:7.2f}  p90 {r[int(0.9 * len(r))]:7.2f}  max {r[-1]:7.2f} deg/s"
        )
        # rate achieved once the roll is established (drop the first 3 steps of each sat run)
        est = []
        prev = -99
        run = 0
        for k in sat:
            run = run + 1 if k == prev + 1 else 0
            prev = k
            if run >= 3:
                est.append(abs(dphi[k]))
        if est:
            est.sort()
            print(
                f"  ...established (>=4th consecutive saturated step): n={len(est)} "
                f"median {est[len(est) // 2]:7.2f}  max {est[-1]:7.2f} deg/s"
            )

    # ---- per-boundary step response -------------------------------------------------------
    print("\n  COMMANDED BANK STEPS (segment boundaries) -- settle to within tol of the target")
    print(
        "    step   t_s    from ->     to    dcmd   phi0   |  t(50%)  t(90%)  t(<10deg)  "
        "peak|p|  meanNz(roll)   phi_end"
    )
    bnds = [0] + [k for k in range(1, n) if cmd[k] != cmd[k - 1]]
    for b in bnds:
        tgt = cmd[b]
        phi0 = phi[b]
        d = tgt - phi0
        if abs(d) < 1.0:
            continue
        # scan forward to the next boundary (or end)
        nxt = n
        for k in range(b + 1, n):
            if cmd[k] != tgt:
                nxt = k
                break
        seg = range(b, nxt)

        def first(pred) -> float:
            for k in seg:
                if pred(phi[k]):
                    return t[k] - t[b]
            return float("nan")

        t50 = first(lambda x, d=d, phi0=phi0: (x - phi0) / d >= 0.5)
        t90 = first(lambda x, d=d, phi0=phi0: (x - phi0) / d >= 0.9)
        t10 = first(lambda x, tgt=tgt: abs(x - tgt) <= 10.0)
        rollend = b
        for k in seg:
            if abs(phi[k] - tgt) <= 10.0:
                rollend = k
                break
        pk = max((abs(p_col[k]) for k in range(b, max(b + 1, rollend + 1))), default=0.0)
        nzs = [nz[k] for k in range(b, max(b + 1, rollend + 1))]
        print(
            f"    {b:4d} {t[b]:6.3f} {phi0:+8.1f} -> {tgt:+7.1f} {d:+7.1f} {phi0:+7.1f}   | "
            f"{t50:7.3f} {t90:7.3f} {t10:10.3f}  {pk:7.1f}  {sum(nzs) / len(nzs):11.2f}  "
            f"{phi[nxt - 1]:+8.1f}"
        )

    # ---- chi / nz co-occurrence, the D1 and D3 masks ---------------------------------------
    chi = [r["chi_deg"] for r in rows]
    for name, bar in (("D1 chi>60 & nz>3", 60.0), ("D3 chi>45 & nz>3", 45.0)):
        mask = [abs(chi[k]) > bar and nz[k] > 3.0 for k in range(n)]
        runs = []
        k = 0
        while k < n:
            if mask[k]:
                j = k
                while j + 1 < n and mask[j + 1]:
                    j += 1
                runs.append((k, j))
                k = j + 1
            else:
                k += 1
        tot = sum(1 for m in mask if m)
        print(f"\n  {name}: {tot} ticks in {len(runs)} run(s)")
        for i, j in runs:
            print(
                f"    steps {i:3d}-{j:3d}  t {t[i]:6.3f}-{t[j]:6.3f}  dur {t[j] - t[i] + DT:5.3f} s "
                f"(>=0.5 s? {'YES' if t[j] - t[i] >= 0.5 else 'no '})  "
                f"|chi| {abs(chi[i]):5.1f}->{max(abs(chi[k]) for k in range(i, j + 1)):5.1f}"
                f"->{abs(chi[j]):5.1f}  nz {min(nz[i:j + 1]):.2f}-{max(nz[i:j + 1]):.2f}  "
                f"alt {rows[i]['alt_m']:.0f}->{rows[j]['alt_m']:.0f} m"
            )
    # closure band for the D2 entry gate
    clo = [r["closure_ms"] for r in rows]
    over = [k for k in range(n) if clo[k] > 120.0]
    rise = [k for k in over if k == 0 or clo[k - 1] <= 120.0]
    print(
        f"\n  D2 entry gate closure>120 m/s: {len(over)}/{n} ticks, {len(rise)} rising edge(s) "
        f"at steps {rise[:12]}  (closure range {min(clo):+.0f}..{max(clo):+.0f} m/s)"
    )
    gam = [r["gamma_deg"] for r in rows]
    print(f"  gamma range {min(gam):+.1f}..{max(gam):+.1f} deg")


if __name__ == "__main__":
    base = "/home/anthony_nguyen/aca/astrum/controllers/rl/runs/d3b"
    report(f"{base}/replay_kill_seed62.csv", "KILL seed 62 (173 steps, 2 segments)")
    report(f"{base}/replay_nokill_seed28.csv", "NOKILL seed 28 (576 steps, all 6 segments)")
    sys.stdout.flush()
