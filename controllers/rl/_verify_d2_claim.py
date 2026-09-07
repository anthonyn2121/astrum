"""Independent recomputation of the D2 (yo-yo) clause cascade from the replay CSVs.

Reimplements `bfm_detect.detect_yoyos` (bfm_detect.py:382-456) verbatim against the raw
`smp_*` columns (native units: radians / m / m/s / g) so the numbers are derived, not
copied. No import of bfm_detect -- this is the independent arm.
"""

import csv
import math
import sys

# thresholds, transcribed from bfm_detect.py:66-91
CLOSURE_OVERSHOOT = 120.0
YOYO_GAMMA_SWING_DEG = 30.0
YOYO_GAMMA_EXCURSION_DEG = 15.0
YOYO_RETURN_S = 6.0
YOYO_GAMMA_RETURN_DEG = 10.0
YOYO_EXIT_RNG_M = 1500.0
YOYO_ASPECT_SLIP_DEG = 10.0
NZ_LOADED = 3.0
LOADED_FRAC = 0.5

D = math.degrees


def load(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def analyse(
    path,
    label,
    swing_deg=YOYO_GAMMA_SWING_DEG,
    ret_deg=YOYO_GAMMA_RETURN_DEG,
    ret_s=YOYO_RETURN_S,
    quiet=False,
):
    rows = load(path)
    n = len(rows)
    t = [float(r["smp_t"]) for r in rows]
    gam = [float(r["smp_gamma_rad"]) for r in rows]
    nz = [float(r["smp_nz"]) for r in rows]
    clo = [float(r["smp_closure"]) for r in rows]
    rng = [float(r["smp_rng"]) for r in rows]
    asp = [float(r["smp_aspect"]) for r in rows]

    if not quiet:
        print(f"--- {label}: {n} ticks, t {t[0]:.3f}..{t[-1]:.3f} s")
        above = [i for i in range(n) if clo[i] > CLOSURE_OVERSHOOT]
        print(
            f"    closure > {CLOSURE_OVERSHOOT}: {len(above)}/{n} ticks; "
            f"range {min(clo):+.0f} .. {max(clo):+.0f} m/s; min = {min(clo):+.1f} m/s"
        )
        # contiguity of the above-band set
        runs = []
        for i in above:
            if runs and i == runs[-1][1] + 1:
                runs[-1][1] = i
            else:
                runs.append([i, i])
        print(f"    above-band runs: {[tuple(r) for r in runs]}")

    swing = math.radians(swing_deg)
    excur = math.radians(YOYO_GAMMA_EXCURSION_DEG)
    ret = math.radians(ret_deg)

    entries = [
        i
        for i in range(n)
        if clo[i] > CLOSURE_OVERSHOOT and (i == 0 or clo[i - 1] <= CLOSURE_OVERSHOOT)
    ]
    if not quiet:
        print(f"    rising-edge entry candidates (line 411): {entries}")

    n_events = 0
    n_clean = 0
    i = 0
    while i < n:
        if not (clo[i] > CLOSURE_OVERSHOOT and (i == 0 or clo[i - 1] <= CLOSURE_OVERSHOOT)):
            i += 1
            continue
        g0 = gam[i]
        window = [k for k in range(i, n) if t[k] - t[i] <= ret_s]
        if len(window) < 3:
            break
        peak = max(window, key=lambda k: abs(gam[k] - g0))
        loaded = [nz[k] > NZ_LOADED for k in range(i, peak + 1)]
        lf = sum(loaded) / len(loaded)
        if not quiet:
            print(
                f"    entry i={i} t={t[i]:.3f}s  g0={D(g0):+.4f} deg  closure {clo[i]:+.1f} m/s  "
                f"window {len(window)} ticks (t {t[i]:.3f}..{t[window[-1]]:.3f} s)"
            )
            print(
                f"      peak k={peak} t={t[peak]:.3f}s  gamma {D(gam[peak]):+.4f} deg  "
                f"swing {D(abs(gam[peak] - g0)):.4f} deg  |gamma| {abs(D(gam[peak])):.4f} deg"
            )
            print(
                f"      (1) swing >= {swing_deg}: {D(abs(gam[peak] - g0)):.4f} -> "
                f"{'PASS' if abs(gam[peak] - g0) >= swing else 'FAIL by %.4f' % (swing_deg - D(abs(gam[peak] - g0)))}"
            )
            print(
                f"      (2) |gamma_peak| >= {YOYO_GAMMA_EXCURSION_DEG}: {abs(D(gam[peak])):.4f} -> "
                f"{'PASS' if abs(gam[peak]) >= excur else 'FAIL'}"
            )
            print(
                f"      (3) loaded_frac >= {LOADED_FRAC}: {lf:.4f} "
                f"({sum(loaded)}/{len(loaded)} over steps {i}-{peak}) -> "
                f"{'PASS' if lf >= LOADED_FRAC else 'FAIL'}"
            )
            # the return clause, evaluated INDEPENDENTLY of whether the swing clause passed
            tail = [k for k in range(peak + 1, n) if t[k] - t[i] <= ret_s]
            if tail:
                best = min(tail, key=lambda k: abs(gam[k] - g0))
                hit = [k for k in tail if abs(gam[k] - g0) < ret]
                print(
                    f"      (4) return |gamma-g0| < {ret_deg} within {ret_s}s: "
                    f"tail = steps {tail[0]}-{tail[-1]} (t {t[tail[0]]:.3f}..{t[tail[-1]]:.3f} s); "
                    f"closest {D(abs(gam[best] - g0)):.4f} deg at step {best} (t {t[best]:.3f} s) -> "
                    f"{'PASS at step %d' % hit[0] if hit else 'FAIL by %.4f deg' % (D(abs(gam[best] - g0)) - ret_deg)}"
                )
            else:
                print("      (4) tail empty")
            # last tick, past the window
            print(
                f"      last tick k={n - 1} t={t[n - 1]:.3f}s gamma {D(gam[n - 1]):+.4f} deg  "
                f"|gamma-g0| = {D(abs(gam[n - 1] - g0)):.4f} deg"
            )
            # global closest approach to g0 AFTER the peak, ignoring the window entirely
            tail_all = list(range(peak + 1, n))
            if tail_all:
                bg = min(tail_all, key=lambda k: abs(gam[k] - g0))
                print(
                    f"      (4') ignoring the {ret_s}s window entirely: closest to g0 anywhere after "
                    f"the peak = {D(abs(gam[bg] - g0)):.4f} deg at step {bg} (t {t[bg]:.3f} s)"
                )
            print(
                f"      D2c bleed: needs closure_exit < {CLOSURE_OVERSHOOT} and rng_exit < {YOYO_EXIT_RNG_M}; "
                f"min closure over trace {min(clo):+.1f} m/s "
                f"-> satisfiable: {min(clo) < CLOSURE_OVERSHOOT}"
            )

        # exact control flow
        if abs(gam[peak] - g0) < swing or abs(gam[peak]) < excur:
            i += 1
            continue
        if not loaded or lf < LOADED_FRAC:
            i += 1
            continue
        exit_idx = None
        for k in range(peak + 1, n):
            if t[k] - t[i] > ret_s:
                break
            if abs(gam[k] - g0) < ret:
                exit_idx = k
                break
        if exit_idx is None:
            i = peak + 1
            continue
        bled = clo[exit_idx] < CLOSURE_OVERSHOOT and rng[exit_idx] < YOYO_EXIT_RNG_M
        kept = D(wrap_pi(asp[exit_idx] - asp[i])) <= YOYO_ASPECT_SLIP_DEG
        n_events += 1
        n_clean += int(bled and kept)
        if not quiet:
            print(f"      EVENT exit={exit_idx} bled={bled} kept={kept}")
        i = exit_idx + 1
    return n_events, n_clean


base = "/workspace/controllers/rl/runs/d3b/"
kill = base + "replay_kill_seed62.csv"
nokill = base + "replay_nokill_seed28.csv"
if len(sys.argv) > 1:
    base = sys.argv[1]
    kill = base + "replay_kill_seed62.csv"
    nokill = base + "replay_nokill_seed28.csv"

print("=" * 100)
print("INDEPENDENT D2 CLAUSE RECOMPUTATION (no bfm_detect import)")
print("=" * 100)
for path, label in ((kill, "KILL seed 62"), (nokill, "NOKILL seed 28")):
    ev, cl = analyse(path, label)
    print(f"    => D2 = {ev}, D2c = {cl}")
    print()

print("=" * 100)
print("COUNTERFACTUAL: lower ONLY the swing bar (return + bleed bars untouched)")
print("=" * 100)
for path, label in ((kill, "KILL seed 62"), (nokill, "NOKILL seed 28")):
    for sw in (30.0, 28.47, 28.0, 25.0, 20.0, 10.0, 1.0, 0.0):
        ev, cl = analyse(path, label, swing_deg=sw, quiet=True)
        print(f"    {label}: SWING={sw:6.2f} RET_DEG=10 RET_S=6 -> D2 = {ev}, D2c = {cl}")
    print()

print("=" * 100)
print("COUNTERFACTUAL: swing bar met (0 deg) AND the return bar swept")
print("=" * 100)
for path, label in ((kill, "KILL seed 62"), (nokill, "NOKILL seed 28")):
    for rd in (10.0, 15.0, 18.03, 20.0, 23.89, 25.0, 30.0):
        ev, cl = analyse(path, label, swing_deg=0.0, ret_deg=rd, quiet=True)
        ev2, cl2 = analyse(path, label, swing_deg=0.0, ret_deg=rd, ret_s=1e9, quiet=True)
        print(
            f"    {label}: SWING=0 RET_DEG={rd:6.2f} RET_S=6 -> D2 = {ev}, D2c = {cl}"
            f"   | RET_S=inf -> D2 = {ev2}, D2c = {cl2}"
        )
    print()
