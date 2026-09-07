"""Independent recomputation of the 'pulse-then-hold, twice' commitment claim.

Stdlib only (host has no numpy). Reads the two dumped replay CSVs.
"""

import csv
import math

DT = 5.0 / 120.0
KILL = "/home/anthony_nguyen/aca/astrum/controllers/rl/runs/d3b/replay_kill_seed62.csv"
NOKILL = "/home/anthony_nguyen/aca/astrum/controllers/rl/runs/d3b/replay_nokill_seed28.csv"

AIL_COMMIT = 0.5
BANK_COMMIT_DEG = 45.0
NZ_LOADED = 3.0
CHI_ROOM_DEG = 45.0  # checked below against source
CHI_OUT_DEG = 60.0
CHI_MIN_HOLD_S = 0.5


def load(path):
    with open(path) as fh:
        rd = csv.DictReader(fh)
        rows = list(rd)
        names = rd.fieldnames or []
    d = {n: [float(r[n]) for r in rows] for n in names}
    d["_n"] = len(rows)
    return d


def runs_of(mask):
    """contiguous True runs -> list of (i, j) inclusive"""
    out = []
    i, n = 0, len(mask)
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


def report(tag, d):
    n = d["_n"]
    ail, nz, phi, alt = d["ail"], d["nz_g"], d["phi_deg"], d["alt_m"]
    t = d["t_s"]
    print("=" * 100)
    print(f"  {tag}   n = {n} steps, {n * DT:.3f} s")
    print("=" * 100)

    # ---- 1. the +1.0 and -1.0 saturated pulses, CONTIGUITY CHECKED
    for label, mask in (
        ("ail >= +0.999", [a >= 0.999 for a in ail]),
        ("ail <= -0.999", [a <= -0.999 for a in ail]),
        ("ail >  +0.5  ", [a > AIL_COMMIT for a in ail]),
        ("ail <  -0.5  ", [a < -AIL_COMMIT for a in ail]),
    ):
        rr = runs_of(mask)
        tot = sum(j - i + 1 for i, j in rr)
        print(
            f"  {label}: {tot} steps ({tot * DT:.3f} s) in {len(rr)} contiguous run(s): "
            + ", ".join(f"{i}-{j}({(j - i + 1) * DT:.3f}s)" for i, j in rr[:12])
            + (" ..." if len(rr) > 12 else "")
        )

    # ---- 2. the claim's OWN statistic: |ail| < 0.10 AND nz > 3
    mask = [abs(a) < 0.10 and z > 3.0 for a, z in zip(ail, nz)]
    rr = sorted(runs_of(mask), key=lambda p: p[1] - p[0], reverse=True)
    print(f"\n  |ail|<0.10 AND nz>3  -> {len(rr)} run(s); longest first:")
    for i, j in rr[:8]:
        print(
            f"    steps {i:3d}-{j:3d}  {(j - i + 1) * DT:6.3f} s  phi {phi[i]:+8.2f} -> {phi[j]:+8.2f}"
            f"  alt {alt[i]:7.1f} -> {alt[j]:7.1f} ({alt[j] - alt[i]:+7.1f} m)"
            f"  nz {min(nz[i : j + 1]):.2f}-{max(nz[i : j + 1]):.2f} g"
            f"  seg {int(d['seg_idx'][i])}->{int(d['seg_idx'][j])}"
            f"{'   *ENDS AT TERMINATION*' if j == n - 1 else ''}"
        )

    # ---- 3. the DETECTOR's OWN D4 bank hold: sign(phi) constant, |phi|>45, nz>3
    def longest_hold(vals, bar, need_valid=None):
        best = (0.0, -1, -1)
        i = 0
        while i < n:
            ok = abs(vals[i]) > bar and nz[i] > NZ_LOADED
            if need_valid is not None:
                ok = ok and need_valid[i] > 0.5
            if ok:
                s = 1 if vals[i] > 0 else -1
                j = i
                while j + 1 < n:
                    ok2 = abs(vals[j + 1]) > bar and nz[j + 1] > NZ_LOADED
                    if need_valid is not None:
                        ok2 = ok2 and need_valid[j + 1] > 0.5
                    if not (ok2 and (1 if vals[j + 1] > 0 else -1) == s):
                        break
                    j += 1
                dur = t[j] - t[i]
                if dur > best[0]:
                    best = (dur, i, j)
                i = j + 1
            else:
                i += 1
        return best

    dur, i, j = longest_hold(phi, BANK_COMMIT_DEG)
    print(
        f"\n  D4 longest_bank_hold (sign(phi) const, |phi|>45, nz>3): {dur:.3f} s"
        f"  steps {i}-{j}  phi {phi[i]:+.2f} -> {phi[j]:+.2f}"
        f"  dalt {alt[j] - alt[i]:+.1f} m"
        f"{'   *ENDS AT TERMINATION*' if j == n - 1 else ''}"
    )
    chi = d["chi_deg"]
    cv = d["chi_valid"]
    dur, i, j = longest_hold(chi, CHI_ROOM_DEG, cv)
    print(
        f"  D4 longest_chi_hold  (sign(chi) const, |chi|>45, valid, nz>3): {dur:.3f} s"
        + (f"  steps {i}-{j}  dalt {alt[j] - alt[i]:+.1f} m" if i >= 0 else "  (none)")
    )
    # rate-cmd run the module actually computes (|ail|>0.5, sign const)
    dur, i, j = (0.0, -1, -1)
    best = (0.0, -1, -1)
    k = 0
    while k < n:
        if abs(ail[k]) > AIL_COMMIT:
            s = 1 if ail[k] > 0 else -1
            m = k
            while m + 1 < n and abs(ail[m + 1]) > AIL_COMMIT and (1 if ail[m + 1] > 0 else -1) == s:
                m += 1
            if t[m] - t[k] > best[0]:
                best = (t[m] - t[k], k, m)
            k = m + 1
        else:
            k += 1
    print(
        f"  D4 longest_rate_cmd_run_s (|ail|>0.5 sign const): {best[0]:.3f} s steps {best[1]}-{best[2]}"
    )

    # ---- 4. is the pulse-then-hold the DECODE, not the solution? per-segment table
    print("\n  PER-SEGMENT: is 'pulse then |ail|~0' present in EVERY segment (i.e. mechanical)?")
    print(
        f"    {'seg':>3} {'steps':>10} {'cmdbank':>8} {'cmdload':>7} {'pulse_s':>8} {'|ail|<0.1_s':>11}"
        f" {'phi_end':>8} {'nz_med':>7} {'nz>3 frac':>9} {'dalt_m':>8}"
    )
    for sg in range(6):
        idx = [k for k in range(n) if int(d["seg_idx"][k]) == sg]
        if not idx:
            print(f"    {sg:3d}  NEVER EXECUTED")
            continue
        a0, b0 = idx[0], idx[-1]
        sub = list(range(a0, b0 + 1))
        pulse = sum(1 for k in sub if abs(ail[k]) > 0.999)
        quiet = sum(1 for k in sub if abs(ail[k]) < 0.10)
        nzs = sorted(nz[k] for k in sub)
        frac = sum(1 for k in sub if nz[k] > 3.0) / len(sub)
        print(
            f"    {sg:3d} {a0:4d}-{b0:4d} {d['cmd_bank_deg'][a0]:8.2f} {d['cmd_load'][a0]:7.4f}"
            f" {pulse * DT:8.3f} {quiet * DT:11.3f} {phi[b0]:8.2f} {nzs[len(nzs) // 2]:7.2f}"
            f" {frac:9.2f} {alt[b0] - alt[a0]:8.1f}"
        )

    # ---- 5. exact claimed windows, verbatim
    print("\n  CLAIM'S EXACT WINDOWS, recomputed:")
    for lo, hi, name in (
        (0, 23, "pulse A (claim: +1.000, 1.000 s)"),
        (38, 95, "hold A (claim 2.417 s)"),
        (96, 109, "pulse B (claim: -1.000, 0.583 s)"),
        (123, 172, "hold B (claim 2.042 s)"),
    ):
        if hi >= n:
            print(f"    {name}: window exceeds trace (n={n})")
            continue
        sub = list(range(lo, hi + 1))
        print(
            f"    {name}: steps {lo}-{hi} = {(hi - lo + 1) * DT:.3f} s"
            f" | ail {min(ail[k] for k in sub):+.3f}..{max(ail[k] for k in sub):+.3f}"
            f" | phi {phi[lo]:+.2f} -> {phi[hi]:+.2f} (min {min(phi[k] for k in sub):+.2f})"
            f" | nz {min(nz[k] for k in sub):.2f}-{max(nz[k] for k in sub):.2f}"
            f" | alt {alt[lo]:.1f} -> {alt[hi]:.1f} ({alt[hi] - alt[lo]:+.1f} m)"
            f" | gamma {d['gamma_deg'][lo]:+.2f} -> {d['gamma_deg'][hi]:+.2f}"
        )
    # gaps in claimed windows
    print("\n  GAPS the claim skips over:")
    for lo, hi in ((24, 37), (110, 122)):
        if hi >= n:
            continue
        sub = list(range(lo, hi + 1))
        print(
            f"    steps {lo}-{hi} ({(hi - lo + 1) * DT:.3f} s): ail {min(ail[k] for k in sub):+.3f}"
            f"..{max(ail[k] for k in sub):+.3f}  nz {min(nz[k] for k in sub):.2f}-{max(nz[k] for k in sub):.2f}"
            f"  phi {phi[lo]:+.2f} -> {phi[hi]:+.2f}"
        )

    # ---- 6. is hold B a TURN or a dive recovery? level-turn nz for that bank
    print("\n  HOLD B: turn or pull-out? (level-turn nz for the held bank = 1/cos(phi))")
    for k in (123, 135, 148, 160, 172):
        if k >= n:
            continue
        need = 1.0 / max(math.cos(math.radians(phi[k])), 1e-6)
        print(
            f"    step {k:3d} t {t[k]:6.3f} phi {phi[k]:+7.2f} nz {nz[k]:5.2f} g (level-turn nz "
            f"{need:5.2f} g) gamma {d['gamma_deg'][k]:+7.2f} pitch_rate {d['pitch_rate_dps'][k]:+7.2f}"
            f" psi {d['psi_deg'][k]:+8.2f} chi {d['chi_deg'][k]:+8.2f} valid {int(cv[k])}"
        )
    # heading change over hold B vs gamma change
    if n > 172:
        pass
    hi = min(172, n - 1)
    print(
        f"    over steps 123-{hi}: psi {d['psi_deg'][123]:+.2f} -> {d['psi_deg'][hi]:+.2f}"
        f" ({d['psi_deg'][hi] - d['psi_deg'][123]:+.2f} deg),"
        f" gamma {d['gamma_deg'][123]:+.2f} -> {d['gamma_deg'][hi]:+.2f}"
        f" ({d['gamma_deg'][hi] - d['gamma_deg'][123]:+.2f} deg)"
    )


def main():
    report("KILL seed 62", load(KILL))
    print()
    report("NON-KILL seed 28 (same commands, 576 steps)", load(NOKILL))


if __name__ == "__main__":
    main()
