"""Adversarial audit of the claim: "the 4.0 s grain never bound that D1 event: the entire hold
sits inside ONE segment, with zero bank changes during it."

Re-flies the D1-existence genome (d1_exists_report.txt:15) through the real `Genome.act`, then
measures, per step: the segment index, the COMMANDED bank target, the FLOWN bank phi, chi, nz, alt.
Recomputes the D1 hold bounds, the clause (a) lookback window and every segment it touches.
"""

from __future__ import annotations

import math

import _diag_d3b_express as E
import _diag_true_mpc as M
import numpy as np
from astrum_dogfight import bfm_detect as B

G = np.array(
    [
        1.0,
        1.0,
        0.2,
        0.582792,
        1.0,
        0.2,
        1.0,
        1.0,
        0.2,
        0.872769,
        1.0,
        0.2,
        -0.007301,
        1.0,
        0.2,
        0.312102,
        1.0,
        0.2,
    ],
    float,
)
NS = E.N_STEPS
DT = E.DT


def seg_of(step: int) -> int:
    return min(5, (step * 6) // NS)


def cmd_bank_deg(step: int) -> float:
    return M.BANK_MAX_DEG * float(np.clip(G[seg_of(step) * 3], -1.0, 1.0))


def fly(seed: int):
    gm = M.Genome(segments=6, channels=3, param="bank", roll_gate=True)
    env, base = M.build_env(E.CELL, seed, NS * DT)
    acts = []

    def ctl(step, me, bd, rel):
        a = gm.act(G, step, NS, me)
        acts.append(np.asarray(a, float).copy())
        return a

    r = M.roll_once(env, base, ctl, NS, keep=False)
    env.close()
    return r, np.asarray(acts)


def main() -> None:
    print("=" * 96)
    print(
        "D1-existence genome, re-flown.  NS=%d  DT=%.6f s  seg len = %d steps = %.3f s"
        % (NS, DT, NS // 6, (NS // 6) * DT)
    )
    print(
        "segment boundaries (step -> t): "
        + ", ".join("%d->%.3f" % (k * NS // 6, k * NS // 6 * DT) for k in range(7))
    )
    print("=" * 96)

    traces = {}
    for sd in (11, 28, 45, 62, 79, 96, 113, 130):
        r, acts = fly(sd)
        det = B.detect_all(r.trace)
        traces[sd] = (r, acts, det)
        d1 = det["displacement_rolls"]
        print(
            "  seed %3d  steps %d  event %-6s kill %d  hp %.1f  dalt %+.0f  D1 %d  "
            "windows %s"
            % (
                sd,
                r.steps,
                str(r.event),
                r.kill,
                r.hp_dealt,
                r.dalt_m,
                len(d1),
                [(e.i0, e.i1) for e in d1],
            )
        )

    # cross-seed identity of the command stream
    a62 = traces[62][1]
    for sd in traces:
        n = min(len(a62), len(traces[sd][1]))
        print(
            "  max|dActions| seed %3d vs 62 over %d steps = %.3e"
            % (sd, n, float(np.max(np.abs(traces[sd][1][:n] - a62[:n]))))
        )

    r, acts, det = traces[62]
    tr = r.trace
    ev = det["displacement_rolls"][0]
    hold = [(abs(s.chi) > math.radians(B.CHI_OUT_DEG) and s.nz > B.NZ_LOADED) for s in tr]
    runs = B._runs(hold)
    print(
        "\n  ALL loaded-out-of-plane runs (|chi|>%.0f & nz>%.1f): %s"
        % (B.CHI_OUT_DEG, B.NZ_LOADED, [(a, b, round(tr[b].t - tr[a].t, 3)) for a, b in runs])
    )
    i = ev.i0
    (hi, hj) = next((a, b) for a, b in runs if a == i)
    j = ev.i1
    print(
        "  D1 event: hold steps %d-%d  t %.3f-%.3f = %.3f s ; exit (clause d) step %d t %.3f"
        % (hi, hj, tr[hi].t, tr[hj].t, ev.detail["hold_s"], j, tr[j].t)
    )
    print(
        "  segments touched: hold %s .. %s ; full event i0..i1 %s .. %s"
        % (seg_of(hi), seg_of(hj), seg_of(i), seg_of(j))
    )
    print(
        "  d_aspect %+.1f deg  d_rng %+.0f m  d_es %+.0f m"
        % (ev.d_aspect_deg, ev.d_rng_m, ev.d_es_m)
    )

    # ---- COMMANDED vs FLOWN bank across the hold ----
    phis = np.array([math.degrees(tr[k].phi) for k in range(hi, hj + 1)])
    cmds = np.array([cmd_bank_deg(k) for k in range(hi, hj + 1)])
    ails = np.array([tr[k].ail for k in range(hi, hj + 1)])
    print(
        "\n  COMMANDED bank over the hold: unique values %s  (constant: %s)"
        % (sorted(set(np.round(cmds, 3).tolist())), len(set(np.round(cmds, 6).tolist())) == 1)
    )
    print(
        "  FLOWN phi over the hold: min %+.2f  max %+.2f  range %.2f deg ; "
        "first %+.2f  last %+.2f  (delta %+.2f)"
        % (phis.min(), phis.max(), phis.max() - phis.min(), phis[0], phis[-1], phis[-1] - phis[0])
    )
    dphi = np.diff(phis) / DT
    print(
        "  FLOWN roll rate over the hold: min %+.1f  max %+.1f  mean|.| %.1f deg/s"
        % (dphi.min(), dphi.max(), float(np.mean(np.abs(dphi))))
    )
    print(
        "  |phi - cmd| over the hold: min %.2f  max %.2f  mean %.2f deg"
        % (
            np.min(np.abs(phis - cmds)),
            np.max(np.abs(phis - cmds)),
            float(np.mean(np.abs(phis - cmds))),
        )
    )
    print(
        "  aileron over the hold: min %+.3f max %+.3f  (nonzero ticks %d/%d)"
        % (ails.min(), ails.max(), int(np.sum(np.abs(ails) > 1e-6)), len(ails))
    )

    # roll-in transient after the seg0->seg1 step
    b1 = NS // 6
    print(
        "\n  seg0->seg1 bank STEP at step %d t %.3f : cmd %+.2f -> %+.2f deg"
        % (b1, b1 * DT, cmd_bank_deg(b1 - 1), cmd_bank_deg(b1))
    )
    print("  step | seg | cmd_bank |   phi   | roll_rate |  chi   | chi_v |  nz  |  alt  ")
    for k in list(range(b1 - 4, b1 + 40, 2)) + [hi - 1, hi, hi + 1, hj - 1, hj, hj + 1, j]:
        if k < 1 or k >= len(tr):
            continue
        rr = (math.degrees(tr[k].phi) - math.degrees(tr[k - 1].phi)) / DT
        tag = ""
        if k == hi:
            tag = "  <-- HOLD ENTRY"
        if k == hj:
            tag = "  <-- HOLD EXIT"
        if k == j:
            tag = "  <-- clause (d)"
        print(
            "  %4d |  %d  | %+8.2f | %+7.2f | %+9.1f | %+6.1f |   %d   | %5.2f | %5.0f%s"
            % (
                k,
                seg_of(k),
                cmd_bank_deg(k),
                math.degrees(tr[k].phi),
                rr,
                math.degrees(tr[k].chi),
                int(tr[k].chi_valid),
                tr[k].nz,
                tr[k].alt,
                tag,
            )
        )

    # ---- clause (a): the in-plane lookback ----
    print(
        "\n  CLAUSE (a) in-plane lookback: needs some k with |chi| < %.0f deg and "
        "t[%d]-t[k] <= %.1f s" % (B.CHI_INPLANE_DEG, hi, B.CHI_LOOKBACK_S)
    )
    ks = [
        k
        for k in range(hi - 1, -1, -1)
        if tr[hi].t - tr[k].t <= B.CHI_LOOKBACK_S
        and abs(tr[k].chi) < math.radians(B.CHI_INPLANE_DEG)
    ]
    win = [k for k in range(hi - 1, -1, -1) if tr[hi].t - tr[k].t <= B.CHI_LOOKBACK_S]
    print(
        "    lookback window = steps %d..%d  (t %.3f..%.3f) -> segments %s"
        % (min(win), max(win), tr[min(win)].t, tr[max(win)].t, sorted({seg_of(k) for k in win}))
    )
    print(
        "    satisfying steps: %s"
        % (ks[::-1] if len(ks) < 30 else "%d steps, %d..%d" % (len(ks), min(ks), max(ks)))
    )
    if ks:
        print("    segments of the satisfying steps: %s" % sorted({seg_of(k) for k in ks}))
        print(
            "    ALL satisfying steps in seg 1? %s   ALL in seg 0? %s"
            % (all(seg_of(k) == 1 for k in ks), all(seg_of(k) == 0 for k in ks))
        )
        for k in ks[::-1][:6] + ks[:3]:
            print(
                "      step %4d seg %d t %.3f |chi| %.2f deg nz %.2f cmd_bank %+.2f"
                % (k, seg_of(k), tr[k].t, abs(math.degrees(tr[k].chi)), tr[k].nz, cmd_bank_deg(k))
            )

    # ---- price the hold ----
    print(
        "\n  PRICE of the hold: alt %+.0f m (%.0f -> %.0f) over %.3f s = %.1f m/s ; "
        "es_rel %+.0f m ; rng %+.0f m ; closure %+.0f -> %+.0f m/s"
        % (
            tr[hj].alt - tr[hi].alt,
            tr[hi].alt,
            tr[hj].alt,
            tr[hj].t - tr[hi].t,
            (tr[hj].alt - tr[hi].alt) / (tr[hj].t - tr[hi].t),
            tr[hj].es_rel - tr[hi].es_rel,
            tr[hj].rng - tr[hi].rng,
            tr[hi].closure,
            tr[hj].closure,
        )
    )
    print(
        "  EPISODE price: dalt %+.0f m over %d steps ; hp_dealt %.1f ; kill %d"
        % (r.dalt_m, r.steps, r.hp_dealt, r.kill)
    )
    # chi_peak / nz at the chi peak
    kp = max(range(hi, hj + 1), key=lambda k: abs(tr[k].chi))
    print(
        "  chi_peak %.2f deg at step %d t %.3f  nz there %.2f g  chi_valid %s  seg %d"
        % (abs(math.degrees(tr[kp].chi)), kp, tr[kp].t, tr[kp].nz, tr[kp].chi_valid, seg_of(kp))
    )
    # closure sign flip inside the event = a PASS
    sgn = [1 if tr[k].closure > 0 else -1 for k in range(hi, j + 1)]
    flips = [k for k in range(hi + 1, j + 1) if sgn[k - hi] != sgn[k - hi - 1]]
    print(
        "  closure sign flips inside i0..i1: %s (CPA inside the event => the bandit PASSED)" % flips
    )
    kmin = min(range(len(tr)), key=lambda k: tr[k].rng)
    print(
        "  min range over the episode: %.0f m at step %d t %.3f" % (tr[kmin].rng, kmin, tr[kmin].t)
    )


if __name__ == "__main__":
    main()
