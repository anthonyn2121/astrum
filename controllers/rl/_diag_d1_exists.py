"""EXISTENCE PROOF: an 18-dim genome INSIDE the D3b search space that scores D1 >= 1.

The L=4.0 s schedule that P2 found is piecewise-constant over exactly six 4.0 s segments with a
constant load and throttle, so it is representable as a 6 x 3 = 18-dim `Genome` vector. This encodes
it, flies it through the real `Genome.act`, and reports the detector verdict, the objective value
`score()` would have given it, and the altitude price -- so the D1 can be compared against the
incumbent kill genome on the search's own terms.

Also audits the two clauses the found event leans on: (d) "rolled back in" and (f) "bought angles".
"""

from __future__ import annotations

import math

import _diag_d3b_express as E
import _diag_true_mpc as M
import numpy as np
from astrum_dogfight import bfm_detect as B

_LOG: list[str] = []


def log(s: str = "") -> None:
    print(s, flush=True)
    _LOG.append(s)


def encode(phases, load: float, thr: float) -> np.ndarray:
    """(bank_deg, load, thr) x 6 -> the 18-dim genome `Genome.act` decodes back to it."""
    g = []
    for _, bank, ld, th in phases[:6]:
        g += [
            float(np.clip(bank / M.BANK_MAX_DEG, -1.0, 1.0)),
            2.0 * float(ld) - 1.0,
            2.0 * float(th) - 1.0,
        ]
    return np.asarray(g, float)


def fly_genome(g: np.ndarray, seed: int):
    gm = M.Genome(segments=6, channels=3, param="bank", roll_gate=True)
    env, base = M.build_env(E.CELL, seed, E.N_STEPS * E.DT)

    def ctl(step, me, bd, rel):
        return gm.act(g, step, E.N_STEPS, me)

    r = M.roll_once(env, base, ctl, E.N_STEPS, keep=False)
    env.close()
    return r, gm


def d1_audit(tr, label: str) -> None:
    d1 = B.detect_displacement_rolls(tr)
    log(f"\n  {label}: D1 = {len(d1)}")
    for ev in d1:
        i, j = ev.i0, ev.i1
        hold_end = None
        hold = [(abs(s.chi) > math.radians(B.CHI_OUT_DEG) and s.nz > B.NZ_LOADED) for s in tr]
        for a, b in B._runs(hold):
            if a == i:
                hold_end = b
        log(
            f"    displacement_roll  hold steps {i}-{hold_end} t {ev.t0:.3f}-"
            f"{tr[hold_end].t:.3f} = {ev.detail['hold_s']:.3f} s "
            f"(bar {B.CHI_MIN_HOLD_S:.1f} s, margin {ev.detail['hold_s'] / B.CHI_MIN_HOLD_S:.1f}x)"
        )
        log(
            f"      chi_peak {ev.detail['chi_peak_deg']:.1f} deg   nz_peak {ev.detail['nz_peak']:.2f} g   "
            f"d_aspect {ev.d_aspect_deg:+.1f} deg   d_rng {ev.d_rng_m:+.0f} m   d_es {ev.d_es_m:+.0f} m"
        )
        log(
            f"      ENTRY  step {i} t {tr[i].t:.3f}  rng {tr[i].rng:.0f} m  closure {tr[i].closure:+.0f} m/s  "
            f"aspect {math.degrees(tr[i].aspect):.1f}  alt {tr[i].alt:.0f} m  chi_valid {tr[i].chi_valid}"
        )
        log(
            f"      EXIT   step {j} t {tr[j].t:.3f}  rng {tr[j].rng:.0f} m  closure {tr[j].closure:+.0f} m/s  "
            f"aspect {math.degrees(tr[j].aspect):.1f}  alt {tr[j].alt:.0f} m  "
            f"|chi| {abs(math.degrees(tr[j].chi)):.1f}  chi_valid {tr[j].chi_valid}   <-- clause (d)"
        )
        log(
            f"      ALTITUDE PRICE of the hold: {tr[hold_end].alt - tr[i].alt:+.0f} m over "
            f"{tr[hold_end].t - tr[i].t:.3f} s"
        )
        # is the (d) exit a VALID chi or the chi=0.0 sentinel?
        if not tr[j].chi_valid:
            log(
                "      NOTE clause (d) is satisfied by the chi = 0.0 SENTINEL that "
                "out_of_plane_angle returns when the LOS is within CHI_VALID_ATA_DEG of the "
                "velocity vector, i.e. by the bandit being DEAD AHEAD, not by rolling back in."
            )
        # first VALID |chi| < 30 inside the return window, if any
        alt_exit = None
        for k in range(hold_end + 1, len(tr)):
            if tr[k].t - tr[hold_end].t > B.CHI_RETURN_S:
                break
            if tr[k].chi_valid and abs(tr[k].chi) < math.radians(B.CHI_RETURN_DEG):
                alt_exit = k
                break
        log(
            f"      first chi_valid |chi| < {B.CHI_RETURN_DEG:.0f} inside the return window: "
            + (
                f"step {alt_exit} t {tr[alt_exit].t:.3f} (so the event survives a validity-gated "
                f"clause (d) too; d_aspect there = "
                f"{math.degrees(tr[alt_exit].aspect - tr[i].aspect):+.1f} deg)"
                if alt_exit is not None
                else "NONE -- the event exists only via the sentinel"
            )
        )


def main() -> None:
    log("#" * 100)
    log("#  D1 EXISTENCE PROOF inside the 18-dim D3b search space")
    log("#" * 100)

    # rebuild the L=4.0 s chain deterministically (same code path as the sweep)
    import _diag_d1_contradiction as K

    r_ref, rows_ref, phases = K.chain(4.0)
    log(f"\n  L=4.0 s schedule from P2 iteration 3 (flown as a Sched): {r_ref.steps} steps, event={r_ref.event}")
    log("  phases as STORED (unclipped) and as FLOWN (clipped to +-BANK_MAX_DEG = 115):")
    for k, p in enumerate(phases):
        log(
            f"    seg {k}  t {k * 4.0:5.1f}-{(k + 1) * 4.0:5.1f} s  bank stored {p[1]:+8.2f} -> "
            f"flown {float(np.clip(p[1], -115.0, 115.0)):+8.2f}  load {p[2]:.2f}  thr {p[3]:.2f}"
        )

    g = encode(phases, 1.0, 0.60)
    log("\n  ENCODED 18-dim genome (what `Genome.act` will decode; identical command stream):")
    log("    " + np.array2string(g, precision=6, separator=", ", max_line_width=200))
    gm_dec = M.Genome(segments=6, channels=3, param="bank", roll_gate=True)
    log("  decode check (bank_deg, load, thr) per segment:")
    for s in range(6):
        p = g[s * 3 : (s + 1) * 3]
        log(
            f"    seg {s}: bank {M.BANK_MAX_DEG * float(np.clip(p[0], -1, 1)):+8.2f}  "
            f"load {0.5 * (float(np.clip(p[1], -1, 1)) + 1.0):.4f}  "
            f"thr {0.5 * (float(np.clip(p[2], -1, 1)) + 1.0):.4f}"
        )
    del gm_dec

    log("\n  Flown through the REAL `Genome.act` over the D3b search seeds:")
    log(
        f"     {'seed':>5} {'steps':>6} {'event':>10} {'kill':>4} {'hp':>6} {'tLeth':>6} "
        f"{'rngMn':>6} {'dalt':>7} {'D1':>3} {'D2':>3} {'D2c':>4} {'D3':>3} {'score':>12}"
    )
    tr_keep = None
    n_d1 = 0
    for sd in (11, 28, 45, 62, 79, 96, 113, 130):
        r, _ = fly_genome(g, sd)
        det = B.detect_all(r.trace)
        sc, comp = M.score(r)
        n_d1 += len(det["displacement_rolls"])
        log(
            f"     {sd:>5} {r.steps:>6} {str(r.event):>10} {r.kill:>4} {r.hp_dealt:>6.1f} "
            f"{r.tleth_s:>6.3f} {r.rng_min:>6.0f} {r.dalt_m:>+7.0f} "
            f"{len(det['displacement_rolls']):>3} {len(det['yoyos']):>3} "
            f"{len(det['yoyos_clean']):>4} {len(det['turning_room']):>3} {sc:>12.3f}"
        )
        if sd == 62:
            tr_keep = r.trace
    log(f"\n  POOLED D1 over the 8 seeds = {n_d1}")
    d1_audit(tr_keep, "seed 62, the encoded 18-dim genome")

    # the incumbent, for the objective comparison
    inc = np.load(f"{E.OUT}/d3b_best.npz")["genome"]
    r_inc, _ = fly_genome(inc, 62)
    sc_inc, _ = M.score(r_inc)
    det_inc = B.detect_all(r_inc.trace)
    log(
        f"\n  OBJECTIVE COMPARISON on seed 62 (the search MAXIMISES `score`):\n"
        f"    incumbent  D1={len(det_inc['displacement_rolls'])}  kill={r_inc.kill}  "
        f"hp={r_inc.hp_dealt:.0f}  dalt={r_inc.dalt_m:+.0f} m  score {sc_inc:+.3f}"
    )
    r_d1, _ = fly_genome(g, 62)
    sc_d1, comp_d1 = M.score(r_d1)
    log(
        f"    D1 genome  D1={len(B.detect_all(r_d1.trace)['displacement_rolls'])}  kill={r_d1.kill}  "
        f"hp={r_d1.hp_dealt:.0f}  dalt={r_d1.dalt_m:+.0f} m  score {sc_d1:+.3f}  components {comp_d1}"
    )
    log(
        f"    => the D1 genome is worth {sc_d1 - sc_inc:+.1f} on the objective, i.e. CMA-ES ranks it "
        f"{'above' if sc_d1 > sc_inc else 'BELOW'} the incumbent."
    )

    path = f"{E.OUT}/d1_exists_report.txt"
    with open(path, "w") as fh:
        fh.write("\n".join(_LOG) + "\n")
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
