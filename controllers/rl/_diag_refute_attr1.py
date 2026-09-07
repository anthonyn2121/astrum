"""REFUTATION TEST for ATTRIBUTION 1 (`the D1 negative measures the OBJECTIVE`).

The claim under test: "the D1 genome is worth -20,006.1 objective points against the incumbent, so
CMA-ES ranks it far below." Three things are measured here, none estimated:

  A. Both genomes re-flown on ALL EIGHT search seeds, with `M.score` per seed AND the MEAN over
     seeds -- the mean is what `M.Evaluator.__call__` returns as `fit` and therefore the only
     number `CMAES.tell` ever sorts. A single-seed score is not the objective the search saw.
  B. Generation 0 of the real search reproduced bit-for-bit (same CMAES seed, same popsize, same
     Evaluator, same seed-skip), so the D1 genome's fitness can be RANKED inside the population
     CMA-ES actually selected from. `CMAES.tell` has no elitism: it sorts the current generation
     only, so a deficit against the eventual incumbent is not a selection pressure.
  C. The deficit decomposed by objective term, under both definitions.
"""

from __future__ import annotations

import _diag_d1_contradiction as K  # noqa: E402
import _diag_d1_exists as X  # noqa: E402
import _diag_d3b_express as E  # noqa: E402
import _diag_true_mpc as M  # noqa: E402
import numpy as np

SEEDS = [11, 28, 45, 62, 79, 96, 113, 130]


def fly(g, seed):
    gm = M.Genome(segments=6, channels=3, param="bank", roll_gate=True)
    env, base = M.build_env(E.CELL, seed, E.N_STEPS * E.DT)

    def ctl(step, me, bd, rel):
        return gm.act(g, step, E.N_STEPS, me)

    r = M.roll_once(env, base, ctl, E.N_STEPS, keep=False)
    env.close()
    return r


def table(label, g):
    print(f"\n  {label}")
    print(
        f"     {'seed':>5} {'steps':>6} {'event':>7} {'kill':>4} {'hp':>6} {'tLeth':>6} "
        f"{'t600':>6} {'rngMn':>7} {'dalt':>7} {'rnds':>5} {'score':>13}"
    )
    scores, rows = [], []
    for sd in SEEDS:
        r = fly(g, sd)
        s, c = M.score(r)
        scores.append(s)
        rows.append((sd, r, s, c))
        print(
            f"     {sd:>5} {r.steps:>6} {str(r.event):>7} {r.kill:>4} {r.hp_dealt:>6.1f} "
            f"{r.tleth_s:>6.3f} {r.t600_s:>6.3f} {r.rng_min:>7.2f} {r.dalt_m:>+7.0f} "
            f"{r.rounds:>5} {s:>13.3f}"
        )
    mean = float(np.mean(scores))
    print(
        f"     {'MEAN':>5} {'':>6} {'':>7} {sum(r.kill for _, r, _, _ in rows):>4} "
        f"{np.mean([r.hp_dealt for _, r, _, _ in rows]):>6.1f} "
        f"{np.mean([r.tleth_s for _, r, _, _ in rows]):>6.3f} "
        f"{np.mean([r.t600_s for _, r, _, _ in rows]):>6.3f} "
        f"{np.mean([r.rng_min for _, r, _, _ in rows]):>7.2f} "
        f"{np.mean([r.dalt_m for _, r, _, _ in rows]):>+7.0f} "
        f"{np.mean([r.rounds for _, r, _, _ in rows]):>5.0f} {mean:>13.3f}"
        "   <-- THIS is `fit`, what CMAES.tell sorts"
    )
    print(
        f"     per-seed spread: min {min(scores):.3f}  max {max(scores):.3f}  "
        f"range {max(scores) - min(scores):.3f}"
    )
    return mean, scores, rows


def main() -> None:
    print("#" * 100)
    print("#  ATTRIBUTION 1 REFUTATION TEST")
    print("#" * 100)

    inc = np.load(f"{E.OUT}/d3b_best.npz")["genome"]
    _, _, phases = K.chain(4.0)
    g_d1 = X.encode(phases, 1.0, 0.60)
    print("\n  D1 genome re-encoded from the same code path as d1_exists_report.txt:")
    print("    " + np.array2string(g_d1, precision=6, separator=", ", max_line_width=200))
    print(f"    inside the search box [-1,1]? {bool(np.all(np.abs(g_d1) <= 1.0))}")

    m_inc, s_inc, r_inc = table("INCUMBENT (npz genome)", inc)
    m_d1, s_d1, r_d1 = table("D1 GENOME (18-dim encoding)", g_d1)

    print("\n" + "=" * 100)
    print("  C. THE DEFICIT, UNDER BOTH DEFINITIONS")
    print("=" * 100)
    i62 = SEEDS.index(62)
    print(
        f"    claim's number   (single seed 62): {s_d1[i62]:.3f} - {s_inc[i62]:.3f} = "
        f"{s_d1[i62] - s_inc[i62]:+.1f}"
    )
    print(f"    search's number  (MEAN over 8)   : {m_d1:.3f} - {m_inc:.3f} = {m_d1 - m_inc:+.1f}")
    print(
        f"    inflation of the claim's deficit : {abs(s_d1[i62] - s_inc[i62]) - abs(m_d1 - m_inc):+.1f} "
        f"points ({100 * (abs(s_d1[i62] - s_inc[i62]) / abs(m_d1 - m_inc) - 1):.1f}% too large)"
    )
    print("    per-seed deficits (D1 - incumbent), all eight:")
    for k, sd in enumerate(SEEDS):
        print(
            f"      seed {sd:>3}: {s_d1[k] - s_inc[k]:+11.1f}"
            + ("   (kill seed)" if r_inc[k][1].kill else "   (NO-kill seed)")
        )
    mk = float(np.mean([r.kill for _, r, _, _ in r_inc]))
    mh = float(np.mean([r.hp_dealt for _, r, _, _ in r_inc]))
    mt = float(np.mean([r.tleth_s for _, r, _, _ in r_inc]))
    print("\n    decomposition of the MEAN deficit by objective term (incumbent minus D1 genome):")
    print(f"      W_KILL * d(kill)   = {M.W_KILL:.0e} * {mk:.3f} = {M.W_KILL * mk:+11.1f}")
    print(f"      W_HP   * d(hp)     = {M.W_HP:.0e} * {mh:.1f} = {M.W_HP * mh:+11.1f}")
    print(f"      W_TLETH* d(tleth)  = {M.W_TLETH:.0e} * {mt:.3f} = {M.W_TLETH * mt:+11.1f}")
    a_i = float(np.mean([c["arrival"] for _, _, _, c in r_inc]))
    a_d = float(np.mean([c["arrival"] for _, _, _, c in r_d1]))
    p_i = float(np.mean([c["alt"] for _, _, _, c in r_inc]))
    p_d = float(np.mean([c["alt"] for _, _, _, c in r_d1]))
    print(
        f"      arrival  {a_i:.3f} - {a_d:.3f} = {a_i - a_d:+11.3f}   (ARRIVAL_CAP = "
        f"{M.ARRIVAL_CAP}, binding on neither)"
    )
    print(f"      alt price -{p_i:.3f} - (-{p_d:.3f}) = {p_d - p_i:+11.3f}")

    print("\n" + "=" * 100)
    print(
        "  B. GENERATION 0 OF THE REAL SEARCH, REPRODUCED -- where does +", f"{m_d1:.3f}", "RANK?"
    )
    print("=" * 100)
    gm = M.Genome(segments=6, channels=3, param="bank", roll_gate=True)
    es = M.CMAES(gm.dim, 0.4, 180, 20260906)  # the run's defaults (argparse: --sigma0, --seed)
    pop = np.clip(es.ask(), -1.0, 1.0)
    ev = M.Evaluator(E.CELL, gm, E.N_STEPS, SEEDS, True, 6)
    try:
        fit, fired, best = ev(pop)
    finally:
        ev.close()
    print(
        f"    reproduced gen 0: fit(best) {np.max(fit):+.3f}   fit(mean) {np.mean(fit):+.3f}   "
        f"fired {int(fired.sum())}"
    )
    print("    prereg gen 0    : fit(best)     +3.212   fit(mean)     +1.862   fired     1")
    rank = int(np.sum(fit > m_d1)) + 1
    print(
        f"\n    D1 genome fitness {m_d1:.3f} ranks {rank} of {len(fit)} in gen 0 "
        f"(mu = {es.mu} are SELECTED)  -> "
        f"{'INSIDE the selected set' if rank <= es.mu else 'outside the selected set'}"
    )
    q = np.percentile(fit, [0, 10, 25, 50, 75, 90, 100])
    print("    gen-0 fitness percentiles 0/10/25/50/75/90/100: " + "  ".join(f"{v:.3f}" for v in q))
    print(f"    candidates in gen 0 scoring BELOW the D1 genome: {int(np.sum(fit < m_d1))} of 180")
    print(f"    gap from the D1 genome to gen 0's OWN best: {np.max(fit) - m_d1:+.3f} points")
    print(
        "\n    CMAES.tell elitism check (source): "
        "`idx = np.argsort(-fitness)` over `self.pop` only; xmean is a weighted mean of the"
    )
    print("    current generation's top mu. No incumbent, no hall-of-fame, enters selection.")


if __name__ == "__main__":
    main()
