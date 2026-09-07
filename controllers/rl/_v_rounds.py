import sys

sys.path.insert(0, "/workspace/controllers/rl")
import statistics as st

import _diag_true_mpc as M
import numpy as np

d = np.load("/workspace/controllers/rl/runs/d3b/d3b_best.npz", allow_pickle=True)
g = d["genome"].astype(float)
cell = M.CELLS["merge1500"]
gen = M.Genome(6, 3, "bank", True)
N = int(round(24.0 * 1000.0 / (M.CONTROL_DT_S * 1000.0)))
print("n_steps", N, " CONTROL_DT_S", M.CONTROL_DT_S)
print(
    "R_LETHAL_M",
    M.R_LETHAL_M,
    " FIRE_LAY_MISS_M",
    M.FIRE_LAY_MISS_M,
    " HIT_RADIUS_M",
    M.C.HIT_RADIUS_M,
    " damage/hit implied",
    2.0,
)
print()
hdr = (
    "seed  ev      kill  hp_dealt  hits  rounds  hit%   fireSteps  hpTaken  oppRounds  oppHits  oppHit%  "
    "fireLay>8  rngFireMin  rngFireMax  steps"
)
print(hdr)
rows = []
for sd in [11, 28, 45, 62, 79, 96, 113, 130]:
    env, base = M.build_env(cell, sd, 24.0)
    nfire = 0
    lay_gt8 = 0
    rngs = []

    def ctrl(step, me, bd, rel):
        return gen.act(g, step, N, me)

    r = M.roll_once(env, base, ctrl, N, keep=False)
    # `fire` is recomputed from the trace with the SAME predicate roll_once used (M._trigger).
    for s in r.trace:
        fired = (s.rng <= M.R_LETHAL_M) and (s.lay_miss_m <= M.FIRE_LAY_MISS_M)
        if fired:
            nfire += 1
            rngs.append(s.rng)
            if s.lay_miss_m > M.C.HIT_RADIUS_M:
                lay_gt8 += 1
    me = base._states[base.learner_id]
    bd = base._states[base.opponent_id]
    opp_rounds = int(round(M.C.AMMO_MAX - float(bd.ammo)))
    hits = r.hp_dealt / 2.0
    opp_hits = r.hp_taken / 2.0
    rows.append(
        (
            sd,
            r.event,
            r.kill,
            r.hp_dealt,
            hits,
            r.rounds,
            100.0 * hits / max(1, r.rounds),
            nfire,
            r.hp_taken,
            opp_rounds,
            opp_hits,
            100.0 * opp_hits / max(1, opp_rounds),
            lay_gt8,
            min(rngs) if rngs else float("nan"),
            max(rngs) if rngs else float("nan"),
            r.steps,
        )
    )
    print(
        "%4d  %-6s  %4d  %8.1f  %4.0f  %6d  %5.1f  %9d  %7.1f  %9d  %7.0f  %7.1f  %9d  %10.1f  %10.1f  %5d"
        % rows[-1]
    )
    env.close()


pct = [x[6] for x in rows]
rnd = [x[5] for x in rows]
print()
print("POOLED over all 8 seeds (same genome, same commanded actions, same oracle):")
print(
    "  rounds total %d  hits total %.0f  pooled hit%% %.1f"
    % (sum(rnd), sum(x[4] for x in rows), 100.0 * sum(x[4] for x in rows) / sum(rnd))
)
print("  per-seed hit%%: min %.1f  median %.1f  max %.1f" % (min(pct), st.median(pct), max(pct)))
print("  kill seeds hit%%: %s" % ["%.1f" % x[6] for x in rows if x[2] == 1])
print("  nokill seeds hit%%: %s" % ["%.1f" % x[6] for x in rows if x[2] == 0])
