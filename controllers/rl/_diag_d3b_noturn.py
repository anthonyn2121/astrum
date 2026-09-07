"""D3b REFUTATION PROBE — is the AGENT's turn what closed the 1621.73 m constant-velocity miss?

`bfm_detect.cpa_miss_m` at step 0 reads 1621.73 m. That number is the miss at the CPA of two
CONSTANT-VELOCITY aircraft, so it is the counterfactual "neither jet turns from here". The pro-BFM
claim reads it as "the no-turn geometry was a 1.6 km miss, and MY maneuver closed it 313x".

The constant-velocity counterfactual is not the flyable one, because the bandit is a scripted
Tier-3 that DOES turn (its own docstring: it commits a lead turn at >=800 m of offset, and
merge1500 sits inside its threat gate). The flyable counterfactual is "I do not turn; he does
whatever he does". This script flies exactly that, in the same cell, at the same sim seed, with the
same honest trigger, so the closure can be ATTRIBUTED rather than assumed.

ARMS (all: cell merge1500, horizon 24 s = 576 control steps, `M._trigger` overwritten by
`M.roll_once` for every arm, so no arm can win on a different fire rule)
  genome        the d3b winner, verbatim from d3b_best.npz — the reproduction control
  noturn_t060   wings-level bank hold + flight-path (gamma) hold at 0, throttle 0.6023
                (= the genome's own segment-0 throttle command)
  noturn_t100   same, throttle 1.0 (so "he ran out of energy" cannot explain the result)
  noturn_hold   bank hold to 0 with elev = 0.0: no pitch command at all, the literal null stick

The two `noturn_*` arms hold BANK at zero, so the agent's heading is held: the measured own-ship
`dpsi` is printed for every arm and is the audit that the no-turn arm did not turn.

Everything printed is measured in this run. Nothing is estimated.
"""

from __future__ import annotations

import math

import numpy as np
from astrum_rl import _paths

_paths.require_astrum_env()
import _diag_true_mpc as M  # noqa: E402
from astrum_dogfight import bfm  # noqa: E402
from astrum_dogfight import bfm_detect as B  # noqa: E402

DEG = 180.0 / math.pi
NPZ = "runs/d3b/d3b_best.npz"
HORIZON_S = 24.0
N_STEPS = int(round(HORIZON_S / M.CONTROL_DT_S))


def _wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def unwrap_deg(seq):
    out = [seq[0]]
    for x in seq[1:]:
        out.append(out[-1] + _wrap180(x - out[-1]))
    return out


def make_genome_ctl(genome: np.ndarray):
    g = M.Genome(6, 3, "bank", True)

    def fn(step, me, bd, rel):
        return g.act(genome, step, N_STEPS, me)

    return fn


def make_noturn(thr: float, mode: str):
    """Wings-level heading hold. `mode='gamma'` holds flight path at 0; `mode='hold'` sends elev=0."""

    def fn(step, me, bd, rel):
        ail = bfm._roll_to(float(me.phi), 0.0)  # bank -> 0: no turn
        if mode == "gamma":
            gam = math.atan2(float(me.vel[2]), math.hypot(float(me.vel[0]), float(me.vel[1])))
            elev = float(np.clip(bfm.KPITCH * gam, -1.0, 1.0))  # gam<0 (sinking) -> pull
        else:
            elev = 0.0
        return np.array([ail, elev, 0.0, thr, -1.0], np.float32)

    return fn


def fly(name, ctl, seed):
    env, base = M.build_env(M.CELLS["merge1500"], seed, HORIZON_S)
    rows = []

    def rec(step, me, bd, rel):
        a = np.asarray(ctl(step, me, bd, rel), np.float32)
        rows.append(
            dict(
                step=step,
                t=step * M.CONTROL_DT_S,
                rng=float(rel.rng),
                cpa=float(B.cpa_miss_m(me.pos, me.vel, bd.pos, bd.vel)),
                aspect=float(rel.aspect) * DEG,
                ata=float(rel.ata) * DEG,
                hca=float(rel.hca) * DEG,
                clos=float(rel.closure),
                losr=float(rel.los_rate) * DEG,
                lay=float(rel.lay_miss_m),
                psi=float(me.psi) * DEG,
                opsi=float(bd.psi) * DEG,
                phi=float(me.phi) * DEG,
                ophi=float(bd.phi) * DEG,
                nz=float(me.nz),
                onz=float(bd.nz),
                alt=float(me.alt),
                vt=float(me.airspeed),
                hp=float(bd.health),
                myhp=float(me.health),
            )
        )
        return a

    r = M.roll_once(env, base, rec, N_STEPS, keep=False)
    env.close()
    psi = unwrap_deg([x["psi"] for x in rows])
    opsi = unwrap_deg([x["opsi"] for x in rows])
    cpas = [x["cpa"] for x in rows]
    rngs = [x["rng"] for x in rows]
    imin = int(np.argmin(rngs))
    icpa = int(np.argmin(cpas))
    print(f"\n=== {name}   seed {seed} ===")
    print(
        f"  steps {r.steps:4d} ({r.steps * M.CONTROL_DT_S:6.3f} s)  event={r.event}  kill={r.kill} "
        f"hp_dealt={r.hp_dealt:6.2f} hp_taken={r.hp_taken:6.2f} rounds={r.rounds} "
        f"tLeth={r.tleth_s:.3f} dalt={r.dalt_m:+8.1f} m"
    )
    print(f"  cpa_miss[0]  = {cpas[0]:10.3f} m      cpa_miss[last] = {cpas[-1]:9.3f} m")
    print(f"  min cpa_miss = {min(cpas):10.3f} m at step {icpa} (t={rows[icpa]['t']:.3f} s)")
    print(f"  min rng      = {min(rngs):10.3f} m at step {imin} (t={rows[imin]['t']:.3f} s)")
    print(f"  ratio cpa[0]/min cpa = {cpas[0] / max(min(cpas), 1e-9):9.2f}x")
    print(
        f"  OWN  dpsi over episode = {psi[-1] - psi[0]:+9.2f} deg   "
        f"|nz|max = {max(abs(x['nz']) for x in rows):5.2f} g   "
        f"|phi|max = {max(abs(x['phi']) for x in rows):7.2f} deg"
    )
    print(
        f"  OPP  dpsi over episode = {opsi[-1] - opsi[0]:+9.2f} deg   "
        f"nz max = {max(x['onz'] for x in rows):5.2f} g   "
        f"|phi|max = {max(abs(x['ophi']) for x in rows):7.2f} deg"
    )
    print(f"  hca: {rows[0]['hca']:+8.2f} -> {rows[imin]['hca']:+8.2f} deg (at min rng)")
    print("   step     t     rng      cpa   aspect     ata      hca    clos   losr    psi   opsi")
    marks = sorted({0, 20, 40, 60, 80, 100, 120, 140, 156, imin, icpa, len(rows) - 1})
    for i in marks:
        if i >= len(rows):
            continue
        x = rows[i]
        print(
            f"  {i:5d} {x['t']:6.3f} {x['rng']:8.1f} {x['cpa']:8.1f} {x['aspect']:8.2f} "
            f"{x['ata']:7.2f} {x['hca']:+8.2f} {x['clos']:7.1f} {x['losr']:6.2f} "
            f"{psi[i]:7.2f} {opsi[i]:7.2f}"
        )
    return dict(name=name, seed=seed, rows=rows, r=r, psi=psi, opsi=opsi)


def main():
    d = np.load(NPZ, allow_pickle=True)
    genome = np.asarray(d["genome"], float)
    npz_actions = np.asarray(d["actions"], np.float32)
    print(f"genome from {NPZ}: {np.round(genome, 6).tolist()}")
    print(f"npz actions shape {npz_actions.shape}")

    out = []
    for seed in (62, 28):
        out.append(fly("genome", make_genome_ctl(genome), seed))
        out.append(fly("noturn_t060", make_noturn(0.6023, "gamma"), seed))
        out.append(fly("noturn_t100", make_noturn(1.0, "gamma"), seed))
        out.append(fly("noturn_hold", make_noturn(0.6023, "hold"), seed))

    print("\n\n================ SUMMARY TABLE ================")
    print(
        f"{'arm':14s} {'seed':>5s} {'steps':>6s} {'own dpsi':>9s} {'opp dpsi':>9s} "
        f"{'cpa[0]':>9s} {'min cpa':>9s} {'min rng':>9s} {'hp':>7s} {'kill':>5s} {'event':>10s}"
    )
    for o in out:
        cpas = [x["cpa"] for x in o["rows"]]
        rngs = [x["rng"] for x in o["rows"]]
        print(
            f"{o['name']:14s} {o['seed']:5d} {o['r'].steps:6d} "
            f"{o['psi'][-1] - o['psi'][0]:+9.2f} {o['opsi'][-1] - o['opsi'][0]:+9.2f} "
            f"{cpas[0]:9.2f} {min(cpas):9.2f} {min(rngs):9.2f} "
            f"{o['r'].hp_dealt:7.2f} {o['r'].kill:5d} {str(o['r'].event):>10s}"
        )


if __name__ == "__main__":
    main()
