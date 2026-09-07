"""REFUTE-pass instrument: log the Tier-3 FSM state, break sign and commanded stick per
control step while re-flying the d3b_best genome. Nothing is reimplemented: the real
`Tier3Opponent.act` is called, we only wrap it to record its internals."""

import sys

import numpy as np

sys.path.insert(0, "/workspace/crates/astrum-env/python")
sys.path.insert(0, "/workspace/controllers/rl")

import _diag_true_mpc as M  # noqa: E402
from astrum_dogfight import bfm_detect as B  # noqa: E402
from astrum_dogfight import constants as C  # noqa: E402
from astrum_dogfight.geometry import relative, specific_energy  # noqa: E402

DEG = 180.0 / np.pi
NPZ = "/workspace/controllers/rl/runs/d3b/d3b_best.npz"


def fly(seed: int, n_steps: int):
    d = np.load(NPZ, allow_pickle=True)
    g = np.asarray(d["genome"], np.float64)
    gm = M.Genome(segments=6, channels=3, param="bank", roll_gate=True)
    env, base = M.build_env(M.CELLS["merge1500"], seed, 24.0)
    opp = base.opponent
    log = []

    real_act = opp.act

    def wrapped(s, learner):
        pre_state, pre_sign, pre_dwell = opp.state, opp._break_sign, opp._dwell
        a4 = real_act(s, learner)
        rg = relative(s, learner)  # learner seen from T3
        rg_t = relative(learner, s)  # T3 seen from learner  (== the recorder's `rel`)
        log.append(
            dict(
                pre_state=pre_state,
                post_state=opp.state,
                pre_sign=pre_sign,
                post_sign=opp._break_sign,
                pre_dwell=pre_dwell,
                t3_ata=rg.ata * DEG,
                t3_ata_az=rg.ata_az * DEG,
                my_ata=rg_t.ata * DEG,
                rng=rg_t.rng,
                aspect=rg_t.aspect * DEG,
                threatened=bool(rg_t.ata < C.THREAT_CONE_RAD and rg_t.rng < C.THREAT_RANGE),
                clear=bool(rg_t.ata > C.THREAT_CONE_EXIT_RAD or rg_t.rng > C.THREAT_RANGE_EXIT),
                opp_vt=s.airspeed,
                opp_phi=s.phi * DEG,
                opp_nz=s.nz,
                opp_alt=s.alt,
                opp_es=specific_energy(s),
                opp_ammo=int(s.ammo),
                opp_fire=float(a4[4]),
                ail=float(a4[0]),
                elev=float(a4[1]),
                thr=float(a4[3]),
            )
        )
        return a4

    opp.act = wrapped

    def ctl(step, me, bd, rel):
        return gm.act(g, step, n_steps, me)

    r = M.roll_once(env, base, ctl, n_steps, keep=False)
    return r, log


def hold_runs(mask):
    out, s = [], None
    for i, m in enumerate(mask):
        if m and s is None:
            s = i
        if not m and s is not None:
            out.append((s, i - 1))
            s = None
    if s is not None:
        out.append((s, len(mask) - 1))
    return out


DT = 1.0 / M.RATE_HZ  # the opponent FSM ticks once per SIM tick (action_repeat is a wrapper)
for seed, n in [(62, 576), (28, 576)]:
    r, L = fly(seed, n)
    print("=" * 92)
    print(
        f"SEED {seed}: event={r.event} kill={r.kill} steps={r.steps} hp_dealt={r.hp_dealt:.1f} "
        f"tleth={r.tleth_s:.3f}  (log rows {len(L)})"
    )
    runs, s0 = [], 0
    for i in range(1, len(L) + 1):
        if i == len(L) or L[i]["post_state"] != L[s0]["post_state"]:
            runs.append((L[s0]["post_state"], s0, i - 1))
            s0 = i
    print("  MEASURED FSM timeline (from the live Tier3Opponent, not a replication):")
    for st, a, b in runs:
        print(
            f"    {st:<16} steps {a:3d}-{b:3d}  t {a * DT:6.3f}-{b * DT:6.3f} s  "
            f"({b - a + 1} steps, {b - a + 1} / {len(L)} = {100.0 * (b - a + 1) / len(L):.1f}%)"
        )
    sgn = [x["post_sign"] for x in L]
    flips = [i for i in range(1, len(sgn)) if sgn[i] != sgn[i - 1]]
    print(f"  _break_sign flips at steps {flips}  (values {[sgn[i] for i in flips]})")
    d = [i for i in range(len(L)) if L[i]["post_state"] == "defensive"]
    if d:
        a, b = d[0], d[-1]
        seg = L[a : b + 1]
        nz = [x["opp_nz"] for x in seg]
        phi = [x["opp_phi"] for x in seg]
        es = [x["opp_es"] for x in seg]
        alt = [x["opp_alt"] for x in seg]
        vt = [x["opp_vt"] for x in seg]
        print(f"  DEFENSIVE span {a}-{b} = {b - a + 1} steps = {(b - a + 1) * DT:.3f} s")
        print(
            f"    opp_nz   first {nz[0]:.3f}  last {nz[-1]:.3f}  min {min(nz):.3f}  max {max(nz):.3f}  mean {sum(nz) / len(nz):.3f}"
        )
        print(
            f"    opp_phi  first {phi[0]:.2f}  last {phi[-1]:.2f}  min {min(phi):.2f}  max {max(phi):.2f}"
        )
        print(
            f"    opp_vt   first {vt[0]:.2f}  last {vt[-1]:.2f}  min {min(vt):.2f}  max {max(vt):.2f}"
        )
        print(
            f"    opp_alt  first {alt[0]:.1f}  last {alt[-1]:.1f}  min {min(alt):.1f}  max {max(alt):.1f}"
        )
        print(
            f"    opp_es   first {es[0]:.1f}  last {es[-1]:.1f}  min {min(es):.1f}  max {max(es):.1f}  -> {es[-1] - es[0]:+.1f} m"
        )
        for lbl, m in [
            ("nz>3", [x > B.NZ_LOADED for x in nz]),
            (
                "nz>3 AND |phi|>76.5",
                [nz[i] > B.NZ_LOADED and abs(phi[i]) > 0.9 * 85.0 for i in range(len(nz))],
            ),
        ]:
            rr = [(x + a, y + a) for x, y in hold_runs(m)]
            tot = sum(y - x + 1 for x, y in rr)
            lng = max((y - x + 1 for x, y in rr), default=0)
            print(
                f"    LOAD-GATED HOLD {lbl:<22} total {tot:3d} steps ({tot * DT:.3f} s), "
                f"longest {lng:3d} ({lng * DT:.3f} s), runs {rr}"
            )
    # whole-episode opponent energy/altitude story
    alt = [x["opp_alt"] for x in L]
    es = [x["opp_es"] for x in L]
    vt = [x["opp_vt"] for x in L]
    rng = [x["rng"] for x in L]
    print(f"  WHOLE EPISODE ({len(L)} steps = {len(L) * DT:.3f} s):")
    print(
        f"    opp_alt  0:{alt[0]:.1f}  end:{alt[-1]:.1f}  min {min(alt):.1f} (step {alt.index(min(alt))})  max {max(alt):.1f} (step {alt.index(max(alt))})"
    )
    print(
        f"    opp_es   0:{es[0]:.1f}  end:{es[-1]:.1f}  min {min(es):.1f} (step {es.index(min(es))})  max {max(es):.1f} (step {es.index(max(es))})  net {es[-1] - es[0]:+.1f}"
    )
    print(
        f"    opp_vt   0:{vt[0]:.1f}  end:{vt[-1]:.1f}  min {min(vt):.1f} (step {vt.index(min(vt))})  max {max(vt):.1f}"
    )
    print(
        f"    rng      0:{rng[0]:.1f}  end:{rng[-1]:.1f}  max {max(rng):.1f} (step {rng.index(max(rng))})"
    )
    print(
        f"    opp_ammo 0:{L[0]['opp_ammo']}  end:{L[-1]['opp_ammo']}  fired {L[0]['opp_ammo'] - L[-1]['opp_ammo']}"
    )
