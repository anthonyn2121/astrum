"""Second refute pass: the break REVERSAL window, the DEFENSIVE release condition, and the
T3's own nose-on-ness / gunfire during the break. Same live-wrap instrument as pass 1."""

import sys

import numpy as np

sys.path.insert(0, "/workspace/crates/astrum-env/python")
sys.path.insert(0, "/workspace/controllers/rl")

import _diag_true_mpc as M  # noqa: E402
from astrum_dogfight import bfm_detect as B  # noqa: E402
from astrum_dogfight import constants as C  # noqa: E402
from astrum_dogfight.geometry import relative, specific_energy  # noqa: E402

DEG = 180.0 / np.pi
DT = 1.0 / M.RATE_HZ


def fly(seed, n_steps):
    d = np.load("/workspace/controllers/rl/runs/d3b/d3b_best.npz", allow_pickle=True)
    g = np.asarray(d["genome"], np.float64)
    gm = M.Genome(segments=6, channels=3, param="bank", roll_gate=True)
    env, base = M.build_env(M.CELLS["merge1500"], seed, 24.0)
    opp, log, real = base.opponent, [], base.opponent.act

    def wrapped(s, learner):
        pre = (opp.state, opp._break_sign, opp._dwell)
        a4 = real(s, learner)
        rg, rg_t = relative(s, learner), relative(learner, s)
        log.append(
            dict(
                pre_state=pre[0],
                pre_sign=pre[1],
                pre_dwell=pre[2],
                state=opp.state,
                sign=opp._break_sign,
                t3_ata=rg.ata * DEG,
                t3_ata_az=rg.ata_az * DEG,
                my_ata=rg_t.ata * DEG,
                rng=rg_t.rng,
                aspect=rg_t.aspect * DEG,
                clear=bool(rg_t.ata > C.THREAT_CONE_EXIT_RAD or rg_t.rng > C.THREAT_RANGE_EXIT),
                nz=s.nz,
                phi=s.phi * DEG,
                alt=s.alt,
                vt=s.airspeed,
                es=specific_energy(s),
                ammo=int(s.ammo),
                fire=float(a4[4]),
                ail=float(a4[0]),
                elev=float(a4[1]),
            )
        )
        return a4

    opp.act = wrapped
    r = M.roll_once(env, base, lambda st, me, bd, rel: gm.act(g, st, n_steps, me), n_steps)
    return r, log


print(
    f"BREAK_REVERSE_DEADBAND = {C.BREAK_REVERSE_DEADBAND_RAD * DEG:.2f} deg   "
    f"BREAK_BANK = {C.BREAK_BANK_RAD * DEG:.2f} deg   NZ_LOADED = {B.NZ_LOADED}"
)
r, L = fly(62, 576)
print(f"seed 62: {r.event} steps={r.steps}  log ticks={len(L)}")
print("\n### THE REVERSAL: ticks 265-350 every 5 ticks (sign flip -> unload)")
print("  tick   t_s   sign  phi_deg   nz    ail    elev   t3_ata_az  t3_ata  my_ata   rng")
for i in range(265, 356, 5):
    x = L[i]
    print(
        "  %4d %6.3f %+5.1f %8.2f %6.3f %+6.3f %+6.3f %+9.2f %7.2f %7.2f %7.1f"
        % (
            i,
            i * DT,
            x["sign"],
            x["phi"],
            x["nz"],
            x["ail"],
            x["elev"],
            x["t3_ata_az"],
            x["t3_ata"],
            x["my_ata"],
            x["rng"],
        )
    )
print("\n### unloaded stretch: ticks where nz <= NZ_LOADED inside DEFENSIVE")
d = [i for i in range(len(L)) if L[i]["state"] == "defensive"]
un = [i for i in d if L[i]["nz"] <= B.NZ_LOADED]
if un:
    print(
        f"  {len(un)} ticks = {len(un) * DT:.3f} s, span {un[0]}-{un[-1]} (t {un[0] * DT:.3f}-{un[-1] * DT:.3f} s), "
        f"min nz {min(L[i]['nz'] for i in un):.3f} g at tick {min(un, key=lambda i: L[i]['nz'])}"
    )
print("\n### T3's own nose-on-ness + gunfire inside the break (ticks %d-%d)" % (d[0], d[-1]))
seg = [L[i] for i in d]
print(
    "  t3_ata  min %.2f max %.2f  |  aspect(me) min %.2f max %.2f"
    % (
        min(x["t3_ata"] for x in seg),
        max(x["t3_ata"] for x in seg),
        min(x["aspect"] for x in seg),
        max(x["aspect"] for x in seg),
    )
)
fires = [i for i in d if L[i]["fire"] > 0.5]
print(
    "  T3 fire ticks inside break: %d (= %.3f s), first %s last %s ; rounds fired over episode %d"
    % (
        len(fires),
        len(fires) * DT,
        fires[0] if fires else None,
        fires[-1] if fires else None,
        L[0]["ammo"] - L[-1]["ammo"],
    )
)
print("\n### seed 28: the DEFENSIVE RELEASE (why it ends one tick after the kill episode would)")
r2, L2 = fly(28, 576)
print(f"seed 28: {r2.event} steps={r2.steps}  log ticks={len(L2)}")
rel_i = next(
    i
    for i in range(1, len(L2))
    if L2[i]["state"] != "defensive" and L2[i - 1]["state"] == "defensive"
)
print("  release at tick %d (t %.3f s)" % (rel_i, rel_i * DT))
print("  tick   t_s   state       my_ata    rng   clear  dwell")
for i in range(rel_i - 4, rel_i + 5):
    x = L2[i]
    print(
        "  %4d %6.3f  %-11s %7.2f %7.1f  %-5s %5d"
        % (i, i * DT, x["state"], x["my_ata"], x["rng"], x["clear"], x["pre_dwell"])
    )
print("\n### seed 28 whole-episode opponent energy after the release")
for lbl, k in [("alt", "alt"), ("es", "es"), ("vt", "vt")]:
    v = [x[k] for x in L2]
    print(
        "  %-4s tick0 %.1f  tick861 %.1f  tick2879 %.1f  min %.1f (tick %d)  max %.1f (tick %d)"
        % (lbl, v[0], v[861], v[2879], min(v), v.index(min(v)), max(v), v.index(max(v)))
    )
print(
    "  rng   tick861 %.1f  max after release %.1f at tick %d"
    % (
        L2[861]["rng"],
        max(x["rng"] for x in L2[rel_i:]),
        rel_i + max(range(len(L2) - rel_i), key=lambda j: L2[rel_i + j]["rng"]),
    )
)
st = {}
for x in L2[rel_i:]:
    st[x["state"]] = st.get(x["state"], 0) + 1
print("  states after release:", {k: f"{v} ticks ({v * DT:.2f} s)" for k, v in st.items()})
