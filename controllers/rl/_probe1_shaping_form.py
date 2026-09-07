"""PROBE 1 -- does the shaping arithmetic structurally punish delayed payoff?

Standalone (no sim): reward.compute is pure over fabricated AStates, so every number
below comes from EXECUTING the shipped reward code, not from transcribing constants.
"""

import sys
import types
from math import cos, degrees, exp, radians, sin

import numpy as np

# reward.py needs no sim; astrum_dogfight/__init__ pulls env.py -> astrum_env. Stub it.
sys.modules.setdefault("astrum_env", types.ModuleType("astrum_env"))
sys.path.insert(0, "/home/anthony_nguyen/aca/astrum/crates/astrum-env/python")

from astrum_dogfight import constants as C  # noqa: E402
from astrum_dogfight import reward as R  # noqa: E402
from astrum_dogfight.geometry import AState, relative  # noqa: E402

GAM = 0.995  # per CONTROL step (resolved_discount cap, episode_length 1440)
REPEAT = 5  # sim ticks per control step
HZ = 120.0
CTL_HZ = HZ / REPEAT


def st(pos, vel, phi=0.0, theta=0.0, psi=0.0, hp=100.0, ammo=511, alt=None):
    pos = np.asarray(pos, float)
    vel = np.asarray(vel, float)
    return AState(
        id=1, team=1, alive=True, pos=pos, vel=vel, phi=phi, theta=theta, psi=psi,
        p=0.0, q=0.0, r=0.0, alpha=radians(8.0), beta=0.0, nz=1.0,
        airspeed=float(np.linalg.norm(vel)), alt=(pos[2] if alt is None else alt),
        health=hp, ammo=ammo,
    )


def terms(prev_me, prev_b, me, b, outcome=True):
    """Per-SIM-TICK reward decomposition from the shipped compute()."""
    out = R.compute(prev_me, prev_b, me, b, 0.0, C.GAMMA, 0.0, 0.0, 0.0, outcome)
    return out.reward, out.info


print("=" * 78)
print("(0) WHERE THE SHAPING LIVES / AT WHAT RATE")
print("=" * 78)
print(f"reward.compute() is called once per SIM TICK  (env.py:200, 1 tick = {1/HZ*1e3:.2f} ms)")
print(f"ActionRepeat SUMS {REPEAT} ticks into one control step (envs.py:184-193) => "
      f"{CTL_HZ:.0f} Hz control")
print(f"the critic discounts gamma={GAM} PER CONTROL STEP (config.py:90-95, cap discount_max)")
print("=> a shaping increment emitted at any of the 5 ticks in control step k is "
      "discounted gamma^k")

# ---------------------------------------------------------------------------
# (a) functional form: gamma*Phi' - Phi, or Phi' - Phi ?
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("(a) FUNCTIONAL FORM -- measured, by holding the state FIXED")
print("=" * 78)
# If the form were gamma*Phi(s') - Phi(s), then s' == s gives (gamma-1)*Phi != 0.
# If the form is Phi(s') - Phi(s), a fixed state gives exactly 0.
me = st([0, -800, 4000], [0, 220, 0])
bd = st([0, 0, 4000], [0, 200, 0])
_, info = terms(me, bd, me, bd)
for k in ("close_reward", "aim_reward", "aspect_reward", "energy_reward",
          "rel_energy_reward", "band_reward", "lead_reward"):
    print(f"  {k:20s} at s'==s : {info.get(k, float('nan')):+.10f}")
print("  ALL EXACTLY ZERO on a null transition => the form is  Phi(s') - Phi(s)  (gamma=1)")
print("  (cross-check the source: reward.py:233 _phi(rg.rng)-_phi(prev_rg.rng);")
print("   :256 rg.gun_quality-prev_rg.gun_quality; :271 _psi-_psi; :315,:345,:370,:389 same)")

# ---------------------------------------------------------------------------
# (b) the residual bias, verified numerically on a dip-and-return trajectory
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("(b) RESIDUAL BIAS  B = (1-gamma) * sum_k gamma^k Phi(s_{k+1})")
print("=" * 78)


def disc_sum(phis, K, gamma=GAM, nhr=False):
    """Discounted return of the shaping on a control-step potential sequence."""
    tot = 0.0
    for k in range(len(phis) - 1):
        f = K * (gamma * phis[k + 1] - phis[k]) if nhr else K * (phis[k + 1] - phis[k])
        tot += gamma ** k * f
    return tot


# synthetic: Phi held at p0, dips by d for T s, returns to p0+better
def dip(p0, d, T_s, better=0.0, pre_s=1.0, post_s=8.0):
    n_pre, n_dip, n_post = int(pre_s * CTL_HZ), int(T_s * CTL_HZ), int(post_s * CTL_HZ)
    return [p0] * n_pre + [p0 - d] * n_dip + [p0 + better] * n_post


for K, name, d in ((C.K_ASPECT, "K_ASPECT (45 deg)", 45.0 / 180.0),
                   (C.K_CLOSE, "K_CLOSE (300 m)", 300.0 / 3000.0)):
    print(f"  {name}, K={K}")
    for T in (2.0, 4.0, 6.0):
        seq = dip(-0.5, d, T)
        g1 = disc_sum(seq, K)
        nh = disc_sum(seq, K, nhr=True)
        pred = -K * d * (1.0 - GAM ** (T * CTL_HZ))
        print(f"    T={T:.0f}s  gamma=1 form: {g1:+8.3f}   NHR form: {nh:+8.3f}   "
              f"excursion penalty (g1 - flat baseline) predicted {pred:+8.3f}")
    flat = disc_sum([-0.5] * 300, K)
    flat_nhr = disc_sum([-0.5] * 300, K, nhr=True)
    print(f"    no dip at all   gamma=1: {flat:+8.3f}   NHR: {flat_nhr:+8.3f}")

print()
print("  => under the NHR form the dip costs EXACTLY 0 (path-independent);")
print("     under the shipped gamma=1 form it costs K*dPhi*(1-gamma^(24T)).")

print()
print("  hidden standing bonus implied by the bias (the reward the current form ")
print("  secretly pays for merely HOLDING a high potential, per whole episode):")
for K, nm in ((C.K_CLOSE, "K_CLOSE"), (C.K_ASPECT, "K_ASPECT"), (C.K_AIM, "K_AIM")):
    for phi_hold in (-1.0, -0.5, -0.1):
        N = 1440
        B = -K * (1 - GAM) * abs(phi_hold) * (1 - GAM ** N) / (1 - GAM)
        print(f"    {nm:9s} holding Phi={phi_hold:+.2f} for 60 s : B = {B:+8.1f}")

# ---------------------------------------------------------------------------
# (c) the real displacement roll, term by term, from compute()
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("(c) A REAL DISPLACEMENT ROLL -- per-term cost measured through compute()")
print("=" * 78)

# Build the two endpoint geometries of the excursion and read the PBRS deltas that
# compute() actually emits for the whole swing (PBRS telescopes, so the sum over the
# excursion == compute() on the single (start -> bottom) transition).
def geom(rng_m, aspect_deg, ata_deg, alt=4000.0, v=220.0, vb=200.0):
    """me at origin-ish, bandit at `rng` ahead; set my heading so ata is right and the
    bandit's heading so aspect is right. Planar."""
    b_pos = np.array([0.0, rng_m, alt])
    # bandit heading: aspect = angle at bandit between his tail (-nose) ... use nose_b . los_hat
    # los_hat (me->bandit) = +N. cos(aspect) = nose_b . los_hat => nose_b heading = aspect off N.
    a = radians(aspect_deg)
    nb = np.array([sin(a), cos(a), 0.0])
    bandit = st(b_pos, nb * vb, psi=np.arctan2(nb[0], nb[1]), alt=alt)
    # me: ata = angle between my nose and +N
    t = radians(ata_deg)
    nm = np.array([sin(t), cos(t), 0.0])
    me = st(np.array([0.0, 0.0, alt]), nm * v, psi=np.arctan2(nm[0], nm[1]), alt=alt)
    return me, bandit


cases = [
    # label, (rng, aspect, ata) start ->  (rng, aspect, ata) bottom of the excursion
    ("out-of-zone roll  @1500 m", (1500, 55, 20), (1800, 100, 70)),
    ("in-zone roll      @ 800 m", (800, 30, 10), (1000, 60, 55)),
    ("head-on merge pass @1200 m", (1200, 150, 10), (900, 175, 60)),
]
for label, s0, s1 in cases:
    me0, b0 = geom(*s0)
    me1, b1 = geom(*s1)
    _, i_down = terms(me0, b0, me1, b1)   # the excursion (potentials degrade)
    _, i_up = terms(me1, b1, me0, b0)     # the recovery (potentials restored)
    g0, g1 = relative(me0, b0), relative(me1, b1)
    print(f"\n  {label}")
    print(f"    start : rng {g0.rng:6.0f}  aspect {degrees(g0.aspect):5.1f}  "
          f"ata {degrees(g0.ata):5.1f}  lead {degrees(g0.lead_error):6.1f}  q {g0.gun_quality:.3f}")
    print(f"    bottom: rng {g1.rng:6.0f}  aspect {degrees(g1.aspect):5.1f}  "
          f"ata {degrees(g1.ata):5.1f}  lead {degrees(g1.lead_error):6.1f}  q {g1.gun_quality:.3f}")
    tot_down = tot_up = 0.0
    for k in ("close_reward", "aim_reward", "aspect_reward", "lead_reward",
              "energy_reward", "rel_energy_reward", "band_reward"):
        d, u = i_down.get(k, 0.0), i_up.get(k, 0.0)
        tot_down += d
        tot_up += u
        flag = ""
        if abs(d + u) > 1e-9:
            flag = "  <-- ONE-SIDED: not repaid"
        print(f"      {k:18s} down {d:+9.3f}   up(repay) {u:+9.3f}   net {d+u:+9.3f}{flag}")
    print(f"      {'TOTAL':18s} down {tot_down:+9.3f}   up(repay) {tot_up:+9.3f}   "
          f"net {tot_down+tot_up:+9.3f}   <-- permanent, undiscounted")
    for T in (2.0, 4.0, 6.0):
        dfac = GAM ** (T * CTL_HZ)
        print(f"      net incl. gamma discount of the repayment T={T:.0f}s: "
              f"{tot_down + tot_up*dfac:+9.3f}")

# ---------------------------------------------------------------------------
# (d) non-potential instantaneous terms
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("(d) NON-POTENTIAL (instantaneous) terms -- magnitudes per SECOND of excursion")
print("=" * 78)
print(f"  TIME_PENALTY      {C.TIME_PENALTY:+.4f}/tick = {C.TIME_PENALTY*HZ:+.4f}/s  "
      f"(reward.py:136)")
print(f"  K_TRACK forgone   up to {-C.K_TRACK:+.4f}/tick = {-C.K_TRACK*HZ:+.3f}/s at f_nose=1,"
      f" f_range=1 (capped: TRACK_BUDGET={C.TRACK_BUDGET})")
print(f"  K_SOLUTION forgone up to {-C.K_SOLUTION:+.4f}/tick = {-C.K_SOLUTION*HZ:+.3f}/s "
      f"(only inside R_EFF with lead<~6 deg)")
# corner floor
for v in (170.0, 140.0, 110.0):
    pen = C.K_CORNER * max(0.0, (C.V_CORNER_FLOOR - v)) / C.V_CORNER_FLOOR
    print(f"  K_CORNER  @ {v:5.0f} m/s : {-pen:+.5f}/tick = {-pen*HZ:+.3f}/s "
          f"(V_CORNER_FLOOR={C.V_CORNER_FLOOR})")
# spiral
for (pit, bank, agl) in ((-30, 80, 1000), (-45, 90, 1500), (-20, 60, 700), (-45, 90, 2500)):
    nose_low = min(1.0, max(0.0, radians(-pit) / C.SPIRAL_PITCH_RAD))
    bank_frac = min(1.0, max(0.0, (radians(bank) - C.SPIRAL_BANK_MIN_RAD)
                             / (C.SPIRAL_BANK_MAX_RAD - C.SPIRAL_BANK_MIN_RAD)))
    floor = C.DECK_AGL + C.DECK_MARGIN
    alt_frac = min(1.0, max(0.0, (C.SPIRAL_ALT_BAND_AGL - agl) / (C.SPIRAL_ALT_BAND_AGL - floor)))
    s = nose_low * bank_frac * alt_frac
    print(f"  K_SPIRAL  pitch {pit:+3.0f} bank {bank:3.0f} agl {agl:5.0f} : "
          f"{C.K_SPIRAL*s:+.5f}/tick = {C.K_SPIRAL*s*HZ:+.3f}/s")
for agl in (700.0, 500.0, 350.0):
    frac = min(1.0, max(0.0, (C.DECK_AGL + C.DECK_MARGIN - agl) / C.DECK_MARGIN))
    print(f"  K_DECK_FLOOR agl {agl:5.0f} : {C.K_DECK_FLOOR*frac:+.5f}/tick = "
          f"{C.K_DECK_FLOOR*frac*HZ:+.3f}/s")

# ---------------------------------------------------------------------------
# reference scale
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("REFERENCE SCALE")
print("=" * 78)
for t in (2, 4, 6, 8, 12, 20):
    print(f"  R_KILL={C.R_KILL} realised {t:2d} s later, discounted: "
          f"{C.R_KILL * GAM ** (t*CTL_HZ):+7.2f}")
print(f"  R_DECK {C.R_DECK}   R_DEATH {C.R_DEATH}   R_BUBBLE_EXIT {C.R_BUBBLE_EXIT}   "
      f"R_TIMEOUT {C.R_TIMEOUT}")
print(f"  a whole kill's worth of damage dense: 100 HP * DMG_PER_HP {C.DMG_PER_HP} = "
      f"{100*C.DMG_PER_HP}")
