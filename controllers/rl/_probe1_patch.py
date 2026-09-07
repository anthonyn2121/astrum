"""PROBE 1e -- verify the proposed patch numerically, and size the two gate forfeitures."""

import sys
import types
from math import cos, degrees, radians, sin

import numpy as np

sys.modules.setdefault("astrum_env", types.ModuleType("astrum_env"))
sys.path.insert(0, "/home/anthony_nguyen/aca/astrum/crates/astrum-env/python")

from astrum_dogfight import constants as C  # noqa: E402
from astrum_dogfight import reward as R  # noqa: E402
from astrum_dogfight.geometry import AState, relative  # noqa: E402

GAM, REPEAT, HZ = 0.995, 5, 120.0
NHR_COEF = (GAM - 1.0) / REPEAT  # the proposed one-line correction coefficient

print("=" * 80)
print("A) gate inertness / gate magnitudes")
print("=" * 80)
print(f"  PURSUIT_FLOOR = {C.PURSUIT_FLOOR}  -> f_pursuit == 1 at every aspect "
      f"(constants.py:421) => the reward-v11 aspect damping on K_TRACK/K_CLOSE is INERT")
print(f"  CONVERT_ASPECT_RAD {degrees(C.CONVERT_ASPECT_RAD):.0f} deg, "
      f"CONVERT_RANGE_MAX {C.CONVERT_RANGE_MAX:.0f} m, "
      f"CONVERT_CLOSURE_MAX {C.CONVERT_CLOSURE_MAX:.0f} m/s")


def st(pos, vel, alt=None, phi=0.0):
    pos = np.asarray(pos, float)
    vel = np.asarray(vel, float)
    return AState(id=1, team=1, alive=True, pos=pos, vel=vel, phi=phi, theta=0.0,
                  psi=float(np.arctan2(vel[0], vel[1])), p=0.0, q=0.0, r=0.0,
                  alpha=radians(8.0), beta=0.0, nz=1.0,
                  airspeed=float(np.linalg.norm(vel)),
                  alt=(pos[2] if alt is None else alt), health=100.0, ammo=511)


def pair(rng, aspect_deg, ata_deg, v_me, v_bd, alt=4000.0):
    a = radians(aspect_deg)
    nb = np.array([sin(a), cos(a), 0.0])
    bd = st([0.0, rng, alt], nb * v_bd, alt=alt)
    t = radians(ata_deg)
    me = st([0.0, 0.0, alt], np.array([sin(t), cos(t), 0.0]) * v_me, alt=alt)
    return me, bd


# K_CLOSE in-zone overtake gate: how much positive closing shaping is forfeited?
print()
print("  K_CLOSE in-zone overtake gate (reward.py:234-241) — forfeited POSITIVE shaping:")
for (r0, r1, dt) in ((900, 400, 3.0), (900, 400, 6.0), (800, 300, 2.0), (1000, 600, 5.0)):
    clo = (r0 - r1) / dt
    dphi = (r0 - r1) / 3000.0
    me0, b0 = pair(r0, 20.0, 5.0, 200.0 + clo, 200.0)
    me1, b1 = pair(r1, 20.0, 5.0, 200.0 + clo, 200.0)
    i = R.compute(me0, b0, me1, b1, 0.0, C.GAMMA, 0.0, 0.0, 0.0, True).info
    g = relative(me1, b1)
    gated = abs(g.closure) > C.CONVERT_CLOSURE_MAX and g.aspect <= C.CONVERT_ASPECT_RAD \
        and g.rng <= C.CONVERT_RANGE_MAX
    print(f"    {r0}->{r1} m in {dt:.0f} s (closure {g.closure:5.1f} m/s): "
          f"would pay {C.K_CLOSE*dphi:+6.2f}, compute() paid {i['close_reward']:+6.2f}"
          f"{'   <-- ZEROED' if gated else ''}")

# K_LEAD ratchet magnitude per nose-off cycle, by range
print()
print("  K_LEAD ratchet (reward.py:388-390) — cost of one nose-off/nose-on cycle:")
for rng in (600, 900, 1100, 1500, 2500):
    for (l0, l1) in ((5.0, 60.0), (5.0, 120.0), (5.0, 180.0)):
        me0, b0 = pair(rng, 20.0, l0, 220.0, 200.0)
        me1, b1 = pair(rng, 20.0, l1, 220.0, 200.0)
        down = R.compute(me0, b0, me1, b1, 0.0, C.GAMMA, 0.0, 0.0, 0.0, True).info
        up = R.compute(me1, b1, me0, b0, 0.0, C.GAMMA, 0.0, 0.0, 0.0, True).info
        net = down["lead_reward"] + up["lead_reward"]
        print(f"    rng {rng:5d} m, ata {l0:.0f}->{l1:.0f}->{l0:.0f} deg : "
              f"down {down['lead_reward']:+7.2f}  up {up['lead_reward']:+7.2f}  "
              f"NET {net:+7.2f}{'   <-- unrepaid ratchet' if net < -1e-6 else ''}")

print()
print("=" * 80)
print("B) THE PATCH: per-tick  K*(Phi'-Phi)  +  ((gamma-1)/REPEAT)*K*Phi'")
print("=" * 80)
print(f"  NHR_COEF = (gamma-1)/REPEAT = {NHR_COEF:+.6f}  (gamma={GAM}, REPEAT={REPEAT})")


def sim_dip(K, phi_lo, phi_hi, T_s, n_s=30.0, t0=2.0, patched=False):
    """Per-TICK emission, per-CONTROL-STEP discounting -- exactly the shipped plumbing."""
    n = int(n_s * HZ)
    a, b = int(t0 * HZ), int((t0 + T_s) * HZ)
    phi = [phi_hi] * (n + 1)
    for i in range(a, min(b, n + 1)):
        phi[i] = phi_lo
    tot = 0.0
    for i in range(n):
        f = K * (phi[i + 1] - phi[i])
        if patched:
            f += NHR_COEF * K * phi[i + 1]
        tot += GAM ** (i // REPEAT) * f
    return tot


for K, nm in ((C.K_ASPECT, "K_ASPECT"), (C.K_CLOSE, "K_CLOSE"), (C.K_LEAD, "K_LEAD")):
    print(f"\n  {nm} (K={K}), potential held at -0.5, dipped by dPhi for T s, then restored:")
    print(f"    {'dPhi':>6s} {'T':>4s} | {'shipped':>9s} {'flat(ship)':>11s} {'pen':>8s} "
          f"| {'patched':>9s} {'flat(patch)':>12s} {'pen':>8s}")
    flat_s = sim_dip(K, -0.5, -0.5, 0.0)
    flat_p = sim_dip(K, -0.5, -0.5, 0.0, patched=True)
    for d in (0.1, 0.25, 0.5):
        for T in (2.0, 4.0, 6.0):
            s = sim_dip(K, -0.5 - d, -0.5, T)
            p = sim_dip(K, -0.5 - d, -0.5, T, patched=True)
            print(f"    {d:6.2f} {T:4.0f} | {s:+9.3f} {flat_s:+11.3f} {s-flat_s:+8.3f} "
                  f"| {p:+9.3f} {flat_p:+12.3f} {p-flat_p:+8.3f}")

print()
print("  => the patch drives the transient-excursion penalty to ~0 (residual is the")
print("     0.2% intra-control-step gain error, bounded by NHR_COEF*REPEAT*K*|dPhi|/2).")
print()
print("  reward-scale impact of the added standing term, per CONTROL step (5 ticks):")
for K, nm in ((C.K_CLOSE, "K_CLOSE"), (C.K_ASPECT, "K_ASPECT"), (C.K_AIM, "K_AIM"),
              (C.K_LEAD, "K_LEAD"), (C.K_ENERGY, "K_ENERGY"), (C.K_REL_ENERGY, "K_REL_ENERGY"),
              (C.K_BAND, "K_BAND")):
    print(f"    {nm:13s} K={K:6.1f}: {(GAM-1)*K:+8.4f} * Phi  per control step "
          f"(|Phi|<=1 => up to {abs((GAM-1)*K)*24:.2f}/s)")
tot = sum(((GAM - 1) * k) for k in (C.K_CLOSE, C.K_ASPECT, C.K_AIM, C.K_LEAD))
print(f"    four position potentials together: up to {abs(tot):.3f} per control step "
      f"= {abs(tot)*24:.2f}/s of NEW standing reward magnitude")
print(f"    for reference, the CURRENT typical |per-control-step| shaping in the 14 s "
      f"close-fight rollout was ~0.1-0.5")
