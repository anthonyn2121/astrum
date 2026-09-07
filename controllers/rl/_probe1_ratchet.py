"""PROBE 1b -- size the one-sided K_LEAD ratchet on kinematically consistent trajectories,
and verify the arithmetic of the proposed control-step NHR patch. No sim needed.
"""

import sys
import types
from math import atan2, cos, degrees, radians, sin, tan

import numpy as np

sys.modules.setdefault("astrum_env", types.ModuleType("astrum_env"))
sys.path.insert(0, "/home/anthony_nguyen/aca/astrum/crates/astrum-env/python")

from astrum_dogfight import constants as C  # noqa: E402
from astrum_dogfight import reward as R  # noqa: E402
from astrum_dogfight.geometry import AState, relative  # noqa: E402

GAM, REPEAT, HZ = 0.995, 5, 120.0
CTL_HZ = HZ / REPEAT
DT = 1.0 / HZ
G = 9.80665


class Jet:
    """3-DoF coordinated-turn point mass at constant speed: bank -> turn rate."""

    def __init__(self, pos, hdg, v, alt):
        self.pos = np.array([pos[0], pos[1], alt], float)
        self.hdg = hdg
        self.v = v
        self.bank = 0.0

    def step(self, bank, dt=DT):
        self.bank = bank
        # coordinated level turn: psi_dot = g tan(phi) / V
        self.hdg += G * tan(bank) / self.v * dt
        self.pos[0] += self.v * sin(self.hdg) * dt
        self.pos[1] += self.v * cos(self.hdg) * dt

    def state(self):
        vel = np.array([self.v * sin(self.hdg), self.v * cos(self.hdg), 0.0])
        return AState(id=1, team=1, alive=True, pos=self.pos.copy(), vel=vel,
                      phi=self.bank, theta=0.0, psi=self.hdg, p=0.0, q=0.0, r=0.0,
                      alpha=radians(8.0), beta=0.0, nz=1.0 / max(cos(self.bank), 1e-3),
                      airspeed=self.v, alt=float(self.pos[2]), health=100.0, ammo=511)


def run(label, me_j, bd_j, me_bank_fn, bd_bank_fn, secs=60.0, outcome=True):
    """Integrate, feed every tick through the SHIPPED reward.compute, total each term."""
    n = int(secs * HZ)
    acc = {}
    disc_acc = {}
    prev_me, prev_bd = me_j.state(), bd_j.state()
    tv_lead_down = 0.0  # total variation of Phi_L on the DOWN side, out of zone
    rngs = []
    for i in range(n):
        t = i * DT
        me_j.step(me_bank_fn(t))
        bd_j.step(bd_bank_fn(t))
        me, bd = me_j.state(), bd_j.state()
        out = R.compute(prev_me, prev_bd, me, bd, 0.0, C.GAMMA, 0.0, 0.0, 0.0, outcome)
        k_ctl = i // REPEAT
        d = GAM ** k_ctl
        for key in ("close_reward", "aim_reward", "aspect_reward", "lead_reward",
                    "energy_reward", "rel_energy_reward", "band_reward", "track_reward"):
            v = out.info.get(key, 0.0)
            acc[key] = acc.get(key, 0.0) + v
            disc_acc[key] = disc_acc.get(key, 0.0) + d * v
        rg = relative(me, bd)
        rngs.append(rg.rng)
        prev_me, prev_bd = me, bd
    print(f"\n  --- {label}  ({secs:.0f} s, rng {min(rngs):.0f}..{max(rngs):.0f} m) ---")
    print(f"      {'term':18s} {'undiscounted':>13s} {'discounted':>12s}")
    tot = totd = 0.0
    for key in ("close_reward", "aim_reward", "aspect_reward", "lead_reward",
                "energy_reward", "rel_energy_reward", "band_reward", "track_reward"):
        tot += acc.get(key, 0.0)
        totd += disc_acc.get(key, 0.0)
        print(f"      {key:18s} {acc.get(key,0.0):+13.2f} {disc_acc.get(key,0.0):+12.2f}")
    print(f"      {'TOTAL':18s} {tot:+13.2f} {totd:+12.2f}")
    return acc, disc_acc


print("=" * 78)
print("1) THE MERGE PASS: K_LEAD charged on the fly-through, never repaid")
print("=" * 78)
# Pure head-on, no turning: both jets fly straight through each other's 3-9 line.
me = Jet((0.0, -1500.0), 0.0, 220.0, 4000.0)
bd = Jet((150.0, 1500.0), radians(180.0), 200.0, 4000.0)
run("straight head-on fly-through, no turns", me, bd, lambda t: 0.0, lambda t: 0.0, secs=20.0)
print("      ^ lead_reward is the ONE-SIDED ratchet: the lay swings 0->180 deg as he passes")
print("        (rng > CONVERT_RANGE_MAX=1000 the whole time => the UP swing pays 0,")
print("         reward.py:388-390 `lead_r = raw_lead if (raw_lead <= 0 or in_zone) else 0`)")

print()
print("=" * 78)
print("2) A 60 s ONE-CIRCLE CO-ORBIT (the neutral merge the program is stuck on)")
print("=" * 78)
me = Jet((0.0, -1500.0), 0.0, 220.0, 4000.0)
bd = Jet((300.0, 1500.0), radians(180.0), 200.0, 4000.0)
# both pull a hard level turn the same way after the pass -> co-orbit at ~equal radius
run("both pull 70 deg bank (~2.9 g level) throughout", me, bd,
    lambda t: radians(70.0), lambda t: radians(70.0), secs=60.0)

print()
print("=" * 78)
print("3) WHAT A DISPLACEMENT ROLL COSTS ON TOP: same co-orbit, one 4 s out-of-plane roll")
print("=" * 78)
for roll_at in (10.0,):
    me = Jet((0.0, -1500.0), 0.0, 220.0, 4000.0)
    bd = Jet((300.0, 1500.0), radians(180.0), 200.0, 4000.0)
    base, based = run("baseline: hold the turn", me, bd,
                      lambda t: radians(70.0), lambda t: radians(70.0), secs=30.0)
    me = Jet((0.0, -1500.0), 0.0, 220.0, 4000.0)
    bd = Jet((300.0, 1500.0), radians(180.0), 200.0, 4000.0)

    def me_bank(t, t0=roll_at):
        # unload + roll away for 4 s (reverse the bank), then re-establish the turn
        return radians(-70.0) if t0 <= t < t0 + 4.0 else radians(70.0)

    roll, rolld = run(f"displacement roll at t={roll_at:.0f}s for 4 s", me, bd,
                      me_bank, lambda t: radians(70.0), secs=30.0)
    print()
    print("      DELTA (roll - baseline), discounted:")
    tot = 0.0
    for k in sorted(set(base) | set(roll)):
        dv = rolld.get(k, 0.0) - based.get(k, 0.0)
        tot += dv
        print(f"        {k:18s} {dv:+9.2f}")
    print(f"        {'TOTAL':18s} {tot:+9.2f}   "
          f"vs R_KILL 6 s later = {C.R_KILL*GAM**(6*CTL_HZ):+.2f}")

print()
print("=" * 78)
print("4) PATCH ARITHMETIC: control-step NHR emitted per tick")
print("=" * 78)
print("  proposed per-tick shaping:  K*(Phi' - Phi)  +  ((gamma_ctl - 1)/REPEAT)*K*Phi'")
print("  summed over the 5 ticks of control step k this is")
print("      K*(Phi_{k+1} - Phi_k) + (gamma-1)*K*mean(Phi over the 5 ticks)")
print("   ~= K*(gamma*Phi_{k+1} - Phi_k)                (the NHR form, at the CONTROL rate)")
print("  approximation error per control step = (gamma-1)*K*(mean(Phi) - Phi_{k+1}):")
for K, nm, rate in ((C.K_ASPECT, "K_ASPECT", "aspect 190 deg/s (the measured merge crossing)"),
                    (C.K_CLOSE, "K_CLOSE", "closure 400 m/s (max)"),
                    (C.K_LEAD, "K_LEAD", "lead 200 deg/s")):
    if nm == "K_ASPECT":
        dphi_ctl = (190.0 / CTL_HZ) / 180.0
    elif nm == "K_CLOSE":
        dphi_ctl = (400.0 / CTL_HZ) / 3000.0
    else:
        dphi_ctl = 0.5 * (200.0 / CTL_HZ) * radians(1.0)
    err = abs((GAM - 1.0) * K * 0.5 * dphi_ctl)
    print(f"    {nm:9s} dPhi/ctl-step = {dphi_ctl:.5f} ({rate}) -> |err| <= {err:.6f} "
          f"per control step ({err*CTL_HZ*60:.3f} over 60 s undiscounted)")

print()
print("  the added standing term's magnitude, per control step:")
for K, nm in ((C.K_CLOSE, "K_CLOSE"), (C.K_ASPECT, "K_ASPECT"), (C.K_AIM, "K_AIM"),
              (C.K_LEAD, "K_LEAD")):
    print(f"    {nm:9s} (1-gamma)*K*|Phi|  at |Phi|=1: {(1-GAM)*K:+.4f}/ctl-step "
          f"= {(1-GAM)*K*CTL_HZ:+.3f}/s")
