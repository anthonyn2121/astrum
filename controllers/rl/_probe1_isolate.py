"""PROBE 1d -- ISOLATE the two defects on identical trajectories.

For each trajectory we compute, at the CONTROL-step rate (the rate the critic discounts at):
  J_shipped  = what reward.compute actually emits (gamma=1 differencing + one-sided gates)
  J_ideal    = K*(Phi' - Phi) with NO gates            (gamma=1 differencing only)
  J_nhr      = K*(gamma*Phi' - Phi)  with NO gates     (policy-invariant NHR form)
  bias       = J_ideal - J_nhr = K(1-gamma) * sum_k gamma^k Phi_{k+1}
  gate_loss  = J_shipped - J_ideal
Comparing bias/gate_loss between a baseline and a maneuvering trajectory isolates the
structural penalty from the genuine geometric consequence of the maneuver.
"""

import sys
import types
from math import cos, degrees, exp, radians, sin

import numpy as np

sys.modules.setdefault("astrum_env", types.ModuleType("astrum_env"))
sys.path.insert(0, "/home/anthony_nguyen/aca/astrum/crates/astrum-env/python")

from astrum_dogfight import constants as C  # noqa: E402
from astrum_dogfight import reward as R  # noqa: E402
from astrum_dogfight.geometry import AState, body_bearing, relative  # noqa: E402

GAM, REPEAT, HZ = 0.995, 5, 120.0
DT, G = 1.0 / HZ, 9.80665


class Jet3:
    def __init__(self, pos, chi, gam_fp, v):
        self.p = np.array(pos, float)
        self.chi, self.g, self.v = chi, gam_fp, v
        self.bank, self.n = 0.0, 1.0

    def step(self, bank, n, dt=DT):
        self.bank, self.n = bank, n
        cg = max(cos(self.g), 0.15)
        self.g += (G / self.v) * (n * cos(bank) - cos(self.g)) * dt
        self.chi += (G / self.v) * (n * sin(bank) / cg) * dt
        self.v = max(self.v + (-G * sin(self.g) - 0.6 * (n - 1.0) ** 2) * dt, 70.0)
        self.p += self.vel() * dt

    def vel(self):
        return np.array([self.v * cos(self.g) * sin(self.chi),
                         self.v * cos(self.g) * cos(self.chi), self.v * sin(self.g)])

    def state(self):
        a = radians(3.0) * self.n
        return AState(id=1, team=1, alive=True, pos=self.p.copy(), vel=self.vel(),
                      phi=self.bank, theta=self.g + a, psi=self.chi, p=0.0, q=0.0, r=0.0,
                      alpha=a, beta=0.0, nz=self.n, airspeed=self.v,
                      alt=float(self.p[2]), health=100.0, ammo=511)


# the four potentials, transcribed from reward.py:212-213, 256, 268-269, 385-386
def phis(rg):
    return {
        "close": -min(max(rg.rng, C.R_CLOSE_MIN), C.TRACK_RANGE) / C.TRACK_RANGE,
        "aim": rg.gun_quality,
        "aspect": -min(max(rg.aspect, 0.0), C.PURSUIT_GATE_HI) / C.PURSUIT_GATE_HI,
        "lead": (1.0 + cos(rg.lead_error)) * 0.5,
    }


KS = {"close": C.K_CLOSE, "aim": C.K_AIM, "aspect": C.K_ASPECT, "lead": C.K_LEAD}
INFOKEY = {"close": "close_reward", "aim": "aim_reward", "aspect": "aspect_reward",
           "lead": "lead_reward"}


def rollout(me_cmd, secs, bd_bank=radians(70.0), bd_n=3.0,
            me0=(0.0, -900.0, 4000.0), bd0=(0.0, 0.0, 4000.0), v0=(230.0, 200.0)):
    bd = Jet3(list(bd0), 0.0, 0.0, v0[1])
    me = Jet3(list(me0), 0.0, 0.0, v0[0])
    n = int(secs * HZ)
    prev_me, prev_bd = me.state(), bd.state()
    ship = {k: 0.0 for k in KS}
    phi_ctl = {k: [phis(relative(prev_me, prev_bd))[k]] for k in KS}
    other = 0.0
    for i in range(n):
        t = i * DT
        b, nz = me_cmd(t, me, bd)
        me.step(b, nz)
        bd.step(bd_bank, bd_n)
        ms, bs = me.state(), bd.state()
        out = R.compute(prev_me, prev_bd, ms, bs, 0.0, C.GAMMA, 0.0, 0.0, 0.0, True)
        d = GAM ** (i // REPEAT)
        for k in KS:
            ship[k] += d * out.info.get(INFOKEY[k], 0.0)
        for k in ("spiral_penalty", "corner_penalty", "deck_penalty", "track_reward",
                  "energy_reward", "rel_energy_reward", "band_reward"):
            other += d * out.info.get(k, 0.0)
        if (i + 1) % REPEAT == 0:
            p = phis(relative(ms, bs))
            for k in KS:
                phi_ctl[k].append(p[k])
        prev_me, prev_bd = ms, bs
    ideal, nhr = {}, {}
    for k in KS:
        seq, K = phi_ctl[k], KS[k]
        ideal[k] = sum(GAM ** j * K * (seq[j + 1] - seq[j]) for j in range(len(seq) - 1))
        nhr[k] = sum(GAM ** j * K * (GAM * seq[j + 1] - seq[j]) for j in range(len(seq) - 1))
    rg = relative(prev_me, prev_bd)
    return dict(ship=ship, ideal=ideal, nhr=nhr, other=other, end=rg, v=prev_me.airspeed)


def pure(t, me, bd):
    ms, bs = me.state(), bd.state()
    off, bank_err, _ = body_bearing(ms, bs.pos)
    return (float(np.clip(ms.phi + 180.0 * bank_err * DT, -radians(85), radians(85))),
            float(np.clip(1.0 + 6.0 * off, 1.0, 5.8)))


def with_roll(t0, dur):
    def f(t, me, bd):
        b, n = pure(t, me, bd)
        if t0 <= t < t0 + dur:
            return (b + radians(120.0) if b < 0 else b - radians(120.0)), 1.0
        return b, n
    return f


SECS = 14.0
print("=" * 84)
print("ISOLATION: same 14 s window, bandit in a 3 g / 70 deg level break, learner 900 m astern")
print("=" * 84)
runs = [("A pure pursuit", pure)]
for t0, dur in ((3.0, 2.0), (3.0, 4.0), (3.0, 6.0)):
    runs.append((f"B roll t={t0:.0f} dur={dur:.0f}s", with_roll(t0, dur)))

base = None
for label, cmd in runs:
    r = rollout(cmd, SECS)
    tot_ship = sum(r["ship"].values())
    tot_ideal = sum(r["ideal"].values())
    tot_nhr = sum(r["nhr"].values())
    bias = tot_ideal - tot_nhr
    gate = tot_ship - tot_ideal
    e = r["end"]
    print(f"\n  {label}")
    print(f"    end: rng {e.rng:6.0f}  aspect {degrees(e.aspect):5.1f}  "
          f"ata {degrees(e.ata):5.1f}  lead {degrees(e.lead_error):5.1f}  V {r['v']:5.1f}")
    print(f"    {'term':8s} {'J_shipped':>10s} {'J_ideal':>10s} {'J_NHR':>10s} "
          f"{'bias':>9s} {'gate_loss':>10s}")
    for k in ("close", "aspect", "aim", "lead"):
        print(f"    {k:8s} {r['ship'][k]:+10.2f} {r['ideal'][k]:+10.2f} {r['nhr'][k]:+10.2f} "
              f"{r['ideal'][k]-r['nhr'][k]:+9.2f} {r['ship'][k]-r['ideal'][k]:+10.2f}")
    print(f"    {'SUM':8s} {tot_ship:+10.2f} {tot_ideal:+10.2f} {tot_nhr:+10.2f} "
          f"{bias:+9.2f} {gate:+10.2f}")
    print(f"    other dense (track/energy/band/spiral/corner/deck), discounted: "
          f"{r['other']:+.2f}")
    if base is None:
        base = (tot_ship, tot_nhr, bias, gate, r)
    else:
        b_ship, b_nhr, b_bias, b_gate, br = base
        print(f"    vs A:  d(J_NHR) = {tot_nhr-b_nhr:+8.2f}   <- the GEOMETRIC consequence "
              f"(what an invariant shaping would say)")
        print(f"           d(bias) = {bias-b_bias:+8.2f}   <- penalty from gamma=1 differencing")
        print(f"           d(gate) = {gate-b_gate:+8.2f}   <- penalty from the one-sided gates")
        print(f"           d(J_shipped) = {tot_ship-b_ship:+8.2f}  (= sum of the three)")
        print(f"           structural share = "
              f"{100*abs((bias-b_bias)+(gate-b_gate))/max(abs(tot_ship-b_ship),1e-9):.0f}% "
              f"of the total penalty")

print()
print("=" * 84)
print("SAME, but a MERGE start (head-on 2500 m, both turning) -- the user's target scenario")
print("=" * 84)
base = None
for label, cmd in runs:
    r = rollout(cmd, 20.0, bd_bank=radians(75.0), bd_n=4.0,
                me0=(0.0, -1250.0, 4200.0), bd0=(250.0, 1250.0, 4000.0), v0=(230.0, 215.0))
    tot_ship = sum(r["ship"].values())
    tot_ideal = sum(r["ideal"].values())
    tot_nhr = sum(r["nhr"].values())
    bias, gate = tot_ideal - tot_nhr, tot_ship - tot_ideal
    e = r["end"]
    print(f"\n  {label}: end rng {e.rng:6.0f} aspect {degrees(e.aspect):5.1f} "
          f"ata {degrees(e.ata):5.1f} lead {degrees(e.lead_error):5.1f}")
    print(f"    J_shipped {tot_ship:+8.2f}  J_NHR {tot_nhr:+8.2f}  bias {bias:+8.2f}  "
          f"gate_loss {gate:+8.2f}   (lead gate alone {r['ship']['lead']-r['ideal']['lead']:+.2f})")
    if base is None:
        base = (tot_ship, tot_nhr, bias, gate)
    else:
        b_ship, b_nhr, b_bias, b_gate = base
        print(f"    vs A: d(J_NHR) {tot_nhr-b_nhr:+8.2f} | d(bias) {bias-b_bias:+8.2f} | "
              f"d(gate) {gate-b_gate:+8.2f} | d(J_shipped) {tot_ship-b_ship:+8.2f}")
