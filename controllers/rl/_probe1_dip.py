"""PROBE 1c -- (i) path-independence check with EQUAL-LENGTH trajectories,
(ii) a 3-DoF lift-vector displacement roll in the close fight, measured through
the shipped reward.compute.
"""

import sys
import types
from math import atan2, cos, degrees, radians, sin

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

# ---------------------------------------------------------------------------
print("=" * 78)
print("(i) PATH INDEPENDENCE, equal-length trajectories, same endpoints")
print("=" * 78)
N = 24 * 30  # 30 s of control steps


def disc(phis, K, nhr):
    tot = 0.0
    for k in range(len(phis) - 1):
        f = K * (GAM * phis[k + 1] - phis[k]) if nhr else K * (phis[k + 1] - phis[k])
        tot += GAM ** k * f
    return tot


def dip_seq(p0, d, T_s, t0_s=2.0, n=N):
    s = [p0] * n
    a, b = int(t0_s * 24), int((t0_s + T_s) * 24)
    for i in range(a, min(b, n)):
        s[i] = p0 - d
    return s


for K, nm, dd in ((C.K_ASPECT, "K_ASPECT", (15 / 180, 30 / 180, 45 / 180, 60 / 180)),
                  (C.K_CLOSE, "K_CLOSE", (100 / 3000, 200 / 3000, 300 / 3000, 400 / 3000)),
                  (C.K_LEAD, "K_LEAD", (0.10, 0.20, 0.30, 0.50)),
                  (C.K_AIM, "K_AIM", (0.25, 0.5, 1.0))):
    print(f"\n  {nm} (K={K})   [gamma=1 shipped form  |  NHR form]")
    flat1, flatn = disc([-0.5] * N, K, False), disc([-0.5] * N, K, True)
    print(f"    no excursion            {flat1:+9.3f}   |  {flatn:+9.3f}")
    for d in dd:
        for T in (2.0, 4.0, 6.0):
            g1 = disc(dip_seq(-0.5, d, T), K, False)
            nh = disc(dip_seq(-0.5, d, T), K, True)
            print(f"    dPhi={d:5.3f} T={T:.0f}s        {g1:+9.3f}   |  {nh:+9.3f}"
                  f"     penalty {g1-flat1:+8.3f}  (NHR {nh-flatn:+.6f})")

# ---------------------------------------------------------------------------
# (ii) 3-DoF lift-vector model
# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("(ii) 3-DoF LIFT-VECTOR DISPLACEMENT ROLL in the close fight")
print("=" * 78)


class Jet3:
    """Point mass with a lift vector: commanded (bank, load factor). Speed from Es
    bookkeeping is out of scope -- speed responds to the flight-path angle only."""

    def __init__(self, pos, chi, gam_fp, v):
        self.p = np.array(pos, float)
        self.chi = chi        # ground track (rad, from N, +E)
        self.g = gam_fp       # flight-path angle
        self.v = v
        self.bank = 0.0
        self.n = 1.0

    def step(self, bank, n, dt=DT):
        self.bank, self.n = bank, n
        cg = max(cos(self.g), 0.15)
        self.g += (G / self.v) * (n * cos(bank) - cos(self.g)) * dt
        self.chi += (G / self.v) * (n * sin(bank) / cg) * dt
        self.v += (-G * sin(self.g) - 0.6 * (self.n - 1.0) ** 2) * dt  # crude induced bleed
        self.v = max(self.v, 70.0)
        self.p += self.vel() * dt

    def vel(self):
        return np.array([self.v * cos(self.g) * sin(self.chi),
                         self.v * cos(self.g) * cos(self.chi),
                         self.v * sin(self.g)])

    def state(self):
        alpha = radians(3.0) * self.n
        return AState(id=1, team=1, alive=True, pos=self.p.copy(), vel=self.vel(),
                      phi=self.bank, theta=self.g + alpha, psi=self.chi,
                      p=0.0, q=0.0, r=0.0, alpha=alpha, beta=0.0, nz=self.n,
                      airspeed=self.v, alt=float(self.p[2]), health=100.0, ammo=511)


def rollout(me_cmd, secs, label, verbose=False):
    """Learner starts 900 m behind a bandit in a 5 g level break turn."""
    bd = Jet3([0.0, 0.0, 4000.0], radians(0.0), 0.0, 200.0)
    me = Jet3([0.0, -900.0, 4000.0], radians(0.0), 0.0, 230.0)
    n = int(secs * HZ)
    prev_me, prev_bd = me.state(), bd.state()
    accd = {}
    acc = {}
    trace = []
    for i in range(n):
        t = i * DT
        bank, nz = me_cmd(t, me, bd)
        me.step(bank, nz)
        bd.step(radians(70.0), 3.0)  # bandit: hard level break turn
        ms, bs = me.state(), bd.state()
        out = R.compute(prev_me, prev_bd, ms, bs, 0.0, C.GAMMA, 0.0, 0.0, 0.0, True)
        d = GAM ** (i // REPEAT)
        for k in ("close_reward", "aim_reward", "aspect_reward", "lead_reward",
                  "energy_reward", "rel_energy_reward", "band_reward", "track_reward",
                  "spiral_penalty", "corner_penalty", "deck_penalty"):
            v = out.info.get(k, 0.0)
            acc[k] = acc.get(k, 0.0) + v
            accd[k] = accd.get(k, 0.0) + d * v
        rg = relative(ms, bs)
        trace.append((t, rg.rng, degrees(rg.aspect), degrees(rg.ata),
                      degrees(rg.lead_error), ms.airspeed))
        prev_me, prev_bd = ms, bs
    tot = sum(acc.values())
    totd = sum(accd.values())
    t_, rng, asp, ata, lead, v = trace[-1]
    print(f"\n  {label}")
    print(f"    end @{secs:.0f}s: rng {rng:6.0f}  aspect {asp:5.1f}  ata {ata:5.1f}  "
          f"lead {lead:6.1f}  V {v:5.1f}")
    if verbose:
        for j in range(0, len(trace), int(2 * HZ)):
            t_, rng, asp, ata, lead, v = trace[j]
            print(f"      t={t_:5.1f} rng {rng:6.0f} aspect {asp:6.1f} ata {ata:6.1f} "
                  f"lead {lead:6.1f} V {v:5.1f}")
    print(f"    {'term':18s} {'undisc':>10s} {'disc':>10s}")
    for k in sorted(acc):
        if abs(acc[k]) > 1e-9:
            print(f"    {k:18s} {acc[k]:+10.2f} {accd[k]:+10.2f}")
    print(f"    {'TOTAL':18s} {tot:+10.2f} {totd:+10.2f}")
    return acc, accd, trace


SECS = 14.0


def pure(t, me, bd):
    """Lag/lead pure pursuit: put the lift vector on him and pull."""
    ms, bs = me.state(), bd.state()
    rg = relative(ms, bs)
    from astrum_dogfight.geometry import body_bearing
    off, bank_err, fwd = body_bearing(ms, bs.pos)
    return float(np.clip(ms.phi + 3.0 * bank_err * DT * 60, -radians(85), radians(85))), \
        float(np.clip(1.0 + 6.0 * off, 1.0, 5.8))


def with_roll(t0, dur):
    def f(t, me, bd):
        b, n = pure(t, me, bd)
        if t0 <= t < t0 + dur:
            # displacement roll: roll the lift vector ~120 deg out of plane and UNLOAD
            return b + radians(120.0) if b < 0 else b - radians(120.0), 1.0
        return b, n
    return f


a0, d0, tr0 = rollout(pure, SECS, "A: pure pursuit (no out-of-plane excursion)", verbose=True)
for t0, dur in ((3.0, 2.0), (3.0, 4.0), (3.0, 6.0)):
    a1, d1, tr1 = rollout(with_roll(t0, dur), SECS,
                          f"B: displacement roll at t={t0:.0f}s for {dur:.0f}s")
    print(f"    DELTA vs A, discounted, per term:")
    tot = 0.0
    for k in sorted(set(a0) | set(a1)):
        dv = d1.get(k, 0.0) - d0.get(k, 0.0)
        if abs(dv) > 1e-6:
            print(f"      {k:18s} {dv:+9.3f}")
        tot += dv
    print(f"      {'TOTAL':18s} {tot:+9.3f}")
