"""REFUTATION probe: run the proposed FREEZE-AND-FLUSH law on the SAME kinematic
displacement roll that _probe1_dip.py (ii) already measured.

Freeze-and-flush claims: during a detected commitment window the four geometry terms
emit 0.0 and the potential's ENTRY value is held; on exit emit K*(Phi_exit - Phi_entry).
Claimed properties: (a) episode undiscounted sum preserved EXACTLY, (b) relief +14.378
discounted for a 4 s recovered excursion.

Both are tested here on a REAL trajectory instead of a synthetic recovered dip.
"""

import sys
import types
from math import cos, degrees, radians, sin

import numpy as np

sys.modules.setdefault("astrum_env", types.ModuleType("astrum_env"))
sys.path.insert(0, "/home/anthony_nguyen/aca/astrum/crates/astrum-env/python")

from astrum_dogfight import constants as C  # noqa: E402
from astrum_dogfight import reward as R  # noqa: E402
from astrum_dogfight.geometry import AState, body_bearing, relative  # noqa: E402

GAM, REPEAT, HZ = 0.995, 5, 120.0
DT = 1.0 / HZ
G = 9.80665

TERMS = ("close_reward", "aim_reward", "aspect_reward", "lead_reward")
KS = {"close_reward": C.K_CLOSE, "aim_reward": C.K_AIM,
      "aspect_reward": C.K_ASPECT, "lead_reward": C.K_LEAD}


def phis(rg):
    """The four shipped potentials, read straight off the shipped formulas."""
    return {
        "close_reward": -min(max(rg.rng, C.R_CLOSE_MIN), C.TRACK_RANGE) / C.TRACK_RANGE,
        "aim_reward": rg.gun_quality,
        "aspect_reward": -min(max(rg.aspect, 0.0), C.PURSUIT_GATE_HI) / C.PURSUIT_GATE_HI,
        "lead_reward": (1.0 + cos(rg.lead_error)) * 0.5,
    }


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
        self.v += (-G * sin(self.g) - 0.6 * (self.n - 1.0) ** 2) * dt
        self.v = max(self.v, 70.0)
        self.p += self.vel() * dt

    def vel(self):
        return np.array([self.v * cos(self.g) * sin(self.chi),
                         self.v * cos(self.g) * cos(self.chi), self.v * sin(self.g)])

    def state(self):
        alpha = radians(3.0) * self.n
        return AState(id=1, team=1, alive=True, pos=self.p.copy(), vel=self.vel(),
                      phi=self.bank, theta=self.g + alpha, psi=self.chi,
                      p=0.0, q=0.0, r=0.0, alpha=alpha, beta=0.0, nz=self.n,
                      airspeed=self.v, alt=float(self.p[2]), health=100.0, ammo=511)


def pure(me, bd):
    ms, bs = me.state(), bd.state()
    off, bank_err, fwd = body_bearing(ms, bs.pos)
    return (float(np.clip(ms.phi + 3.0 * bank_err * DT * 60, -radians(85), radians(85))),
            float(np.clip(1.0 + 6.0 * off, 1.0, 5.8)))


def rollout(secs, roll_at=None, roll_dur=0.0):
    """Returns per-tick rows: shipped emissions, the four potentials, in-window flag."""
    bd = Jet3([0.0, 0.0, 4000.0], radians(0.0), 0.0, 200.0)
    me = Jet3([0.0, -900.0, 4000.0], radians(0.0), 0.0, 230.0)
    n = int(secs * HZ)
    prev_me, prev_bd = me.state(), bd.state()
    rows = []
    for i in range(n):
        t = i * DT
        b, nz = pure(me, bd)
        inwin = roll_at is not None and roll_at <= t < roll_at + roll_dur
        if inwin:
            b, nz = (b + radians(120.0) if b < 0 else b - radians(120.0)), 1.0
        me.step(b, nz)
        bd.step(radians(70.0), 3.0)
        ms, bs = me.state(), bd.state()
        out = R.compute(prev_me, prev_bd, ms, bs, 0.0, C.GAMMA, 0.0, 0.0, 0.0, True)
        rg = relative(ms, bs)
        rows.append(dict(i=i, t=t, inwin=inwin,
                         ship={k: out.info.get(k, 0.0) for k in TERMS},
                         phi=phis(rg), rng=rg.rng, aspect=degrees(rg.aspect),
                         lead=degrees(rg.lead_error)))
        prev_me, prev_bd = ms, bs
    return rows


def freeze_flush(rows):
    """The candidate's law, exactly: emit 0 in-window, flush K*(Phi_exit-Phi_entry) on exit."""
    out = []
    entry = None
    for j, r in enumerate(rows):
        e = dict(r["ship"])
        if r["inwin"]:
            if entry is None:
                entry = rows[j - 1]["phi"] if j > 0 else r["phi"]
            for k in TERMS:
                e[k] = 0.0
        elif entry is not None:
            for k in TERMS:
                e[k] = KS[k] * (r["phi"][k] - entry[k])
            entry = None
        out.append(e)
    if entry is not None:                       # window still open at the terminal
        for k in TERMS:
            out[-1][k] = KS[k] * (rows[-1]["phi"][k] - entry[k])
    return out


def tot(rows, emis):
    u = {k: 0.0 for k in TERMS}
    d = {k: 0.0 for k in TERMS}
    peak = 0.0
    for r, e in zip(rows, emis):
        g = GAM ** (r["i"] // REPEAT)
        for k in TERMS:
            u[k] += e[k]
            d[k] += g * e[k]
        peak = max(peak, max(abs(e[k]) for k in TERMS))
    return u, d, peak


print("=" * 92)
print("FREEZE-AND-FLUSH on the RECORDED kinematic displacement roll (_probe1_dip.py (ii))")
print("  learner 900 m astern, bandit in a 5 g level break; roll = lift vector 120 deg")
print("  out of plane + unload, i.e. the maneuver the waiver is meant to protect.")
print("=" * 92)

base = rollout(14.0)
bu, bd_, _ = tot(base, [r["ship"] for r in base])
print(f"\n  A: no excursion            end rng {base[-1]['rng']:6.0f}  "
      f"aspect {base[-1]['aspect']:5.1f}  lead {base[-1]['lead']:5.1f}")
print(f"     four terms   undisc {sum(bu.values()):+9.3f}   disc {sum(bd_.values()):+9.3f}")

for dur in (2.0, 4.0, 6.0):
    rows = rollout(14.0, roll_at=3.0, roll_dur=dur)
    su, sd, speak = tot(rows, [r["ship"] for r in rows])
    fe = freeze_flush(rows)
    fu, fd, fpeak = tot(rows, fe)
    ent = rows[int(3.0 * HZ) - 1]["phi"]
    ex = rows[int((3.0 + dur) * HZ)]["phi"]
    print(f"\n  B: {dur:.0f} s out-of-plane roll at t=3 s   "
          f"end rng {rows[-1]['rng']:6.0f}  aspect {rows[-1]['aspect']:5.1f}  "
          f"lead {rows[-1]['lead']:5.1f}")
    print("     RECOVERY CHECK  Phi_exit - Phi_entry  (0 == the 'recovered excursion' the "
          "relief figure assumes)")
    for k in TERMS:
        print(f"       {k:14s} entry {ent[k]:+7.4f}  exit {ex[k]:+7.4f}  "
              f"dPhi {ex[k]-ent[k]:+7.4f}   K*dPhi {KS[k]*(ex[k]-ent[k]):+8.3f}")
    print(f"     {'term':14s} {'ship undisc':>12s} {'froz undisc':>12s} {'SUM BREAK':>10s}"
          f" {'ship disc':>10s} {'froz disc':>10s} {'relief':>9s}")
    for k in TERMS:
        print(f"     {k:14s} {su[k]:+12.3f} {fu[k]:+12.3f} {fu[k]-su[k]:+10.3f}"
              f" {sd[k]:+10.3f} {fd[k]:+10.3f} {fd[k]-sd[k]:+9.3f}")
    print(f"     {'TOTAL':14s} {sum(su.values()):+12.3f} {sum(fu.values()):+12.3f} "
          f"{sum(fu.values())-sum(su.values()):+10.3f} {sum(sd.values()):+10.3f} "
          f"{sum(fd.values()):+10.3f} {sum(fd.values())-sum(sd.values()):+9.3f}")
    print(f"     max |single-tick emission|:  shipped {speak:7.3f}   frozen {fpeak:7.3f}")

print()
print("=" * 92)
print("CONTROL: the SYNTHETIC recovered dip the +14.378 figure was measured on")
print("=" * 92)
N = 24 * 30
for k, K, d in (("aspect_reward", C.K_ASPECT, 0.167), ("lead_reward", C.K_LEAD, 0.30),
                ("aim_reward", C.K_AIM, 0.25), ("close_reward", C.K_CLOSE, 0.10)):
    for T in (0.6, 1.3, 4.0):
        p0 = -0.5
        s = [p0] * N
        a, b = int(2.0 * 24), int((2.0 + T) * 24)
        for i in range(a, min(b, N)):
            s[i] = p0 - d
        ship = sum(GAM ** i * K * (s[i + 1] - s[i]) for i in range(N - 1))
        print(f"  {k:14s} dPhi {d:5.3f}  T={T:4.1f}s   shipped {ship:+8.3f}   "
              f"frozen  +0.000   relief {-ship:+8.3f}")
