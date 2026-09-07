"""D3b EXPRESSIBILITY audit -- could a 6 x 3 / 4.0 s piecewise-constant bank schedule have
WRITTEN DOWN a detector-passing displacement roll (D1) or yo-yo (D2) at all?

The negative half of D3b ("0 D1/D2 in 15,187 rollouts") only measures the environment if the
parameterisation could have expressed the event. Four measurements, in order:

  P1  ROLL AUTHORITY, isolated. Fly single bank STEPS through the exact `Genome.act` decode and
      measure the achieved roll rate and the time to accumulate 45/90/135/180/230 deg of bank,
      loaded and unloaded, from wings level and from an established loaded bank. Compare against
      the 4.0 s segment.

  P2  RESOLUTION SWEEP, constructive. chi is NOT commanded: chi = phi_lift - phi_los, so a constant
      bank target holds the lift vector while the LOS plane rotates out from under it. So the
      fair question is "what is the BEST open-loop piecewise-constant bank schedule at segment
      length L that tries to hold |chi| in the D1 band?". Built by iteration: fly, read the
      required bank phi_req(t) = phi(t) - chi(t) + CHI_TGT off the flown trace, project it onto the
      piecewise-constant basis at length L, re-fly. L in {4.0, 2.0, 1.0, 0.5} s. Report the longest
      LOADED |chi|>60 hold each L achieves, and run the detector.

  P3  CLAUSE ATTRIBUTION. Walk D1's six clauses and D2's entry/swing/return/exit gates verbatim on
      every trace and report WHICH clause rejects each duration-qualifying hold. A hold that is
      long enough and is rejected for being flown while OPENING is not a resolution failure.

  P4  D2 STRUCTURE. Count the rising edges of the `closure > CLOSURE_OVERSHOOT` entry gate and
      measure the gamma swing and exit closure available inside YOYO_RETURN_S of each edge.

Everything is flown through `_diag_true_mpc`'s own `build_env` / `roll_once` / `Genome.act`, so no
dynamics, trigger or detector code is reimplemented here.
"""

from __future__ import annotations

import math
import os
import sys

import _diag_true_mpc as M
import numpy as np
from astrum_dogfight import bfm
from astrum_dogfight import bfm_detect as B

OUT = "/workspace/controllers/rl/runs/d3b"
CELL = M.CELLS["merge1500"]
SEED = 62
DT = M.CONTROL_DT_S
HORIZON_S = 24.0
N_STEPS = int(round(HORIZON_S / DT))
_LOG: list[str] = []


def log(s: str = "") -> None:
    print(s, flush=True)
    _LOG.append(s)


# =================================================================================================
# A schedule controller that uses the EXACT `Genome.act` decode at an arbitrary time resolution
# =================================================================================================
class Sched:
    """Piecewise-constant (bank_deg, load, thr) over time, decoded exactly as `Genome.act` does.

    `phases` is a list of (duration_s, bank_deg, load, thr). The final phase is held to the end.
    The bank target is clipped to +-BANK_MAX_DEG so the schedule stays inside the genome's own
    reachable command set -- an expressibility test may not use commands the genome cannot name.
    """

    def __init__(self, phases: list[tuple[float, float, float, float]]):
        self.phases = phases
        self.edges = np.cumsum([p[0] for p in phases])

    def cmd(self, t: float) -> tuple[float, float, float]:
        i = int(np.searchsorted(self.edges, t, side="right"))
        i = min(i, len(self.phases) - 1)
        _, bank, load, thr = self.phases[i]
        return float(np.clip(bank, -M.BANK_MAX_DEG, M.BANK_MAX_DEG)), float(load), float(thr)

    def __call__(self, step, me, bd, rel):
        bank_deg, load, thr = self.cmd(step * DT)
        phi_tgt = math.radians(bank_deg)
        bank_err = phi_tgt - float(me.phi)
        ail = float(np.clip(bfm.KROLL * bank_err, -1.0, 1.0))
        mag = max(math.cos(bank_err), 0.0)
        elev = -float(np.clip(load * mag, 0.0, 1.0))
        return np.array([ail, elev, 0.0, thr, -1.0], np.float32)


def fly(controller, n_steps: int = N_STEPS, seed: int = SEED):
    """One rollout; returns (Rollout, per-step dict-of-arrays read off the detector trace)."""
    env, base = M.build_env(CELL, seed, n_steps * DT)
    rows: list[dict] = []

    def rec(step, me, bd, rel):
        a = np.asarray(controller(step, me, bd, rel), np.float32)
        bank_deg, load, thr = controller.cmd(step * DT) if hasattr(controller, "cmd") else (
            float("nan"),
            float("nan"),
            float("nan"),
        )
        rows.append(
            dict(
                step=step,
                t=step * DT,
                cmd_bank=bank_deg,
                cmd_load=load,
                cmd_thr=thr,
                ail=float(a[0]),
                elev=float(a[1]),
                phi=math.degrees(float(me.phi)),
                theta=math.degrees(float(me.theta)),
                p=math.degrees(float(me.p)),
                nz=float(me.nz),
                alt=float(me.alt),
                vt=float(me.airspeed),
            )
        )
        return a

    r = M.roll_once(env, base, rec, n_steps, keep=False)
    env.close()
    for k, row in enumerate(rows[: r.steps]):
        s = r.trace[k]
        row["chi"] = math.degrees(s.chi)
        row["chi_valid"] = bool(s.chi_valid)
        row["gamma"] = math.degrees(s.gamma)
        row["rng"] = s.rng
        row["aspect"] = math.degrees(s.aspect)
        row["ata_az"] = math.degrees(s.ata_az)
        row["closure"] = s.closure
        row["cpa"] = s.cpa_miss
    return r, rows[: r.steps]


def unwrap_deg(v: list[float]) -> list[float]:
    out, off = [v[0]], 0.0
    for k in range(1, len(v)):
        d = v[k] - v[k - 1]
        if d > 180.0:
            off -= 360.0
        elif d < -180.0:
            off += 360.0
        out.append(v[k] + off)
    return out


def runs_of(mask) -> list[tuple[int, int]]:
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


# =================================================================================================
# P1 -- roll authority
# =================================================================================================
def p1_roll_authority() -> None:
    log("\n" + "=" * 100)
    log("  P1  ROLL AUTHORITY -- the achieved roll rate of the F-16 under Control::Raw")
    log("=" * 100)
    log(
        "  CONFIG FACT (.jsbsim-variants/aircraft/f16/f16.xml:356-359): the Roll channel forms\n"
        "    fcs/roll-rate-norm = 0.31821 * velocities/p-aero-rad_sec and subtracts it from\n"
        "    fcs/aileron-cmd-norm, i.e. a[0] IS a normalised ROLL-RATE command whose full-scale\n"
        f"    value is 1/0.31821 = {1.0 / 0.31821:.4f} rad/s = {math.degrees(1.0 / 0.31821):.2f} deg/s."
    )
    log(
        f"  DECODE FACT (_diag_true_mpc.py:217): ail = clip(KROLL*(phi_tgt-phi), -1, 1) with\n"
        f"    bfm.KROLL = {bfm.KROLL}, so the aileron SATURATES for any bank error >= "
        f"{math.degrees(1.0 / bfm.KROLL):.2f} deg. Every segment change bigger than that commands\n"
        "    full roll rate for the whole roll-in."
    )

    probes = [
        ("wings-level -> +115, load 0.0 (unloaded roll)", [(8.0, 115.0, 0.0, 0.60)]),
        ("wings-level -> +115, load 1.0 (roll gate on)", [(8.0, 115.0, 1.0, 0.60)]),
        ("wings-level ->  +60, load 1.0", [(8.0, 60.0, 1.0, 0.60)]),
        ("+115 (4 s) -> +48, load 1.0 (loaded roll-back, 67 deg)", [(4.0, 115.0, 1.0, 0.60), (8.0, 48.0, 1.0, 0.60)]),
        ("+115 (4 s) ->   0, load 1.0 (loaded roll-back, 115 deg)", [(4.0, 115.0, 1.0, 0.60), (8.0, 0.0, 1.0, 0.60)]),
        ("+115 (4 s) -> -115, load 1.0 (full reversal, 230 deg)", [(4.0, 115.0, 1.0, 0.60), (8.0, -115.0, 1.0, 0.60)]),
    ]
    log(
        "\n  Each probe flown through the SAME decode. `t(X)` = wall time from the command edge to\n"
        "  accumulate X deg of |bank change|; `p` from AState.p (the sim's own body roll rate).\n"
    )
    log(
        "  probe                                                  dphi   t(45)  t(90) t(135) t(180) t(230)"
        "   pk|p|  est|p|  meanNz"
    )
    results = {}
    for name, phases in probes:
        r, rows = fly(Sched(phases), n_steps=int(round(12.0 / DT)))
        t0 = phases[0][0] if len(phases) > 1 else 0.0
        i0 = min(range(len(rows)), key=lambda k: abs(rows[k]["t"] - t0))
        phi = unwrap_deg([w["phi"] for w in rows])
        tgt = phases[-1][1]
        dtot = tgt - phi[i0]
        times = {}
        for X in (45.0, 90.0, 135.0, 180.0, 230.0):
            if abs(dtot) + 1e-9 < X:
                times[X] = float("nan")
                continue
            got = float("nan")
            for k in range(i0, len(rows)):
                if abs(phi[k] - phi[i0]) >= X:
                    got = rows[k]["t"] - rows[i0]["t"]
                    break
            times[X] = got
        # roll window = until within 10 deg of the target
        kend = len(rows) - 1
        for k in range(i0, len(rows)):
            if abs(phi[k] - tgt) <= 10.0:
                kend = k
                break
        win = range(i0, max(i0 + 1, kend + 1))
        pk = max(abs(rows[k]["p"]) for k in win)
        # "established": drop the first 0.25 s of the roll
        est_win = [k for k in win if rows[k]["t"] - rows[i0]["t"] >= 0.25]
        est = max((abs(rows[k]["p"]) for k in est_win), default=float("nan"))
        mnz = sum(rows[k]["nz"] for k in win) / len(list(win))
        results[name] = dict(times=times, pk=pk, est=est, dtot=dtot, t_settle=rows[kend]["t"] - rows[i0]["t"])
        log(
            f"  {name:52s} {dtot:+7.1f} "
            + " ".join(f"{times[X]:6.3f}" for X in (45.0, 90.0, 135.0, 180.0, 230.0))
            + f"  {pk:6.1f}  {est:6.1f}  {mnz:6.2f}"
        )
    log(
        "\n  Reference: ONE SEGMENT of the D3b parameterisation is 96 control steps = 4.000 s.\n"
        "  A segment boundary is the ONLY place the bank target can change."
    )
    return results


# =================================================================================================
# P2 -- resolution sweep: the best open-loop chi-hold schedule at segment length L
# =================================================================================================
CHI_TGT_DEG = 75.0  # inside the D1 band (>60) with margin for the drift between segment edges


def build_required_bank(rows: list[dict], chi_tgt: float) -> tuple[list[float], list[float]]:
    """phi_req(t) that would have held chi at +chi_tgt, read off a flown trace.

    chi = phi_lift - phi_los, and phi tracks phi_lift up to the (theta, alpha) tilt, so
    phi_los_equiv(t) = phi(t) - chi(t) and phi_req(t) = phi_los_equiv(t) + chi_tgt. This is a
    first-order inversion, which is why P2 ITERATES: the flown trace after tracking phi_req is a
    different trajectory, so phi_req is recomputed from it and the schedule re-projected.
    """
    t = [w["t"] for w in rows]
    phi = unwrap_deg([w["phi"] for w in rows])
    chi = [w["chi"] for w in rows]
    return t, [phi[k] - chi[k] + chi_tgt for k in range(len(rows))]


def project(t: list[float], req: list[float], L: float, t_end: float) -> list[tuple[float, float, float, float]]:
    """Project a continuous required-bank signal onto a piecewise-constant basis of length L."""
    phases = []
    n = int(math.ceil(t_end / L))
    for i in range(n):
        a, b = i * L, (i + 1) * L
        vals = [req[k] for k in range(len(t)) if a <= t[k] < b]
        v = float(np.mean(vals)) if vals else (req[-1] if req else 0.0)
        phases.append((L, v, 1.0, 0.60))
    if not phases:
        phases = [(t_end, 0.0, 1.0, 0.60)]
    return phases


def loaded_chi_holds(rows: list[dict], bar_deg: float) -> list[tuple[int, int, float]]:
    mask = [abs(w["chi"]) > bar_deg and w["nz"] > B.NZ_LOADED for w in rows]
    out = []
    for i, j in runs_of(mask):
        out.append((i, j, rows[j]["t"] - rows[i]["t"]))
    return out


def p2_resolution_sweep() -> dict:
    log("\n" + "=" * 100)
    log("  P2  RESOLUTION SWEEP -- best open-loop piecewise-constant bank schedule at length L")
    log("=" * 100)
    log(
        f"  Target: hold chi at +{CHI_TGT_DEG:.0f} deg (D1 needs |chi| > {B.CHI_OUT_DEG:.0f} for\n"
        f"  >= {B.CHI_MIN_HOLD_S:.1f} s with nz > {B.NZ_LOADED:.0f}). load = 1.0, thr = 0.60 fixed, so the only\n"
        "  variable is the bank schedule -- exactly the channel the genome parameterises.\n"
        "  3 iterations per L: fly -> re-read phi_req -> re-project -> re-fly.\n"
    )
    # seed the iteration with the incumbent genome's own trace (its seg 0 is the in-plane pull)
    gm = M.Genome(segments=6, channels=3, param="bank", roll_gate=True)
    g = np.load(f"{OUT}/d3b_best.npz")["genome"]

    class GenomeCtl:
        def __init__(self, g, gm, n):
            self.g, self.gm, self.n = g, gm, n

        def cmd(self, t):
            s = self.gm.seg_of(int(round(t / DT)), self.n)
            p = self.g[s * 3 : (s + 1) * 3]
            return (
                M.BANK_MAX_DEG * float(np.clip(p[0], -1, 1)),
                0.5 * (float(np.clip(p[1], -1, 1)) + 1.0),
                0.5 * (float(np.clip(p[2], -1, 1)) + 1.0),
            )

        def __call__(self, step, me, bd, rel):
            return self.gm.act(self.g, step, self.n, me)

    r0, rows0 = fly(GenomeCtl(g, gm, N_STEPS))
    log(
        f"  seed trace = the incumbent genome, {r0.steps} steps, event={r0.event}, "
        f"longest loaded |chi|>60 hold "
        f"{max([h[2] for h in loaded_chi_holds(rows0, B.CHI_OUT_DEG)], default=0.0):.3f} s"
    )

    out = {}
    log(
        "\n     L(s) segs_in_7.2s iter  steps  event   longest loaded |chi|>60 hold   #holds>=0.5s"
        "   D1  D2  D2c  D3   maxNz  dalt(m)"
    )
    for L in (4.0, 2.0, 1.0, 0.5):
        rows = rows0
        for it in range(3):
            t, req = build_required_bank(rows, CHI_TGT_DEG)
            phases = project(t, req, L, HORIZON_S)
            r, rows_new = fly(Sched(phases))
            hold = loaded_chi_holds(rows_new, B.CHI_OUT_DEG)
            longest = max([h[2] for h in hold], default=0.0)
            nq = sum(1 for h in hold if h[2] >= B.CHI_MIN_HOLD_S)
            det = B.detect_all(r.trace)
            log(
                f"     {L:4.1f} {int(math.ceil(7.208 / L)):11d} {it:5d} {r.steps:6d} "
                f"{str(r.event):>7s} {longest:29.3f} s {nq:11d}   "
                f"{len(det['displacement_rolls']):2d}  {len(det['yoyos']):2d}   "
                f"{len(det['yoyos_clean']):2d}  {len(det['turning_room']):2d} "
                f"{max(w['nz'] for w in rows_new):7.2f} {rows_new[-1]['alt'] - rows_new[0]['alt']:8.0f}"
            )
            rows = rows_new
            out[(L, it)] = (r, rows_new, phases, det)
    return out


# =================================================================================================
# P3 -- clause attribution
# =================================================================================================
def d1_attribute(tr: list[B.Sample], label: str) -> None:
    """Walk `detect_displacement_rolls` clause by clause and say which one rejects each hold."""
    hold = [(abs(s.chi) > math.radians(B.CHI_OUT_DEG) and s.nz > B.NZ_LOADED) for s in tr]
    rr = B._runs(hold)
    log(f"\n  D1 attribution -- {label}: {len(rr)} loaded |chi|>60 run(s)")
    if not rr:
        log("    (no run at all: the chi/nz mask never went true)")
        return
    for i, j in rr:
        dur = tr[j].t - tr[i].t
        why = []
        if dur < B.CHI_MIN_HOLD_S:
            why.append(f"(a-dur) hold {dur:.3f}s < {B.CHI_MIN_HOLD_S}s")
        inplane = any(
            abs(tr[k].chi) < math.radians(B.CHI_INPLANE_DEG)
            for k in range(i - 1, -1, -1)
            if tr[i].t - tr[k].t <= B.CHI_LOOKBACK_S
        )
        if not inplane:
            why.append("(a-lookback) never |chi|<20 within 4 s before")
        if tr[i].closure <= 0.0:
            why.append(f"(c) closure at entry {tr[i].closure:+.1f} m/s <= 0 (OPENING)")
        exit_idx = None
        for k in range(j + 1, len(tr)):
            if tr[k].t - tr[j].t > B.CHI_RETURN_S:
                break
            if abs(tr[k].chi) < math.radians(B.CHI_RETURN_DEG):
                exit_idx = k
                break
        if exit_idx is None:
            why.append("(d) never |chi|<30 within 4 s after")
        else:
            sides = {1 if tr[k].ata_az >= 0.0 else -1 for k in range(i, exit_idx + 1)}
            if len(sides) != 1:
                why.append("(e) ata_az SIDE FLIPPED (bandit crossed the nose)")
            dasp = math.degrees(tr[exit_idx].aspect - tr[i].aspect)
            if dasp >= -B.ASPECT_GAIN_DEG:
                why.append(f"(f) d_aspect {dasp:+.1f} deg, needs <= {-B.ASPECT_GAIN_DEG:.0f}")
        log(
            f"    steps {i:3d}-{j:3d}  t {tr[i].t:6.3f}-{tr[j].t:6.3f}  dur {dur:5.3f} s  "
            f"|chi| {abs(math.degrees(tr[i].chi)):5.1f}->{abs(math.degrees(tr[j].chi)):5.1f}  "
            f"nz {min(tr[k].nz for k in range(i, j + 1)):.2f}-{max(tr[k].nz for k in range(i, j + 1)):.2f}  "
            f"clo {tr[i].closure:+6.0f}  asp {math.degrees(tr[i].aspect):5.1f}"
        )
        log(f"        -> {'PASSES ALL CLAUSES' if not why else ' ; '.join(why)}")


def d2_attribute(tr: list[B.Sample], label: str) -> None:
    n = len(tr)
    edges = [
        i
        for i in range(n)
        if tr[i].closure > B.CLOSURE_OVERSHOOT and (i == 0 or tr[i - 1].closure <= B.CLOSURE_OVERSHOOT)
    ]
    log(
        f"\n  D2 attribution -- {label}: {len(edges)} rising edge(s) of closure > "
        f"{B.CLOSURE_OVERSHOOT:.0f} m/s at steps {edges}"
    )
    for i in edges:
        g0 = tr[i].gamma
        window = [k for k in range(i, n) if tr[k].t - tr[i].t <= B.YOYO_RETURN_S]
        if len(window) < 3:
            log(f"    edge step {i}: window shorter than 3 samples")
            continue
        peak = max(window, key=lambda k: abs(tr[k].gamma - g0))
        swing = math.degrees(abs(tr[peak].gamma - g0))
        loaded = [tr[k].nz > B.NZ_LOADED for k in range(i, peak + 1)]
        lf = sum(loaded) / max(1, len(loaded))
        ret = None
        for k in range(peak + 1, n):
            if tr[k].t - tr[i].t > B.YOYO_RETURN_S:
                break
            if abs(tr[k].gamma - g0) < math.radians(B.YOYO_GAMMA_RETURN_DEG):
                ret = k
                break
        why = []
        if swing < B.YOYO_GAMMA_SWING_DEG:
            why.append(f"swing {swing:.1f} < {B.YOYO_GAMMA_SWING_DEG:.0f} deg")
        if abs(math.degrees(tr[peak].gamma)) < B.YOYO_GAMMA_EXCURSION_DEG:
            why.append(
                f"|gamma_peak| {abs(math.degrees(tr[peak].gamma)):.1f} < {B.YOYO_GAMMA_EXCURSION_DEG:.0f}"
            )
        if lf < B.LOADED_FRAC:
            why.append(f"loaded_frac {lf:.2f} < {B.LOADED_FRAC}")
        if ret is None:
            why.append(f"no return to within {B.YOYO_GAMMA_RETURN_DEG:.0f} deg of gamma0 in {B.YOYO_RETURN_S:.0f} s")
        log(
            f"    edge step {i} t {tr[i].t:6.3f}  gamma0 {math.degrees(g0):+6.1f}  "
            f"peak step {peak} t {tr[peak].t:6.3f} gamma {math.degrees(tr[peak].gamma):+6.1f} "
            f"swing {swing:5.1f}  loaded_frac {lf:4.2f}  return {ret}"
        )
        if ret is not None:
            bled = tr[ret].closure < B.CLOSURE_OVERSHOOT and tr[ret].rng < B.YOYO_EXIT_RNG_M
            kept = math.degrees(tr[ret].aspect - tr[i].aspect) <= B.YOYO_ASPECT_SLIP_DEG
            log(
                f"        exit: closure {tr[ret].closure:+7.1f} m/s (clean needs < "
                f"{B.CLOSURE_OVERSHOOT:.0f}), rng {tr[ret].rng:7.0f} m (clean needs < "
                f"{B.YOYO_EXIT_RNG_M:.0f}) -> bled={bled} kept={kept}"
            )
        log(f"        -> {'YO-YO DETECTED' if not why else ' ; '.join(why)}")
    # what gamma swing is available AT ALL over the whole trace
    g = [math.degrees(s.gamma) for s in tr]
    log(f"    gamma over the whole trace: {min(g):+.1f} .. {max(g):+.1f} deg (range {max(g) - min(g):.1f})")


# =================================================================================================
# P4 -- a constructed yo-yo attempt at fine resolution
# =================================================================================================
def p4_yoyo_attempt() -> None:
    log("\n" + "=" * 100)
    log("  P4  CONSTRUCTED YO-YO ATTEMPTS -- gamma swing >= 30 deg AND back, inside 6 s of t=0")
    log("=" * 100)
    log(
        "  D2's entry gate is a RISING edge of closure > 120 m/s. In this cell closure is +381 m/s\n"
        "  on tick 0, so the only edge the detector ever sees is step 0 and the whole yo-yo must\n"
        "  fit in [0, 6.0] s. These probes command a pure vertical (bank 0 = lift in the vertical)\n"
        "  pull then push, at phase lengths from 4.0 s (one segment) down to 0.5 s.\n"
    )
    log("     phases                                             steps  event  swing  |gpk|  lf   ret  D2 D2c")
    trials = [
        ("4.0 up / 4.0 down (segment-grained)", [(4.0, 0.0, 1.0, 1.0), (4.0, 0.0, 0.0, 1.0), (16.0, 0.0, 1.0, 1.0)]),
        ("2.0 up / 2.0 down", [(2.0, 0.0, 1.0, 1.0), (2.0, 0.0, 0.0, 1.0), (20.0, 0.0, 1.0, 1.0)]),
        ("1.5 up / 1.5 down", [(1.5, 0.0, 1.0, 1.0), (1.5, 0.0, 0.0, 1.0), (21.0, 0.0, 1.0, 1.0)]),
        ("1.0 up / 1.0 down", [(1.0, 0.0, 1.0, 1.0), (1.0, 0.0, 0.0, 1.0), (22.0, 0.0, 1.0, 1.0)]),
        ("2.0 up / 2.0 down, inverted push (bank 115)", [(2.0, 0.0, 1.0, 1.0), (2.0, 115.0, 1.0, 1.0), (20.0, 0.0, 1.0, 1.0)]),
        ("1.0 up / 1.0 down, inverted push (bank 115)", [(1.0, 0.0, 1.0, 1.0), (1.0, 115.0, 1.0, 1.0), (22.0, 0.0, 1.0, 1.0)]),
    ]
    for name, phases in trials:
        r, rows = fly(Sched(phases))
        tr = r.trace
        g0 = tr[0].gamma
        win = [k for k in range(len(tr)) if tr[k].t <= B.YOYO_RETURN_S]
        peak = max(win, key=lambda k: abs(tr[k].gamma - g0))
        swing = math.degrees(abs(tr[peak].gamma - g0))
        loaded = [tr[k].nz > B.NZ_LOADED for k in range(0, peak + 1)]
        lf = sum(loaded) / max(1, len(loaded))
        ret = None
        for k in range(peak + 1, len(tr)):
            if tr[k].t > B.YOYO_RETURN_S:
                break
            if abs(tr[k].gamma - g0) < math.radians(B.YOYO_GAMMA_RETURN_DEG):
                ret = k
                break
        det = B.detect_all(tr)
        log(
            f"     {name:50s} {r.steps:5d} {str(r.event):>6s} {swing:6.1f} "
            f"{abs(math.degrees(tr[peak].gamma)):6.1f} {lf:4.2f} {str(ret):>5s} "
            f"{len(det['yoyos']):2d} {len(det['yoyos_clean']):3d}"
        )
        if det["yoyos"]:
            for ev in det["yoyos"]:
                log(f"        {ev.kind} t {ev.t0:.3f}-{ev.t1:.3f} detail={ev.detail}")
        d2_attribute(tr, name)


def main() -> None:
    log("#" * 100)
    log("#  D3b EXPRESSIBILITY AUDIT -- can a 4.0 s piecewise-constant bank schedule write a D1/D2?")
    log("#" * 100)
    log(
        f"  cell merge1500  lateral {CELL['lateral_m']:.0f} m  sep {CELL['separation_m']:.0f} m  "
        f"aspect {CELL['aspect_deg']:.2f} deg  perch {CELL['me_dalt_ft']:.0f} ft  seed {SEED}"
    )
    log(
        f"  horizon {HORIZON_S:.1f} s = {N_STEPS} control steps x {DT * 1e3:.3f} ms; "
        f"one D3b segment = {N_STEPS // 6} steps = {N_STEPS // 6 * DT:.3f} s"
    )
    log(
        f"  detector bars: CHI_OUT {B.CHI_OUT_DEG:.0f} CHI_ROOM {B.CHI_ROOM_DEG:.0f} "
        f"CHI_RETURN {B.CHI_RETURN_DEG:.0f} CHI_MIN_HOLD {B.CHI_MIN_HOLD_S:.1f}s "
        f"CHI_RETURN_S {B.CHI_RETURN_S:.1f}s CHI_LOOKBACK {B.CHI_LOOKBACK_S:.1f}s "
        f"NZ_LOADED {B.NZ_LOADED:.1f}"
    )
    log(
        f"                 YOYO swing {B.YOYO_GAMMA_SWING_DEG:.0f} excursion "
        f"{B.YOYO_GAMMA_EXCURSION_DEG:.0f} return {B.YOYO_GAMMA_RETURN_DEG:.0f} in "
        f"{B.YOYO_RETURN_S:.0f}s  CLOSURE_OVERSHOOT {B.CLOSURE_OVERSHOOT:.0f} m/s "
        f"exit_rng {B.YOYO_EXIT_RNG_M:.0f} m"
    )

    p1_roll_authority()
    sweep = p2_resolution_sweep()

    log("\n" + "=" * 100)
    log("  P3  CLAUSE ATTRIBUTION")
    log("=" * 100)
    gm = M.Genome(segments=6, channels=3, param="bank", roll_gate=True)
    g = np.load(f"{OUT}/d3b_best.npz")["genome"]
    for sd, tag in ((62, "incumbent genome, KILL seed 62"), (28, "incumbent genome, NOKILL seed 28")):
        env, base = M.build_env(CELL, sd, N_STEPS * DT)

        def ctl(step, me, bd, rel):
            return gm.act(g, step, N_STEPS, me)

        r = M.roll_once(env, base, ctl, N_STEPS)
        env.close()
        det = B.detect_all(r.trace)
        log(
            f"\n  {tag}: {r.steps} steps event={r.event} D1={len(det['displacement_rolls'])} "
            f"D2={len(det['yoyos'])} D2c={len(det['yoyos_clean'])} D3={len(det['turning_room'])}"
        )
        d1_attribute(r.trace, tag)
        d2_attribute(r.trace, tag)
    for (L, it), (r, rows, phases, det) in sorted(sweep.items()):
        if it != 2:
            continue
        d1_attribute(r.trace, f"P2 sweep L={L:.1f}s iter 2")

    p4_yoyo_attempt()

    path = f"{OUT}/express_report.txt"
    with open(path, "w") as fh:
        fh.write("\n".join(_LOG) + "\n")
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    main()
    sys.stdout.flush()
