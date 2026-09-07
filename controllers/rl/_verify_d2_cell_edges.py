"""Independent audit of: "the D2 entry gate opens exactly ONCE per 24 s episode, on tick 0,
so the whole yo-yo must fit in [0, 6.0] s" -- attributed to the CELL (merge1500).

Three measurements, none of which re-implements a detector (traces are `B.make_sample` objects
produced inside `M.roll_once`, i.e. exactly what `B.detect_yoyos` reads):

  (A) REPRODUCE the 16 probe schedules of `_diag_d2_express.py` and print the rising-edge INDICES
      (the cited report prints only the COUNT and appends the literal string "(all of them at
      step 0)" -- line 154 of that script -- so the index claim was never measured there).
  (B) Ask whether a SECOND rising edge is reachable IN THIS CELL, inside 24 s, using only commands
      the D3b genome can name (Sched = the verbatim `Genome.act` decode). The 16 published probes
      all command load 1.0 AND throttle 1.0, so the throttle and unload channels were never probed.
  (C) The same question with closed-loop controllers and with longer horizons, to separate
      "the CELL" from "the 24 s horizon" and from "the open-loop parameterisation".

Run:
  docker compose -f /home/anthony_nguyen/aca/astrum/docker-compose.yml --profile rl run --rm \
      --no-deps dogfight-train bash -lc 'cd /workspace/controllers/rl && python _verify_d2_cell_edges.py'
"""

from __future__ import annotations

import math

import _diag_d3b_express as E
import _diag_true_mpc as M
import numpy as np
from astrum_dogfight import bfm as BFM
from astrum_dogfight import bfm_detect as B

DEG = 180.0 / math.pi
DT = M.CONTROL_DT_S
_LOG: list[str] = []


def log(s: str = "") -> None:
    print(s, flush=True)
    _LOG.append(s)


def rising_edges(tr) -> list[int]:
    """bfm_detect.py:411 verbatim."""
    return [
        i
        for i in range(len(tr))
        if tr[i].closure > B.CLOSURE_OVERSHOOT
        and (i == 0 or tr[i - 1].closure <= B.CLOSURE_OVERSHOOT)
    ]


def runs_above(tr) -> list[tuple[int, int]]:
    mask = [s.closure > B.CLOSURE_OVERSHOOT for s in tr]
    out, i, n = [], 0, len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        out.append((i, j))
        i = j + 1
    return out


def row(name: str, tr, event, extra: str = "") -> tuple[int, list[int]]:
    ed = rising_edges(tr)
    ra = runs_above(tr)
    nb = sum(1 for s in tr if s.closure > B.CLOSURE_OVERSHOOT)
    d2 = B.detect_yoyos(tr)
    clo = [s.closure for s in tr]
    log(
        f"  {name:<46} {len(tr):>5} {nb:>6} {len(ra):>5} {len(ed):>4} "
        f"{str(ed[:4]):>18} {min(clo):>+8.1f} {max(clo):>+8.1f} {len(d2):>3} {str(event):>11} {extra}"
    )
    if len(ed) > 1:
        for i in ed:
            log(
                f"        entry {i:>4} t {tr[i].t:6.3f} s  clo {tr[i].closure:+7.1f} m/s "
                f" rng {tr[i].rng:8.1f} m  gamma {tr[i].gamma * DEG:+7.2f} deg  nz {tr[i].nz:+5.2f}"
            )
        log(f"        runs above band: {ra}")
    return len(ed), ed


HDR = (
    f"  {'arm':<46} {'steps':>5} {'>band':>6} {'runs':>5} {'ed':>4} {'edge idx':>18} "
    f"{'cloMin':>8} {'cloMax':>8} {'D2':>3} {'event':>11}"
)


# ------------------------------------------------------------------ (A) the published 16 probes
def published_trials() -> list[tuple[str, list]]:
    trials: list[tuple[str, list]] = []
    for d_up in (1.0, 1.5, 2.0, 2.5, 3.0):
        trials.append(
            (
                f"HIGH pull {d_up:.1f}s @bank0 then bank115 pull",
                [(d_up, 0.0, 1.0, 1.0), (24.0, 115.0, 1.0, 1.0)],
            )
        )
    for d_dn in (2.0, 2.5, 3.0, 3.5, 4.0):
        trials.append(
            (
                f"LOW  bank115 pull {d_dn:.1f}s then bank0 pull",
                [(d_dn, 115.0, 1.0, 1.0), (24.0, 0.0, 1.0, 1.0)],
            )
        )
    trials.append(
        (
            "SEG  4.0s bank0 pull then 4.0s bank115 pull",
            [(4.0, 0.0, 1.0, 1.0), (20.0, 115.0, 1.0, 1.0)],
        )
    )
    trials.append(
        (
            "SEG  4.0s bank115 pull then 4.0s bank0 pull",
            [(4.0, 115.0, 1.0, 1.0), (20.0, 0.0, 1.0, 1.0)],
        )
    )
    trials.append(
        (
            "FINE 1.0 bank0 / 1.5 bank115 / rest bank0",
            [(1.0, 0.0, 1.0, 1.0), (1.5, 115.0, 1.0, 1.0), (21.5, 0.0, 1.0, 1.0)],
        )
    )
    trials.append(
        (
            "FINE 0.5 bank0 / 1.0 bank115 / rest bank0",
            [(0.5, 0.0, 1.0, 1.0), (1.0, 115.0, 1.0, 1.0), (22.5, 0.0, 1.0, 1.0)],
        )
    )
    trials.append(
        (
            "FINE 1.5 bank115 / 1.5 bank0 / rest bank115",
            [(1.5, 115.0, 1.0, 1.0), (1.5, 0.0, 1.0, 1.0), (21.0, 115.0, 1.0, 1.0)],
        )
    )
    trials.append(
        (
            "FINE 2.5 bank115 / 2.0 bank0 / rest bank115",
            [(2.5, 115.0, 1.0, 1.0), (2.0, 0.0, 1.0, 1.0), (19.5, 115.0, 1.0, 1.0)],
        )
    )
    return trials


# ---------------------------------------------------- (B) genome-expressible extend / re-attack
def probe_trials() -> list[tuple[str, list]]:
    """Schedules inside the genome's OWN command set that try to break the band and re-cross it.

    All of these are (duration_s, bank_deg, load, thr) with |bank| <= 115, load in [0,1],
    thr in [0,1] -- exactly what `Genome.act` can name at 4 s resolution (the 2.0 s ones are
    finer than the genome, and are marked, so the verdict can be reported at both resolutions).
    """
    t: list[tuple[str, list]] = []
    # hard nose-up-ish break at idle, then coast: kill own closing speed, let the bandit re-close
    t.append(
        ("BRK  4s bank0 pull idle, then coast idle", [(4.0, 0.0, 1.0, 0.0), (24.0, 0.0, 0.0, 0.0)])
    )
    t.append(
        ("BRK  4s bank0 pull idle, then coast mil", [(4.0, 0.0, 1.0, 0.0), (24.0, 0.0, 0.0, 1.0)])
    )
    t.append(
        ("BRK  8s bank0 pull idle, then coast idle", [(8.0, 0.0, 1.0, 0.0), (24.0, 0.0, 0.0, 0.0)])
    )
    # 180-out: bank hard, pull, then unload and run
    t.append(
        (
            "OUT  4s bank115 pull, 4s bank0 unload idle",
            [(4.0, 115.0, 1.0, 0.0), (4.0, 0.0, 0.0, 0.0), (24.0, 0.0, 0.0, 0.0)],
        )
    )
    t.append(
        (
            "OUT  8s bank115 pull idle, then unload idle",
            [(8.0, 115.0, 1.0, 0.0), (24.0, 0.0, 0.0, 0.0)],
        )
    )
    t.append(
        (
            "OUT  8s bank-115 pull idle, then unload idle",
            [(8.0, -115.0, 1.0, 0.0), (24.0, 0.0, 0.0, 0.0)],
        )
    )
    t.append(
        (
            "OUT  12s bank115 pull idle, then unload idle",
            [(12.0, 115.0, 1.0, 0.0), (24.0, 0.0, 0.0, 0.0)],
        )
    )
    # extend then turn back in (a re-attack): the classic second-edge shape
    t.append(
        (
            "RE   4s b115 pull, 8s b0 idle, 4s b115 pull, rest b0",
            [
                (4.0, 115.0, 1.0, 0.0),
                (8.0, 0.0, 0.0, 0.0),
                (4.0, 115.0, 1.0, 1.0),
                (24.0, 0.0, 0.0, 1.0),
            ],
        )
    )
    t.append(
        (
            "RE   4s b-115 pull, 8s b0 idle, 4s b-115 pull, rest b0",
            [
                (4.0, -115.0, 1.0, 0.0),
                (8.0, 0.0, 0.0, 0.0),
                (4.0, -115.0, 1.0, 1.0),
                (24.0, 0.0, 0.0, 1.0),
            ],
        )
    )
    t.append(
        (
            "RE   8s b115 pull idle, 4s b0 idle, 4s b115 pull mil",
            [
                (8.0, 115.0, 1.0, 0.0),
                (4.0, 0.0, 0.0, 0.0),
                (4.0, 115.0, 1.0, 1.0),
                (24.0, 0.0, 0.0, 1.0),
            ],
        )
    )
    # pure drag: idle, wings level, whole episode (bandit does all the work)
    t.append(("DRAG idle wings-level all 24 s", [(24.0, 0.0, 0.0, 0.0)]))
    t.append(("DRAG mil  wings-level all 24 s", [(24.0, 0.0, 0.0, 1.0)]))
    # the incumbent genome's own first segment, then idle coast (isolate seg 0 from seg 1)
    t.append(
        (
            "SEG0 4s b115 load1 thr0.60, then idle level",
            [(4.0, 115.0, 1.0, 0.6023), (24.0, 0.0, 0.0, 0.0)],
        )
    )
    return t


# --------------------------------------------------------------- (C) closed loop + longer clocks
def make_pursuit():
    def ctl(step, me, bd, rel):
        ail, elev = BFM._point_at(rel, lag_rad=math.radians(2.0))
        ail, elev = BFM._deck_guard(me, ail, elev)
        return np.array([ail, elev, 0.0, 1.0, -1.0], np.float32)

    return "pure pursuit, mil power (closed loop)", ctl


def make_lag_track():
    mv = BFM.LagPursuitTracker()
    mv.max_steps = 10**9

    def ctl(step, me, bd, rel):
        if step == 0:
            mv.reset(me, rel)
        a, _done = mv.step(me, rel)
        return np.array([a[0], a[1], a[2], a[3], -1.0], np.float32)

    return "lag-pursuit tracker (bfm.LagPursuitTracker)", ctl


def fly_raw(ctl, n_steps: int, seed: int):
    env, base = M.build_env(
        M.CELL if hasattr(M, "CELL") else M.CELLS["merge1500"], seed, n_steps * DT
    )
    try:
        return M.roll_once(env, base, ctl, n_steps, keep=False)
    finally:
        env.close()


def main() -> None:
    log("#" * 118)
    log("#  D2 ENTRY-GATE AUDIT -- is 'exactly one rising edge, at tick 0' a CELL fact?")
    log(
        f"#  cell merge1500   CLOSURE_OVERSHOOT = {B.CLOSURE_OVERSHOOT:.1f} m/s   "
        f"YOYO_RETURN_S = {B.YOYO_RETURN_S:.1f} s   dt = {DT * 1000:.3f} ms"
    )
    log("#" * 118)

    log("\n(A) THE 16 PUBLISHED PROBES, re-flown, with the edge INDICES printed\n")
    log(HDR)
    counts_a = []
    for name, phases in published_trials():
        r, _rows = E.fly(E.Sched(phases))
        c, _ed = row(name, r.trace, r.event)
        counts_a.append(c)
    log(
        f"\n  edges per episode over {len(counts_a)} published probes: min {min(counts_a)} max {max(counts_a)}"
    )

    log("\n(B) EXTEND / RE-ATTACK PROBES inside the genome's own command set (seed 62, 24 s)\n")
    log(HDR)
    counts_b = []
    for name, phases in probe_trials():
        r, _rows = E.fly(E.Sched(phases))
        c, _ed = row(name, r.trace, r.event)
        counts_b.append(c)
    log(
        f"\n  edges per episode over {len(counts_b)} extend/re-attack probes: min {min(counts_b)} max {max(counts_b)}"
    )

    log("\n(C) CLOSED LOOP and LONGER CLOCKS, same cell, seed 62\n")
    arms = [make_pursuit(), make_lag_track()]
    for horizon_s in (24.0, 60.0):
        n_steps = int(round(horizon_s / DT))
        log(f"  --- horizon {horizon_s:.0f} s = {n_steps} control steps")
        log(HDR)
        for name, ctl in arms:
            r = fly_raw(ctl, n_steps, 62)
            row(f"{name} [{horizon_s:.0f}s]", r.trace, r.event)
        # the incumbent genome itself, on the longer clock (segments 2-5 finally matter)
        z = np.load("runs/d3b/d3b_best.npz", allow_pickle=False)
        g = np.asarray(z["genome"], float)
        gm = M.Genome(segments=6, channels=3, param="bank", roll_gate=True)

        def gctl(step, me, bd, rel, _g=g, _gm=gm):
            return _gm.act(_g, min(step, 575), 576, me)

        r = fly_raw(gctl, n_steps, 62)
        row(f"d3b genome (held after 24 s) [{horizon_s:.0f}s]", r.trace, r.event)

    path = "/workspace/controllers/rl/runs/d3b/verify_d2_cell_edges.txt"
    with open(path, "w") as fh:
        fh.write("\n".join(_LOG) + "\n")
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
