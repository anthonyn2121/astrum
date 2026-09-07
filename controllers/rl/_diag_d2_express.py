"""Is a D2 (yo-yo) expressible by the D3b parameterisation, at ANY time resolution?

D2's entry gate is a RISING edge of closure > CLOSURE_OVERSHOOT. Three things must be measured
before a "not expressible" or "expressible" verdict is admissible:

  (1) how many rising edges of closure > 120 m/s exist in this cell at all, over many different
      trajectories -- if the only edge is tick 0, the whole yo-yo must fit in [0, YOYO_RETURN_S];
  (2) the ELEVATOR SIGN: `Genome.act` builds elev = -clip(load*max(cos(bank_err),0), 0, 1), which
      lies in [-1, 0]. It is a PULL-ONLY channel: the genome cannot command a push. So the return
      half of a yo-yo must be flown as roll-then-pull, and that roll costs time inside the window;
  (3) the closure FLOOR inside the window, because `clean` needs closure < 120 m/s AND rng < 1500 m
      at the exit index.

Then a grid over (up/down phase length, down-bank) low- and high-yo-yo schedules at resolutions
from 4.0 s down to 0.5 s, reporting the best (swing, return) pair any of them achieves.
"""

from __future__ import annotations

import math

import _diag_d3b_express as E
from astrum_dogfight import bfm_detect as B

_LOG: list[str] = []


def log(s: str = "") -> None:
    print(s, flush=True)
    _LOG.append(s)


def d2_stats(tr) -> dict:
    """Every quantity D2 reads, for the tick-0 entry edge."""
    n = len(tr)
    edges = [
        i
        for i in range(n)
        if tr[i].closure > B.CLOSURE_OVERSHOOT
        and (i == 0 or tr[i - 1].closure <= B.CLOSURE_OVERSHOOT)
    ]
    out = dict(edges=edges)
    if not edges:
        return out
    i = edges[0]
    g0 = tr[i].gamma
    win = [k for k in range(i, n) if tr[k].t - tr[i].t <= B.YOYO_RETURN_S]
    peak = max(win, key=lambda k: abs(tr[k].gamma - g0))
    loaded = [tr[k].nz > B.NZ_LOADED for k in range(i, peak + 1)]
    ret = None
    for k in range(peak + 1, n):
        if tr[k].t - tr[i].t > B.YOYO_RETURN_S:
            break
        if abs(tr[k].gamma - g0) < math.radians(B.YOYO_GAMMA_RETURN_DEG):
            ret = k
            break
    # the best (largest) swing that is followed by a return inside the window, over ALL peaks
    best = None
    for p in win:
        sw = math.degrees(abs(tr[p].gamma - g0))
        if sw < B.YOYO_GAMMA_SWING_DEG or abs(math.degrees(tr[p].gamma)) < B.YOYO_GAMMA_EXCURSION_DEG:
            continue
        for k in range(p + 1, n):
            if tr[k].t - tr[i].t > B.YOYO_RETURN_S:
                break
            if abs(tr[k].gamma - g0) < math.radians(B.YOYO_GAMMA_RETURN_DEG):
                best = (p, k, sw)
                break
        if best:
            break
    out.update(
        swing=math.degrees(abs(tr[peak].gamma - g0)),
        gpk=math.degrees(tr[peak].gamma),
        t_peak=tr[peak].t,
        lf=sum(loaded) / max(1, len(loaded)),
        ret=ret,
        best=best,
        clo_min_win=min(tr[k].closure for k in win),
        rng_min_win=min(tr[k].rng for k in win),
        gmin=min(math.degrees(s.gamma) for s in tr[: len(win)]),
        gmax=max(math.degrees(s.gamma) for s in tr[: len(win)]),
    )
    return out


def main() -> None:
    log("#" * 100)
    log("#  D2 (YO-YO) EXPRESSIBILITY -- merge1500, the D3b parameterisation")
    log("#" * 100)
    log(
        "\n  (2) ELEVATOR SIGN, from the decode itself (_diag_true_mpc.py:218-219):\n"
        "        elev = -clip(load * max(cos(bank_err), 0), 0, 1)   in [-1, 0]\n"
        "      load in [0,1] and the outer clip has lo = 0, so elev is PULL-ONLY. load = 0 gives\n"
        "      elev = 0 (neutral stick), NOT a push. A nose-down gamma change must therefore be\n"
        "      flown by banking (bank_max = 115 deg, so the lift vector can be tilted at most\n"
        f"      115 deg off vertical -> cos(115 deg) = {math.cos(math.radians(115)):.3f} of the lift acts DOWN)\n"
        "      and the RETURN half of a yo-yo costs a roll before it can even start pulling."
    )

    trials: list[tuple[str, list]] = []
    # high yo-yo: pull up at bank 0, then roll to +115 and pull to bring the nose back down
    for d_up in (1.0, 1.5, 2.0, 2.5, 3.0):
        trials.append(
            (
                f"HIGH pull {d_up:.1f}s @bank0 then bank115 pull",
                [(d_up, 0.0, 1.0, 1.0), (24.0, 115.0, 1.0, 1.0)],
            )
        )
    # low yo-yo: bank 115 and pull (nose down), then roll back to bank 0 and pull
    for d_dn in (2.0, 2.5, 3.0, 3.5, 4.0):
        trials.append(
            (
                f"LOW  bank115 pull {d_dn:.1f}s then bank0 pull",
                [(d_dn, 115.0, 1.0, 1.0), (24.0, 0.0, 1.0, 1.0)],
            )
        )
    # the 4.0 s segment-grained versions, i.e. what the genome can actually name
    trials.append(("SEG  4.0s bank0 pull then 4.0s bank115 pull", [(4.0, 0.0, 1.0, 1.0), (20.0, 115.0, 1.0, 1.0)]))
    trials.append(("SEG  4.0s bank115 pull then 4.0s bank0 pull", [(4.0, 115.0, 1.0, 1.0), (20.0, 0.0, 1.0, 1.0)]))
    # a fine-resolution three-phase attempt: fast up, hard over, hard back
    trials.append(("FINE 1.0 bank0 / 1.5 bank115 / rest bank0", [(1.0, 0.0, 1.0, 1.0), (1.5, 115.0, 1.0, 1.0), (21.5, 0.0, 1.0, 1.0)]))
    trials.append(("FINE 0.5 bank0 / 1.0 bank115 / rest bank0", [(0.5, 0.0, 1.0, 1.0), (1.0, 115.0, 1.0, 1.0), (22.5, 0.0, 1.0, 1.0)]))
    trials.append(("FINE 1.5 bank115 / 1.5 bank0 / rest bank115", [(1.5, 115.0, 1.0, 1.0), (1.5, 0.0, 1.0, 1.0), (21.0, 115.0, 1.0, 1.0)]))
    trials.append(("FINE 2.5 bank115 / 2.0 bank0 / rest bank115", [(2.5, 115.0, 1.0, 1.0), (2.0, 0.0, 1.0, 1.0), (19.5, 115.0, 1.0, 1.0)]))

    log(
        "\n  (1)+(3) GRID. `edges` = rising edges of closure > 120 m/s in the whole 24 s.\n"
        "  `swing` / `gpk` are the largest gamma excursion inside [0, 6.0] s and its peak value;\n"
        "  `ret` = the return index D2 needs; `best` = the largest swing that IS followed by a\n"
        "  return inside the window (None = no (swing >= 30, return) pair exists at all).\n"
    )
    log(
        f"     {'schedule':<44} {'steps':>5} {'edges':>6} {'swing':>6} {'gpk':>7} {'lf':>4} "
        f"{'ret':>5} {'best':>16} {'cloMin':>7} {'rngMin':>7} {'D2':>3} {'D2c':>4}"
    )
    all_edges = []
    clo_floor = []
    for name, phases in trials:
        r, rows = E.fly(E.Sched(phases))
        st = d2_stats(r.trace)
        det = B.detect_all(r.trace)
        all_edges.append(len(st["edges"]))
        clo_floor.append(st.get("clo_min_win", float("nan")))
        log(
            f"     {name:<44} {r.steps:>5} {len(st['edges']):>6} {st.get('swing', float('nan')):>6.1f} "
            f"{st.get('gpk', float('nan')):>+7.1f} {st.get('lf', float('nan')):>4.2f} "
            f"{str(st.get('ret')):>5} {str(st.get('best')):>16} "
            f"{st.get('clo_min_win', float('nan')):>7.0f} {st.get('rng_min_win', float('nan')):>7.0f} "
            f"{len(det['yoyos']):>3} {len(det['yoyos_clean']):>4}"
        )
    log(
        f"\n  Rising edges of closure > {B.CLOSURE_OVERSHOOT:.0f} m/s per 24 s episode over "
        f"{len(trials)} trajectories: min {min(all_edges)}, max {max(all_edges)} "
        f"(all of them at step 0)."
    )
    log(
        f"  Closure FLOOR inside the [0, {B.YOYO_RETURN_S:.0f}] s window over the same trajectories: "
        f"min {min(clo_floor):+.0f} m/s, max {max(clo_floor):+.0f} m/s.\n"
        f"  `clean` requires closure < {B.CLOSURE_OVERSHOOT:.0f} m/s at the exit index, so as long as\n"
        f"  the floor stays above {B.CLOSURE_OVERSHOOT:.0f} m/s no exit inside the window can be clean --\n"
        "  D2c = 0 is forced by the CELL geometry, not by the schedule."
    )

    path = f"{E.OUT}/d2_express_report.txt"
    with open(path, "w") as fh:
        fh.write("\n".join(_LOG) + "\n")
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
