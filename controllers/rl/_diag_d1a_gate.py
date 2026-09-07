#!/usr/bin/env python3
"""D1a — measure the reward-waiver PREDICATE before anyone touches the reward.

    cd controllers/rl && uv run python _diag_d1a_gate.py --cells chase,headon --episodes 8

WHAT IS MEASURED

The tick-wise gate the D1 family would hang a `K_LEAD` waiver on:

    GATE  ==  chi_valid  AND  |chi| > CHI_ROOM_DEG (45 deg)
                         AND  nz    > NZ_LOADED    (3.0)
                         AND  NOT in_zone

with `in_zone` read verbatim off the reward's own gate (`reward.py:388`):
`rg.aspect <= C.CONVERT_ASPECT_RAD (50 deg) and rg.rng <= C.CONVERT_RANGE_MAX (1000 m)`.
Every constant is imported from source (`bfm_detect.CHI_ROOM_DEG`, `bfm_detect.NZ_LOADED`,
`constants.CONVERT_ASPECT_RAD`, `constants.CONVERT_RANGE_MAX`) — nothing is hardcoded.

`chi_valid` is REQUIRED. `bfm_detect.out_of_plane_angle` forces `chi = 0.0` when the LOS lies within
`CHI_VALID_ATA_DEG = 5 deg` of the velocity vector, because there the LOS plane is degenerate and chi
is undefined, NOT zero. An invalid tick must therefore never be counted as in-plane evidence either,
so the invalid-tick count is reported separately per arm.

Reported per arm per cell:
  * the GATE fraction, and EACH CLAUSE's own fraction separately (|chi|>45 over valid ticks, nz>3,
    NOT in_zone) so the BINDING clause is visible rather than inferred;
  * the number of distinct gated EXCURSIONS (maximal runs of GATE) and their duration distribution
    (p50/p90/max) in BOTH seconds and CONTROL STEPS — one control step = action_repeat 5 / 120 Hz =
    41.7 ms, so 36 steps = 1.5 s and 12 = 0.5 s. A fraction alone cannot distinguish one long hold
    from many blips;
  * ALTITUDE SPENT during gated ticks — `alt[i_end] - alt[i_start]` per excursion (the same sign
    convention as `bfm_detect.commitment`'s `chi_hold_dalt_m`; negative = spent), p50 and total.
    This is the discriminator that matters: c87 is recorded holding ~39.76 s of LOADED constant-sign
    bank at -2642 m per hold while decking 5/5 — a loaded nose-off descending break, not a
    displacement roll. A gate that c87 clears is a gate cleared by the program's dominant failure
    mode. PRICE HOLDS IN ALTITUDE.
  * a SIGN-CONSTANT subset of the excursions (`sign(chi)` constant throughout), which is the
    skeptic's "sawing-chi spiral" test — a spiral that alternates sides fragments into many short
    same-predicate runs, a displacement roll does not;
  * the outcome line (kills/deaths/decks/bubbles/timeouts, dealt, taken, mean episode length) beside
    it, so the fraction is interpretable. Episode length is printed ONLY as the tick-fraction
    denominator, never as evidence of BFM.
  * a threshold sensitivity sweep over chi bar {30,45,60} deg x nz bar {2,3,4} g, so a future reader
    can see whether the outcome is a threshold artifact or structural.

THE FIVE ARMS, ALL IN ONE INVOCATION (D6 is mandatory; a control measured in another session is not
a control). Order: the two CONTROLS first.

  1. `inplane`  -- scripted `bfm.LagPursuitTracker`, the in-plane pursuit arm A2 used. D6 control (i).
  2. `c87`      -- `runs/curriculum-87/c87_arrival_obs_5000k.pt` greedy. D6 control (iii) and the
                   program's dominant failure mode (loaded descending break, decks 5/5).
  3. `highyoyo` -- the FIXED `train_parallel._highyoyo_stick` (throttle converted from the power
                   fraction its constants were written as into agent space, i.e. A2's `--thr fixed`).
                   A2 already proved this stick flies NO completed yo-yo (D1 = D2 = 0.00 at every
                   threshold); that is expected and is NOT the question here. The question is only
                   whether the GATE PREDICATE separates a deliberate out-of-plane attempt from
                   in-plane pursuit.
  4. `c96`      -- `runs/curriculum-96/run.pt` greedy (the current champion, 45 kills).
  5. `c95`      -- `runs/curriculum-95/run.pt` greedy.

Both cells, same seeds per sub-cell across all five arms:
  * chase  -- sep 700 m, 60 s, co-heading, aspects 30 and 55, no perch.
  * headon -- sep 3000 m, 180 s, `opp_heading_deg=180`, +2000 ft perch, lateral 1400 m expressed as
              the FIXED bearing `asin(1400/3000) = 27.8 deg` (the repeatable eval/probe form used by
              `_bfm_verdict_tmp.py --cell headon`, not scenario's random `lateral_m_max` band).
Opponent is the native aggressive T3 (`opponent_tier` left at its default 0), jitter 0.5 always.

WHAT THIS DOES NOT MEASURE

  * It does NOT score BFM. It contains no D1/D2/D3/D5 counts and makes no claim about whether any arm
    flies a maneuver — A2 and the policy-side verdict already answered that. It measures ONE
    predicate's selectivity.
  * It does NOT model what a rebate would PAY. D1b's salvaged form rebates at excursion CLOSE gated
    on completion (chi returns under 30 deg within 4 s AND aspect gained >= 15 deg); this harness
    deliberately measures the RAW attitude-and-load predicate, because that is the object whose
    selectivity is in question. If the raw predicate does not separate the arms, no completion clause
    layered on top can rescue it as an ATTITUDE-AND-LOAD predicate.
  * The scripted arms hold fire (agent trigger -1). A2 recorded zero rounds for every stick, so this
    reproduces its effective behaviour and keeps the scripted arms pure flight-geometry arms; their
    outcome line therefore cannot show kills and their episodes run longer than a killing policy's.
    Compare EXCURSIONS/EPISODE and DURATIONS across arms, not raw episode tick counts.
  * c87 and c95 predate B1, so their obs column 29 (`gun_quality`) is out of distribution here. Read
    them as "does the PREDICATE fire on these policies", not as a fair re-score.

PRE-REGISTERED INTERPRETATION (written before the run; the record predicts this box FAILS)

`inplane` runs nz p50 5.26 (A2) and already scores 0.25 D1/episode once the chi bar drops to 30 deg;
c87 holds 39.76 s of loaded constant-sign bank while decking 5/5.

  * PASS requires `inplane` AND `c87` both NEAR ZERO on the gate fraction, AND the out-of-plane arms
    (`highyoyo`, and c96/c95 to the extent they maneuver) MATERIALLY ABOVE them.
  * FAIL if `inplane` >= `highyoyo`, or if c87 clears the gate as often as c95/c96. In that case THE
    WHOLE DETECTOR-GATED-WAIVER FAMILY IS DEAD AS AN ATTITUDE-AND-LOAD PREDICATE — say so
    explicitly, do not go looking for a threshold that rescues it, and redirect the program to
    D3/D4. The sensitivity sweep is reported only so a reader can see whether the failure is a
    threshold artifact or structural; it is NOT a search for a rescuing setting.
  * A gate fraction below 0.5% on the arms that are supposed to clear it also kills D1b (too rare to
    matter), per the D1b judging note.
"""

from __future__ import annotations

import argparse
import math
import statistics

import numpy as np
import torch
from astrum_rl import _paths

DEG = 180.0 / math.pi

CHI_BARS_DEG = (30.0, 45.0, 60.0)
NZ_BARS = (2.0, 3.0, 4.0)

# One control step at the trainer's cadence: action_repeat 5 at 120 Hz => 24 Hz => 41.7 ms.
# NOT 208 ms; three prior candidates mis-sized their interventions 5x on that error.
CONTROL_DT_S = 5.0 / 120.0

ARM_SPECS = [
    # (label, kind, target, role)
    ("inplane", "stick", "inplane", "CONTROL"),
    ("c87", "ckpt", "runs/curriculum-87/c87_arrival_obs_5000k.pt", "CONTROL"),
    ("highyoyo", "stick", "highyoyo", "ARM"),
    ("c96", "ckpt", "runs/curriculum-96/run.pt", "ARM"),
    ("c95", "ckpt", "runs/curriculum-95/run.pt", "ARM"),
]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="D1a: selectivity of the |chi|/nz/not-in-zone gate.")
    p.add_argument("--cells", default="chase,headon", help="comma list of chase,headon")
    p.add_argument(
        "--arms", default="inplane,c87,highyoyo,c96,c95", help="comma list; controls first"
    )
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--chase-aspects", default="30,55")
    p.add_argument("--headon-lateral-m", type=float, default=1400.0)
    p.add_argument("--headon-separation", type=float, default=3000.0)
    p.add_argument("--jitter", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=4)
    return p.parse_args(argv)


# ======================================================================================
# Cells
# ======================================================================================
def cells_for(name: str, args) -> list[tuple[str, dict, int]]:
    """`(label, env_kwargs, seed0)`. `seed0` depends ONLY on the sub-cell, never on the arm."""
    out = []
    if name == "chase":
        for a in args.chase_aspects.split(","):
            if not a.strip():
                continue
            asp = float(a)
            kw = dict(
                posture="offensive",
                aspect_deg=asp,
                separation_m=700.0,
                max_time_s=60.0,
                rate_hz=120.0,
                jitter=args.jitter,
            )
            out.append((f"chase/asp{asp:.0f}", kw, args.seed + int(round(asp)) * 1000))
    else:
        sep = args.headon_separation
        lat = args.headon_lateral_m
        asp = math.degrees(math.asin(min(1.0, lat / sep)))
        kw = dict(
            posture="offensive",
            aspect_deg=asp,
            separation_m=sep,
            max_time_s=180.0,
            rate_hz=120.0,
            jitter=args.jitter,
            me_dalt_ft=2000.0,
            opp_heading_deg=180.0,
        )
        out.append((f"headon/lat{lat:.0f}m", kw, args.seed + int(round(asp)) * 1000))
    return out


# ======================================================================================
# Arms
# ======================================================================================
def make_stick_actor(which: str, TP, BFM, C):
    """Return `fn(me, bd, rel, st) -> agent action[5]` for a scripted arm, flying 100% of ticks.

    Trigger is pinned to -1 (hold fire): see the docstring's "does not measure" note.
    """

    def _fix_thr(a4):
        """A2's `--thr fixed`: reinterpret the stick's throttle as the POWER FRACTION its constants
        were written as and convert to agent space (inverse of `native = 0.5(a+1)`)."""
        a = np.asarray(a4, np.float32).copy()
        a[3] = float(np.clip(2.0 * float(a[3]) - 1.0, -1.0, 1.0))
        return a

    if which == "highyoyo":

        def fn(me, bd, rel, st):
            sgn = 1.0 if rel.ata_az >= 0.0 else -1.0
            a4 = _fix_thr(
                TP._highyoyo_stick(float(me.phi), float(me.theta), float(rel.closure), sgn)
            )
            return np.asarray([a4[0], a4[1], a4[2], a4[3], -1.0], np.float32)

    elif which == "inplane":

        def fn(me, bd, rel, st):
            trk = st.get("trk")
            if trk is None:
                trk = st["trk"] = BFM.LagPursuitTracker()
                trk.reset(me, rel)  # `Maneuver.step` reads the entry frame `reset` captures
            a4, _ = trk.step(me, rel)
            a4 = np.asarray(a4, np.float32)
            return np.asarray([a4[0], a4[1], a4[2], a4[3], -1.0], np.float32)

    else:
        raise ValueError(which)
    return fn


def roll_stick(actor_fn, env_kwargs, seed0, episodes, B, C, classify):
    from astrum_dogfight.geometry import relative
    from astrum_rl.envs import make_tdmpc_env

    rows, traces = [], []
    for ep in range(episodes):
        env = make_tdmpc_env(
            3, action_repeat=5, frame_stack=1, jitter_seed=seed0 + ep, **env_kwargs
        )
        base = env.unwrapped
        lnr, opp = base.learner_id, base.opponent_id
        env.reset(seed=seed0 + ep)
        st: dict = {}
        t, tr, event = 0.0, [], None
        term = trunc = False
        while True:
            me, bd = base._states[lnr], base._states[opp]
            rel = relative(me, bd)
            a = actor_fn(me, bd, rel, st)
            tr.append(B.make_sample(t, me, bd, a, rel))
            _o, _r, term, trunc, info = env.step(a)
            t += CONTROL_DT_S
            if info.get("event"):
                event = info["event"]
            if term or trunc:
                break
        me, bd = base._states[lnr], base._states[opp]
        rows.append(
            dict(
                outcome=classify(event, term, trunc),
                dealt=C.HP_MAX - float(bd.health),
                taken=C.HP_MAX - float(me.health),
                steps=len(tr),
            )
        )
        traces.append(tr)
        env.close()
    return rows, traces


def roll_policy(agent, env_kwargs, seed0, episodes, B, C, classify):
    from astrum_dogfight.geometry import relative
    from astrum_rl.envs import make_tdmpc_env

    rows, traces = [], []
    for ep in range(episodes):
        env = make_tdmpc_env(
            3, action_repeat=5, frame_stack=1, jitter_seed=seed0 + ep, **env_kwargs
        )
        base = env.unwrapped
        lnr, opp = base.learner_id, base.opponent_id
        obs, _ = env.reset(seed=seed0 + ep)
        agent._prev_mean.zero_()
        t0, t, tr, event = True, 0.0, [], None
        term = trunc = False
        while True:
            a = agent.act(obs, t0=t0, eval_mode=True).numpy()
            t0 = False
            # Sample the state the action was CHOSEN in, before the step (same cadence as A2).
            me, bd = base._states[lnr], base._states[opp]
            tr.append(B.make_sample(t, me, bd, a, relative(me, bd)))
            obs, _r, term, trunc, info = env.step(a)
            t += CONTROL_DT_S
            if info.get("event"):
                event = info["event"]
            if term or trunc:
                break
        me, bd = base._states[lnr], base._states[opp]
        rows.append(
            dict(
                outcome=classify(event, term, trunc),
                dealt=C.HP_MAX - float(bd.health),
                taken=C.HP_MAX - float(me.health),
                steps=len(tr),
            )
        )
        traces.append(tr)
        env.close()
    return rows, traces


# ======================================================================================
# The measurement
# ======================================================================================
def _runs(mask: list[bool]) -> list[tuple[int, int]]:
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


def _pct(xs, q):
    return float(np.percentile(xs, q)) if len(xs) else float("nan")


def gate_stats(traces, chi_bar_deg: float, nz_bar: float, C) -> dict:
    """Per-tick clause fractions + gated-excursion statistics over a list of per-episode traces."""
    chi_bar = math.radians(chi_bar_deg)
    n_tick = n_invalid = n_chi = n_nz = n_out = n_gate = 0
    durs_steps, dalts, sc_durs_steps, sc_dalts = [], [], [], []
    n_exc = n_exc_sc = 0
    nz_all, chi_valid_deg = [], []
    g_ata, g_rng, g_aspect, g_nz, g_chi = [], [], [], [], []
    for tr in traces:
        gate = []
        for s in tr:
            n_tick += 1
            nz_all.append(s.nz)
            if not s.chi_valid:
                n_invalid += 1
            else:
                chi_valid_deg.append(abs(s.chi) * DEG)
            chi_ok = bool(s.chi_valid and abs(s.chi) > chi_bar)
            nz_ok = bool(s.nz > nz_bar)
            in_zone = bool(s.aspect <= C.CONVERT_ASPECT_RAD and s.rng <= C.CONVERT_RANGE_MAX)
            n_chi += chi_ok
            n_nz += nz_ok
            n_out += not in_zone
            g = chi_ok and nz_ok and not in_zone
            n_gate += g
            gate.append(g)
            if g:
                g_ata.append(s.ata * DEG)
                g_rng.append(s.rng)
                g_aspect.append(s.aspect * DEG)
                g_nz.append(s.nz)
                g_chi.append(abs(s.chi) * DEG)
        for i, j in _runs(gate):
            n_exc += 1
            durs_steps.append(j - i + 1)
            dalts.append(tr[j].alt - tr[i].alt)
            # Sign-constant subset: sign(chi) unchanged across the whole excursion.
            sgn = 1 if tr[i].chi > 0 else -1
            if all((1 if tr[k].chi > 0 else -1) == sgn for k in range(i, j + 1)):
                n_exc_sc += 1
                sc_durs_steps.append(j - i + 1)
                sc_dalts.append(tr[j].alt - tr[i].alt)
    n_valid = n_tick - n_invalid
    ne = max(1, len(traces))
    edges = [0, 20, 45, 60, 90, 135, 180.001]
    hist = {
        f"{lo:.0f}-{hi:.0f}": sum(1 for v in chi_valid_deg if lo <= v < hi) / max(1, n_valid)
        for lo, hi in zip(edges[:-1], edges[1:])
    }
    return dict(
        nz_p50_all=_pct(nz_all, 50),
        chi_hist=hist,
        g_ata_p50=_pct(g_ata, 50),
        g_ata_p90=_pct(g_ata, 90),
        g_rng_p50=_pct(g_rng, 50),
        g_aspect_p50=_pct(g_aspect, 50),
        g_nz_p50=_pct(g_nz, 50),
        g_chi_p50=_pct(g_chi, 50),
        n_ep=len(traces),
        n_tick=n_tick,
        n_invalid=n_invalid,
        f_invalid=n_invalid / max(1, n_tick),
        f_chi=n_chi
        / max(1, n_valid),  # over VALID ticks: an undefined chi is not in-plane evidence
        f_nz=n_nz / max(1, n_tick),
        f_out=n_out / max(1, n_tick),
        f_gate=n_gate / max(1, n_tick),
        n_gate=n_gate,
        n_exc=n_exc,
        exc_per_ep=n_exc / ne,
        dur_p50=_pct(durs_steps, 50),
        dur_p90=_pct(durs_steps, 90),
        dur_max=(max(durs_steps) if durs_steps else float("nan")),
        dalt_p50=_pct(dalts, 50),
        dalt_total=float(sum(dalts)) if dalts else 0.0,
        dalt_worst=(min(dalts) if dalts else float("nan")),
        n_exc_sc=n_exc_sc,
        sc_per_ep=n_exc_sc / ne,
        sc_dur_p50=_pct(sc_durs_steps, 50),
        sc_dur_max=(max(sc_durs_steps) if sc_durs_steps else float("nan")),
        sc_dalt_total=float(sum(sc_dalts)) if sc_dalts else 0.0,
    )


def _both(steps: float) -> str:
    if not np.isfinite(steps):
        return "  nan"
    return f"{steps * CONTROL_DT_S:5.2f}s({steps:.0f})"


def outcome_line(rows) -> str:
    n = len(rows)
    kinds = ("kill", "death", "deck", "bubble", "timeout")
    parts = " ".join(f"{k} {sum(r['outcome'] == k for r in rows)}" for k in kinds)
    other = sorted({r["outcome"] for r in rows} - set(kinds))
    return (
        f"{n} eps | {parts}"
        + (f" other {other}" if other else "")
        + f" | dealt {statistics.fmean(r['dealt'] for r in rows):5.1f}"
        f" taken {statistics.fmean(r['taken'] for r in rows):5.1f}"
        f" | mean len {statistics.fmean(r['steps'] for r in rows) * CONTROL_DT_S:5.1f}s"
        f" ({statistics.fmean(r['steps'] for r in rows):.0f} steps)"
    )


def report(label: str, role: str, cell: str, rows, traces, B, C) -> dict:
    g = gate_stats(traces, B.CHI_ROOM_DEG, B.NZ_LOADED, C)
    print(f"\n-- ARM {label} [{role}] / {cell} --")
    print(f"   score: {outcome_line(rows)}")
    print(
        f"   ticks {g['n_tick']} | chi INVALID {g['n_invalid']} ({100 * g['f_invalid']:.1f}%)"
        f" -> chi clause read over {g['n_tick'] - g['n_invalid']} valid ticks"
    )
    print(
        f"   clause frac: |chi|>{B.CHI_ROOM_DEG:.0f} {g['f_chi']:.4f} | nz>{B.NZ_LOADED:.1f}"
        f" {g['f_nz']:.4f} | NOT in_zone {g['f_out']:.4f}  ==>  GATE {g['f_gate']:.4f}"
        f" ({g['n_gate']} ticks)"
    )
    print(
        f"   excursions: {g['n_exc']} ({g['exc_per_ep']:.2f}/ep) | dur p50 {_both(g['dur_p50'])}"
        f" p90 {_both(g['dur_p90'])} max {_both(g['dur_max'])}"
    )
    print(
        f"   dalt/excursion: p50 {g['dalt_p50']:8.1f} m | total {g['dalt_total']:9.0f} m"
        f" | worst {g['dalt_worst']:8.1f} m"
    )
    print(
        f"   sign-constant chi subset: {g['n_exc_sc']} ({g['sc_per_ep']:.2f}/ep)"
        f" | dur p50 {_both(g['sc_dur_p50'])} max {_both(g['sc_dur_max'])}"
        f" | dalt total {g['sc_dalt_total']:9.0f} m"
    )
    print(
        f"   ON GATED TICKS: ata p50 {g['g_ata_p50']:5.1f} p90 {g['g_ata_p90']:5.1f} deg"
        f" | rng p50 {g['g_rng_p50']:6.0f} m | aspect p50 {g['g_aspect_p50']:5.1f} deg"
        f" | nz p50 {g['g_nz_p50']:4.2f} | |chi| p50 {g['g_chi_p50']:5.1f} deg"
        f"   [all-tick nz p50 {g['nz_p50_all']:4.2f}]"
    )
    print(
        "   |chi| hist over VALID ticks: "
        + "  ".join(f"{k} {v:.3f}" for k, v in g["chi_hist"].items())
    )
    return g


def sensitivity(label: str, cell: str, traces, C) -> None:
    print(f"   sensitivity {label} / {cell}:  chi>  nz>   GATEfrac  exc/ep  durp50(steps)")
    for chi in CHI_BARS_DEG:
        for nz in NZ_BARS:
            g = gate_stats(traces, chi, nz, C)
            tag = "  <- shipped" if (chi, nz) == (45.0, 3.0) else ""
            print(
                f"                             {chi:4.0f} {nz:4.1f}   {g['f_gate']:8.4f}"
                f"  {g['exc_per_ep']:6.2f}  {_both(g['dur_p50'])}{tag}"
            )


# ======================================================================================
def main() -> None:
    args = parse_args()
    _paths.require_astrum_env()
    from astrum_dogfight import bfm as BFM
    from astrum_dogfight import bfm_detect as B
    from astrum_dogfight import constants as C
    from astrum_rl.tdmpc2 import train_parallel as TP
    from astrum_rl.tdmpc2.agent import TDMPC2, resolve_device
    from astrum_rl.tdmpc2.config import TDMPC2Config
    from astrum_rl.tdmpc2.eval_policy import _classify

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, args.threads))
    device = resolve_device(args.device)

    print(
        f"D1a GATE PREDICATE | GATE = chi_valid & |chi|>{B.CHI_ROOM_DEG:.0f}deg &"
        f" nz>{B.NZ_LOADED:.1f} & NOT(aspect<={DEG * C.CONVERT_ASPECT_RAD:.0f}deg &"
        f" rng<={C.CONVERT_RANGE_MAX:.0f}m)"
    )
    print(
        f"CHI_VALID_ATA_DEG {B.CHI_VALID_ATA_DEG:.0f} | 1 control step ="
        f" {1000 * CONTROL_DT_S:.1f} ms (36 = 1.5 s, 12 = 0.5 s) |"
        f" {args.episodes} eps/arm/sub-cell | jitter {args.jitter:g}"
    )
    print("D6: `inplane` and `c87` are the CONTROLS and run FIRST, same invocation/env/seeds/bars.")
    print(
        "PRE-REGISTERED: PASS needs inplane AND c87 near zero with the out-of-plane arms materially"
        " above. FAIL if inplane >= highyoyo, or if c87 clears as often as c95/c96 -> the"
        " detector-gated-waiver family is DEAD as an attitude-and-load predicate."
    )

    want = [a.strip() for a in args.arms.split(",") if a.strip()]
    arms = [s for s in ARM_SPECS if s[0] in want]
    cell_names = [c.strip() for c in args.cells.split(",") if c.strip()]

    summary: dict[tuple[str, str], dict] = {}
    pooled_traces: dict[tuple[str, str], list] = {}

    for cell_name in cell_names:
        subcells = cells_for(cell_name, args)
        print(f"\n{'#' * 100}\n# CELL {cell_name}: {subcells[0][1]}\n{'#' * 100}")
        for label, kind, target, role in arms:
            agent = None
            if kind == "ckpt":
                sd = torch.load(target, map_location=device, weights_only=False)
                print(
                    f"\n[{label}] {target} stage_idx={int(sd.get('stage_idx', -1))}"
                    f" collected={int(sd.get('collected', -1)):,}"
                )
                cfg = TDMPC2Config(**sd["cfg"])
                cfg.mpc = False  # greedy = the out-of-band read
                agent = TDMPC2(cfg, device)
                agent.load_state_dict(sd)
                agent.model.eval()
            all_rows, all_traces = [], []
            for sub_label, kw, seed0 in subcells:
                if kind == "ckpt":
                    rows, traces = roll_policy(agent, kw, seed0, args.episodes, B, C, _classify)
                else:
                    fn = make_stick_actor(target, TP, BFM, C)
                    rows, traces = roll_stick(fn, kw, seed0, args.episodes, B, C, _classify)
                report(label, role, sub_label, rows, traces, B, C)
                all_rows.extend(rows)
                all_traces.extend(traces)
            if len(subcells) > 1:
                g = report(label, role, f"{cell_name}/POOLED", all_rows, all_traces, B, C)
            else:
                g = gate_stats(all_traces, B.CHI_ROOM_DEG, B.NZ_LOADED, C)
            summary[(cell_name, label)] = dict(g, role=role, rows=all_rows)
            pooled_traces[(cell_name, label)] = all_traces
            sensitivity(label, f"{cell_name}/POOLED", all_traces, C)

    # ---- final comparison tables --------------------------------------------------------
    for cell_name in cell_names:
        print(f"\n{'=' * 100}\nSUMMARY / cell {cell_name} (pooled over sub-cells)\n{'=' * 100}")
        print(
            "arm        role     GATEfrac  |chi|>45  nz>3    ~zone   invalid  exc/ep"
            "  durp50   durmax   dalt_p50  dalt_tot"
        )
        for label, _k, _t, _r in arms:
            g = summary.get((cell_name, label))
            if not g:
                continue
            print(
                f"{label:10s} {g['role']:8s} {g['f_gate']:8.4f}  {g['f_chi']:8.4f}"
                f"  {g['f_nz']:6.4f}  {g['f_out']:6.4f}  {g['f_invalid']:7.4f}"
                f"  {g['exc_per_ep']:6.2f}  {_both(g['dur_p50'])}  {_both(g['dur_max'])}"
                f"  {g['dalt_p50']:8.1f}  {g['dalt_total']:9.0f}"
            )
        print("\narm        outcome")
        for label, _k, _t, _r in arms:
            g = summary.get((cell_name, label))
            if g:
                print(f"{label:10s} {outcome_line(g['rows'])}")

    # ---- the verdict arithmetic, printed so it cannot be re-read after the fact ---------
    print(f"\n{'=' * 100}\nPRE-REGISTERED VERDICT ARITHMETIC\n{'=' * 100}")
    for cell_name in cell_names:
        ip = summary.get((cell_name, "inplane"))
        hy = summary.get((cell_name, "highyoyo"))
        c87 = summary.get((cell_name, "c87"))
        c96 = summary.get((cell_name, "c96"))
        c95 = summary.get((cell_name, "c95"))
        if not all((ip, hy, c87, c96, c95)):
            print(f"{cell_name}: incomplete arm set, no verdict")
            continue
        f1 = ip["f_gate"] >= hy["f_gate"]
        f2 = c87["f_gate"] >= min(c95["f_gate"], c96["f_gate"])
        print(
            f"{cell_name}: inplane {ip['f_gate']:.4f} vs highyoyo {hy['f_gate']:.4f}"
            f" -> inplane>=highyoyo {f1} | c87 {c87['f_gate']:.4f} vs c95 {c95['f_gate']:.4f} /"
            f" c96 {c96['f_gate']:.4f} -> c87 clears as often {f2}"
            f" ==> {'FAIL' if (f1 or f2) else 'PASS-candidate'}"
        )


if __name__ == "__main__":
    main()
