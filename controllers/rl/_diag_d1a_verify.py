#!/usr/bin/env python3
"""D1a ADVERSARIAL VERIFICATION — the two things `_diag_d1a_gate.py`'s numbers cannot settle.

    cd controllers/rl && uv run python _diag_d1a_verify.py --cells headon,chase30 --episodes 12

WHAT IS MEASURED (and why the original run cannot answer it)

`_diag_d1a_gate.py` reported ONE pooled tick fraction per arm per cell and no dispersion. Its whole
headon verdict ("PASS-candidate") rests on a single unbracketed ordering, c87 0.0649 < c95 0.0946 <
c96 0.1083, over 12 episodes. Two specific attacks:

  1. DISPERSION on the decisive comparison. This harness reports the PER-EPISODE gate fraction for
     every arm (mean / p50 / min / max) and, because seeds are paired by episode index across arms,
     a PAIRED sign test of c87 vs c96 and c87 vs c95 (how many of the 12 paired episodes does the
     control clear MORE than the arm). If the control wins a substantial minority of paired
     episodes, "c87 does not clear as often" is not a measured fact at n=12.

  2. THE "AS OFTEN" READING. The box says "if c87 clears the gate AS OFTEN AS c95/c96". The original
     harness implemented that as a DUTY CYCLE (tick fraction). The plainer reading of "how often" is
     a RATE: gated excursions per episode. Both are reported here side by side, per arm, so a reader
     can see whether the verdict is statistic-dependent.

  3. CHI CONDITIONING. `out_of_plane_angle` returns valid=False only inside
     `CHI_VALID_ATA_DEG = 5 deg`. Just outside it the LOS projection perpendicular to the velocity
     vector has norm ~sin(ata), so at ata = 6-10 deg `phi_los` is computed from a near-zero vector
     and `chi` is defined but ILL-CONDITIONED — a tiny geometry change swings it by tens of degrees.
     The original run's gated ticks for c96 chase/asp30 sit at ata p50 8.6 / p90 9.2 deg with
     |chi| p50 156 deg, and c95's at ata p50 8.9 deg with |chi| p50 139 deg, i.e. essentially the
     whole clearance in the champion's kill cell is inside 10 deg of the degeneracy boundary. This
     harness therefore reports the gate fraction ALSO under a well-conditioned restriction
     `ata >= ATA_FLOOR_DEG` (default 15 deg), and the ata histogram on gated ticks, so a reader can
     see how much of each arm's clearance is real out-of-plane flying versus reference-frame noise.
     The restriction is applied IDENTICALLY to every arm including the controls.

Everything else is held to the original: same GATE (`chi_valid & |chi| > bfm_detect.CHI_ROOM_DEG &
nz > bfm_detect.NZ_LOADED & NOT(aspect <= CONVERT_ASPECT_RAD & rng <= CONVERT_RANGE_MAX)`), same
constants imported from source, same cells, same per-sub-cell seeds (`seed + round(aspect_deg)*1000`,
independent of the arm), same greedy `cfg.mpc = False` read, same `_prev_mean.zero_()` per episode,
same jitter 0.5, same 41.7 ms control step (action_repeat 5 at 120 Hz; 36 steps = 1.5 s, 12 = 0.5 s).
D6: `inplane` and `c87` are CONTROLS and run FIRST in the same invocation.

WHAT THIS DOES NOT MEASURE

  * It does NOT re-derive the chase verdict. The chase already triggers BOTH pre-registered FAIL
    clauses at every one of the 9 threshold settings in the original sweep; nothing here can rescue
    or worsen that. `chase30` is included ONLY as the cell where the chi-conditioning artifact is
    visible, not as a verdict cell.
  * It does NOT score BFM, and contains no D1/D2/D3/D5 detector counts.
  * It does NOT model what a D1b rebate would PAY. Note in passing that the shipped `K_LEAD` term
    (`reward.py:388-390`) only ever CHARGES out of zone (`raw_lead <= 0.0 or in_zone`), so the ticks
    on which a waiver could pay anything are (GATE and lead degrading) — a strict subset of the GATE
    fraction measured here and by the original. Every gate fraction in this record is an UPPER BOUND
    on the payable fraction.
  * Episode length is used only as the tick-fraction denominator, never as evidence.

PRE-REGISTERED INTERPRETATION (written before this run)

  * If the paired sign test has c87 clearing MORE than the arm in >= 4 of 12 paired headon episodes
    for either arm, the headon ordering is not resolved at this n and the "PASS-candidate" label
    cannot be banked either way.
  * If under the excursion-RATE reading c87 >= min(c96, c95) on the headon, then the headon cell
    FAILS the box's FAIL clause 2 as written ("as often as"), the verdict is statistic-dependent,
    and — since the box pre-registers no per-cell decomposition at all — the box's single verdict is
    FAIL on both cells.
  * If restricting to ata >= 15 deg collapses c96/c95's chase30 clearance toward zero while leaving
    the controls roughly unchanged, then the arms' apparent chase clearance is substantially a
    frame-conditioning artifact and no arm/control ordering from that sub-cell may be quoted.
  * These predictions can come out the wrong way: if c87 loses every paired episode, the rate
    reading agrees with the duty-cycle reading, and the ata restriction changes nothing, then the
    original headon reading survives all three attacks and I will say so.
"""

from __future__ import annotations

import argparse
import math
import statistics

import numpy as np
import torch
from astrum_rl import _paths

DEG = 180.0 / math.pi
CONTROL_DT_S = 5.0 / 120.0  # action_repeat 5 at 120 Hz => 24 Hz => 41.7 ms. NOT 208 ms.

ARM_SPECS = [
    ("inplane", "stick", "inplane", "CONTROL"),
    ("c87", "ckpt", "runs/curriculum-87/c87_arrival_obs_5000k.pt", "CONTROL"),
    ("highyoyo", "stick", "highyoyo", "ARM"),
    ("c96", "ckpt", "runs/curriculum-96/run.pt", "ARM"),
    ("c95", "ckpt", "runs/curriculum-95/run.pt", "ARM"),
]


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--cells", default="headon,chase30")
    p.add_argument("--arms", default="inplane,c87,highyoyo,c96,c95")
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--ata-floor-deg", type=float, default=15.0)
    p.add_argument("--headon-lateral-m", type=float, default=1400.0)
    p.add_argument("--headon-separation", type=float, default=3000.0)
    p.add_argument("--jitter", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=4)
    return p.parse_args(argv)


def cells_for(name: str, args):
    if name == "chase30":
        kw = dict(
            posture="offensive",
            aspect_deg=30.0,
            separation_m=700.0,
            max_time_s=60.0,
            rate_hz=120.0,
            jitter=args.jitter,
        )
        return [("chase/asp30", kw, args.seed + 30 * 1000)]
    if name == "chase55":
        kw = dict(
            posture="offensive",
            aspect_deg=55.0,
            separation_m=700.0,
            max_time_s=60.0,
            rate_hz=120.0,
            jitter=args.jitter,
        )
        return [("chase/asp55", kw, args.seed + 55 * 1000)]
    sep, lat = args.headon_separation, args.headon_lateral_m
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
    return [(f"headon/lat{lat:.0f}m", kw, args.seed + int(round(asp)) * 1000)]


def make_stick_actor(which, TP, BFM):
    def _fix_thr(a4):
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
                trk.reset(me, rel)
            a4, _ = trk.step(me, rel)
            a4 = np.asarray(a4, np.float32)
            return np.asarray([a4[0], a4[1], a4[2], a4[3], -1.0], np.float32)

    else:
        raise ValueError(which)
    return fn


def roll(kind, target, agent, env_kwargs, seed0, episodes, B, C, classify, TP, BFM):
    """One trace list per episode, plus the per-episode outcome row. Seeds paired by ep index."""
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
        if kind == "ckpt":
            agent._prev_mean.zero_()
        else:
            fn = make_stick_actor(target, TP, BFM)
        st: dict = {}
        t0, t, tr, event = True, 0.0, [], None
        while True:
            me, bd = base._states[lnr], base._states[opp]
            rel = relative(me, bd)
            if kind == "ckpt":
                a = agent.act(obs, t0=t0, eval_mode=True).numpy()
                t0 = False
            else:
                a = fn(me, bd, rel, st)
            tr.append(B.make_sample(t, me, bd, a, rel))
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


def _runs(mask):
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


def per_episode(traces, B, C, ata_floor_deg):
    """Per-episode gate fraction / excursion count, both unrestricted and ata-restricted."""
    chi_bar = math.radians(B.CHI_ROOM_DEG)
    ata_floor = math.radians(ata_floor_deg)
    ep = []
    g_ata_all = []
    for tr in traces:
        n = len(tr)
        gate, gate_r = [], []
        for s in tr:
            in_zone = bool(s.aspect <= C.CONVERT_ASPECT_RAD and s.rng <= C.CONVERT_RANGE_MAX)
            g = bool(s.chi_valid and abs(s.chi) > chi_bar and s.nz > B.NZ_LOADED and not in_zone)
            gate.append(g)
            gate_r.append(g and s.ata >= ata_floor)
            if g:
                g_ata_all.append(s.ata * DEG)
        exc = _runs(gate)
        ep.append(
            dict(
                n=n,
                frac=sum(gate) / max(1, n),
                frac_r=sum(gate_r) / max(1, n),
                n_gate=sum(gate),
                n_gate_r=sum(gate_r),
                n_exc=len(exc),
                n_exc_r=len(_runs(gate_r)),
                dur_max=(max(j - i + 1 for i, j in exc) if exc else 0),
                dalt_tot=float(sum(tr[j].alt - tr[i].alt for i, j in exc)) if exc else 0.0,
            )
        )
    return ep, g_ata_all


def main():
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
        f"D1a VERIFY | GATE = chi_valid & |chi|>{B.CHI_ROOM_DEG:.0f} & nz>{B.NZ_LOADED:.1f} &"
        f" NOT(aspect<={DEG * C.CONVERT_ASPECT_RAD:.0f} & rng<={C.CONVERT_RANGE_MAX:.0f}m)"
        f" | ata floor for the WELL-CONDITIONED variant = {args.ata_floor_deg:.0f} deg"
        f" (CHI_VALID_ATA_DEG = {B.CHI_VALID_ATA_DEG:.0f})"
    )
    print(
        f"1 control step = {1000 * CONTROL_DT_S:.1f} ms (36 = 1.5 s, 12 = 0.5 s) |"
        f" {args.episodes} eps/arm/sub-cell | jitter {args.jitter:g} | seeds paired by ep index"
    )
    print("D6: inplane and c87 are CONTROLS, run FIRST, same invocation / env / seeds / bars.\n")

    want = [a.strip() for a in args.arms.split(",") if a.strip()]
    arms = [s for s in ARM_SPECS if s[0] in want]
    store: dict[tuple[str, str], list] = {}
    outc: dict[tuple[str, str], list] = {}

    for cell_name in [c.strip() for c in args.cells.split(",") if c.strip()]:
        for sub_label, kw, seed0 in cells_for(cell_name, args):
            print(f"{'#' * 96}\n# CELL {sub_label}: {kw}\n{'#' * 96}")
            for label, kind, target, role in arms:
                agent = None
                if kind == "ckpt":
                    sd = torch.load(target, map_location=device, weights_only=False)
                    print(
                        f"[{label}] {target} stage_idx={int(sd.get('stage_idx', -1))}"
                        f" collected={int(sd.get('collected', -1)):,}"
                    )
                    cfg = TDMPC2Config(**sd["cfg"])
                    cfg.mpc = False
                    agent = TDMPC2(cfg, device)
                    agent.load_state_dict(sd)
                    agent.model.eval()
                rows, traces = roll(
                    kind, target, agent, kw, seed0, args.episodes, B, C, _classify, TP, BFM
                )
                ep, g_ata = per_episode(traces, B, C, args.ata_floor_deg)
                store[(sub_label, label)] = ep
                outc[(sub_label, label)] = rows
                fr = [e["frac"] for e in ep]
                frr = [e["frac_r"] for e in ep]
                ex = [e["n_exc"] for e in ep]
                kinds = ("kill", "death", "deck", "bubble", "timeout")
                sc = " ".join(f"{k} {sum(r['outcome'] == k for r in rows)}" for k in kinds)
                print(f"-- {label} [{role}] / {sub_label} -- {sc}")
                print(
                    f"   GATEfrac  pooled {sum(e['n_gate'] for e in ep) / max(1, sum(e['n'] for e in ep)):.4f}"
                    f" | per-ep mean {statistics.fmean(fr):.4f} p50 {_pct(fr, 50):.4f}"
                    f" min {min(fr):.4f} max {max(fr):.4f}"
                )
                print(
                    f"   GATEfrac ata>={args.ata_floor_deg:.0f}  pooled"
                    f" {sum(e['n_gate_r'] for e in ep) / max(1, sum(e['n'] for e in ep)):.4f}"
                    f" | per-ep mean {statistics.fmean(frr):.4f}"
                    f" | kept {sum(e['n_gate_r'] for e in ep)}/{sum(e['n_gate'] for e in ep)} gated ticks"
                )
                print(
                    f"   excursions/ep (the 'AS OFTEN' RATE) mean {statistics.fmean(ex):.2f}"
                    f" p50 {_pct(ex, 50):.1f} min {min(ex)} max {max(ex)}"
                    f" | eps with ZERO gated ticks {sum(1 for e in ep if e['n_gate'] == 0)}/{len(ep)}"
                )
                print(
                    f"   ata on GATED ticks: p10 {_pct(g_ata, 10):5.1f} p50 {_pct(g_ata, 50):5.1f}"
                    f" p90 {_pct(g_ata, 90):5.1f} deg | frac of gated ticks with ata<10deg"
                    f" {sum(1 for v in g_ata if v < 10.0) / max(1, len(g_ata)):.3f}"
                )
                print(
                    f"   dalt over gated excursions: total {sum(e['dalt_tot'] for e in ep):9.0f} m"
                    f" | per-ep p50 {_pct([e['dalt_tot'] for e in ep], 50):8.1f} m\n"
                )

    # ---- paired comparisons: the control vs each arm, episode by episode ------------------
    print(
        f"{'=' * 96}\nPAIRED SIGN TEST (same seed per episode index) — control c87 vs each arm\n{'=' * 96}"
    )
    for sub_label, label in [k for k in store if k[1] == "c87"]:
        for arm in ("c96", "c95", "highyoyo"):
            a = store.get((sub_label, "c87"))
            b = store.get((sub_label, arm))
            if not a or not b:
                continue
            n = min(len(a), len(b))
            wins_f = sum(1 for i in range(n) if a[i]["frac"] > b[i]["frac"])
            wins_e = sum(1 for i in range(n) if a[i]["n_exc"] > b[i]["n_exc"])
            print(
                f"{sub_label}: c87 vs {arm:9s} | c87 clears a LARGER tick fraction in"
                f" {wins_f}/{n} paired eps | MORE excursions in {wins_e}/{n}"
            )
            print(
                "                              per-ep frac c87 "
                + " ".join(f"{a[i]['frac']:.3f}" for i in range(n))
            )
            print(
                f"                              per-ep frac {arm:5s} "
                + " ".join(f"{b[i]['frac']:.3f}" for i in range(n))
            )

    # ---- the two readings of "as often", side by side --------------------------------------
    print(f"\n{'=' * 96}\nTHE TWO READINGS OF THE BOX'S 'AS OFTEN', SIDE BY SIDE\n{'=' * 96}")
    print("cell / arm            DUTYCYCLE(tickfrac)   RATE(exc/ep)   DUTY ata>=floor")
    for (sub_label, label), ep in store.items():
        tot = max(1, sum(e["n"] for e in ep))
        print(
            f"{sub_label:20s} {label:9s} {sum(e['n_gate'] for e in ep) / tot:10.4f}"
            f"        {statistics.fmean([e['n_exc'] for e in ep]):8.2f}"
            f"       {sum(e['n_gate_r'] for e in ep) / tot:8.4f}"
        )
    for sub_label in sorted({k[0] for k in store}):
        g = {k[1]: v for k, v in store.items() if k[0] == sub_label}
        if not {"c87", "c96", "c95"} <= set(g):
            continue

        def duty(lb):
            return sum(e["n_gate"] for e in g[lb]) / max(1, sum(e["n"] for e in g[lb]))

        def rate(lb):
            return statistics.fmean([e["n_exc"] for e in g[lb]])

        f_duty = duty("c87") >= min(duty("c95"), duty("c96"))
        f_rate = rate("c87") >= min(rate("c95"), rate("c96"))
        print(
            f"\n{sub_label}: FAIL clause 2 ('c87 clears as often as c95/c96')"
            f" -> under DUTY CYCLE {f_duty} | under RATE {f_rate}"
            + ("   <== STATISTIC-DEPENDENT VERDICT" if f_duty != f_rate else "")
        )


if __name__ == "__main__":
    main()
