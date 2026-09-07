"""Audit the claim: "D1=D2=0 does not license 'no BFM' because the suite has no LEAD-TURN detector."

Reads the two dumped D3b traces (kill seed 62, non-kill seed 28) back into real
`bfm_detect.Sample` objects -- every one of the 18 Sample fields is a dumped `smp_*` column, so the
trace handed to the detectors here is the SAME trace they read during the search, not a
reconstruction. Then:

  1. runs `B.detect_all` verbatim and prints the counts;
  2. audits D1 and D3 CLAUSE BY CLAUSE, so we can say which clause actually rejects each candidate
     hold rather than guessing (the claim blames D1 clause (f), the aspect-gain clause);
  3. prices every hold in ALTITUDE;
  4. measures LEAD vs LAG pursuit directly from pos/vel, independently of the detector suite, so the
     question "was a lead turn flown at all?" is answered by geometry and not by the instrument
     whose coverage is under dispute.

    docker compose --profile rl run --rm --no-deps dogfight-train \
        bash -lc 'cd /workspace/controllers/rl && python _diag_leadturn_refute.py'
"""

from __future__ import annotations

import csv
import math

import numpy as np
from astrum_rl import _paths

_paths.require_astrum_env()
from astrum_dogfight import bfm_detect as B  # noqa: E402

DEG = 180.0 / math.pi
RUNS = "/workspace/controllers/rl/runs/d3b"
TRACES = [
    ("KILL seed62", f"{RUNS}/replay_kill_seed62.csv"),
    ("NOKILL seed28", f"{RUNS}/replay_nokill_seed28.csv"),
]


def load_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def to_trace(rows):
    """Rebuild the exact `Sample` list the detectors read. All 18 fields come from `smp_*`."""
    return [
        B.Sample(
            t=float(r["smp_t"]),
            chi=float(r["smp_chi_rad"]),
            chi_valid=bool(float(r["smp_chi_valid"])),
            gamma=float(r["smp_gamma_rad"]),
            nz=float(r["smp_nz"]),
            phi=float(r["smp_phi_rad"]),
            alt=float(r["smp_alt"]),
            ail=float(r["smp_ail"]),
            rng=float(r["smp_rng"]),
            ata=float(r["smp_ata"]),
            ata_az=float(r["smp_ata_az"]),
            aspect=float(r["smp_aspect"]),
            closure=float(r["smp_closure"]),
            cpa_miss=float(r["smp_cpa_miss"]),
            es_rel=float(r["smp_es_rel"]),
            lay_miss_m=float(r["smp_lay_miss_m"]),
            lethal=bool(float(r["smp_lethal"])),
        )
        for r in rows
    ]


def audit_d1(tr):
    """Re-walk `detect_displacement_rolls` recording the FIRST clause that rejects each hold."""
    hold = [(abs(s.chi) > math.radians(B.CHI_OUT_DEG) and s.nz > B.NZ_LOADED) for s in tr]
    out = []
    for i, j in B._runs(hold):
        rec = {
            "i": i,
            "j": j,
            "t0": tr[i].t,
            "t1": tr[j].t,
            "dur": tr[j].t - tr[i].t,
            "dalt": tr[j].alt - tr[i].alt,
            "nz_pk": max(tr[k].nz for k in range(i, j + 1)),
            "chi_pk": max(abs(tr[k].chi) for k in range(i, j + 1)) * DEG,
        }
        if rec["dur"] < B.CHI_MIN_HOLD_S:
            rec["reject"] = f"HOLD DURATION {rec['dur']:.3f}s < CHI_MIN_HOLD_S {B.CHI_MIN_HOLD_S}"
            out.append(rec)
            continue
        if not any(
            abs(tr[k].chi) < math.radians(B.CHI_INPLANE_DEG)
            for k in range(i - 1, -1, -1)
            if tr[i].t - tr[k].t <= B.CHI_LOOKBACK_S
        ):
            rec["reject"] = "(a) no in-plane lookback"
            out.append(rec)
            continue
        if tr[i].closure <= 0.0:
            rec["reject"] = f"(c) not closing at entry (closure {tr[i].closure:+.1f} m/s)"
            out.append(rec)
            continue
        exit_idx = None
        for k in range(j + 1, len(tr)):
            if tr[k].t - tr[j].t > B.CHI_RETURN_S:
                break
            if abs(tr[k].chi) < math.radians(B.CHI_RETURN_DEG):
                exit_idx = k
                break
        if exit_idx is None:
            rec["reject"] = "(d) never rolled back in"
            out.append(rec)
            continue
        rec["exit"] = exit_idx
        if len({1 if tr[k].ata_az >= 0.0 else -1 for k in range(i, exit_idx + 1)}) != 1:
            rec["reject"] = "(e) ata_az side FLIPPED (bandit crossed the nose)"
            out.append(rec)
            continue
        d_aspect = (tr[exit_idx].aspect - tr[i].aspect) * DEG
        rec["d_aspect"] = d_aspect
        rec["reject"] = (
            f"(f) ASPECT CLAUSE: d_aspect {d_aspect:+.2f} deg >= -{B.ASPECT_GAIN_DEG}"
            if d_aspect >= -B.ASPECT_GAIN_DEG
            else "NONE -- EVENT ACCEPTED"
        )
        out.append(rec)
    return out


def audit_d3(tr):
    hold = [(abs(s.chi) > math.radians(B.CHI_ROOM_DEG) and s.nz > B.NZ_LOADED) for s in tr]
    out = []
    for i, j in B._runs(hold):
        rec = {
            "i": i,
            "j": j,
            "dur": tr[j].t - tr[i].t,
            "dalt": tr[j].alt - tr[i].alt,
            "cpa0": tr[i].cpa_miss,
            "cpa1": tr[j].cpa_miss,
            "nz_pk": max(tr[k].nz for k in range(i, j + 1)),
        }
        if rec["dur"] < B.CHI_MIN_HOLD_S:
            rec["reject"] = f"HOLD DURATION {rec['dur']:.3f}s < {B.CHI_MIN_HOLD_S}"
        elif len({1 if tr[k].ata_az >= 0.0 else -1 for k in range(i, j + 1)}) != 1:
            rec["reject"] = "ata_az side flipped"
        elif tr[j].cpa_miss <= tr[i].cpa_miss:
            rec["reject"] = f"CPA NOT BUILT ({rec['cpa0']:.1f} -> {rec['cpa1']:.1f} m)"
        else:
            after = [k for k in range(j + 1, len(tr)) if tr[k].t - tr[j].t <= B.CHI_RETURN_S]
            if not after or min(tr[k].cpa_miss for k in after) >= tr[j].cpa_miss:
                rec["reject"] = "room built but never CASHED"
            else:
                rec["reject"] = "NONE -- EVENT ACCEPTED"
        out.append(rec)
    return out


def lead_lag(rows):
    """Signed LEAD component of the nose, independent of the detector suite.

    u   = unit LOS me->bandit.  vh = unit own velocity.  wh = unit bandit velocity.
    wperp = component of wh perpendicular to the LOS (the direction the bandit is DRIFTING across
    the LOS).  lead_deg = signed angle of the nose off the LOS, positive when it is displaced
    TOWARD where the bandit is going (lead pursuit / lead turn) and negative when it trails him
    (lag pursuit).  |lead_deg| == ata by construction; only the SIGN is new information.
    """
    out = []
    for r in rows:
        p = np.array([float(r["pos_e_m"]), float(r["pos_n_m"]), float(r["pos_u_m"])])
        v = np.array([float(r["vel_e_ms"]), float(r["vel_n_ms"]), float(r["vel_u_ms"])])
        q = np.array([float(r["opp_pos_e_m"]), float(r["opp_pos_n_m"]), float(r["opp_pos_u_m"])])
        d = q - p
        rng = np.linalg.norm(d)
        u = d / max(rng, 1e-9)
        vh = v / max(np.linalg.norm(v), 1e-9)
        # bandit velocity from its own heading/pitch is not dumped as a vector; difference its
        # position instead (done by the caller via consecutive rows) -- here use own-ship only when
        # the bandit vector is supplied.
        out.append((u, vh, rng))
    lead = []
    for k, r in enumerate(rows):
        u, vh, rng = out[k]
        if k + 1 < len(rows):
            q0 = np.array(
                [
                    float(rows[k]["opp_pos_e_m"]),
                    float(rows[k]["opp_pos_n_m"]),
                    float(rows[k]["opp_pos_u_m"]),
                ]
            )
            q1 = np.array(
                [
                    float(rows[k + 1]["opp_pos_e_m"]),
                    float(rows[k + 1]["opp_pos_n_m"]),
                    float(rows[k + 1]["opp_pos_u_m"]),
                ]
            )
            w = q1 - q0
        else:
            w = None
        if w is None or np.linalg.norm(w) < 1e-9:
            lead.append(float("nan"))
            continue
        wh = w / np.linalg.norm(w)
        wperp = wh - float(np.dot(wh, u)) * u
        nrm = np.linalg.norm(wperp)
        if nrm < 1e-6:
            lead.append(float("nan"))
            continue
        wperp /= nrm
        ata = math.degrees(math.acos(float(np.clip(np.dot(vh, u), -1.0, 1.0))))
        sgn = 1.0 if float(np.dot(vh, wperp)) > 0.0 else -1.0
        lead.append(sgn * ata)
    return lead


def main():
    for name, path in TRACES:
        rows = load_rows(path)
        tr = to_trace(rows)
        res = B.detect_all(tr)
        print("=" * 100)
        print(f"{name}: {len(tr)} ticks, {res['duration_s']:.3f} s")
        print(
            f"  detect_all -> D1 {len(res['displacement_rolls'])}  D2 {len(res['yoyos'])} "
            f"D2c {len(res['yoyos_clean'])}  D3 {len(res['turning_room'])}  lethal_ticks {res['lethal_ticks']}"
        )
        print(
            f"  chi_valid ticks: {sum(1 for s in tr if s.chi_valid)}/{len(tr)}; "
            f"max|chi| {max(abs(s.chi) for s in tr) * DEG:.1f} deg at tick "
            f"{max(range(len(tr)), key=lambda k: abs(tr[k].chi))}"
        )
        print(f"  episode dalt {tr[-1].alt - tr[0].alt:+.1f} m")
        print("  -- D1 candidate holds (|chi|>60 & nz>3), first rejecting clause:")
        d1 = audit_d1(tr)
        if not d1:
            print("     (no candidate holds at all)")
        for r in d1:
            print(
                f"     ticks {r['i']}-{r['j']} t {r['t0']:.3f}-{r['t1']:.3f} dur {r['dur']:.3f}s "
                f"nzPk {r['nz_pk']:.2f} chiPk {r['chi_pk']:.1f} dalt {r['dalt']:+.1f} m -> {r['reject']}"
            )
        print("  -- D3 candidate holds (|chi|>45 & nz>3), first rejecting clause:")
        for r in audit_d3(tr):
            print(
                f"     ticks {r['i']}-{r['j']} dur {r['dur']:.3f}s nzPk {r['nz_pk']:.2f} "
                f"cpa {r['cpa0']:.1f}->{r['cpa1']:.1f} dalt {r['dalt']:+.1f} m -> {r['reject']}"
            )
        # ---- lead vs lag, detector-independent -------------------------------------------------
        lead = lead_lag(rows)
        fin = [(k, v) for k, v in enumerate(lead) if not math.isnan(v)]
        pos = [k for k, v in fin if v > 0]
        print("  -- LEAD(+) / LAG(-) pursuit, signed nose displacement off the LOS:")
        print(
            f"     ticks with LEAD sign: {len(pos)}/{len(fin)}; "
            f"max lead {max((v for _, v in fin), default=float('nan')):+.2f} deg; "
            f"min (lag) {min((v for _, v in fin), default=float('nan')):+.2f} deg"
        )
        for k in (0, 24, 48, 72, 96, 120, 144, 156, 165, len(fin) - 1):
            if k < len(lead) and not math.isnan(lead[k]):
                r = rows[k]
                print(
                    f"     tick {k:3d} t {float(r['t_s']):6.3f} rng {float(r['rng_m']):7.1f} "
                    f"lead {lead[k]:+7.2f} deg  hca {float(r['hca_deg']):7.2f}  aspect {float(r['aspect_deg']):7.2f} "
                    f"cpa {float(r['cpa_miss_m']):7.1f}"
                )
        print()


if __name__ == "__main__":
    main()


def fly_controls():
    """Fly the RECORDED hand-coded one-circle lead turn on merge1500 and audit the suite on IT.

    This is the sharpest form of the claim under test: "the suite contains no detector for the lead
    turn". `recorded` fires 0 rounds on this cell (d3b_prereg.txt:79-87), so the gun PRNG cannot
    touch it and one seed is the whole story.
    """
    import _diag_true_mpc as M

    cell = M.CELLS["merge1500"]
    for arm in ("recorded", "recorded_two"):
        env, base = M.build_env(cell, 62, 24.0)
        r = M.roll_once(env, base, M.make_control(arm), 576, keep=False)
        env.close()
        res = B.detect_all(r.trace)
        tr = r.trace
        asp = [s.aspect * DEG for s in tr]
        print("=" * 100)
        print(
            f"ARM {arm}: {r.steps} steps, event {r.event}, hp {r.hp_dealt:.1f}, rounds {r.rounds}, "
            f"rngMn {r.rng_min:.0f} m, dalt {r.dalt_m:+.0f} m"
        )
        print(
            f"  detect_all -> D1 {len(res['displacement_rolls'])}  D2 {len(res['yoyos'])} "
            f"D3 {len(res['turning_room'])}  lethal_ticks {res['lethal_ticks']}"
        )
        print(
            f"  aspect: first {asp[0]:.1f} min {min(asp):.1f} last {asp[-1]:.1f}; "
            f"most negative d_aspect over any window "
            f"{min(asp[j] - asp[i] for i in range(len(asp)) for j in range(i + 1, len(asp))):+.1f} deg"
        )
        ch = [abs(s.chi) * DEG for s in tr]
        ld = [i for i in range(len(tr)) if tr[i].nz > B.NZ_LOADED]
        print(
            f"  |chi| on LOADED ticks (nz>3, {len(ld)}/{len(tr)}): "
            f"max {max((ch[i] for i in ld), default=float('nan')):.1f}, "
            f"ticks in 30-45 band {sum(1 for i in ld if 30 <= ch[i] < 45)}, "
            f"45-60 band {sum(1 for i in ld if 45 <= ch[i] < 60)}, >=60 {sum(1 for i in ld if ch[i] >= 60)}"
        )
        print("  -- D1 candidate holds:")
        for x in audit_d1(tr) or [None]:
            print(
                "     (none)"
                if x is None
                else f"     ticks {x['i']}-{x['j']} dur {x['dur']:.3f}s nzPk {x['nz_pk']:.2f} "
                f"chiPk {x['chi_pk']:.1f} dalt {x['dalt']:+.1f} m -> {x['reject']}"
            )
        print("  -- D3 candidate holds:")
        for x in audit_d3(tr) or [None]:
            print(
                "     (none)"
                if x is None
                else f"     ticks {x['i']}-{x['j']} dur {x['dur']:.3f}s nzPk {x['nz_pk']:.2f} "
                f"cpa {x['cpa0']:.1f}->{x['cpa1']:.1f} dalt {x['dalt']:+.1f} m -> {x['reject']}"
            )
        cpa = [s.cpa_miss for s in tr]
        print(
            f"  cpa_miss first {cpa[0]:.1f} max {max(cpa):.1f}@{cpa.index(max(cpa))} min {min(cpa):.1f}@{cpa.index(min(cpa))}"
        )


if __name__ == "__main__":
    import sys

    if "--controls" in sys.argv:
        fly_controls()
