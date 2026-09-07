"""Verify/refute: "the D3b optimum IS LeadTurnArm's TURN phase, and the missing APPROACH is why D3=0".

Three measurements, all on the SAME cell (`merge1500`) and the SAME search seeds the D3b genome was
optimised and re-flown on, through the SAME `roll_once` (so the same honest trigger):

  A  LTA1000+  the reference arm AS SHIPPED (APPROACH -> TURN -> TRACK). Does the arm that the D3
                detector was DESIGNED around register D3 > 0 here? And does its APPROACH phase
                actually BUILD cpa_miss on a cell that spawns with 1621.7 m of predicted CPA when
                D_TURNROOM is 800 m?
  B  LTAturn0   the same arm with `turn_at` = +inf, i.e. TURN committed at step 0 with NO APPROACH.
                This is the claim's own counterfactual: "the TURN phase without the APPROACH phase".
                If the claim is right, B should look like the genome and read D3 = 0 for the same
                reason.
  C  the genome, for the side-by-side (bank command, cpa_miss, D3).

Nothing here is estimated: every number printed is read off a rollout of the real env.
"""

import math

import _diag_merge_convert_tmp as CV
import _diag_true_mpc as M
import numpy as np
from astrum_dogfight import bfm_detect as B

DEG = 180.0 / math.pi
SEEDS = [11, 28, 45, 62, 79, 96, 113, 130]
HORIZON_S = 24.0
N_STEPS = int(round(HORIZON_S / M.CONTROL_DT_S))
CELL = M.CELLS["merge1500"]
GENOME = np.array(
    [
        1.0, 1.0, 0.20466533, 0.42106359, 0.04260145, 1.0,
        -1.0, 0.81890272, -1.0, -0.45391245, -0.49038442, 0.90924598,
        1.0, -1.0, -1.0, -1.0, 0.7947789, 0.53075154,
    ]
)
GM = M.Genome(6, 3, "bank", True)


def det(r):
    d = B.detect_all(r.trace)
    return (
        len(d["displacement_rolls"]),
        len(d["yoyos"]),
        len(d["yoyos_clean"]),
        len(d["turning_room"]),
        len(d["lethal_ticks"]) if hasattr(d["lethal_ticks"], "__len__") else d["lethal_ticks"],
    )


def cpa_stats(r):
    c = [s.cpa_miss for s in r.trace]
    n = len(c)
    rises = sum(1 for i in range(n - 1) if c[i + 1] > c[i])
    return c, rises


def run_lta(seed, turn_at):
    env, base = M.build_env(CELL, seed, HORIZON_S)
    arm = CV.LeadTurnArm(turn_at, True)
    phases = []

    def ctl(step, me, bd, rel):
        phases.append(arm.phase)
        return arm.act(me, bd, rel)

    r = M.roll_once(env, base, ctl, N_STEPS, keep=True)
    env.close()
    return r, phases


def run_genome(seed):
    env, base = M.build_env(CELL, seed, HORIZON_S)

    def ctl(step, me, bd, rel):
        return GM.act(GENOME, step, N_STEPS, me)

    r = M.roll_once(env, base, ctl, N_STEPS, keep=True)
    env.close()
    return r, None


def main():
    print("#" * 100)
    print("#  VERIFY the 'rediscovered LeadTurnArm TURN phase' claim -- merge1500, D3b seeds")
    print("#" * 100)
    print(f"  cell merge1500 = {CELL}")
    print(f"  D_TURNROOM = {CV.D_TURNROOM:.0f} m   TURN_BANK = {CV.TURN_BANK * DEG:.1f} deg")
    print(f"  BANK_MAX_DEG (search bound) = {M.BANK_MAX_DEG:.1f} deg")
    print(f"  N_STEPS = {N_STEPS} @ {M.CONTROL_DT_S * 1000:.3f} ms\n")

    # --- the APPROACH control law, evaluated ALGEBRAICALLY at the spawn geometry ------------------
    # phi_cmd = side * clip(2*(miss - D)/D, -1, 1) * 45 deg. At miss = 1621.7, D = 800:
    miss0 = 1621.73
    g = float(np.clip(2.0 * (miss0 - CV.D_TURNROOM) / CV.D_TURNROOM, -1.0, 1.0))
    print(f"  APPROACH law at the SPAWN miss {miss0:.1f} m: clip(2*({miss0:.1f}-{CV.D_TURNROOM:.0f})"
          f"/{CV.D_TURNROOM:.0f}) = {g:+.3f}  -> phi_cmd = side * {g * 45.0:+.1f} deg")
    print("     sign is +side = bank TOWARD his pass side = predicted CPA is SPENT, not built.\n")

    rows = []
    for seed in SEEDS:
        # A: reference arm as shipped
        rA, phA = run_lta(seed, 1000.0)
        # B: TURN from step 0, no APPROACH (the claim's counterfactual)
        rB, phB = run_lta(seed, float("inf"))
        # C: the genome
        rC, _ = run_genome(seed)
        for name, r, ph in (("LTA1000+", rA, phA), ("LTAturn0", rB, phB), ("genome", rC, None)):
            d1, d2, d2c, d3, lt = det(r)
            c, rises = cpa_stats(r)
            napp = ph.count("APPROACH") if ph else 0
            nturn = ph.count("TURN") if ph else 0
            ntrk = ph.count("TRACK") if ph else 0
            rows.append(
                dict(
                    arm=name, seed=seed, steps=r.steps, event=str(r.event), kill=r.kill,
                    hp=r.hp_dealt, d1=d1, d2=d2, d2c=d2c, d3=d3, lt=lt,
                    cpa0=c[0], cpamin=min(c), cpamax=max(c), cpalast=c[-1], rises=rises,
                    ntr=len(c) - 1, dalt=r.dalt_m, rngmin=r.rng_min,
                    napp=napp, nturn=nturn, ntrk=ntrk,
                )
            )
        print(f"  seed {seed} done", flush=True)

    print("\n" + "=" * 118)
    print("  arm        seed steps      event kill     hp  D1 D2 D2c D3 lethTk   cpa0  cpaMin  cpaMax "
          "cpaLast rises/tr  rngMin    dalt  APPR TURN TRCK")
    print("=" * 118)
    for r in rows:
        print(
            f"  {r['arm']:<9} {r['seed']:>5} {r['steps']:>5} {r['event']:>10} {r['kill']:>4} "
            f"{r['hp']:>6.1f}  {r['d1']:>2} {r['d2']:>2} {r['d2c']:>3} {r['d3']:>2} {r['lt']:>6} "
            f"{r['cpa0']:>6.0f} {r['cpamin']:>7.0f} {r['cpamax']:>7.0f} {r['cpalast']:>7.0f} "
            f"{r['rises']:>4}/{r['ntr']:<4} {r['rngmin']:>7.0f} {r['dalt']:>7.0f} "
            f"{r['napp']:>5} {r['nturn']:>4} {r['ntrk']:>4}"
        )

    print("\n  POOLED over the 8 D3b seeds:")
    for name in ("LTA1000+", "LTAturn0", "genome"):
        sub = [r for r in rows if r["arm"] == name]
        print(
            f"    {name:<9} D1={sum(r['d1'] for r in sub):>3} D2={sum(r['d2'] for r in sub):>3} "
            f"D2c={sum(r['d2c'] for r in sub):>3} D3={sum(r['d3'] for r in sub):>3} | "
            f"kills={sum(r['kill'] for r in sub)}/8 hp={np.mean([r['hp'] for r in sub]):.1f} "
            f"dalt={np.mean([r['dalt'] for r in sub]):+.0f} m"
        )

    # --- per-step APPROACH detail on one seed: does APPROACH BUILD cpa_miss? ----------------------
    print("\n" + "=" * 100)
    print("  APPROACH-PHASE DETAIL, LTA1000+ seed 62: what the missing phase would have done to cpa_miss")
    print("=" * 100)
    env, base = M.build_env(CELL, 62, HORIZON_S)
    arm = CV.LeadTurnArm(1000.0, True)
    log = []

    def ctl(step, me, bd, rel):
        ph = arm.phase
        a = arm.act(me, bd, rel)
        miss, side = CV.cpa(me, bd)
        vm = np.asarray(me.vel, float)
        sp = float(np.linalg.norm(vm))
        gam = math.asin(max(-1.0, min(1.0, vm[2] / sp))) if sp > 1e-6 else 0.0
        # the TURN law's own bank target, verbatim from _diag_merge_convert_tmp.py:191
        btgt = float(np.clip(CV.TURN_BANK + 2.0 * gam, math.radians(40.0), math.radians(115.0)))
        log.append(
            (step, ph, float(me.phi) * DEG, miss, side, float(rel.rng), float(a[0]), float(a[1]),
             gam * DEG, btgt * DEG)
        )
        return a

    r = M.roll_once(env, base, ctl, N_STEPS, keep=True)
    env.close()
    print("   step     t_s  phase       phi_deg   cpa_m  side    rng_m    ail   elev  gam_deg  LTA_bank_tgt")
    for row in log:
        if row[0] % 12 == 0 or row[0] in (0, 1, 2):
            print(
                f"  {row[0]:>5} {row[0] * M.CONTROL_DT_S:>7.3f}  {row[1]:<9} {row[2]:>8.2f} "
                f"{row[3]:>7.0f} {row[4]:>+5.0f} {row[5]:>8.0f} {row[6]:>+6.2f} {row[7]:>+6.2f} "
                f"{row[8]:>+8.2f} {row[9]:>13.2f}"
            )
        if row[0] > 300:
            break
    bt = [row[9] for row in log]
    print(f"\n   LTA TURN-law bank target over the WHOLE episode: min {min(bt):.2f} max {max(bt):.2f} deg; "
          f"ticks at the 115 ceiling {sum(1 for b in bt if b >= 114.999)}/{len(bt)}, "
          f"at the 40 floor {sum(1 for b in bt if b <= 40.001)}/{len(bt)}")
    print("   (the search's seg 0 commands a FIXED +115.00 deg; the genome has no gamma feedback at all)")
    appr = [row for row in log if row[1] == "APPROACH"]
    if appr:
        cs = [row[3] for row in appr]
        print(f"\n   APPROACH lasted {len(appr)} steps ({len(appr) * M.CONTROL_DT_S:.3f} s)")
        print(f"   cpa_miss during APPROACH: {cs[0]:.1f} -> {cs[-1]:.1f} m   (min {min(cs):.1f}, max {max(cs):.1f})")
        ups = sum(1 for i in range(len(cs) - 1) if cs[i + 1] > cs[i])
        print(f"   ticks where cpa_miss RISES during APPROACH: {ups}/{len(cs) - 1}")
        print(f"   net change over APPROACH: {cs[-1] - cs[0]:+.1f} m  "
              f"({'BUILT room' if cs[-1] > cs[0] else 'SPENT room'})")
    turn = [row for row in log if row[1] == "TURN"]
    if turn:
        print(f"\n   TURN lasted {len(turn)} steps ({len(turn) * M.CONTROL_DT_S:.3f} s), "
              f"|phi| {min(abs(t[2]) for t in turn):.1f}-{max(abs(t[2]) for t in turn):.1f} deg")
        print(f"   ticks with |phi| > 90 deg (PAST VERTICAL): "
              f"{sum(1 for t in turn if abs(t[2]) > 90.0)}/{len(turn)}")
    print(f"\n   outcome: event={r.event} kill={r.kill} hp={r.hp_dealt:.1f} dalt={r.dalt_m:+.0f} m "
          f"rngMin={r.rng_min:.0f} m  D1/D2/D2c/D3 = {det(r)[:4]}")


if __name__ == "__main__":
    main()
