"""curriculum-97 PRE-FLIGHT — does the OBS_DIM 41 -> 45 append actually resume BIT-IDENTICALLY?

The whole claim to a single-variable experiment rests on one property: after `_widen_resume_obs`
zero-pads the encoder's first layer, the warm champion must produce THE SAME ACTION it produced
before the append, so the four new features arrive silent and have to be earned rather than
delivering a step change in behaviour.  Every collapse in curriculum-65 began with a warm start
that perturbed the policy on contact, so this is checked, not assumed.

The test is exact because the append is at the END: `build_observation` now returns 45 floats whose
first 41 ARE the old vector, so `obs45[:41]` is a byte-exact reconstruction of what c96 trained on.
Feed that to c96 as loaded, feed the full 45 to c96 as widened, and the two actions must agree to
float precision.  If they do not, the append is not a controlled edit and the run must not launch.

Also checked, because a silent version of either would invalidate the run rather than crash it:
  (a) the padded weight equals the original on the trained columns and is EXACTLY zero on the new
      ones (a nonzero pad would inject an untrained bias through LayerNorm);
  (b) the four new features are finite, O(1) and CORRECTLY SIGNED on a real merge rollout, not just
      on the synthetic states the unit test uses;
  (c) Adam's moments were padded rather than reset (a reset would change the effective LR on the
      encoder and confound the edit).
"""

from __future__ import annotations

import copy
import math
import os

import numpy as np
import torch
from astrum_dogfight import constants as C
from astrum_dogfight.geometry import build_observation, flight_path_angle, relative
from astrum_rl.tdmpc2.agent import TDMPC2
from astrum_rl.tdmpc2.config import TDMPC2Config
from astrum_rl.tdmpc2.train_parallel import _widen_resume_obs

CKPT = "/workspace/controllers/rl/runs/curriculum-96/run.pt"
OUT = "/workspace/controllers/rl/runs/c97"
NEW = 4  # features appended by curriculum-97
_LOG: list[str] = []


def log(s: str = "") -> None:
    print(s, flush=True)
    _LOG.append(s)


def _build(sd: dict) -> TDMPC2:
    cfg = TDMPC2Config(**sd["cfg"])
    cfg.mpc = False  # act() must be the policy prior, not the planner (E3: MPC is death-seeking)
    agent = TDMPC2(cfg, device=torch.device("cpu"))
    agent.model.load_state_dict(sd["model"])
    agent.model.eval()
    return agent


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    log("#" * 100)
    log("#  curriculum-97 pre-flight — OBS_DIM 41 -> 45 must resume bit-identically")
    log("#" * 100)

    assert C.OBS_DIM == 41 + NEW, f"expected OBS_DIM 45, got {C.OBS_DIM}"
    raw = torch.load(CKPT, map_location="cpu", weights_only=False)
    fs = int(raw["cfg"].get("frame_stack", 1) or 1)
    old_per_frame = int(raw["cfg"]["obs_dim"]) // fs
    log(f"\nckpt {CKPT}")
    log(f"  cfg obs_dim {raw['cfg']['obs_dim']}  frame_stack {fs}  => per-frame {old_per_frame}")
    log(f"  collected {raw.get('step', raw.get('collected', '?'))}")
    assert old_per_frame == 41, f"c96 must be 41-wide per frame, got {old_per_frame}"

    old_sd = copy.deepcopy(raw)
    new_sd = copy.deepcopy(raw)
    w_old = old_sd["model"]["_encoder.0.weight"].clone()
    _widen_resume_obs(new_sd, old_per_frame, C.OBS_DIM, fs)
    w_new = new_sd["model"]["_encoder.0.weight"]

    # ---- (a) the pad itself -------------------------------------------------------------------
    log("\n" + "=" * 100)
    log("(a) ENCODER PAD")
    log("=" * 100)
    log(f"  shape {tuple(w_old.shape)} -> {tuple(w_new.shape)}")
    kept = torch.equal(w_new[:, :41], w_old)
    zeroed = bool(torch.all(w_new[:, 41:] == 0.0))
    log(f"  trained columns preserved EXACTLY : {kept}")
    log(
        f"  new columns exactly zero          : {zeroed}  (max |w| = {w_new[:, 41:].abs().max():.3e})"
    )
    assert kept and zeroed

    # ---- (c) optimizer moments ---------------------------------------------------------------
    log("\n" + "=" * 100)
    log("(c) ADAM MOMENTS")
    log("=" * 100)
    enc_group = new_sd["optim"]["param_groups"][0]["params"]
    padded = 0
    for i in enc_group:
        st = new_sd["optim"]["state"].get(i, {})
        ea = st.get("exp_avg")
        if getattr(ea, "shape", None) == w_new.shape:
            padded += 1
            log(
                f"  param {i}: exp_avg {tuple(ea.shape)}  new-column moment max "
                f"{ea[:, 41:].abs().max():.3e} (must be 0)"
            )
            assert bool(torch.all(ea[:, 41:] == 0.0))
    log(f"  encoder tensors with padded moments: {padded}  (0 would mean momentum was RESET)")

    # ---- (b) + the headline: identical action on a REAL rollout ------------------------------
    import _diag_f2_poscontrol as F
    import _diag_true_mpc as M

    agent_old, agent_new = _build(old_sd), _build(new_sd)

    log("\n" + "=" * 100)
    log("(b) THE HEADLINE — same state, 41-wide c96 vs 45-wide c96, greedy action")
    log("=" * 100)
    log(
        "     the 45-obs is fed to the widened net; obs45[:41] (byte-exact old vector) to the original."
    )

    worst = 0.0
    rows: list[tuple] = []
    for cell_name, cell in (("merge@2000", F.merge_cell(2000.0)), ("merge@0", F.merge_cell(0.0))):
        env, base = M.build_env(cell, 11, 24.0)
        env.reset()
        for step in range(int(round(24.0 / M.CONTROL_DT_S))):
            me = base._states[base.learner_id]
            bd = base._states[base.opponent_id]
            rel = relative(me, bd)
            o45 = build_observation(me, bd)
            assert o45.shape == (45,) and np.all(np.isfinite(o45))
            o41 = o45[:41].copy()
            with torch.no_grad():
                a_old = agent_old.act(o41, t0=(step == 0), eval_mode=True).numpy()
                a_new = agent_new.act(o45, t0=(step == 0), eval_mode=True).numpy()
            d = float(np.max(np.abs(a_old - a_new)))
            worst = max(worst, d)
            if step % 96 == 0:
                rows.append(
                    (
                        cell_name,
                        step * M.CONTROL_DT_S,
                        float(rel.rng),
                        o45[41],
                        o45[42],
                        o45[43],
                        o45[44],
                        d,
                    )
                )
            a = np.asarray(a_new, np.float32).copy()
            a = np.append(a, np.float32(0.0))[:5] if a.shape[0] < 5 else a
            a[4] = M._trigger(rel)
            _o, _r, term, trunc, _i = env.step(a)
            if term or trunc:
                break
        env.close()

    log("")
    log(
        "   cell          t      rng   | idx41 relVS  idx42 sin(g_me)  idx43 sin(g_bd)  idx44 q_bd "
        "| max|dAction|"
    )
    for cn, t, rng, f41, f42, f43, f44, d in rows:
        log(
            f"   {cn:<12} {t:5.2f} {rng:8.1f} | {f41:+12.5f} {f42:+13.5f} {f43:+15.5f} "
            f"{f44:+11.5f} | {d:.3e}"
        )

    log("")
    log(f"  WORST max|action_old - action_new| over both rollouts: {worst:.3e}")
    log(
        f"  VERDICT: {'PASS — the append is silent' if worst < 1e-6 else 'FAIL — the append PERTURBS the policy'}"
    )

    # ---- sign sanity on the live features -----------------------------------------------------
    log("\n" + "=" * 100)
    log("SIGN SANITY on the live features (idx43 must track the T3's measured climb)")
    log("=" * 100)
    env, base = M.build_env(F.merge_cell(2000.0), 11, 24.0)
    env.reset()
    g_bd, g_me, relvs = [], [], []
    for step in range(int(round(24.0 / M.CONTROL_DT_S))):
        me = base._states[base.learner_id]
        bd = base._states[base.opponent_id]
        rel = relative(me, bd)
        o = build_observation(me, bd)
        g_bd.append(math.degrees(flight_path_angle(bd)))
        g_me.append(math.degrees(flight_path_angle(me)))
        relvs.append(float(o[41]) * C.V_REF)
        assert abs(o[43] - math.sin(flight_path_angle(bd))) < 1e-6
        with torch.no_grad():
            a = agent_new.act(o, t0=(step == 0), eval_mode=True).numpy()
        a = np.asarray(a, np.float32).copy()
        a[4] = M._trigger(rel)
        _o, _r, term, trunc, _i = env.step(a)
        if term or trunc:
            break
    env.close()
    log(
        f"  T3  gamma deg: p05 {np.percentile(g_bd, 5):+6.2f}  p50 {np.median(g_bd):+6.2f}  "
        f"p95 {np.percentile(g_bd, 95):+6.2f}  max {np.max(g_bd):+6.2f}"
    )
    log(
        f"  ME   gamma deg: p05 {np.percentile(g_me, 5):+6.2f}  p50 {np.median(g_me):+6.2f}  "
        f"p95 {np.percentile(g_me, 95):+6.2f}  min {np.min(g_me):+6.2f}"
    )
    log(
        f"  rel vertical speed m/s (idx41 x V_REF): p05 {np.percentile(relvs, 5):+7.2f}  "
        f"p50 {np.median(relvs):+7.2f}  p95 {np.percentile(relvs, 95):+7.2f}"
    )
    log("  => idx43 > 0 while idx42 < 0 is the 'he climbs, I dive' state the box exists to expose;")
    log("     under c96 (which is what flew this) that state must be VISIBLE in these percentiles.")

    with open(os.path.join(OUT, "c97_preflight.txt"), "w") as fh:
        fh.write("\n".join(_LOG) + "\n")
    log(f"\n[written] {os.path.join(OUT, 'c97_preflight.txt')}")


if __name__ == "__main__":
    main()
