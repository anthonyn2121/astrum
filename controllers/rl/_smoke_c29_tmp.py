"""Focused CPU smoke for the curriculum-29 additions: the honest FRONTIER-slice gate + the
reward-v9 deck-cushion / expert-hit-gated-fire reward, all on the real Tier-3 env.

    cd controllers/rl && uv run python _smoke_c29_tmp.py

Mirrors _smoke_c27_tmp but swaps in a 2-rung always-[0,hi] BANDED net ladder and turns the
probe cadence up, so the new paths run end-to-end without crashing:
  * the collector records each probe's START aspect (probe_aspect) from the live geometry,
  * the learner's ingest votes ONLY frontier-slice probes ([hi-FRONTIER_SLICE_DEG, hi]),
  * reward.compute applies the dense deck cushion + expert-hit-gated fire bonus (imported).
Tiny model, short episodes; the step-backstop advances the stage so the env rebuilds at the
wider band (the frontier gate alone accrues too few votes in a 3k-step smoke — the backstop is
the safety net, exactly as in a real run).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch
from astrum_rl import _paths
from astrum_rl.tdmpc2.config import TDMPC2Config
from astrum_rl.tdmpc2.train_parallel import (
    FRONTIER_SLICE_DEG,
    _in_frontier,
    build_stages,
    run_parallel,
)


def _tiny_cfg() -> TDMPC2Config:
    from astrum_dogfight import constants as C

    return TDMPC2Config(
        obs_dim=C.OBS_DIM,
        action_dim=C.ACTION_DIM,
        enc_dim=64,
        mlp_dim=64,
        latent_dim=64,
        simnorm_dim=8,
        num_q=2,
        horizon=3,
        batch_size=32,
        buffer_size=8192,
        num_samples=64,
        num_pi_trajs=6,
        num_elites=8,
        iterations=3,
        episode_length=96,
        mpc=False,
        device="cpu",
    )


def main() -> None:
    _paths.require_astrum_env()
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(1)

    # reward-v9 sanity (pure-Python, no sim): fire gate mirrors the expert's hit gate, and the
    # deck cushion ramps in only below (DECK_AGL + DECK_MARGIN) — the two c29 reward changes.
    from astrum_dogfight import reward as R
    from astrum_dogfight.geometry import AState

    def st(pos, *, vel, ammo=511, alt=None):
        return AState(
            id=1,
            team=0,
            alive=True,
            pos=np.asarray(pos, float),
            vel=np.asarray(vel, float),
            phi=0,
            theta=0,
            psi=0,
            p=0,
            q=0,
            r=0,
            alpha=0,
            beta=0,
            nz=1.0,
            airspeed=180.0,
            alt=(pos[2] if alt is None else alt),
            health=100.0,
            ammo=ammo,
        )

    north = [0.0, 180.0, 0.0]
    me0 = st([0, 0, 4500], vel=north, ammo=511)
    me_fired = st([0, 0, 4500], vel=north, ammo=501)
    near = AState(
        id=2,
        team=0,
        alive=True,
        pos=np.asarray([0, 600, 4500], float),
        vel=np.asarray(north, float),
        phi=0,
        theta=0,
        psi=0,
        p=0,
        q=0,
        r=0,
        alpha=0,
        beta=0,
        nz=1.0,
        airspeed=180.0,
        alt=4500.0,
        health=100.0,
        ammo=511,
    )
    wez = R.compute(me0, near, me_fired, near)  # co-moving dead-ahead: lead≈0 → hit → WEZ bonus
    assert wez.info.get("rounds_fired") == 10 and wez.reward > 0.4, wez
    cross = AState(
        id=2,
        team=0,
        alive=True,
        pos=np.asarray([0, 1000, 4500], float),
        vel=np.asarray([180.0, 0.0, 0.0], float),
        phi=0,
        theta=0,
        psi=0,
        p=0,
        q=0,
        r=0,
        alpha=0,
        beta=0,
        nz=1.0,
        airspeed=180.0,
        alt=4500.0,
        health=100.0,
        ammo=511,
    )
    spray = R.compute(me0, cross, me_fired, cross)  # 1 km crossing: big lateral miss → WASTE
    assert spray.reward < wez.reward, (
        spray,
        wez,
    )  # the hit gate makes the long crossing spray cost
    me_cush = st([0, 0, 600], vel=north, alt=600.0)  # inside the 800 m cushion, above the 300 deck
    cush = R.compute(me0, near, me_cush, near)
    assert cush.info.get("deck_penalty", 0.0) < 0.0, (
        cush
    )  # dense floor penalty fires in the cushion
    me_high = st([0, 0, 4500], vel=north, alt=4500.0)
    assert "deck_penalty" not in R.compute(me0, near, me_high, near).info  # none up high
    print(
        f"reward-v9 OK — WEZ {wez.reward:.3f} > spray {spray.reward:.3f}; "
        f"deck cushion {cush.info['deck_penalty']:.4f}"
    )

    # _in_frontier gate-vote filter (pure): on a [0,80] band only the outer slice votes.
    band = build_stages(
        rows=(
            dict(
                tier=3,
                opponent_tier=0,
                posture="offensive",
                aspect_deg=0.0,
                aspect_deg_max=80.0,
                opp_heading_deg=None,
                separation_m=700.0,
                metric="net",
                winrate=0.1,
                net_threshold=3.0,
            ),
        ),
        backstop_steps=1,
    )[0]
    assert _in_frontier(80.0, band) and not _in_frontier(80.0 - FRONTIER_SLICE_DEG - 1, band)

    steps = 3000
    # 2 always-[0,hi] banded net rungs vs the native T3; low bars + tiny backstop so the stage
    # flips mid-run and the env rebuilds at the wider band. probe_every=2 → many probes so the
    # collector's probe_aspect recording + the frontier-vote path both execute.
    stages = build_stages(
        rows=(
            dict(
                tier=3,
                opponent_tier=0,
                posture="offensive",
                aspect_deg=0.0,
                aspect_deg_max=60.0,
                opp_heading_deg=None,
                separation_m=700.0,
                metric="net",
                winrate=0.10,
                net_threshold=3.0,
            ),
            dict(
                tier=3,
                opponent_tier=0,
                posture="offensive",
                aspect_deg=0.0,
                aspect_deg_max=70.0,
                opp_heading_deg=None,
                separation_m=700.0,
                metric="net",
                winrate=0.10,
                net_threshold=3.0,
            ),
        ),
        backstop_steps=steps // 2,
        damage_timeout_penalty=-40.0,
    )
    assert stages[0].aspect_deg == 0.0 and stages[0].aspect_deg_max == 60.0  # always-[0,hi]

    tmp = Path(tempfile.mkdtemp(prefix="c29_smoke_"))
    save = str(tmp / "run.pt")
    summary = run_parallel(
        _tiny_cfg(),
        workers=2,
        steps=steps,
        seed_steps=150,
        stages=stages,
        env_kwargs=dict(action_repeat=5, frame_stack=1, max_time_s=10.0, rate_hz=120.0, jitter=0.5),
        weight_sync_every=10,
        weight_check_every=20,
        utd=1.0,
        queue_max=4,
        save=save,
        save_every=0,
        log_every=100,
        device=torch.device("cpu"),
        probe_every=2,
    )
    print(f"summary: {summary}")
    assert summary["collected"] >= steps
    assert summary["updates"] > 150
    assert summary["episodes"] > 0
    assert np.isfinite(summary["ret_last"])
    assert summary["stage"] >= 1, (
        f"frontier-gated banded ladder never advanced (stage={summary['stage']})"
    )
    print("c29 (frontier-slice gate + probe_aspect + reward-v9) smoke OK")


if __name__ == "__main__":
    main()
