"""Smoke test for the async multi-collector trainer (`train_parallel`), on CPU.

    cd controllers/rl && uv run python -m astrum_rl.tdmpc2.smoke_parallel
    (or `make rl-tdmpc2-parallel-smoke`, which builds the .so first)

Spins up 2 collector processes against the real Tier-1 JSBSim env with a deliberately
tiny model, runs the full pipeline — random seed phase → pretrain burst → weight
broadcast → policy-prior collection → back-to-back learner updates — for a few hundred
collected steps, then checks the run made progress and that the saved checkpoint reloads
into an agent that emits a valid `[-1,1]` action. Exercises the whole parallel plumbing
(spawn, whole-episode transport, batched buffer.add_episode, weight sync) on genuine sim
transitions without touching the GPU.

Uses a **2-stage ladder** with a tiny step-backstop so the curriculum advances mid-run:
that exercises the stage-change path (shared `stage_value` flip → collector closes and
rebuilds its env at the new separation → guided-fire re-burst) and the survive-metric
success log, which a single-stage run would not. Then reloads the checkpoint and does a
short **--resume** run to cover the warm-start path (resume load → immediate broadcast →
no cold seed burst). Takes ~40-80 s (dominated by JSBSim init per worker, twice).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from .. import _paths
from .agent import TDMPC2
from .config import TDMPC2Config
from .train_parallel import _SELFPLAY, build_stages, run_parallel


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
        mpc=False,  # collectors act prior-only (the throughput default)
        device="cpu",
    )


# `steps` must be comfortably larger than the most the queue can pre-buffer while the
# learner cold-starts — `queue_max` episodes × up to `max_time_s·rate_hz/action_repeat`
# control steps each (here 4×240≈960). Otherwise the learner's first drain ingests the
# whole backlog at once, meets the budget, and the loop exits after only the seed burst
# — never exercising steady-state training (real runs use millions of steps, so the
# backlog is a negligible sliver). See the `updates > seed_steps` assertion below.
def run(steps: int = 4000, workers: int = 2, seed_steps: int = 150) -> None:
    _paths.require_astrum_env()
    torch.manual_seed(0)
    np.random.seed(0)
    # The smoke model is tiny; on a many-core box torch's default all-core threading is
    # pure dispatch overhead (each update fires ~ncpu threads for a 64-dim matmul) and
    # thrashes against the collector processes. Pin the learner to one thread so the
    # pretrain burst is milliseconds, not minutes, and steady-state training actually runs.
    torch.set_num_threads(1)

    cfg = _tiny_cfg()
    tmp = Path(tempfile.mkdtemp(prefix="tdmpc2_par_"))
    save = str(tmp / "run.pt")

    # A 2-rung ladder (both Tier 1, different separations) with a tiny backstop so the step-
    # backstop advances stage 0→1 partway through the run — exercising the env-rebuild-on-stage-
    # change path a single stage would leave untested. The first rung uses the numeric DAMAGE
    # metric (5-field row: ≥5 HP dealt) so the collector's opp-HP capture and `_stage_win`'s
    # damage branch run end-to-end; the second is gated on *survive* to cover that metric too.
    # backstop_steps is well inside `steps` so the flip actually fires regardless of the gate.
    stages = build_stages(
        rows=((1, 700, "damage", 0.30, 5.0), (1, 400, "survive", 0.85)),
        backstop_steps=steps // 2,
    )

    summary = run_parallel(
        cfg,
        workers=workers,
        steps=steps,
        seed_steps=seed_steps,
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
        # Exercise the guided-fire override (idea 2) end-to-end: p0=0.5 with quality=0
        # forces the trigger on ~half of all steps (gun_quality ≥ 0 always holds), so the
        # collector's index math, action mutation, and forced-fire→buffer path all run.
        fire_override_p0=0.5,
        fire_override_steps=steps,
        fire_override_quality=0.0,
        # Exercise the honest deterministic-probe gate (curriculum-14): every 3rd episode is
        # flown greedy (no expert/fire-override) and tagged probe=True — the only episode the
        # gate counts. With per-episode start jitter (env_kwargs above) this runs the full
        # rebuild-on-reset + probe-tag + probe-gate path end-to-end on the tiny model.
        probe_every=3,
    )
    print(f"summary: {summary}")

    assert summary["collected"] >= steps, f"only collected {summary['collected']} < {steps}"
    # Must have trained *past* the pretrain burst — a burst-only run (updates == seed_steps)
    # means the learner never reached steady state (starved, lagging, or the loop raced
    # past the budget while it was busy). That is a failure even though data was collected.
    assert summary["updates"] > seed_steps, (
        f"only {summary['updates']} updates (== seed burst {seed_steps}); steady-state "
        f"training never ran — learner starved or the step budget was met during the burst"
    )
    assert summary["episodes"] > 0, "no episodes were collected"
    assert np.isfinite(summary["ret_last"]), f"non-finite return {summary['ret_last']}"
    # The step-backstop must have advanced the ladder off stage 0 (env rebuilt at the new
    # separation) — the whole reason this smoke uses a multi-stage ladder.
    assert summary["stage"] >= 1, (
        f"curriculum never advanced past stage 0 (stage={summary['stage']})"
    )

    # The checkpoint must reload into a fresh agent and produce a valid action.
    assert Path(save).exists(), "no checkpoint written"
    sd = torch.load(save, map_location="cpu", weights_only=False)
    assert sd.get("stage_idx") == summary["stage"], "checkpoint didn't record the curriculum stage"
    agent = TDMPC2(TDMPC2Config(**sd["cfg"]), torch.device("cpu"))
    agent.load_state_dict(sd)
    a = agent.act(np.zeros(cfg.obs_dim, np.float32), t0=True, eval_mode=True).numpy()
    assert a.shape == (cfg.action_dim,)
    assert a.min() >= -1.0 - 1e-5 and a.max() <= 1.0 + 1e-5, a

    # --resume path: warm-start from the checkpoint (model + optimizers + stage), which must
    # skip the cold seed burst (broadcasts trained weights immediately) and keep training.
    resume_sd = torch.load(save, map_location="cpu", weights_only=False)
    start_stage = int(resume_sd["stage_idx"])
    resume_save = str(tmp / "resume.pt")
    resumed = run_parallel(
        cfg,
        workers=workers,
        steps=steps,
        seed_steps=seed_steps,
        stages=stages,
        env_kwargs=dict(action_repeat=5, frame_stack=1, max_time_s=10.0, rate_hz=120.0, jitter=0.5),
        weight_sync_every=10,
        weight_check_every=20,
        utd=1.0,
        queue_max=4,
        save=resume_save,
        save_every=0,
        log_every=100,
        device=torch.device("cpu"),
        start_stage=start_stage,
        resume_sd=resume_sd,
        probe_every=3,
    )
    print(f"resume summary: {resumed}")
    assert resumed["collected"] >= steps, f"resume only collected {resumed['collected']} < {steps}"
    assert resumed["updates"] > 0, "resume produced no learner updates"
    assert resumed["stage"] >= start_stage, "resume regressed the curriculum stage"

    print("parallel TD-MPC2 smoke OK")


def run_selfplay(steps: int = 3000, workers: int = 2, seed_steps: int = 150) -> None:
    """Smoke the SELF-PLAY pipeline (curriculum-21) end-to-end on CPU.

    Exercises the wiring the scripted-curriculum smoke does not: the learner seeds the opponent
    pool before collectors spawn, each collector re-samples a `ScriptedPolicyOpponent` from the
    pool and swaps it into the sim per episode, and the learner periodically snapshots a fresh
    prior into the pool (so it grows past the seed). A successful run with episodes collected
    implies the pooled-opponent sim path flew without crashing (a dead collector aborts the run).
    """
    _paths.require_astrum_env()
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(1)

    cfg = _tiny_cfg()
    tmp = Path(tempfile.mkdtemp(prefix="tdmpc2_sp_"))
    save = str(tmp / "run.pt")
    pool_dir = str(tmp / "pool")

    # The single neutral-merge self-play rung (huge backstop → the top-stage gate stays a no-op).
    stages = build_stages(rows=_SELFPLAY, backstop_steps=10_000_000, damage_timeout_penalty=-40.0)

    snapshot_every = steps // 3  # small so ≥1 periodic snapshot fires inside the budget
    summary = run_parallel(
        cfg,
        workers=workers,
        steps=steps,
        seed_steps=seed_steps,
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
        probe_every=3,
        selfplay=True,
        pool_dir=pool_dir,
        selfplay_snapshot_every=snapshot_every,
        selfplay_recency_bias=0.0,
        selfplay_max_pool=8,
        # curriculum-24 anchoring: half the episodes (and every probe) face the SCRIPTED T3 instead
        # of a pooled copy. Exercises the scripted-opponent-restore path AND the pooled path in one
        # run (the pool still grows on the snapshot clock, so the snaps≥2 assertion below holds).
        selfplay_scripted_prob=0.5,
    )
    print(f"selfplay summary: {summary}")

    assert summary["collected"] >= steps, f"only collected {summary['collected']} < {steps}"
    assert summary["updates"] > seed_steps, (
        f"steady-state training never ran ({summary['updates']})"
    )
    assert summary["episodes"] > 0, (
        "no episodes were collected (pooled-opponent path may have died)"
    )
    assert np.isfinite(summary["ret_last"]), f"non-finite return {summary['ret_last']}"

    # The pool must have grown past the single seed snapshot — the learner's periodic in-process
    # export fired and dropped a fresh frozen prior for collectors to face.
    snaps = sorted(Path(pool_dir).glob("snap_*.pt"))
    assert len(snaps) >= 2, f"pool never grew past the seed (snaps={[p.name for p in snaps]})"
    assert not list(Path(pool_dir).glob("*.tmp")), "a snapshot staging .tmp file was left behind"

    # Each snapshot must reload as a usable TorchScript prior (what a collector loads).
    from .export import load_prior

    prior = load_prior(str(snaps[-1]))
    a = prior(torch.zeros(1, cfg.obs_dim))
    assert a.shape == (1, cfg.action_dim) and a.abs().max() <= 1.0

    print("parallel TD-MPC2 SELF-PLAY smoke OK")


if __name__ == "__main__":
    import sys

    if "--selfplay" in sys.argv:
        run_selfplay()
    else:
        run()
