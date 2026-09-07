"""Export a dogfight EPISODE (embedded env) to an `astrum-replay-bag` JSON the
browser renderer opens offline via its "Open file…" control.

The RL episodes run in the EMBEDDED headless env (astrum_env in-process), not the
Zenoh sim → telemetry → bag pipeline that `astrum-web-export` bakes. So we roll the
greedy policy here and emit the SAME JSON shape (crates/astrum-web/src/lib.rs ::
ExportDoc / FrameJson) directly — one frame per aircraft per agent-rate step.

Run inside the container (RL uv env):
  cd /workspace/controllers/rl && uv run python _replay_export_tmp.py \
      --checkpoint runs/BASELINE_merge_survivor_2480k.pt \
      --opp-heading 180 --opp-tier 0 --separation 1200 --aspects 0 \
      --episodes 6 --jitter 0.5 --max-time-s 120 --out-dir /workspace/replays

Default geometry = the canonical NOSE-TO-NOSE head-on merge vs the scripted Tier-3
(aspect 0 + opp_heading 180). Prints a per-episode summary and writes the most
illustrative episodes (longest engagements) as separate .json files.

Throwaway diagnostic (untracked), like _verdict_tmp.py / _verdict2p_tmp.py.
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch

from astrum_rl import _paths
from astrum_rl.tdmpc2.agent import TDMPC2, resolve_device
from astrum_rl.tdmpc2.config import TDMPC2Config

# Ground-map anchor: the renderer back-computes the tile origin from each frame's
# (lat,lon) minus its ENU offset (web/js/scene.js), so a consistent reference here
# places the aerial imagery correctly under the fight. R-2508 / Edwards airspace.
_LAT0 = 34.90
_LON0 = -117.90
_M_PER_DEG_LAT = 111320.0  # must match web/js/scene.js


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export a dogfight episode to astrum-replay-bag JSON.")
    p.add_argument("--checkpoint", default="runs/BASELINE_merge_survivor_2480k.pt")
    p.add_argument("--episodes", type=int, default=6, help="episodes to roll per aspect")
    p.add_argument("--export-top", type=int, default=2, help="write the N longest engagements")
    p.add_argument("--aspects", default="0", help="comma list of start aspects (deg)")
    p.add_argument("--separation", type=float, default=1200.0)
    p.add_argument("--opp-heading", type=float, default=180.0,
                   help="opponent ABSOLUTE heading deg; 180 = head-on merge; <0 = co-heading CHASE (T3 ahead)")
    p.add_argument("--opp-tier", type=int, default=0, help="0/3 = full scripted Tier-3")
    p.add_argument("--action-repeat", type=int, default=5)
    p.add_argument("--frame-stack", type=int, default=1)
    p.add_argument("--max-time-s", type=float, default=120.0)
    p.add_argument("--rate-hz", type=float, default=120.0)
    p.add_argument("--jitter", type=float, default=0.5)
    p.add_argument("--out-dir", default="/workspace/replays")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def _load(path: str, device) -> TDMPC2:
    sd = torch.load(path, map_location=device, weights_only=False)
    cfg = TDMPC2Config(**sd["cfg"])
    cfg.mpc = False  # greedy policy-prior mean (matches the verdict harnesses)
    agent = TDMPC2(cfg, device)
    agent.load_state_dict(sd)
    agent.model.eval()
    return agent, int(sd.get("stage_idx", -1)), int(sd.get("collected", -1))


def _frame(s, t_s: float) -> dict:
    """Project one ENU AState into the browser's FrameJson shape (+ wall_ns)."""
    e, n, u = float(s.pos[0]), float(s.pos[1]), float(s.pos[2])
    lat = _LAT0 + n / _M_PER_DEG_LAT
    m_per_deg_lon = _M_PER_DEG_LAT * math.cos(math.radians(_LAT0))
    lon = _LON0 + e / m_per_deg_lon
    ve, vn, vu = float(s.vel[0]), float(s.vel[1]), float(s.vel[2])
    return {
        "id": int(s.id),
        "kind": "fixed-wing",
        "sim_time_s": t_s,
        "lat_deg": lat,
        "lon_deg": lon,
        "alt_m": float(s.alt),
        "enu_e": e,
        "enu_n": n,
        "enu_u": u,
        "vel_north_mps": vn,          # AState.vel is ENU → NED for the wire
        "vel_east_mps": ve,
        "vel_down_mps": -vu,
        "phi_rad": float(s.phi),
        "theta_rad": float(s.theta),
        "psi_rad": float(s.psi),
        "airspeed_mps": float(s.airspeed),
        "health": int(max(0, min(255, round(float(s.health))))),
        "wall_ns": int(max(0.0, t_s) * 1e9),
    }


def _roll_episode(agent, env, base, C, seed: int, dt: float) -> tuple[list[dict], dict]:
    from astrum_dogfight.geometry import relative

    from astrum_rl.tdmpc2.eval_policy import _classify

    obs, _ = env.reset(seed=seed)
    agent._prev_mean.zero_()
    frames: list[dict] = []
    k = 0
    # t=0 snapshot (post-reset) so the merge entry is on the timeline.
    for s in base._states.values():
        frames.append(_frame(s, 0.0))
    min_rng = float("inf")
    event = None
    t0 = True
    while True:
        a = agent.act(obs, t0=t0, eval_mode=True).numpy()
        t0 = False
        obs, _r, terminated, truncated, info = env.step(a)
        k += 1
        if info.get("event"):
            event = info["event"]
        t_s = k * dt
        for s in base._states.values():
            frames.append(_frame(s, t_s))
        me = base._states.get(base.learner_id)
        opp = base._states.get(base.opponent_id)
        if me is not None and opp is not None and me.alive and opp.alive:
            min_rng = min(min_rng, float(relative(me, opp).rng))
        if terminated or truncated:
            break
    me = base._states.get(base.learner_id)
    opp = base._states.get(base.opponent_id)
    outcome = _classify(event, terminated, truncated)
    summary = {
        "steps": k,
        "dur_s": k * dt,
        "outcome": outcome,
        "event": event,
        "min_rng_m": (None if min_rng == float("inf") else min_rng),
        "dealt": (C.HP_MAX - float(opp.health)) if opp is not None else None,
        "taken": (C.HP_MAX - float(me.health)) if me is not None else None,
        "me_end_alt": (float(me.alt) if me is not None else None),
        "me_alive": (bool(me.alive) if me is not None else None),
    }
    return frames, summary


def _write_doc(path: str, frames: list[dict], meta_note: str) -> None:
    ids = sorted({f["id"] for f in frames})
    ts = [f["sim_time_s"] for f in frames]
    doc = {
        "kind": "astrum-replay-bag",
        "version": 1,
        "meta": {
            "frames": len(frames),
            "aircraft": ids,
            "sim_time_start": min(ts) if ts else 0.0,
            "sim_time_end": max(ts) if ts else 0.0,
            "note": meta_note,
        },
        "frames": frames,
    }
    with open(path, "w") as fh:
        json.dump(doc, fh)


def main() -> None:
    args = parse_args()
    _paths.require_astrum_env()
    from astrum_dogfight import constants as C

    from astrum_rl.envs import make_tdmpc_env

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, args.threads))
    device = resolve_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    agent, stage, coll = _load(args.checkpoint, device)
    aspects = [float(x) for x in args.aspects.split(",") if x.strip()]
    dt = args.action_repeat / args.rate_hz

    print(
        f"REPLAY EXPORT | {args.checkpoint} (stage {stage}, {coll:,}) greedy | "
        f"vs T{args.opp_tier or 3} opp_hdg {args.opp_heading:.0f}° sep {args.separation:.0f}m "
        f"jitter{args.jitter:g} | {args.episodes} eps/aspect | agent-rate {1 / dt:.0f} Hz"
    )

    episodes: list[dict] = []
    for aspect in aspects:
        env = make_tdmpc_env(
            3,
            posture="offensive",
            aspect_deg=aspect,
            separation_m=args.separation,
            action_repeat=args.action_repeat,
            frame_stack=args.frame_stack,
            max_time_s=args.max_time_s,
            rate_hz=args.rate_hz,
            **({"jitter": args.jitter} if args.jitter > 0 else {}),
            **({"opp_heading_deg": args.opp_heading} if args.opp_heading is not None and args.opp_heading >= 0 else {}),
            **({"opponent_tier": args.opp_tier} if args.opp_tier > 0 else {}),
        )
        base = env.unwrapped
        for ep in range(args.episodes):
            frames, summ = _roll_episode(agent, env, base, C, args.seed + int(aspect) * 1000 + ep, dt)
            summ["aspect"] = aspect
            summ["ep"] = ep
            summ["frames"] = frames
            episodes.append(summ)
            mr = summ["min_rng_m"]
            print(
                f"  asp{aspect:>3.0f} ep{ep:>2} | {summ['dur_s']:>5.1f}s {summ['outcome']:>8} "
                f"| min-rng {('%5.0f' % mr) if mr is not None else '  n/a'}m "
                f"| dealt {summ['dealt']:>5.1f} taken {summ['taken']:>5.1f} "
                f"| end_alt {summ['me_end_alt']:>6.0f}m alive={summ['me_alive']}"
            )
        env.close()

    # Rank: KILLS first (cleanest = shortest), then survivals (longest = most to watch), losses last.
    _pri = {"kill": 3, "timeout": 2, "deck": 1, "death": 0}

    def _key(s):
        p = _pri.get(s["outcome"], 1)
        # kills: shorter is cleaner (negate dur so shorter sorts higher within the kill tier)
        secondary = -s["dur_s"] if s["outcome"] == "kill" else s["dur_s"]
        return (p, secondary)

    ranked = sorted(episodes, key=_key, reverse=True)
    top = ranked[: max(1, args.export_top)]
    print(f"\n  writing {len(top)} episode(s) to {args.out_dir}:")
    for s in top:
        mr = s["min_rng_m"]
        note = (
            f"BASELINE greedy vs T{args.opp_tier or 3} head-on merge (opp_hdg {args.opp_heading:.0f}, "
            f"sep {args.separation:.0f}m, aspect {s['aspect']:.0f}); {s['dur_s']:.1f}s {s['outcome']}, "
            f"min-range {mr:.0f}m" if mr is not None else "n/a"
        )
        fname = f"nose2nose_asp{int(s['aspect'])}_ep{s['ep']}_{s['outcome']}_{int(s['dur_s'])}s.json"
        path = os.path.join(args.out_dir, fname)
        _write_doc(path, s["frames"], note)
        print(f"    {path}  ({len(s['frames'])} frames, {s['dur_s']:.1f}s, {s['outcome']})")


if __name__ == "__main__":
    main()
