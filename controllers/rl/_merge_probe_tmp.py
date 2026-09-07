"""FEASIBILITY PROBE (temp): can a scripted BFM 'beam-merge' demonstrator convert the
neutral merge vs the native Tier-3 that lead-pursuit (PursuitExpert) cannot?

This is the CHEAP CPU gate before any GPU training. It flies a parameterized
`BeamMergeExpert` (defined inline so the law can be tuned fast without touching the
env package) vs a freshly-built native Tier-3 across the beam aspects, through the
EXACT production expert path: compute a native `[ail,elev,rud,thr,trigger]`, invert it
to agent-space with `invert_tdmpc_action`, and step the wrapped env (so `_LinearAction
Wrapper` re-maps it to the identical native action and `ActionRepeat` holds it 5 ticks
— the same cadence the collector flies `info["expert_action"]` at).

    cd controllers/rl && uv run python -u _merge_probe_tmp.py --aspects 60,75,80,85,90

If a config converts a meaningful fraction of merges (kills / real HP dealt), promote
`BeamMergeExpert` into astrum_dogfight.opponent + wire it as a selectable env expert.
If nothing converts, the merge is beyond scripted BFM and we report that with evidence.
"""

from __future__ import annotations

import argparse
import statistics
from math import cos, radians, sin

import numpy as np

from astrum_rl import _paths


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


class BeamMergeExpert:
    """Scripted BFM merge demonstrator. Unified continuous law (no phase switch):

    Aim along the bandit's flight path, blended by ASPECT — LAG (a point behind the
    bandit, along its velocity) at high aspect (the neutral merge: get to the rear
    hemisphere / control zone without overshooting), smoothly transitioning to gun-LEAD
    (the drag-free lead point) as aspect drops and I gain his six. Roll the lift vector
    onto that aim and pull a corner-limited hard turn; hold full AB at the merge and the
    proven guns-trail speed law once behind. A DECK GUARD (roll level + pull up when low
    and nose-low) prevents the deck death that kills the plain lead-pursuit expert. Fires
    on the point-blank hit gate (predicted lateral miss inside the target's hit sphere).

    Sign conventions (sim): aileron+ = roll right; elevator NEGATIVE = pull/g (aft stick);
    throttle in [0,1] (1.0 = AB); trigger fires when >= 0.5. Reads godmode AState like
    every scripted law. Returns native [ail, elev, rud, thr, trigger].
    """

    def __init__(
        self,
        gains=None,
        *,
        lag_gain: float = 0.6,
        deck_floor_agl: float = 1200.0,
        ground_alt: float = 0.0,
    ):
        from astrum_dogfight.opponent import F16Gains

        self.g = gains or F16Gains()
        self.lag_gain = lag_gain
        self.deck_floor = deck_floor_agl
        self.ground_alt = ground_alt

    def reset(self, s) -> None:  # stateless; interface symmetry
        pass

    def act(self, me, target) -> list:
        from astrum_dogfight import constants as C
        from astrum_dogfight.geometry import _unit, body_bearing, relative
        from astrum_dogfight.opponent import _corner_limited_pull

        rg = relative(me, target)

        # --- Deck guard: low & nose-low -> roll wings level and pull up, full power. ---
        agl = me.alt - self.ground_alt
        if agl < self.deck_floor and me.theta < radians(-3.0):
            ail = _clamp(-2.0 * me.phi - self.g.k_roll_rate * me.p, -1.0, 1.0)
            elev = _corner_limited_pull(-0.6, me)  # negative = pull the nose up
            return [ail, elev, 0.0, 1.0, 0.0]

        # --- Aim point (BFM): CLOSE aggressively when far, LAG only when close-and-fast to
        # avoid the overshoot, gun-LEAD once behind. aspect: angle at bandit tail->LOS-to-me,
        # ~pi at the beam/head-on merge, ~0 on his six. ---
        aspect_frac = _clamp(rg.aspect / radians(70.0), 0.0, 1.0)  # 1 = still ahead/abreast, 0 = behind
        close_frac = _clamp((1500.0 - rg.rng) / 1500.0, 0.0, 1.0)  # 0 far, 1 point-blank
        v_hat = _unit(target.vel)
        tof = rg.rng / C.MUZZLE_MPS if rg.rng > 1e-6 else 0.0
        lead_off = target.vel * tof  # drag-free gun lead (for the shot when behind)
        # lag ONLY bites when high-aspect AND close (the overshoot regime); ~0 far out (=> close).
        lag_mag = self.lag_gain * aspect_frac * close_frac * min(rg.rng, 1500.0)
        aim = target.pos + (1.0 - aspect_frac) * lead_off - v_hat * lag_mag

        off, bank_err, _fwd = body_bearing(me, aim)
        roll_gate = min(off / C.PURSUE_ROLL_DEADBAND, 1.0)
        ail = _clamp(C.K_PURSUE_ROLL * bank_err * roll_gate - self.g.k_roll_rate * me.p, -1.0, 1.0)
        aligned = max(cos(bank_err), 0.0)  # pull hardest once rolled onto the aim
        elev = _corner_limited_pull(_clamp(-C.K_PURSUE_PULL * off * aligned, -1.0, 0.0), me)

        # Throttle: AB at the merge (hold turn energy) -> guns-trail speed law once behind.
        closure = _clamp(C.K_CHASE_RANGE * (rg.rng - C.R_CHASE_TRAIL), 0.0, C.V_CHASE_MARGIN)
        thr_closer = _clamp(
            self.g.throttle_trim + self.g.k_spd_to_thr * (target.airspeed + closure - me.airspeed),
            0.0,
            1.0,
        )
        thr = aspect_frac * 1.0 + (1.0 - aspect_frac) * thr_closer

        return [ail, elev, 0.0, thr, 1.0 if self._fire(me, target, rg) else 0.0]

    @staticmethod
    def _fire(me, target, rg) -> bool:
        from astrum_dogfight import constants as C

        if me.ammo <= 0 or not me.alive or not target.alive or not rg.in_range:
            return False
        return rg.rng * sin(rg.lead_error) <= C.GUN_FIRE_MISS_M


def _roll(expert, env, base, C, invert, low, high, episodes: int, seed0: int) -> list:
    from astrum_rl.tdmpc2.eval_policy import _classify

    rows = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        expert.reset(base._states[base.learner_id])
        event = None
        terminated = truncated = False
        while True:
            me = base._states[base.learner_id]
            opp = base._states[base.opponent_id]
            native = expert.act(me, opp)
            a = invert(np.asarray(native, dtype=np.float32), low, high)
            obs, _r, terminated, truncated, info = env.step(a)
            if info.get("event"):
                event = info["event"]
            if terminated or truncated:
                break
        me = base._states[base.learner_id]
        opp = base._states[base.opponent_id]
        dealt = C.HP_MAX - float(opp.health)
        taken = C.HP_MAX - float(me.health)
        rows.append(
            {
                "outcome": _classify(event, terminated, truncated),
                "event": event,
                "dealt": dealt,
                "taken": taken,
                "net": dealt - taken,
            }
        )
    return rows


def parse_args():
    p = argparse.ArgumentParser(description="Scripted beam-merge demonstrator feasibility probe.")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--aspects", default="60,75,80,85,90")
    p.add_argument(
        "--demonstrator",
        default="beam",
        choices=["beam", "t3"],
        help="'beam' = hand-crafted BeamMergeExpert; 't3' = the native Tier3Opponent FSM "
        "flying the learner's jet (scripted self-play as expert).",
    )
    p.add_argument("--lag-gains", default="0.0,0.6,1.0", help="sweep of lag_gain configs (beam only)")
    p.add_argument("--separation", type=float, default=700.0)
    p.add_argument("--deck-floor", type=float, default=1200.0)
    p.add_argument("--jitter", type=float, default=0.5)
    p.add_argument(
        "--action-repeat",
        type=int,
        default=5,
        help="sim ticks each command is held (control cadence = rate_hz/this). The collector "
        "flies the expert at the training action_repeat; sweep this to isolate control-bandwidth.",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _paths.require_astrum_env()
    from astrum_dogfight import constants as C

    from astrum_rl.envs import invert_tdmpc_action, make_tdmpc_env

    np.random.seed(args.seed)
    aspects = [float(x) for x in args.aspects.split(",") if x.strip()]

    # 't3' = the competent BFM FSM flying the learner's jet (scripted self-play as expert):
    # a single config, no lag sweep. dt set to the action-repeat cadence so its FSM dwell
    # timers tick at the rate the collector would actually fly it (once per 5 sim ticks).
    if args.demonstrator == "t3":
        from astrum_dogfight.opponent import Tier3Opponent

        configs = [("t3", Tier3Opponent(dt=args.action_repeat / 120.0))]
    else:
        lag_gains = [float(x) for x in args.lag_gains.split(",") if x.strip()]
        configs = [
            (f"lag_gain={lag}", BeamMergeExpert(lag_gain=lag, deck_floor_agl=args.deck_floor))
            for lag in lag_gains
        ]

    print(
        f"MERGE PROBE [{args.demonstrator}] vs native T3 | sep {args.separation:.0f}m "
        f"jitter{args.jitter:g} deck_floor {args.deck_floor:.0f}m | {args.episodes} eps/cell"
    )
    for label, expert in configs:
        print(f"\n=== {label} ===")
        print(
            f"{'aspect':>6} {'N':>3} {'kill':>4} {'death':>5} {'deck':>4} {'trade':>5} "
            f"{'t/out':>5} {'dealt':>6} {'taken':>6} {'NET':>7}"
        )
        overall = []
        for aspect in aspects:
            env = make_tdmpc_env(
                3,
                posture="offensive",
                aspect_deg=aspect,
                separation_m=args.separation,
                action_repeat=args.action_repeat,
                frame_stack=1,
                max_time_s=60.0,
                rate_hz=120.0,
                jitter=args.jitter,
            )
            base = env.unwrapped
            low = np.asarray(base.action_space.low, dtype=np.float32)
            high = np.asarray(base.action_space.high, dtype=np.float32)
            rows = _roll(
                expert, env, base, C, invert_tdmpc_action, low, high,
                args.episodes, args.seed + int(aspect) * 1000,
            )
            env.close()
            overall.extend(rows)
            n = len(rows)
            kill = sum(r["outcome"] == "kill" for r in rows)
            death = sum(r["outcome"] == "death" for r in rows)
            deck = sum(r["outcome"] == "deck" for r in rows)
            trade = sum(r["event"] == "trade" for r in rows)
            tout = sum(r["outcome"] == "timeout" for r in rows)
            dealt = statistics.fmean(r["dealt"] for r in rows)
            taken = statistics.fmean(r["taken"] for r in rows)
            net = statistics.fmean(r["net"] for r in rows)
            print(
                f"{aspect:>6.0f} {n:>3} {kill:>4} {death:>5} {deck:>4} {trade:>5} "
                f"{tout:>5} {dealt:>6.1f} {taken:>6.1f} {net:>+7.1f}"
            )
        n = len(overall)
        kill = sum(r["outcome"] == "kill" for r in overall)
        loss = sum(r["outcome"] in ("death", "deck") for r in overall)
        net = statistics.fmean(r["net"] for r in overall)
        print(f"  ALL({label}): {n} eps | kills {kill} losses {loss} | mean NET {net:+.1f}")


if __name__ == "__main__":
    main()
