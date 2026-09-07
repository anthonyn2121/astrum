"""`DogfightGymEnv` — the Gymnasium-compatible training adapter (design doc §8).

Single-agent from the trainer's view: the learner flies one f16, a **scripted**
opponent (`opponent.Tier1Opponent` by default) flies the other, driven internally
through the same raw action interface. Wraps `astrum_env.DogfightEnv` and adds the
observation vector (§8.2), reward/penalties (§8.3), and the wrapper-enforced
boundaries (§8.4).

numpy-only: subclasses `gymnasium.Env` iff gymnasium is importable, else exposes
the identical surface over a tiny space shim so it runs in the offline container.
"""

from __future__ import annotations

import astrum_env
import numpy as np

from . import constants as C
from . import reward as _reward
from .geometry import AState, build_observation
from .opponent import PursuitExpert, Tier1Opponent, Tier2Opponent, Tier3Opponent
from .scenario import tier1_scenario, tier3_scenario

try:  # Gymnasium is optional — present it if installed, shim it otherwise.
    import gymnasium as _gym
    from gymnasium import spaces as _spaces

    _Base = _gym.Env
    _HAS_GYM = True
except Exception:  # pragma: no cover - exercised only in the minimal container
    _Base = object
    _HAS_GYM = False


class _Box:
    """Minimal Gymnasium-`Box`-compatible space for the no-gymnasium container."""

    def __init__(self, low, high, shape):
        self.low = np.broadcast_to(np.asarray(low, np.float32), shape).copy()
        self.high = np.broadcast_to(np.asarray(high, np.float32), shape).copy()
        self.shape = tuple(shape)
        self.dtype = np.float32

    def sample(self) -> np.ndarray:
        lo = np.where(np.isfinite(self.low), self.low, -1.0)
        hi = np.where(np.isfinite(self.high), self.high, 1.0)
        return np.random.uniform(lo, hi).astype(np.float32)

    def contains(self, x) -> bool:
        x = np.asarray(x)
        return x.shape == self.shape and bool(np.all(x >= self.low) and np.all(x <= self.high))


def _box(low, high, shape):
    # Broadcast scalar bounds up to `shape` first: gymnasium >=1.0's Box rejects a
    # 0-d `low`/`high` against an explicit shape (np.asarray(-np.inf) has shape ()),
    # and the offline shim expects arrays anyway.
    low = np.broadcast_to(np.asarray(low, np.float32), shape).copy()
    high = np.broadcast_to(np.asarray(high, np.float32), shape).copy()
    if _HAS_GYM:
        return _spaces.Box(low=low, high=high, shape=shape, dtype=np.float32)
    return _Box(low, high, shape)


class DogfightGymEnv(_Base):
    """A 1v1 gun-dogfight vs a scripted opponent."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario_toml: str | None = None,
        *,
        learner_id: int = 1,
        opponent_id: int | None = None,
        opponent=None,
        ground_alt: float = 0.0,
        gamma: float = C.GAMMA,
        timeout_penalty: float = 0.0,
        convert_bonus: float = 0.0,
        outcome_mode: bool = False,
        scenario_fn=None,
        jitter_seed: int = 0,
    ):
        super().__init__()
        # Per-episode start-geometry jitter (curriculum-14): if a `scenario_fn(rng) -> toml`
        # is given, every reset() rebuilds the underlying sim from a freshly-jittered scenario
        # (env construction is ~5 ms, negligible vs an episode), so a deterministic policy can
        # no longer memorise one fixed start and the deterministic-probe gate measures a real
        # kill-RATE over the distribution. Legacy static-TOML path (scenario_fn=None) is
        # unchanged — reset() just re-runs the fixed scenario. The RNG is seeded once (per
        # worker) and advances each reset, so runs stay reproducible; reset(seed=) reseeds it.
        self._scenario_fn = scenario_fn
        self._jitter_rng = np.random.default_rng(jitter_seed) if scenario_fn is not None else None
        if scenario_fn is not None:
            scenario_toml = scenario_fn(self._jitter_rng)  # initial build (for id discovery)
        self._env = astrum_env.DogfightEnv(scenario_toml or tier1_scenario())
        ids = self._env.agent_ids()
        if learner_id not in ids:
            raise ValueError(f"learner_id {learner_id} not in scenario ids {ids}")
        self.learner_id = learner_id
        if opponent_id is None:
            others = [i for i in ids if i != learner_id]
            if len(others) != 1:
                raise ValueError(f"expected exactly one opponent, got ids {ids}")
            opponent_id = others[0]
        self.opponent_id = opponent_id
        self.opponent = opponent or Tier1Opponent()
        # DAgger-style exploration guide surfaced (env-space) in every step/reset info as
        # "expert_action"; the RL collector uses it for whole-episode expert rollouts to seed
        # close→kill trajectories on the long-range rungs. Never drives the sim itself — the
        # learner (or a scripted opponent) always supplies the flown action.
        self._expert = PursuitExpert()
        self.ground_alt = ground_alt
        self.gamma = gamma
        # reward-v7: penalty added on a clock timeout (no kill/death, inside the bubble). Set
        # per-stage by the trainer (negative on kill rungs, 0 on survive rungs). See constants.py.
        self.timeout_penalty = float(timeout_penalty)
        # reward-v17: the merge-converter (M) terminal bonus. >0 enables the survive-convert terminal
        # in reward.compute (reach the bandit's rear quarter → episode ends, hand off to frozen F) and
        # suppresses the fire bonus (M saves ammo for F). 0 (default) = a legacy finisher/chase run.
        self.convert_bonus = float(convert_bonus)
        # reward-v19b (curriculum-62 BFM pivot, as revised): the OUTCOME reward. When set, reward.compute
        # KEEPS all five dense pure-pursuit terms and ADDS the conversion-dwell + specific-energy PBRS on
        # top (v19's original pure-outcome form dropped them and decked the champion; see reward.py's v19b
        # note). Run-level like convert_bonus; 0/False = legacy.
        self._outcome_mode = bool(outcome_mode)

        self.observation_space = _box(-np.inf, np.inf, (C.OBS_DIM,))
        # [aileron, elevator, rudder] ∈ [-1,1]; throttle, trigger ∈ [0,1].
        self.action_space = _box(
            [-1.0, -1.0, -1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0], (C.ACTION_DIM,)
        )
        self._states: dict[int, AState] = {}
        # reward-v4: running per-episode total of credited dense tracking reward, clamped
        # at C.TRACK_BUDGET (reward.compute is stateless, so the cap lives here).
        self._track_used = 0.0
        # reward-v10: running length of the current out-of-solution firing burst, so the waste
        # penalty can ramp with a sustained mag-dump (reward.compute is stateless — it takes this
        # in and returns the updated value; the env owns the per-episode state).
        self._waste_streak = 0.0
        # reward-v12: running per-episode HP dealt to the bandit, so a no-kill clock timeout can be
        # split into a PASSIVE standoff draw (dealt < PASSIVE_DRAW_HP — never ran him down) vs a
        # hard-fought no-finish draw (chased + connected). See step()'s timeout block.
        self._damage_dealt = 0.0
        # reward-v19: running per-episode conversion-dwell reward already credited, clamped at
        # C.CONVERT_DWELL_BUDGET (reward.compute is stateless — it takes this in and returns the
        # updated value; the env owns the per-episode state, exactly like _track_used/_waste_streak).
        self._convert_dwell_used = 0.0

    # -- helpers -----------------------------------------------------------
    def _snapshot(self, obs_list) -> dict[int, AState]:
        return {int(o.id): AState.from_obs(o) for o in obs_list}

    def _obs_vec(self) -> np.ndarray:
        return build_observation(
            self._states[self.learner_id], self._states[self.opponent_id], self.ground_alt
        )

    def _expert_action(self) -> np.ndarray:
        """The pursuit expert's action for the CURRENT state (matching the returned obs),
        in the RAW env action space. Exposed via info so the collector can substitute it."""
        a = self._expert.act(self._states[self.learner_id], self._states[self.opponent_id])
        return np.asarray(a, dtype=np.float32)

    # -- Gymnasium API -----------------------------------------------------
    def reset(self, *, seed=None, options=None):
        # With jitter (scenario_fn) each reset rebuilds the sim from a freshly-perturbed
        # scenario; the RNG advances per episode (reproducible), and an explicit `seed`
        # reseeds it so eval can request a specific, repeatable engagement. Without a
        # scenario_fn the scenario carries its own deterministic sim.seed and reset is a no-op.
        if self._scenario_fn is not None:
            if seed is not None:
                self._jitter_rng = np.random.default_rng(seed)
            self._env = astrum_env.DogfightEnv(self._scenario_fn(self._jitter_rng))
        self._states = self._snapshot(self._env.reset())
        self.opponent.reset(self._states[self.opponent_id])
        self._expert.reset(self._states[self.learner_id])
        self._track_used = 0.0  # reward-v4: new episode → refill the tracking budget
        self._waste_streak = 0.0  # reward-v10: new episode → reset the burst counter
        self._damage_dealt = 0.0  # reward-v12: new episode → reset the HP-dealt accumulator
        self._convert_dwell_used = 0.0  # reward-v19: new episode → refill the conversion-dwell budget
        return self._obs_vec(), {"expert_action": self._expert_action()}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        me_prev = self._states[self.learner_id]
        bandit_prev = self._states[self.opponent_id]

        opp_action = self.opponent.act(bandit_prev, me_prev)
        actions = [
            (self.learner_id, [float(x) for x in action[: C.ACTION_DIM]]),
            (self.opponent_id, [float(x) for x in opp_action]),
        ]
        obs_list, info = self._env.step(actions)
        self._states = self._snapshot(obs_list)
        me = self._states[self.learner_id]
        bandit = self._states[self.opponent_id]

        r = _reward.compute(
            me_prev,
            bandit_prev,
            me,
            bandit,
            self.ground_alt,
            self.gamma,
            self._waste_streak,
            self.convert_bonus,
            self._convert_dwell_used,
            self._outcome_mode,
        )
        # reward-v10: carry the updated burst counter into the next step (compute is stateless).
        self._waste_streak = float(r.info.get("waste_streak", 0.0))
        # reward-v19: carry the updated conversion-dwell accumulator (only touched in outcome_mode).
        self._convert_dwell_used = float(r.info.get("convert_dwell_used", self._convert_dwell_used))
        # reward-v12: accumulate real HP dealt this episode (info carries it only on a hit),
        # for the engagement-conditioned timeout split below.
        self._damage_dealt += float(r.info.get("damage_dealt", 0.0))

        # reward-v4: clamp cumulative dense tracking at C.TRACK_BUDGET so it can never
        # rival a kill. Trim only the portion of this step's tracking reward that would
        # push the episode total past the budget; everything else (damage, fire bonus,
        # penalties, terminals) passes through untouched.
        reward = float(r.reward)
        track_r = float(r.info.get("track_reward", 0.0))
        if track_r > 0.0:
            granted = min(track_r, max(0.0, C.TRACK_BUDGET - self._track_used))
            reward -= track_r - granted  # subtract the over-budget remainder
            self._track_used += granted

        terminated = r.terminated or bool(info.terminated)
        truncated = r.truncated or bool(info.truncated)

        # reward-v7: an UNFINISHED engagement — the sim clock ran out with no kill/death
        # (not terminated) and the learner still inside the bubble (reward.compute did not
        # already tag a bubble exit). Tag it "timeout" (safe for the win-rate gate: not a
        # "kill", still a non-loss "survive") and, on kill-metric stages, apply the penalty
        # so coasting to a draw is EV-negative and the policy must commit to the kill.
        event = r.info.get("event")
        timed_out = bool(info.truncated) and not terminated and event != "left_bubble"
        passive_draw = False
        if timed_out:
            event = "timeout"
            # reward-v12: a PASSIVE standoff draw = timed out having dealt < PASSIVE_DRAW_HP over
            # the whole episode (never connected — never ran the bandit down). Surfaced in info for
            # monitoring regardless of the per-stage penalty.
            passive_draw = self._damage_dealt < C.PASSIVE_DRAW_HP
            if self.timeout_penalty:
                # Penalise the passive draw PASSIVE_TIMEOUT_MULT× harder; a hard-fought no-finish
                # draw (chased + landed real damage but couldn't gun the kill) keeps the base
                # penalty. Keys on DAMAGE DEALT (real hits), so it can't be gamed by spraying from
                # standoff (reward-v10 keeps that −EV). Encodes "connect, don't just survive."
                if passive_draw:
                    reward += self.timeout_penalty * C.PASSIVE_TIMEOUT_MULT
                else:
                    reward += self.timeout_penalty

        info_out = {
            **r.info,
            "event": event,
            "episode_damage_dealt": self._damage_dealt,  # reward-v12: running HP dealt this episode
            "passive_draw": passive_draw,  # reward-v12: timed out without connecting (dealt < PASSIVE_DRAW_HP)
            "track_used": self._track_used,
            "expert_action": self._expert_action(),
            "sim_terminated": bool(info.terminated),
            "sim_truncated": bool(info.truncated),
            "sim_time_s": float(info.sim_time_s),
        }
        return self._obs_vec(), reward, terminated, truncated, info_out


def _jitter_fn(builder, jitter: float, scenario_kwargs: dict):
    """A `scenario_fn(rng) -> toml` closure that re-jitters `builder`'s geometry per reset."""

    def fn(rng):
        return builder(jitter=jitter, rng=rng, **scenario_kwargs)

    return fn


def make_tier1_env(
    *,
    timeout_penalty: float = 0.0,
    convert_bonus: float = 0.0,
    outcome_mode: bool = False,
    jitter: float = 0.0,
    jitter_seed: int = 0,
    **scenario_kwargs,
) -> DogfightGymEnv:
    """Convenience: a Tier-1 (straight-and-level dummy) dogfight env.

    `jitter`>0 rebuilds a perturbed start every episode (see `scenario._jitter_geometry`);
    `jitter_seed` seeds that per-episode RNG (set distinctly per collector worker)."""
    if jitter > 0.0:
        return DogfightGymEnv(
            scenario_fn=_jitter_fn(tier1_scenario, jitter, scenario_kwargs),
            jitter_seed=jitter_seed,
            timeout_penalty=timeout_penalty,
            convert_bonus=convert_bonus,
            outcome_mode=outcome_mode,
        )
    return DogfightGymEnv(
        tier1_scenario(**scenario_kwargs),
        timeout_penalty=timeout_penalty,
        convert_bonus=convert_bonus,
        outcome_mode=outcome_mode,
    )


def make_tier2_env(
    *,
    maneuver: str = "break",
    timeout_penalty: float = 0.0,
    convert_bonus: float = 0.0,
    outcome_mode: bool = False,
    jitter: float = 0.0,
    jitter_seed: int = 0,
    **scenario_kwargs,
) -> DogfightGymEnv:
    """Convenience: a Tier-2 (predictable defensive maneuvers) dogfight env.

    The opponent starts on the learner's tail-side (same setup as Tier 1) and, on
    detecting the learner in its rear hemisphere, breaks or Split-Ses away.
    `maneuver` ∈ {"break", "split_s", "alternate"}. `jitter`>0 → per-episode start jitter.
    """
    rate_hz = scenario_kwargs.get("rate_hz", 120.0)
    opponent = Tier2Opponent(maneuver=maneuver, dt=1.0 / rate_hz)
    if jitter > 0.0:
        return DogfightGymEnv(
            scenario_fn=_jitter_fn(tier1_scenario, jitter, scenario_kwargs),
            jitter_seed=jitter_seed,
            opponent=opponent,
            timeout_penalty=timeout_penalty,
            convert_bonus=convert_bonus,
            outcome_mode=outcome_mode,
        )
    return DogfightGymEnv(
        tier1_scenario(**scenario_kwargs),
        opponent=opponent,
        timeout_penalty=timeout_penalty,
        convert_bonus=convert_bonus,
        outcome_mode=outcome_mode,
    )


def make_tier3_env(
    *,
    posture: str = "offensive",
    opponent_tier: int = 0,
    timeout_penalty: float = 0.0,
    convert_bonus: float = 0.0,
    outcome_mode: bool = False,
    jitter: float = 0.0,
    jitter_seed: int = 0,
    **scenario_kwargs,
) -> DogfightGymEnv:
    """Convenience: a Tier-3 (FSM tactician + WEZ gun) dogfight env.

    `posture` picks the starting geometry: "offensive" puts the shooting bot on the
    learner's six (`tier3_scenario`); "defensive" puts it ahead with the learner on
    its tail (`tier1_scenario`), so the bot opens by breaking. `jitter`>0 → per-episode
    start jitter.

    `opponent_tier` DECOUPLES the opponent AI from the tier-3 start GEOMETRY
    (curriculum-20): 0 (default) or 3 → the full `Tier3Opponent`; 1 → a straight-and-level
    `Tier1Opponent` dummy; 2 → a `Tier2Opponent` breaker. This lets a neutral/off-axis merge
    geometry (`tier3_scenario`'s `aspect_deg`) be flown against a WEAKER AI, so the beam→six
    turn-to-guns CONVERSION skill is learnable against a non-evading target before the same
    merge is faced against the full T3 evader (which the geometry-only ladder could not teach —
    a lead-pursuit expert deals ~0 HP off-axis vs T3, and RL stalled passive there)."""
    rate_hz = scenario_kwargs.get("rate_hz", 120.0)
    if opponent_tier == 1:
        opponent = Tier1Opponent()
    elif opponent_tier == 2:
        opponent = Tier2Opponent(maneuver="break", dt=1.0 / rate_hz)
    else:
        opponent = Tier3Opponent(dt=1.0 / rate_hz)
    if posture == "offensive":
        builder = tier3_scenario
    elif posture == "defensive":
        builder = tier1_scenario
    else:
        raise ValueError(f"unknown posture {posture!r}")
    if jitter > 0.0:
        return DogfightGymEnv(
            scenario_fn=_jitter_fn(builder, jitter, scenario_kwargs),
            jitter_seed=jitter_seed,
            opponent=opponent,
            timeout_penalty=timeout_penalty,
            convert_bonus=convert_bonus,
            outcome_mode=outcome_mode,
        )
    return DogfightGymEnv(
        builder(**scenario_kwargs),
        opponent=opponent,
        timeout_penalty=timeout_penalty,
        convert_bonus=convert_bonus,
        outcome_mode=outcome_mode,
    )
