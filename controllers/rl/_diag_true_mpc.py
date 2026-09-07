"""D3b — TRUE-SIM TRAJECTORY SEARCH: is a gun kill from the neutral merge REACHABLE at all?

The question this box exists to answer is *winnability*, and it is the only instrument on the
board that can. Every other diagnostic measures what the learner (or a hand stick) DOES; this one
searches the action space directly, with the real sim in the loop, for ANY 15-30 s trajectory that
converts a neutral head-on merge into a gun kill against the fixed Tier-3.

    docker compose --profile rl run --rm -T -w /workspace/controllers/rl dogfight-train \
        uv run python -u _diag_true_mpc.py --cells merge1500 --gens 40

PRE-REGISTRATION (this is an EXISTENCE / LOWER-BOUND search, and the wording of its verdict is
fixed in advance):
  * If it produces a trajectory with >=1 kill from the neutral merge carrying a loaded D1 or D2
    event and nonzero `tLeth`, then a feasible, physics-respecting out-of-plane demonstration
    EXISTS and the program's remaining search space has been learner-side all along.
  * If it does not, the ONLY admissible statement is **"no solution found at horizon H under
    objective J"**. It may never conclude "unwinnable" and may never motivate changing the task.
    A search bounds achievability from BELOW only. (This is D3a's P2 lesson, paid for once
    already: D3a's honest verdict was "no dwell budget found under model M", not "no dwell
    budget exists".)

THE FOUR REQUIRED FIXES, and where each lives:

(1) HORIZON ABOVE THE MANEUVER BAND. One open-loop parameterised trajectory over the WHOLE
    `--horizon-s` (default 24 s = 576 control steps), never a 3 s receding window — 3 s sits
    below the 120-360-step maneuver band a displacement roll or yo-yo occupies, so a 3 s planner
    cannot represent the thing we are testing for. See `Genome`.

(2) TERMINAL-OUTCOME OBJECTIVE ONLY. `score()` pays kill, HP dealt, and cumulative `tLeth`
    (lay inside `HIT_RADIUS_M` AND range inside `LETHAL_RANGE_M`), prices altitude spent, and
    scores `hard_deck`/`killed` as -inf. It NEVER pays a per-step or terminal aim/p_hit potential:
    `range_factor` is 0.002 at 900 m and 0.000 past 1200 m, so on a 3000 m run-in an aim potential
    is identically flat and every candidate ties through exactly the one-circle/two-circle commit
    window that is the thing under test.
    The one concession, declared: a small CAPPED arrival term (seconds inside `LETHAL_RANGE_M`,
    plus closest-approach) so the search has gradient on "arrive at the fight". It is pure
    GEOMETRY — no aim, no lay, no p_hit — and it is capped at `ARRIVAL_CAP` = 5.0, which is what
    0.5 s of `tLeth` alone is worth, so it cannot outbid the outcome. Every component is printed
    separately; the VERDICT is read off kill / hp / tLeth, never off the blended scalar.

(3) BUDGET TO THE DIMENSION, AND MARGINALISE THE PRNG. CMA-ES with popsize >= 10*dim (default
    `10*dim`; 18 dims at 6 segments x 3 channels -> 180/generation), and every candidate scored as
    the MEAN over `--seeds` sim seeds (default 8), because gun dispersion is a shared seeded
    SplitMix64 advanced per round fired at 8 mrad = 4.8 m 1-sigma at 600 m against a
    `HIT_RADIUS_M` of 8.0 m. A max over a stochastic objective across thousands of selections is
    the c13 lucky-noise gate. `sim.seed` is a `tier3_scenario` kwarg INDEPENDENT of the geometry,
    so seeds vary the gun PRNG with the start held exactly fixed.
    Cheap and exact: a candidate that fires ZERO rounds cannot be touched by the gun PRNG, so it
    is scored once. `--precheck` VERIFIES that claim (cross-seed max|div| must be 0) rather than
    assuming it; if it fails the harness disables the skip itself.

(4) A CONTROL WITH POWER. The chase cannot fail for the reason the merge can (c96 kills 6/6 there
    with D1=D2=D3=0.00, i.e. in-plane pursuit already solves the chase), so the controls are (a)
    the RECORDED hand-coded one-circle lead turn, imported verbatim from
    `_diag_merge_convert_tmp.LeadTurnArm` -- the arm that measured aspect 14 deg, `alpha_req`
    9 deg, a 1 m lay committed at ~1000 m -- and (b) a pure IN-PLANE arm (`bfm.LagPursuitTracker`).
    The search must rediscover (a) and beat (b) on a cell where in-plane provably fails. All three
    fly the SAME honest trigger (below), so the trigger can never explain a difference.

WHAT THIS BOX DOES NOT CLAIM. It does **not** unblock a behaviour-cloning channel. `update_pi`
(`agent.py:241-259`) has no BC/AWR/demo term and `buffer.py:42-47` has no demo label, so a
demonstration provably trains only encoder/dynamics/reward/Q. The demo is NECESSARY, not
sufficient; the BC channel is a separate, unbuilt change. `--out` writes the winning trajectory to
an .npz for that future change, and for nothing else today.

DELIBERATELY NOT TOUCHED: `R_EFF`, the scripted fire gates, the reward, tier 3, the learner.

D3a's four defects, checked against this design: THREE ARE MOOT BY CONSTRUCTION here, because the
true sim replaces the analytic model -- there is no `ALPHA_CEIL_DEG` to sweep, no closed-form
`E[hits]` needing a magazine cap (the sim owns the magazine), and no `_coturn` treating n and
alpha as independent. The fourth carries over and is fixed: every seed here is EXPLICIT, never
`hash(cell)`, which is not stable under randomised `PYTHONHASHSEED`.
"""

from __future__ import annotations

import argparse
import math
import queue
import time

import numpy as np
from astrum_rl import _paths

_paths.require_astrum_env()
from astrum_dogfight import bfm  # noqa: E402
from astrum_dogfight import bfm_detect as B  # noqa: E402
from astrum_dogfight import constants as C  # noqa: E402
from astrum_dogfight.geometry import relative  # noqa: E402
from astrum_rl.envs import make_tdmpc_env  # noqa: E402

DEG = 180.0 / math.pi
RATE_HZ = 120.0
ACTION_REPEAT = 5
CONTROL_DT_S = ACTION_REPEAT / RATE_HZ  # 41.667 ms

# --- the honest trigger, uniform across the searched arm and BOTH controls ------------------------
# `rng <= LETHAL_RANGE_M` and a corrected-lay miss inside ~2 hit spheres. Not `in_range` (R_EFF =
# 1200 m), where p_hit/round is 0.000 and a 511-round magazine buys nothing: an existence search
# must not be able to score sprays that cannot connect, and holding fire outside 600 m also keeps
# the magazine for the one pass that can. -1.0 is HOLD in agent space (0.0 maps to native 0.5,
# which is exactly ON the sim's gate and fires every tick).
FIRE_LAY_MISS_M = 15.0
R_LETHAL_M = B.LETHAL_RANGE_M  # 600.0

# --- objective weights (see fix (2); the tiers are deliberate and wide) --------------------------
W_KILL = 1.0e4
W_HP = 1.0e2  # a kill is 100 HP -> 1e4, i.e. the same order as W_KILL by construction
W_TLETH = 1.0e1  # 0.5 s of tLeth = 5.0 = the ENTIRE arrival cap
ARRIVAL_CAP = 5.0
W_T600 = 0.2  # s inside LETHAL_RANGE_M
W_CPA = 3.0  # closest approach, normalised over 3000 m
ALT_PRICE_PER_KM = 0.5  # altitude spent, priced in the tie-break tier only
DEAD_SCORE = -1.0e6  # hard_deck / killed / trade == -inf, ordered below every live candidate

# --- maneuver-space parameterisation bounds (`--param bank`) -------------------------------------
BANK_MAX_DEG = 115.0  # matches LeadTurnArm's own bank clip (past vertical is a real BFM state)


# =================================================================================================
# Cells: the neutral head-on merge, with the lateral separation set DETERMINISTICALLY
# =================================================================================================
def _merge_cell(lateral_m: float, separation_m: float = 3000.0) -> dict:
    """A neutral head-on merge carrying exactly `lateral_m` of cross-range, no rng anywhere.

    `tier3_scenario` places the bandit at compass bearing `aspect_deg` and `separation_m` from the
    learner, so lateral = separation*sin(aspect) is available WITHOUT the jitter rng — which is the
    only way to hold the start fixed while varying `sim.seed`. `opp_heading_deg=180` points him
    back at us (the merge, not a co-heading tail chase); `me_dalt_ft=2000` is the standing slight
    altitude edge; `alt_ft=15000` is the block every checkpoint from c87 on was trained in.

    Lateral separation is the quantity the merge had measurably wrong by omission: at <=500 m the
    T3 flies a nose-to-nose face pass with no turning room in it at all, and it only COMMITS A LEAD
    TURN at >=800 m of offset. 1500 m is the band c94 trains on and is inside the T3's own threat
    gate (the offset must stay under 1659 m at the +2000 ft perch or the T3 never breaks and it is
    a materially easier opponent still labelled T3).
    """
    frac = float(np.clip(lateral_m / separation_m, 0.0, 1.0))
    return dict(
        separation_m=separation_m,
        aspect_deg=math.degrees(math.asin(frac)),
        opp_heading_deg=180.0,
        me_dalt_ft=2000.0,
        alt_ft=15000.0,
        lateral_m=lateral_m,
    )


CELLS = {
    "merge0": _merge_cell(0.0),  # collision course: no one-circle/two-circle decision exists
    "merge800": _merge_cell(800.0),  # the T3's own regime change (it starts lead-turning here)
    "merge1500": _merge_cell(1500.0),  # the c94 trained cell, inside the T3 threat gate
}


# =================================================================================================
# The genome: ONE open-loop parameterised trajectory over the whole horizon (fix (1))
# =================================================================================================
class Genome:
    """`segments x channels` piecewise-constant commands spanning the full horizon.

    Two parameterisations, because the choice is not obvious and should be measured rather than
    assumed:

    `bank` (default) -- each segment names a BANK TARGET, a LOAD and a THROTTLE, and an inner
        P-controller (`bfm._roll_to`, the same gain every primitive in `bfm.py` uses) turns the
        bank target into an aileron command. This matters because `Control::Raw` is an
        Nz/roll-rate/throttle interface, NOT bare surfaces: a constant open-loop aileron is a
        constant ROLL RATE, which integrates through several revolutions over a 4 s segment, so a
        raw genome cannot hold a bank angle and therefore cannot express a turn. The inner loop is
        opponent-BLIND -- it reads only own-ship attitude -- so the trajectory is still open loop
        in the sense that matters: nothing about it reacts to the bandit.

    `raw` -- the literal action vector, piecewise constant. Kept as the null parameterisation so
        the `bank` choice above is a measured claim and not a preference.

    The `roll_gate` (default on) withholds the pull while the roll is still going, by
    `cos(bank_err)` -- the house convention (`lead_pursuit`, both scripted breaks). Holding max aft
    stick through an incomplete roll dumps the whole g demand into PITCH, which is exactly the
    defect that made the scripted T3 zoom-climb and that an ungated lead-turn arm reproduced at
    +4,446 m of climb. It does not forbid vertical maneuvers (roll to 0 deg of bank, then pull, is a
    pure vertical) -- it only forbids pulling sideways-and-up at once. `--no-roll-gate` disables it.
    """

    def __init__(self, segments: int, channels: int, param: str, roll_gate: bool):
        if channels not in (3, 4):
            raise ValueError(f"--channels must be 3 or 4, got {channels}")
        if param not in ("bank", "raw"):
            raise ValueError(f"--param must be bank|raw, got {param!r}")
        self.segments, self.channels = int(segments), int(channels)
        self.param, self.roll_gate = param, bool(roll_gate)
        self.dim = self.segments * self.channels

    def seg_of(self, step: int, n_steps: int) -> int:
        return min(self.segments - 1, (step * self.segments) // max(1, n_steps))

    def act(self, g: np.ndarray, step: int, n_steps: int, me) -> np.ndarray:
        """The flown 5-vector for control `step`. `me` is read for own-ship attitude only."""
        s = self.seg_of(step, n_steps)
        p = g[s * self.channels : (s + 1) * self.channels]
        rud = float(np.clip(p[3], -1.0, 1.0)) if self.channels == 4 else 0.0
        if self.param == "raw":
            return np.array(
                [
                    float(np.clip(p[0], -1.0, 1.0)),
                    float(np.clip(p[1], -1.0, 1.0)),
                    rud,
                    0.5 * (float(np.clip(p[2], -1.0, 1.0)) + 1.0),
                    -1.0,
                ],
                np.float32,
            )
        phi_tgt = math.radians(BANK_MAX_DEG) * float(np.clip(p[0], -1.0, 1.0))
        load = 0.5 * (float(np.clip(p[1], -1.0, 1.0)) + 1.0)  # [0,1]; 0 = unload, 1 = max pull
        thr = 0.5 * (float(np.clip(p[2], -1.0, 1.0)) + 1.0)
        bank_err = phi_tgt - float(me.phi)
        ail = float(np.clip(bfm.KROLL * bank_err, -1.0, 1.0))
        mag = max(math.cos(bank_err), 0.0) if self.roll_gate else 1.0
        elev = -float(np.clip(load * mag, 0.0, 1.0))
        return np.array([ail, elev, rud, thr, -1.0], np.float32)


# =================================================================================================
# Rollout
# =================================================================================================
def _trigger(rel) -> float:
    """The honest ballistic gate, identical for the search and both controls."""
    return 1.0 if (rel.rng <= R_LETHAL_M and rel.lay_miss_m <= FIRE_LAY_MISS_M) else -1.0


def _state_vec(me, bd) -> np.ndarray:
    """Everything a determinism check needs to see, for both jets."""
    out = []
    for a in (me, bd):
        out.extend(
            [
                *np.asarray(a.pos, float),
                *np.asarray(a.vel, float),
                a.phi,
                a.theta,
                a.psi,
                a.p,
                a.q,
                a.r,
                a.nz,
                a.airspeed,
                a.health,
                float(a.ammo),
            ]
        )
    return np.asarray(out, float)


class Rollout:
    """One evaluated trajectory: the outcome row, the detector trace, and the flown actions."""

    __slots__ = (
        "event",
        "kill",
        "dead",
        "hp_dealt",
        "hp_taken",
        "tleth_s",
        "t600_s",
        "rng_min",
        "dalt_m",
        "rounds",
        "steps",
        "trace",
        "actions",
        "obs",
        "states",
    )


def roll_once(env, base, controller, n_steps: int, keep=False) -> Rollout:
    """Fly `n_steps` control steps. `controller(step, me, bd, rel) -> agent-space 5-vector`.

    The trigger is overwritten here for EVERY arm, so no arm can win on a different fire rule.
    """
    obs, _ = env.reset()
    r = Rollout()
    r.trace, r.actions, r.obs, r.states = [], [], [], []
    r.event, r.kill, r.dead = None, 0, False
    tleth = t600 = 0
    rng_min = float("inf")
    me = base._states[base.learner_id]
    alt0 = float(me.alt)
    t = 0.0
    for step in range(n_steps):
        me = base._states[base.learner_id]
        bd = base._states[base.opponent_id]
        rel = relative(me, bd)
        a = np.asarray(controller(step, me, bd, rel), np.float32).copy()
        a[4] = _trigger(rel)
        r.trace.append(B.make_sample(t, me, bd, a, rel))
        if keep:
            r.actions.append(a.copy())
            r.obs.append(np.asarray(obs, np.float32).copy())
            r.states.append(_state_vec(me, bd))
        if rel.rng <= R_LETHAL_M:
            t600 += 1
            if rel.lay_miss_m < C.HIT_RADIUS_M:
                tleth += 1
        rng_min = min(rng_min, float(rel.rng))
        obs, _rew, term, trunc, info = env.step(a)
        t += CONTROL_DT_S
        if info.get("event"):
            r.event = info["event"]
        if term or trunc:
            break
    me = base._states[base.learner_id]
    bd = base._states[base.opponent_id]
    r.steps = len(r.trace)
    r.kill = 1 if r.event == "kill" else 0
    r.dead = r.event in ("hard_deck", "killed", "trade")
    r.hp_dealt = C.HP_MAX - float(bd.health)
    r.hp_taken = C.HP_MAX - float(me.health)
    r.tleth_s = tleth * CONTROL_DT_S
    r.t600_s = t600 * CONTROL_DT_S
    r.rng_min = rng_min if math.isfinite(rng_min) else float("nan")
    r.dalt_m = float(me.alt) - alt0
    r.rounds = int(round(C.AMMO_MAX - float(me.ammo)))
    return r


def score(r: Rollout) -> tuple[float, dict]:
    """The objective of fix (2). Returns (scalar, components) -- report the components, always."""
    if r.dead:
        return DEAD_SCORE, dict(
            kill=0, hp=r.hp_dealt, tleth=r.tleth_s, arrival=0.0, alt=0.0, dead=1
        )
    arrival = W_T600 * r.t600_s + W_CPA * (1.0 - min(r.rng_min, 3000.0) / 3000.0)
    arrival = min(arrival, ARRIVAL_CAP)
    alt = ALT_PRICE_PER_KM * max(0.0, -r.dalt_m) / 1000.0
    s = W_KILL * r.kill + W_HP * r.hp_dealt + W_TLETH * r.tleth_s + arrival - alt
    return s, dict(kill=r.kill, hp=r.hp_dealt, tleth=r.tleth_s, arrival=arrival, alt=alt, dead=0)


# =================================================================================================
# CMA-ES (numpy only -- `cma`, `scipy` and `nevergrad` are all absent from the container)
# =================================================================================================
class CMAES:
    """Standard (mu/mu_w, lambda)-CMA-ES with rank-mu and rank-one updates, MAXIMISING.

    The canonical reference implementation, transcribed: weighted recombination, cumulative
    step-size adaptation with the Hansen-Ostermeier constants, and a lazily-refreshed eigen-
    decomposition. It is here rather than as a dependency because the container has no optimiser
    package; the constants are not tuned and must not be.
    """

    def __init__(self, dim: int, sigma0: float, popsize: int, seed: int):
        n = self.n = int(dim)
        self.sigma = float(sigma0)
        self.xmean = np.zeros(n)
        self.lam = int(popsize)
        self.mu = self.lam // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / np.sum(self.w**2)
        self.cc = (4.0 + self.mueff / n) / (n + 4.0 + 2.0 * self.mueff / n)
        self.cs = (self.mueff + 2.0) / (n + self.mueff + 5.0)
        self.c1 = 2.0 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(
            1.0 - self.c1,
            2.0 * (self.mueff - 2.0 + 1.0 / self.mueff) / ((n + 2.0) ** 2 + self.mueff),
        )
        self.damps = 1.0 + 2.0 * max(0.0, math.sqrt((self.mueff - 1.0) / (n + 1.0)) - 1.0) + self.cs
        self.pc, self.ps = np.zeros(n), np.zeros(n)
        self.B, self.D = np.eye(n), np.ones(n)
        self.Cov, self.invsqrtC = np.eye(n), np.eye(n)
        self.counteval, self.eigeneval = 0, 0
        self.chiN = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))
        self.rng = np.random.default_rng(seed)
        self.pop = None

    def ask(self) -> np.ndarray:
        z = self.rng.standard_normal((self.lam, self.n))
        self.pop = self.xmean + self.sigma * (z @ (self.B * self.D).T)
        return self.pop

    def tell(self, fitness) -> None:
        idx = np.argsort(-np.asarray(fitness, float))  # descending: we MAXIMISE
        self.counteval += self.lam
        xold = self.xmean.copy()
        xsel = self.pop[idx[: self.mu]]
        self.xmean = self.w @ xsel
        y = (self.xmean - xold) / self.sigma
        self.ps = (1.0 - self.cs) * self.ps + math.sqrt(self.cs * (2.0 - self.cs) * self.mueff) * (
            self.invsqrtC @ y
        )
        gen = self.counteval / self.lam
        hsig = np.linalg.norm(self.ps) / math.sqrt(
            1.0 - (1.0 - self.cs) ** (2.0 * gen)
        ) / self.chiN < 1.4 + 2.0 / (self.n + 1.0)
        self.pc = (1.0 - self.cc) * self.pc + (
            math.sqrt(self.cc * (2.0 - self.cc) * self.mueff) * y if hsig else 0.0
        )
        artmp = (xsel - xold) / self.sigma
        self.Cov = (
            (1.0 - self.c1 - self.cmu) * self.Cov
            + self.c1
            * (np.outer(self.pc, self.pc) + (0.0 if hsig else self.cc * (2.0 - self.cc)) * self.Cov)
            + self.cmu * ((artmp.T * self.w) @ artmp)
        )
        self.sigma *= math.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1.0))
        if self.counteval - self.eigeneval > self.lam / (self.c1 + self.cmu) / self.n / 10.0:
            self.eigeneval = self.counteval
            self.Cov = np.triu(self.Cov) + np.triu(self.Cov, 1).T
            d, self.B = np.linalg.eigh(self.Cov)
            self.D = np.sqrt(np.maximum(d, 1e-20))
            self.invsqrtC = (self.B * (1.0 / self.D)) @ self.B.T


# =================================================================================================
# Control arms (fix (4))
# =================================================================================================
def make_control(which: str):
    """`recorded` = the hand-coded one-circle lead turn, imported VERBATIM; `inplane` = pursuit.

    Importing `_diag_merge_convert_tmp.LeadTurnArm` rather than copying it is deliberate: the
    control has to be the arm that actually produced the recorded aspect-14 deg / 1 m-lay
    conversion, and a copy can drift from it. Its own trigger is discarded -- `roll_once`
    overwrites `a[4]` for every arm.
    """
    if which in ("recorded", "recorded_two"):
        import _diag_merge_convert_tmp as MC

        arm = MC.LeadTurnArm(1000.0, which == "recorded")  # +: one-circle (into his pass side)

        def fn(step, me, bd, rel):
            return arm.act(me, bd, rel)

        return fn
    if which == "inplane":
        trk = bfm.LagPursuitTracker()
        live = {"on": False}

        def fn(step, me, bd, rel):
            if not live["on"]:
                trk.reset(me, rel)
                live["on"] = True
            a4, _ = trk.step(me, rel)
            a4 = np.asarray(a4, np.float32)
            return np.array([a4[0], a4[1], a4[2], a4[3], -1.0], np.float32)

        return fn
    raise ValueError(which)


# =================================================================================================
# Env plumbing
# =================================================================================================
def build_env(cell: dict, sim_seed: int, horizon_s: float):
    """One env per (cell, sim seed). jitter=0 => `scenario_fn is None` => `reset()` re-runs the
    FIXED scenario without rebuilding, so the same env can be reset per candidate and the start
    geometry is bit-identical across every candidate at that seed."""
    kw = {k: v for k, v in cell.items() if k != "lateral_m"}
    env = make_tdmpc_env(
        3,
        action_repeat=ACTION_REPEAT,
        frame_stack=1,
        record_stats=False,
        jitter=0.0,
        rate_hz=RATE_HZ,
        max_time_s=float(horizon_s) + 1.0,
        seed=int(sim_seed),
        **kw,
    )
    return env, env.unwrapped


def precheck(cell: dict, gm: Genome, n_steps: int, seeds: list[int], rng) -> dict:
    """Two things the search's correctness rests on, MEASURED, not assumed.

    (a) Replay determinism: the same genome on the same env, twice, must agree to 0.
        (`crates/astrum-sim/tests/env.rs:179-208` pins this on the Rust side; this is the check
        through the Python adapter, the reset path and the scripted opponent.)
    (b) Seed independence when no round is fired: `sim.seed` seeds gun dispersion, so a candidate
        that never pulls the trigger must be identical across seeds. The whole 8x cost saving
        rests on it, so if it fails the harness disables the skip rather than trusting it.
    """
    g = rng.uniform(-0.6, 0.6, gm.dim)
    env, base = build_env(cell, seeds[0], n_steps * CONTROL_DT_S)

    def ctl(step, me, bd, rel):
        return gm.act(g, step, n_steps, me)

    t0 = time.time()
    a = roll_once(env, base, ctl, n_steps, keep=True)
    wall = time.time() - t0
    b = roll_once(env, base, ctl, n_steps, keep=True)
    n = min(len(a.states), len(b.states))
    div_replay = (
        float(np.max(np.abs(np.asarray(a.states[:n]) - np.asarray(b.states[:n]))))
        if n
        else float("nan")
    )
    env.close()
    div_seed, seed_skip_ok = float("nan"), False
    if len(seeds) > 1:
        env2, base2 = build_env(cell, seeds[1], n_steps * CONTROL_DT_S)
        c = roll_once(env2, base2, ctl, n_steps, keep=True)
        env2.close()
        n = min(len(a.states), len(c.states))
        div_seed = (
            float(np.max(np.abs(np.asarray(a.states[:n]) - np.asarray(c.states[:n]))))
            if n
            else float("nan")
        )
        seed_skip_ok = a.rounds == 0 and c.rounds == 0 and div_seed == 0.0
    return dict(
        div_replay=div_replay,
        div_seed=div_seed,
        rounds=a.rounds,
        steps=a.steps,
        seed_skip_ok=bool(seed_skip_ok),
        ticks_per_s=(a.steps * ACTION_REPEAT / wall) if wall > 0 else float("nan"),
        wall_s=wall,
    )


# =================================================================================================
# The search
# =================================================================================================
def _eval_one(env, base, g, gm, n_steps):
    def ctl(step, me, bd, rel):
        return gm.act(g, step, n_steps, me)

    r = roll_once(env, base, ctl, n_steps)
    s, comp = score(r)
    return s, comp, r.rounds


def _shard_worker(cell, gm, n_steps, in_q, out_q):
    """One JSBSim per PROCESS (it is process-global and not thread-safe), rebuilt only when the
    `sim.seed` of the job changes -- which is why the parent dispatches one seed per barrier round."""
    env = base = None
    cur = None
    try:
        while True:
            job = in_q.get()
            if job is None:
                break
            sd, idx, g = job
            if cur != sd:
                if env is not None:
                    env.close()
                env, base = build_env(cell, sd, n_steps * CONTROL_DT_S)
                cur = sd
            s, comp, rounds = _eval_one(env, base, np.asarray(g, float), gm, n_steps)
            out_q.put((idx, s, comp, rounds))
    finally:
        if env is not None:
            env.close()


class Evaluator:
    """Scores a population as the MEAN over `seeds` (fix (3)), serially or sharded over processes.

    Seeds are the OUTER loop in both paths: with `jitter=0` an env is bound to one `sim.seed`, so a
    seed-per-round dispatch means each worker rebuilds at most once per round. Rounds 2..N carry
    only the candidates that actually FIRED (see `precheck`: a candidate that fires no round is
    provably seed-independent), so early generations cost one rollout per candidate, not `seeds`.
    """

    def __init__(self, cell, gm, n_steps, seeds, seed_skip: bool, workers: int):
        self.cell, self.gm, self.n_steps = cell, gm, n_steps
        self.seeds, self.seed_skip = seeds, seed_skip
        self.workers = max(1, int(workers))
        self.procs, self.in_q, self.out_q = [], None, None
        if self.workers > 1:
            import multiprocessing as mp

            ctx = mp.get_context("spawn")  # never fork: it would copy an initialised JSBSim
            self.in_q, self.out_q = ctx.Queue(), ctx.Queue()
            for _ in range(self.workers):
                p = ctx.Process(
                    target=_shard_worker,
                    args=(cell, gm, n_steps, self.in_q, self.out_q),
                    daemon=True,
                )
                p.start()
                self.procs.append(p)

    def close(self):
        for _ in self.procs:
            self.in_q.put(None)
        for p in self.procs:
            p.join(timeout=20)
            if p.is_alive():
                p.terminate()

    def _collect(self, n):
        """Gather `n` results, ABORTING on a dead worker instead of blocking forever.

        A worker killed by a signal leaves no traceback and never answers, so a bare
        `out_q.get()` waits for a result that can never arrive — which is exactly how the
        first sharded run sat at 1.9% CPU for 5 minutes still *looking* alive. Poll instead,
        and on every silent interval ask the OS whether the workers still exist.
        """
        out = []
        while len(out) < n:
            try:
                out.append(self.out_q.get(timeout=30.0))
            except queue.Empty:
                dead = [(i, p.exitcode) for i, p in enumerate(self.procs) if not p.is_alive()]
                if dead:
                    raise RuntimeError(
                        f"{len(dead)} of {self.workers} shard workers died without returning "
                        f"a result — (worker, exitcode): {dead}. A NEGATIVE exitcode is a "
                        f"signal: -7 = SIGBUS, -11 = SIGSEGV. SIGBUS here means something "
                        f"rewrote controllers/rl/vendor/astrum_env.so in place while these "
                        f"processes had it mmap'd — `make rl-env`'s `cp` did exactly that "
                        f"until it was changed to an atomic `mv` on 2026-09-06. Do not run a "
                        f"`make rl-*` target while this search or a trainer is live."
                    ) from None
        return out

    def _round(self, sd, jobs, pop):
        """One barrier round at a single `sim.seed`. `jobs` = candidate indices."""
        if self.workers == 1:
            env, base = build_env(self.cell, sd, self.n_steps * CONTROL_DT_S)
            out = []
            for i in jobs:
                s, comp, rounds = _eval_one(env, base, pop[i], self.gm, self.n_steps)
                out.append((i, s, comp, rounds))
            env.close()
            return out
        for i in jobs:
            self.in_q.put((sd, i, pop[i].tolist()))
        return self._collect(len(jobs))

    def __call__(self, pop):
        m = len(pop)
        tot, cnt = np.zeros(m), np.zeros(m, int)
        fired = np.zeros(m, bool)
        comps0: list = [None] * m  # round-0 components, kept per candidate
        seen = dict(
            kill=0, tleth=0.0
        )  # the best ANY single rollout did -- labelled, not reported as fit
        for k, sd in enumerate(self.seeds):
            jobs = (
                list(range(m))
                if k == 0
                else [i for i in range(m) if fired[i] or not self.seed_skip]
            )
            if not jobs:
                break
            for i, s, comp, rounds in self._round(sd, jobs, pop):
                tot[i] += s
                cnt[i] += 1
                if k == 0:
                    comps0[i] = comp
                if rounds > 0:
                    fired[i] = True
                seen["kill"] = max(seen["kill"], int(comp.get("kill", 0)))
                seen["tleth"] = max(seen["tleth"], float(comp.get("tleth", 0.0)))
        fit = np.where(cnt > 0, tot / np.maximum(cnt, 1), DEAD_SCORE)
        # Report the components of the best MEAN candidate, never of the best single rollout: a max
        # over a stochastic objective is the c13 lucky-noise gate, and it is just as wrong in a
        # progress column as it is in a selection rule.
        i = int(np.argmax(fit))
        return fit, fired, dict(i=i, s=float(fit[i]), comp=comps0[i] or {}, seen=seen)


def run_cell(name: str, cell: dict, args, log) -> dict:
    gm = Genome(args.segments, args.channels, args.param, not args.no_roll_gate)
    n_steps = int(round(args.horizon_s / CONTROL_DT_S))
    popsize = args.popsize if args.popsize > 0 else 10 * gm.dim
    seeds = [args.sim_seed0 + 17 * i for i in range(args.seeds)]
    rng = np.random.default_rng(args.seed)

    log("")
    log("=" * 92)
    log(
        f"   CELL {name}   lateral {cell['lateral_m']:.0f} m   sep {cell['separation_m']:.0f} m   "
        f"aspect {cell['aspect_deg']:.2f} deg   perch {cell['me_dalt_ft']:.0f} ft"
    )
    log(
        f"   horizon {args.horizon_s:.1f} s = {n_steps} control steps   param={gm.param}  "
        f"segments={gm.segments}  channels={gm.channels}  dim={gm.dim}  roll_gate={gm.roll_gate}"
    )
    log(
        f"   popsize {popsize} (>= 10*dim = {10 * gm.dim})   gens {args.gens}   "
        f"seeds {seeds}   sigma0 {args.sigma0}"
    )
    log("=" * 92)

    pc = precheck(cell, gm, n_steps, seeds, rng)
    log("")
    log("   PRECHECK (mandated: report the determinism max|div|)")
    log(
        f"     replay determinism, same genome twice   max|div| = {pc['div_replay']:.3e}   "
        f"({pc['steps']} control steps)"
    )
    log(
        f"     cross-seed, {pc['rounds']} rounds fired            max|div| = {pc['div_seed']:.3e}"
        f"   -> seed-skip {'VALID' if pc['seed_skip_ok'] else 'NOT ESTABLISHED'}"
    )
    log(
        f"     throughput {pc['ticks_per_s']:,.0f} sim ticks/s "
        f"({pc['wall_s'] * 1e3 / max(1, pc['steps']):.2f} ms per control step)"
    )
    if pc["div_replay"] != 0.0:
        log("     !! replay is NOT deterministic -- prefix replay is unsound; abort this cell")
        return dict(name=name, aborted="nondeterministic")
    seed_skip = pc["seed_skip_ok"] and not args.no_seed_skip

    es = CMAES(gm.dim, args.sigma0, popsize, args.seed)
    ev = Evaluator(cell, gm, n_steps, seeds, seed_skip, args.workers)
    hall = dict(s=-np.inf, g=None, comp=None)
    log("")
    log(f"   SEARCH   ({args.workers} worker process(es); fit = MEAN over seeds -- the verdict")
    log("             columns are kill/hp/tLeth of the best-MEAN candidate, never of a lucky seed)")
    log(
        "     gen      fit(best)     fit(mean)   sigma |  kill    hp  tLeth   arr   alt | fired    s"
    )
    t_start = time.time()
    rolls = 0
    try:
        for gen in range(args.gens):
            pop = np.clip(es.ask(), -1.0, 1.0)
            t0 = time.time()
            fit, fired, best = ev(pop)
            rolls += len(pop) + int(fired.sum()) * (len(seeds) - 1)
            es.tell(fit)
            if best["s"] > hall["s"]:
                hall.update(s=best["s"], g=pop[best["i"]].copy(), comp=best["comp"])
            c = best["comp"] or {}
            log(
                f"     {gen:3d}  {np.max(fit):+12.3f}  {np.mean(fit):+12.3f}  {es.sigma:6.3f} | "
                f"{c.get('kill', 0):5d} {c.get('hp', 0.0):5.1f} {c.get('tleth', 0.0):6.3f} "
                f"{c.get('arrival', 0.0):5.2f} {c.get('alt', 0.0):5.2f} | "
                f"{int(fired.sum()):5d} {time.time() - t0:4.0f}"
            )
    finally:
        ev.close()
    log(f"   search wall {time.time() - t_start:.0f} s, ~{rolls:,} rollouts")

    # --- the winner, re-flown over EVERY seed with the trace kept ---------------------------
    log("")
    log("   BEST GENOME, re-flown over every seed (this is the row the verdict is read from)")
    rows = _replay_all_seeds("SEARCH", hall["g"], gm, cell, n_steps, seeds, log)
    out = dict(name=name, arms={"SEARCH": rows}, genome=hall["g"], n_steps=n_steps, gm=gm)

    if not args.no_controls:
        log("")
        log("   CONTROLS (fix (4)): `recorded` = the hand-coded one-circle lead turn, verbatim;")
        log(
            "             `recorded_two` = the same arm turning AWAY (two-circle); `inplane` = pursuit"
        )
        for which in ("recorded", "recorded_two", "inplane"):
            out["arms"][which] = _replay_all_seeds(
                which, None, gm, cell, n_steps, seeds, log, control=which
            )
    return out


def _replay_all_seeds(label, g, gm, cell, n_steps, seeds, log, control=None):
    """One row per seed plus a pooled row. Keeps traces so the detector can score the trajectory."""
    rows, traces, keep = [], [], []
    log(
        f"     {'arm':<13} {'seed':>5} {'ev':>10} {'kill':>4} {'hp':>5} {'tkn':>4} "
        f"{'tLeth':>6} {'t600':>6} {'rngMn':>6} {'dalt':>7} {'rnds':>5} {'st':>4}"
    )
    for sd in seeds:
        env, base = build_env(cell, sd, n_steps * CONTROL_DT_S)
        ctl = (
            make_control(control)
            if control
            else (lambda step, me, bd, rel: gm.act(g, step, n_steps, me))
        )
        r = roll_once(env, base, ctl, n_steps, keep=True)
        env.close()
        rows.append(r)
        traces.append(r.trace)
        keep.append(r)
        log(
            f"     {label:<13} {sd:>5} {str(r.event):>10} {r.kill:>4} {r.hp_dealt:>5.1f} "
            f"{r.hp_taken:>4.0f} {r.tleth_s:>6.3f} {r.t600_s:>6.2f} {r.rng_min:>6.0f} "
            f"{r.dalt_m:>+7.0f} {r.rounds:>5d} {r.steps:>4d}"
        )
    n = len(rows)
    log(
        f"     {label + ' POOL':<13} {n:>5} {'-':>10} {sum(r.kill for r in rows):>4} "
        f"{np.mean([r.hp_dealt for r in rows]):>5.1f} "
        f"{np.mean([r.hp_taken for r in rows]):>4.0f} "
        f"{np.mean([r.tleth_s for r in rows]):>6.3f} "
        f"{np.mean([r.t600_s for r in rows]):>6.2f} "
        f"{np.min([r.rng_min for r in rows]):>6.0f} "
        f"{np.mean([r.dalt_m for r in rows]):>+7.0f} "
        f"{np.mean([r.rounds for r in rows]):>5.0f} "
        f"{np.mean([r.steps for r in rows]):>4.0f}"
    )
    return dict(rows=keep, traces=traces)


# =================================================================================================
# Detector verdict
# =================================================================================================
def detector_block(name: str, arms: dict, log) -> None:
    log("")
    log("-" * 92)
    log(f"   DETECTOR -- {name}   (holds priced in ALTITUDE; never HP dealt alone)")
    log("-" * 92)
    log(
        f"     {'arm':<14} {'eps':>4} {'D1':>4} {'D2':>4} {'D2c':>4} {'D3':>4} "
        f"{'lethTk':>7} {'chiMx':>6} {'nzMx':>5} {'dalt':>7} {'kills':>6}"
    )
    for label, blk in arms.items():
        traces = [t for t in blk["traces"] if t]
        if not traces:
            continue
        per = [B.detect_all(t) for t in traces]
        chimx = max(
            (max((abs(s.chi) for s in t if s.chi_valid), default=0.0) for t in traces), default=0.0
        )
        nzmx = max((max((s.nz for s in t), default=0.0) for t in traces), default=0.0)
        log(
            f"     {label:<14} {len(traces):>4} "
            f"{sum(len(p['displacement_rolls']) for p in per):>4} "
            f"{sum(len(p['yoyos']) for p in per):>4} "
            f"{sum(len(p['yoyos_clean']) for p in per):>4} "
            f"{sum(len(p['turning_room']) for p in per):>4} "
            f"{sum(p['lethal_ticks'] for p in per):>7} "
            f"{chimx * DEG:>6.1f} {nzmx:>5.1f} "
            f"{np.mean([r.dalt_m for r in blk['rows']]):>+7.0f} "
            f"{sum(r.kill for r in blk['rows']):>6d}"
        )


def verdict(name: str, arms: dict, log) -> None:
    s = arms.get("SEARCH")
    if not s:
        return
    rows = s["rows"]
    kills = sum(r.kill for r in rows)
    tleth = float(np.mean([r.tleth_s for r in rows]))
    per = [B.detect_all(t) for t in s["traces"] if t]
    loaded = sum(len(p["displacement_rolls"]) + len(p["yoyos_clean"]) for p in per)
    log("")
    log("=" * 92)
    log(f"   PRE-REGISTERED VERDICT -- {name}")
    log("=" * 92)
    log(f"     kills over {len(rows)} seeds ...... {kills}")
    log(f"     mean tLeth ................. {tleth:.3f} s")
    log(f"     loaded D1 + clean D2 events  {loaded}")
    if kills >= 1 and tleth > 0.0 and loaded >= 1:
        log("     >> EXISTENCE ESTABLISHED: a feasible, physics-respecting out-of-plane")
        log("        demonstration of the merge kill EXISTS. The remaining search space is")
        log("        learner-side. (This does NOT unblock BC -- `update_pi` has no demo term.)")
    else:
        log("     >> NO SOLUTION FOUND at this horizon under this objective. This is the ONLY")
        log("        admissible statement: it bounds achievability from BELOW and motivates no")
        log("        task change, no reward change, and no benchmark change.")
        miss = []
        if kills < 1:
            miss.append("no kill")
        if tleth <= 0.0:
            miss.append("zero tLeth")
        if loaded < 1:
            miss.append("no loaded D1/clean D2")
        log(f"        unmet: {', '.join(miss)}")


# =================================================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cells", default="merge1500", help=f"comma list of {sorted(CELLS)}")
    ap.add_argument("--horizon-s", type=float, default=24.0, help="fix (1): never below 12 s")
    ap.add_argument("--segments", type=int, default=6)
    ap.add_argument("--channels", type=int, default=3, help="3 = ail/elev/thr; 4 adds RUDDER")
    ap.add_argument("--param", default="bank", choices=("bank", "raw"))
    ap.add_argument("--no-roll-gate", action="store_true")
    ap.add_argument("--popsize", type=int, default=0, help="0 => 10*dim (fix (3))")
    ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--sigma0", type=float, default=0.4)
    ap.add_argument("--seeds", type=int, default=8, help="fix (3): sim.seed values to average over")
    ap.add_argument("--workers", type=int, default=1, help="shard the population; 6 is the budget")
    ap.add_argument("--sim-seed0", type=int, default=11)
    ap.add_argument("--seed", type=int, default=20260906, help="CMA-ES rng; EXPLICIT, never hash()")
    ap.add_argument("--no-seed-skip", action="store_true")
    ap.add_argument("--no-controls", action="store_true")
    ap.add_argument("--out", default="", help="npz path for the winning trajectory (optional)")
    ap.add_argument(
        "--log",
        default="",
        help="tee the REPORT to this path. Strongly recommended: every DogfightEnv build dumps the "
        "full JSBSim aircraft config to stdout, so the report is otherwise unreadable.",
    )
    args = ap.parse_args()

    if args.horizon_s < 12.0:
        raise SystemExit(
            f"--horizon-s {args.horizon_s} is below the 12 s floor of fix (1): a horizon under the "
            "120-360-step maneuver band cannot represent the maneuver under test."
        )

    fh = open(args.log, "w") if args.log else None

    def log(s=""):
        print(s, flush=True)
        if fh:
            fh.write(s + "\n")
            fh.flush()

    log("")
    log("#" * 92)
    log("#  D3b  TRUE-SIM TRAJECTORY SEARCH -- is the neutral-merge gun kill REACHABLE?")
    log("#  EXISTENCE / LOWER-BOUND search. A negative result reads 'no solution found at")
    log("#  horizon H under objective J' and NEVER 'unwinnable'.")
    log("#" * 92)

    results = {}
    for name in [c.strip() for c in args.cells.split(",") if c.strip()]:
        if name not in CELLS:
            raise SystemExit(f"unknown cell {name!r}; have {sorted(CELLS)}")
        res = run_cell(name, CELLS[name], args, log)
        if res.get("aborted"):
            continue
        detector_block(name, res["arms"], log)
        verdict(name, res["arms"], log)
        results[name] = res

    if args.out and results:
        name, res = next(iter(results.items()))
        rows = res["arms"]["SEARCH"]["rows"]
        best = max(rows, key=lambda r: (r.kill, r.tleth_s))
        np.savez(
            args.out,
            cell=name,
            genome=res["genome"],
            actions=np.asarray(best.actions, np.float32),
            obs=np.asarray(best.obs, np.float32),
            kill=best.kill,
            tleth_s=best.tleth_s,
            hp_dealt=best.hp_dealt,
        )
        log("")
        log(f"   wrote {args.out}  ({len(best.actions)} control steps).  NOTE: nothing consumes")
        log("   this yet -- `update_pi` has no BC/AWR/demo term and the buffer has no demo label.")
    log("")
    if fh:
        fh.close()


if __name__ == "__main__":
    main()
