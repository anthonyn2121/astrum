"""E3 — does the world model's REWARD head know that killing pays and decking costs?

    cd controllers/rl && uv run python -u _diag_e3_reward.py \
        --checkpoints runs/curriculum-96/run.pt,runs/curriculum-95/run.pt,runs/curriculum-87/c87_arrival_obs_5000k.pt \
        --aspects 0,30,55 --episodes 20 --jitter 0.5 --log runs/e3/gate.txt

WHY THIS EXISTS

E2b established that the *termination* head is healthy at its trained depth (AUC 0.94-0.99; 62.7x higher
and roughly calibrated at depth 2; latches on 13/16 terminals at depth 3). The same question has never
been asked of the **reward** head, and it is the bigger one. E2a measured the episode-final transition at
**0.013% of the reward/value loss, one terminal per 4.3 updates** — so the head may simply never have
learned that `R_KILL = +150` or `R_DECK = -200`.

If it has not, the learned reward function is **outcome-blind**. ⚠ **SCOPE, checked before running rather
than after:** this does NOT mean "every path is optimising a blind reward". `_td_target` (`agent.py:299-304`)
is called on the tuple straight out of `buffer.sample()` (`agent.py:308,:313`), so **the critic's TD target
uses the buffer's GROUND-TRUTH reward** — the value path sees the true +150/-200 and is not blind. What a
blind reward head actually costs is exactly two things:
  1. **MPPI `_estimate_value` is unusable** — moot today at `cfg.mpc = False`, decisive if MPC is ever
     switched on (see the branch below);
  2. **one of the three gradients shaping the shared latent carries no outcome information.** The encoder is
     shaped by reward-prediction, Q, and consistency; if reward-prediction is blind to the outcome, the
     latent is organised around the outcome only via Q and consistency. Since `act()` is
     `pi(encode(obs))`, the policy reads that same encoder — so this is a **representation** claim, which is
     the one family that has ever moved this program ([[curriculum-87-arrival-obs]],
     [[curriculum-96-honest-wez]]). State it as that and not as more.

It also decides a live branch. `_estimate_value` (`agent.py:274-296`) is **correct for an episodic MDP**:
`G += discount*(1-term)*reward` adds the reward on the transition INTO the terminal while `term` is still
0, and value-after-terminal genuinely is 0. So the recorded "dying is worth +100..170" planner defect is
NOT unconditional — it exists only if this head mis-prices the terminal:
  - predicts ~= R_DECK  ->  a terminating plan scores ~= -200*g^t against a non-terminating G + g^H*Q,
                            Q < 0  =>  death correctly avoided, NO defect, MPC "expect a null" stands;
  - predicts ~= 0       ->  the plan drops the -200 AND the negative bootstrap  =>  death is worth
                            ~= |g^H * Q|  =>  the elite set is captured by death-seeking plans.
M4 measures both sides of that comparison from the same states instead of quoting c85's Q from memory.

THE QUERY POINT, WHICH IS THE LESSON E2b PAID FOR

Training is `reward_logits = reward(_zs, action)` with `_zs = zs[:-1]` (`agent.py:328-331`), so
`reward(zs[t], action[t])` predicts `reward[t]`. Unlike `termination(zs[1:])`, this slice DOES include
`zs[0]`, so depth 0 is a legitimate query point for this head **in general**.
BUT: E2a showed the episode-final transition is reachable from exactly one start and always lands at
**t = 2**. So the *terminal* reward is only ever supervised at `zs[2]` — two `next()` steps deep — while
ordinary mid-episode rewards are supervised at every depth 0..H-1. The terminal therefore has the same
query-point exposure that inverted E2b, and M2 runs the paired depth ramp for exactly that reason.

PRE-REGISTERED. Residuals are measured against the reward the env ACTUALLY returned on that step, not
against the constants +150/-200: `ActionRepeat` SUMS reward across repeats and dense terms ride along, so
the env's own number is the only ground truth the head was ever trained on. The constants are printed
separately so a reader can see the terminal step really is dominated by the outcome bonus.

  K1  |predicted - actual| <= 25% of |actual| on KILL terminals, at the trained depth   => head prices kills
  K2  same on DECK/TRADE terminals                                                      => head prices death
  K3  the DENSE control: median |residual| on ordinary mid-fight ticks is small relative to the terminal's.
      This is what isolates the terminal as the failure. Without it, a large terminal residual could just
      mean the head is bad everywhere, which is a different (and much less interesting) claim.
  K4  MPC branch: is |predicted terminal reward| >= |g^H * Q| at the same state? If yes, a terminating plan
      is correctly scored worse and the planner defect does not exist.

D6 IS MANDATORY: historical checkpoints in the SAME invocation. A reward-head number that also appears in
c87 is a property of the architecture and the terminal's base rate, not of anything this program changed.

NOT A LEVER. A failure here does not authorise changing a reward CONSTANT (gamma and the reward constants
are a coupled pair — see curriculum-88/89). It authorises fixing the SUPERVISION, which is E2b form (i):
oversample terminal-hosting starts, sampler-only, already costed.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from astrum_rl import _paths

_paths.require_astrum_env()

from astrum_rl.envs import make_tdmpc_env  # noqa: E402
from astrum_rl.tdmpc2 import functional as fn  # noqa: E402
from astrum_rl.tdmpc2.agent import TDMPC2  # noqa: E402
from astrum_rl.tdmpc2.config import TDMPC2Config  # noqa: E402

# The sim's terminal bonuses, printed for orientation only. Residuals are ALWAYS taken against the
# reward the env actually returned, which includes dense terms and the action_repeat sum.
R_KILL, R_DECK, R_BUBBLE_EXIT = 150.0, -200.0, -50.0

# The canonical event strings, read off `reward.py:484-547` rather than guessed: `hard_deck`, `trade`,
# `kill`, `left_bubble`, plus `timeout` synthesised in `env.py:243`.
# ⚠ The pilot ran with `KILL_EVENTS = ("killed",)` and silently classified every kill as "other" — the
# per-episode table was right and the aggregate said `kills 0`. Read the producer, do not guess the string.
KILL_EVENTS = ("kill",)
DEATH_EVENTS = ("hard_deck", "trade")
# `left_bubble` is a THIRD outcome class with its own bonus (-50) and, per `env.py:240`, it is not a
# timeout. Three classes at +150 / -200 / -50 test outcome-blindness far better than two, because a head
# that merely regressed to the dense mean would miss all three by different, predictable amounts.
EGRESS_EVENTS = ("left_bubble",)


def _pred_reward(agent, z, a) -> float:
    """`two_hot_inv(reward(z, a))` — the model's reward in raw units."""
    with torch.no_grad():
        return float(fn.two_hot_inv(agent.model.reward(z, a), agent.cfg).item())


def _reward_ramp(agent, t_obs, t_act, j, max_d: int = 3) -> dict:
    """Predicted reward for the transition (obs_j, act_j), from latents made at depth 0..max_d.

    Paired by construction: every depth predicts the SAME transition, so the only thing that varies is
    how the latent was produced. Rolls with the actions actually taken, so policy drift is excluded.
    """
    row = {}
    a = torch.as_tensor(t_act[j], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        z0 = agent.model.encode(torch.as_tensor(t_obs[j], dtype=torch.float32).unsqueeze(0))
        row[0] = _pred_reward(agent, z0, a)
        for d in range(1, min(j, max_d) + 1):
            zd = agent.model.encode(torch.as_tensor(t_obs[j - d], dtype=torch.float32).unsqueeze(0))
            for k in range(j - d, j):
                zd = agent.model.next(
                    zd, torch.as_tensor(t_act[k], dtype=torch.float32).unsqueeze(0)
                )
            row[d] = _pred_reward(agent, zd, a)
    return row


def _tail_value(agent, z, horizon: int) -> float:
    """`g^H * Q(z_H, pi(z_H))` — the bootstrap a NON-terminating plan would collect from here.

    This is the quantity a predicted terminal ZEROES, so it is exactly the size of the bonus that a
    mis-priced terminal would hand to a death-seeking plan. Measured here rather than quoted from c85.
    """
    with torch.no_grad():
        zz, disc = z, 1.0
        for _ in range(horizon):
            zz = agent.model.next(zz, agent.model.pi(zz)[1])
            disc *= agent.discount
        q = float(agent.model.Q(zz, agent.model.pi(zz)[1], return_type="avg").item())
    return disc * q


def _roll(agent, env, episodes: int, seed0: int, horizon: int):
    """Greedy episodes. Records the ACTUAL reward per step alongside the model's prediction."""
    dense_res, dense_act, terms = [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        agent._prev_mean.zero_()
        t0, event = True, None
        t_obs: list = [obs.copy()]
        t_act: list = []
        t_rew: list = []
        n = 0
        while True:
            a = agent.act(obs, t0=t0, eval_mode=True).numpy()
            t0 = False
            prev = obs
            obs, r, terminated, truncated, info = env.step(a)
            t_act.append(a.copy())
            t_rew.append(float(r))
            t_obs.append(obs.copy())
            n += 1
            if info.get("event"):
                event = info["event"]
            # K3's dense control, at depth 0 -- a LEGITIMATE query point for this head (`_zs` includes
            # zs[0]). Sub-sampled: every 7th tick is plenty and keeps the pass cheap.
            if not (terminated or truncated) and n % 7 == 0:
                with torch.no_grad():
                    z = agent.model.encode(torch.as_tensor(prev, dtype=torch.float32).unsqueeze(0))
                    pr = _pred_reward(
                        agent, z, torch.as_tensor(a, dtype=torch.float32).unsqueeze(0)
                    )
                dense_res.append(pr - float(r))
                dense_act.append(float(r))
            if terminated or truncated:
                j = n - 1  # index of the FINAL transition, the one carrying the outcome bonus
                ramp = _reward_ramp(agent, t_obs, t_act, j)
                with torch.no_grad():
                    zj = agent.model.encode(
                        torch.as_tensor(t_obs[j], dtype=torch.float32).unsqueeze(0)
                    )
                terms.append(
                    dict(
                        ep=ep,
                        event=event or ("timeout" if truncated else "?"),
                        terminated=bool(terminated),
                        steps=n,
                        actual=float(r),
                        ramp=ramp,
                        tail=_tail_value(agent, zj, horizon),
                    )
                )
                break
    return np.asarray(dense_res), np.asarray(dense_act), terms


def _ramp_line(log, label, rows, depth_keys=(0, 1, 2, 3)):
    """Print predicted-vs-actual at each depth for a class of terminals."""
    if not rows:
        log(f"       {label:<22s}  (none in sample)")
        return
    act = np.asarray([r["actual"] for r in rows])
    log(f"       {label:<22s}  n {len(rows):3d}   actual reward p50 {np.median(act):+9.2f}")
    for d in depth_keys:
        v = [r["ramp"][d] for r in rows if d in r["ramp"]]
        if not v:
            continue
        a = [r["actual"] for r in rows if d in r["ramp"]]
        rel = [abs(p - q) / max(abs(q), 1e-9) for p, q in zip(v, a)]
        # ABSOLUTE residual is reported alongside because it is the statistic that ISOLATES the terminal:
        # relative error flatters a class whose actual reward is tiny (the dense ticks sit at ~0.07, so a
        # 0.05 miss reads as 70% there while a 195-unit miss on the terminal reads as 97%). The two
        # together say "accurate to 0.05 where it looks constantly, off by ~195 where it looks once".
        absr = [abs(p - q) for p, q in zip(v, a)]
        mark = "  <- TRAINED DEPTH for the terminal (E2a: t=2)" if d == 2 else ""
        log(
            f"         depth {d}  predicted p50 {np.median(v):+9.2f}   "
            f"|abs err| p50 {np.median(absr):8.3f}   |rel err| p50 {np.median(rel):7.3f}   "
            f"within 25%: {sum(x <= 0.25 for x in rel)}/{len(rel)}{mark}"
        )


def _block(log, tag, ck, sd, dense_res, dense_act, terms, horizon):
    log("")
    log("=" * 96)
    log(
        f"[{tag}] {ck}  (collected={int(sd.get('collected', -1)):,}, "
        f"mpc={bool(sd['cfg'].get('mpc'))}, episodic={bool(sd['cfg'].get('episodic'))})"
    )
    log("=" * 96)

    # Classify on the EVENT, not on `terminated`: `left_bubble` carries a -50 bonus but is a truncation,
    # so a `terminated`-based split would have hidden it among the timeouts.
    def _ev(r):
        return r["event"].split("@")[0]

    kills = [r for r in terms if _ev(r) in KILL_EVENTS]
    deaths = [r for r in terms if _ev(r) in DEATH_EVENTS]
    egress = [r for r in terms if _ev(r) in EGRESS_EVENTS]
    touts = [r for r in terms if _ev(r) == "timeout"]
    # NOTHING may be silently dropped -- that is exactly how the pilot reported `kills 0` while the
    # per-episode table was full of them. Any unclassified event is printed by name.
    known = set(KILL_EVENTS) | set(DEATH_EVENTS) | set(EGRESS_EVENTS) | {"timeout"}
    other = sorted({_ev(r) for r in terms if _ev(r) not in known})

    log(
        f"   episodes {len(terms)}   kills {len(kills)}   deaths {len(deaths)}   "
        f"egress {len(egress)}   timeouts {len(touts)}   dense control ticks {len(dense_res):,}"
    )
    if other:
        log(f"   ⚠ UNCLASSIFIED events present, not in any reported class: {other}")
    assert len(kills) + len(deaths) + len(egress) + len(touts) + sum(
        1 for r in terms if _ev(r) in other
    ) == len(terms), "episode classification lost a row"
    log("")
    log("   M1  DENSE CONTROL (K3) — ordinary mid-fight ticks, depth 0, a legitimate query point")
    if len(dense_res):
        log(
            f"       actual  p50 {np.median(dense_act):+8.4f}  "
            f"|actual| p90 {np.quantile(np.abs(dense_act), 0.9):8.4f}"
        )
        log(
            f"       residual (pred - actual)  p50 {np.median(dense_res):+8.4f}  "
            f"|res| p50 {np.median(np.abs(dense_res)):8.4f}  |res| p90 "
            f"{np.quantile(np.abs(dense_res), 0.9):8.4f}"
        )
        den = np.median(np.abs(dense_act))
        log(
            f"       |res| p50 as a fraction of |actual| p50: "
            f"{np.median(np.abs(dense_res)) / max(den, 1e-9):.4f}"
        )
    log("")
    log("   M2  TERMINAL TRANSITION — the outcome-bearing step, paired depth ramp")
    log(
        "       (the terminal is only ever SUPERVISED at t=2, so depth 2 is the number that counts)"
    )
    _ramp_line(log, "KILL   (R_KILL +150)", kills)
    _ramp_line(log, "DEATH  (R_DECK -200)", deaths)
    _ramp_line(log, "EGRESS (exit -50)", egress)
    _ramp_line(log, "timeout (no bonus)", touts)

    log("")
    log("   M4  MPC BRANCH — would a predicted terminal be scored WORSE than continuing?")
    log("       compares |predicted terminal reward| against |g^H * Q| = the bootstrap it zeroes")
    res = {}
    for name, rows in (("KILL", kills), ("DEATH", deaths), ("EGRESS", egress)):
        if not rows:
            continue
        pr = np.asarray([r["ramp"].get(2, r["ramp"][0]) for r in rows])
        tl = np.asarray([r["tail"] for r in rows])
        # A terminating plan scores G_partial + pred; a continuing one G_partial + tail. Terminating is
        # ATTRACTIVE exactly when pred > tail.
        # ⚠ Sign convention: `adv > 0` is CORRECT for KILL (a kill should attract) and is the DEFECT for
        # DEATH/EGRESS. K4 therefore tests the bad classes only -- an earlier draft required adv <= 0 for
        # every class, which would have failed a perfectly healthy head for pricing kills properly.
        adv = pr - tl
        want = "want > 0" if name == "KILL" else "want <= 0, DEFECT if > 0"
        log(
            f"       {name:<6s} predicted p50 {np.median(pr):+9.2f}   "
            f"g^H*Q p50 {np.median(tl):+9.2f}   "
            f"advantage-of-terminating p50 {np.median(adv):+9.2f}   "
            f"attractive in {int((adv > 0).sum())}/{len(adv)}   [{want}]"
        )
        res[name] = dict(
            pred=float(np.median(pr)), tail=float(np.median(tl)), adv=float(np.median(adv))
        )
    log(f"       (rolled {horizon} latent steps for the bootstrap, matching cfg.horizon)")

    log("")
    log("   per-episode (a head's number is only interpretable beside what happened)")
    log("     ep  event              term  steps      actual    pred@d0    pred@d2      g^H*Q")
    for r in terms:
        log(
            f"     {r['ep']:2d}  {r['event']:<18s} {'Y' if r['terminated'] else '.':>4s}  "
            f"{r['steps']:5d}  {r['actual']:+10.2f}  {r['ramp'][0]:+9.2f}  "
            f"{r['ramp'].get(2, float('nan')):+9.2f}  {r['tail']:+9.2f}"
        )

    def _rel(rows):
        """(median |rel err|, count within 25%, n, median |abs err|) at the TRAINED depth 2."""
        if not rows:
            return float("nan"), 0, 0, float("nan")
        d = [abs(r["ramp"].get(2, r["ramp"][0]) - r["actual"]) for r in rows]
        v = [x / max(abs(r["actual"]), 1e-9) for x, r in zip(d, rows)]
        return float(np.median(v)), sum(x <= 0.25 for x in v), len(v), float(np.median(d))

    k_rel, k_ok, k_n, k_abs = _rel(kills)
    d_rel, d_ok, d_n, d_abs = _rel(deaths)
    e_rel, e_ok, e_n, e_abs = _rel(egress)
    return dict(
        k_rel=k_rel,
        k_ok=k_ok,
        k_n=k_n,
        k_abs=k_abs,
        d_rel=d_rel,
        d_ok=d_ok,
        d_n=d_n,
        d_abs=d_abs,
        e_rel=e_rel,
        e_ok=e_ok,
        e_n=e_n,
        e_abs=e_abs,
        dense_rel=(
            float(np.median(np.abs(dense_res)) / max(np.median(np.abs(dense_act)), 1e-9))
            if len(dense_res)
            else float("nan")
        ),
        dense_abs=float(np.median(np.abs(dense_res))) if len(dense_res) else float("nan"),
        mpc=res,
    )


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints", required=True, help="comma-separated; the FIRST is the run")
    p.add_argument("--aspects", default="0,30,55")
    p.add_argument("--episodes", type=int, default=20, help="per aspect")
    p.add_argument("--separation", type=float, default=700.0)
    p.add_argument("--max-time-s", type=float, default=60.0)
    p.add_argument("--jitter", type=float, default=0.5, help="ALWAYS 0.5 in a verdict")
    p.add_argument("--action-repeat", type=int, default=5)
    p.add_argument("--frame-stack", type=int, default=1)
    p.add_argument("--rate-hz", type=float, default=120.0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log", default="", help="tee to this path (a fresh container loses /tmp)")
    args = p.parse_args(argv)

    torch.set_num_threads(max(1, args.threads))
    sink = open(args.log, "w", buffering=1) if args.log else None

    def log(s: str = "") -> None:
        print(s, flush=True)
        if sink:
            sink.write(s + "\n")

    log("#" * 96)
    log("#  E3 -- does the REWARD head price the terminal? (kill +150 / deck -200)")
    log("#  K1 kill within 25% at depth 2 | K2 death within 25% | K3 dense control small")
    log("#  K4 |pred terminal| >= |g^H*Q| => a terminating plan is correctly scored worse")
    log(
        "#  NOT A LEVER: a failure authorises fixing the SUPERVISION (E2b form (i)), not a constant."
    )
    log("#" * 96)

    ckpts = [c.strip() for c in args.checkpoints.split(",") if c.strip()]
    aspects = [float(a) for a in args.aspects.split(",") if a.strip()]
    log(
        f"   cell chase | sep {args.separation:.0f} m | mt {args.max_time_s:.0f} s | "
        f"jitter {args.jitter:g} | aspects {aspects} | {args.episodes} eps/aspect"
    )
    log("   D6: the FIRST checkpoint is the run; the rest are controls, same env build and seeds.")

    out = {}
    for ci, ck in enumerate(ckpts):
        sd = torch.load(ck, map_location="cpu", weights_only=False)
        cfg = TDMPC2Config(**sd["cfg"])
        cfg.mpc = False  # we call the heads directly; planning is irrelevant here
        agent = TDMPC2(cfg, torch.device("cpu"))
        agent.load_state_dict(sd)
        agent.model.eval()

        dres, dact, terms = [], [], []
        for asp in aspects:
            kw = dict(
                posture="offensive",
                aspect_deg=asp,
                separation_m=args.separation,
                action_repeat=args.action_repeat,
                frame_stack=args.frame_stack,
                max_time_s=args.max_time_s,
                rate_hz=args.rate_hz,
            )
            if args.jitter > 0:
                kw["jitter"] = args.jitter
            env = make_tdmpc_env(3, **kw)
            r_res, r_act, r_terms = _roll(
                agent, env, args.episodes, args.seed + int(round(asp)) * 1000, cfg.horizon
            )
            env.close()
            for r in r_terms:
                r["event"] = f"{r['event']}@a{asp:.0f}"
            dres.append(r_res)
            dact.append(r_act)
            terms += r_terms

        out[ck] = _block(
            log,
            "RUN" if ci == 0 else "CONTROL",
            ck,
            sd,
            np.concatenate(dres) if dres else np.array([]),
            np.concatenate(dact) if dact else np.array([]),
            terms,
            cfg.horizon,
        )

    log("")
    log("=" * 96)
    log("   PRE-REGISTERED GATE")
    log("=" * 96)
    for ck, r in out.items():
        k1 = r["k_n"] > 0 and r["k_rel"] <= 0.25
        k2 = r["d_n"] > 0 and r["d_rel"] <= 0.25
        # K3 is stated on the ABSOLUTE residual: the dense reward sits at ~0.07, so a relative gate there
        # is dominated by its own tiny denominator and says nothing about whether the head is accurate.
        # The isolating claim is "accurate to ~0.05 absolute on the dense terms, off by ~10^2 on the
        # terminal", so the gate is 1.0 reward unit -- well below any terminal bonus, well above the
        # dense scale.
        k3 = not np.isnan(r["dense_abs"]) and r["dense_abs"] <= 1.0
        # K4 tests the BAD classes only. A kill SHOULD be attractive; only DEATH/EGRESS being attractive
        # is the planner defect.
        bad = [v for k, v in r["mpc"].items() if k in ("DEATH", "EGRESS")]
        k4 = all(v["adv"] <= 0 for v in bad) if bad else False
        log(
            f"   {ck}\n"
            f"      KILL   n {r['k_n']:3d}  @d2 |abs err| p50 {r['k_abs']:8.2f}  |rel| "
            f"{r['k_rel']:.3f}  within25% {r['k_ok']}/{r['k_n']}\n"
            f"      DEATH  n {r['d_n']:3d}  @d2 |abs err| p50 {r['d_abs']:8.2f}  |rel| "
            f"{r['d_rel']:.3f}  within25% {r['d_ok']}/{r['d_n']}\n"
            f"      EGRESS n {r['e_n']:3d}  @d2 |abs err| p50 {r['e_abs']:8.2f}  |rel| "
            f"{r['e_rel']:.3f}  within25% {r['e_ok']}/{r['e_n']}\n"
            f"      DENSE  |res| p50 {r['dense_abs']:.4f} absolute  ({r['dense_rel']:.3f} relative)\n"
            f"      [{'X' if k1 else ' '}] K1 kill priced   "
            f"[{'X' if k2 else ' '}] K2 death priced   "
            f"[{'X' if k3 else ' '}] K3 dense control OK   "
            f"[{'X' if k4 else ' '}] K4 terminating scored worse (no planner defect)"
        )
    log("")
    log("   If K1 and K2 both FAIL while K3 PASSES, the learned reward function is OUTCOME-BLIND:")
    log(
        "   accurate on the dense terms it sees constantly, ignorant of the outcome it sees once per"
    )
    log(
        "   4.3 updates. That is a structural finding on the value path, and the fix is SUPERVISION."
    )
    if sink:
        sink.close()


if __name__ == "__main__":
    main()
