"""PROBE-2 empirical arm: measure the ACTUAL exploration noise the collector injects,
the autocorrelation of the COLLECTED action sequence, and the trigger duty cycle.

Runs the c95 head-on merge env exactly as the collector does (agent.act(eval_mode=False),
i.e. MPPI + the per-step Gaussian at agent.py:121) and, for each control step, ALSO
recomputes the same plan with the RNG state and _prev_mean restored so the paired
difference isolates the injected noise:  a_explore - a_greedy = std[0] * z.

Arms:
  white  = as shipped
  ou     = torch.randn(A) monkeypatched to an OU(tau) unit-variance state (the PROPOSED patch)
  pink   = torch.randn(A) monkeypatched to a precomputed 1/f sequence (unit variance)
Only the FINAL per-step draw (shape (A,)) is patched; the MPPI sample noise
(shape (H, n_sample, A)) is untouched.

Also measures the jet's per-control-step roll authority (open loop) so "1.5 s of held
roll" can be converted to degrees of bank.

    cd controllers/rl && uv run python -u _diag_explore_noise.py \
        --checkpoint runs/curriculum-95/run.pt --episodes 3 --max-time-s 60
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from astrum_rl import _paths

_paths.require_astrum_env()
from astrum_dogfight import constants as C  # noqa: E402
from astrum_dogfight.geometry import relative  # noqa: E402
from astrum_rl.envs import make_tdmpc_env  # noqa: E402
from astrum_rl.tdmpc2.agent import TDMPC2, resolve_device  # noqa: E402
from astrum_rl.tdmpc2.config import TDMPC2Config  # noqa: E402

DT = None  # set from action_repeat / rate_hz
CH = ["ail", "elev", "rud", "thr", "trig"]


# ---------------------------------------------------------------- noise arms
class WhiteNoise:
    def reset(self):
        pass

    def draw(self, a_dim, gen):
        return torch.randn(a_dim, generator=gen)


class OUNoise:
    """x <- rho x + sqrt(1-rho^2) w, rho = exp(-1/tau_steps); unit marginal variance."""

    def __init__(self, tau_steps: float):
        self.rho = math.exp(-1.0 / tau_steps)
        self.s = math.sqrt(1.0 - self.rho**2)
        self.x = None

    def reset(self):
        self.x = None

    def draw(self, a_dim, gen):
        w = torch.randn(a_dim, generator=gen)
        if self.x is None or self.x.shape != w.shape:
            self.x = w
        else:
            self.x = self.rho * self.x + self.s * w
        return self.x


class PinkNoise:
    """Precomputed per-episode 1/f^(beta/2) amplitude-shaped sequence, unit variance
    per channel (Timmer & Koenig 1995)."""

    def __init__(self, n: int, beta: float = 1.0, seed: int = 0):
        self.n = n
        self.beta = beta
        self.rng = np.random.default_rng(seed)
        self.buf = None
        self.i = 0

    def _make(self, a_dim):
        nfft = 1 << (2 * self.n - 1).bit_length()
        f = np.fft.rfftfreq(nfft, d=1.0)
        sc = np.zeros_like(f)
        sc[1:] = f[1:] ** (-self.beta / 2.0)
        spec = (
            self.rng.standard_normal((a_dim, f.size)) + 1j * self.rng.standard_normal((a_dim, f.size))
        ) * sc
        x = np.fft.irfft(spec, n=nfft, axis=1)[:, : self.n]
        x /= x.std(axis=1, keepdims=True)
        return x.astype(np.float32)

    def reset(self):
        self.buf = None
        self.i = 0

    def draw(self, a_dim, gen):
        if self.buf is None:
            self.buf = self._make(a_dim)
        v = self.buf[:, min(self.i, self.n - 1)]
        self.i += 1
        return torch.from_numpy(v.copy())


# ---------------------------------------------------------------- stats
def acf(x, maxlag):
    x = np.asarray(x, float)
    x = x - x.mean()
    v = (x * x).mean()
    if v <= 0:
        return np.zeros(maxlag + 1)
    return np.array([float((x[: len(x) - k] * x[k:]).mean() / v) for k in range(maxlag + 1)])


def runs_above(x, thresh):
    """Lengths of maximal runs with x > thresh."""
    a = np.asarray(x) > thresh
    out, run = [], 0
    for v in a:
        if v:
            run += 1
        elif run:
            out.append(run)
            run = 0
    if run:
        out.append(run)
    return np.array(out) if out else np.array([0])


def window_mean_std(x, T):
    x = np.asarray(x, float)
    if len(x) <= T:
        return float("nan")
    c = np.concatenate([[0.0], np.cumsum(x)])
    wm = (c[T:] - c[:-T]) / T
    return float(wm.std())


# ---------------------------------------------------------------- rollout
def paired_step(agent, obs, t0, noise, gen):
    """Return (a_explore, a_greedy) from ONE plan, with the noise draw isolated."""
    o = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
    z = agent.model.encode(o)
    prev = agent._prev_mean.clone()
    state = torch.get_rng_state()
    a_g = agent.plan(z, t0=t0, eval_mode=True)  # greedy: elite pick, no added noise
    agent._prev_mean = prev.clone()
    torch.set_rng_state(state)
    # explore: same plan (same rng, same prev_mean) + the noise draw
    a_e = _plan_explore(agent, z, t0, noise, gen)
    return a_e.numpy().copy(), a_g.numpy().copy()


@torch.no_grad()
def _plan_explore(agent, z, t0, noise, gen):
    """agent.plan(eval_mode=False) with the FINAL draw supplied by `noise`.

    Verbatim copy of TDMPC2.plan (agent.py:65-122) except the last two lines, which
    take the noise sample from the arm instead of torch.randn(A). Kept as a copy so
    the shipped agent is untouched by the probe."""
    cfg = agent.cfg
    H, A = cfg.horizon, cfg.action_dim
    pi_actions = torch.empty(H, cfg.num_pi_trajs, A, device=agent.device)
    if cfg.num_pi_trajs > 0:
        _z = z.repeat(cfg.num_pi_trajs, 1)
        for t in range(H - 1):
            pi_actions[t] = agent.model.pi(_z)[1]
            _z = agent.model.next(_z, pi_actions[t])
        pi_actions[H - 1] = agent.model.pi(_z)[1]
    z = z.repeat(cfg.num_samples, 1)
    mean = torch.zeros(H, A, device=agent.device)
    std = torch.full((H, A), cfg.max_std, device=agent.device)
    if not t0:
        mean[:-1] = agent._prev_mean[1:]
    actions = torch.empty(H, cfg.num_samples, A, device=agent.device)
    if cfg.num_pi_trajs > 0:
        actions[:, : cfg.num_pi_trajs] = pi_actions
    score = None
    for _ in range(cfg.iterations):
        n_sample = cfg.num_samples - cfg.num_pi_trajs
        nz = torch.randn(H, n_sample, A, device=agent.device)
        actions[:, cfg.num_pi_trajs :] = (mean.unsqueeze(1) + std.unsqueeze(1) * nz).clamp(-1, 1)
        value = agent._estimate_value(z, actions).nan_to_num_(0.0)
        elite_idxs = torch.topk(value.squeeze(1), cfg.num_elites, dim=0).indices
        elite_value = value[elite_idxs]
        elite_actions = actions[:, elite_idxs]
        max_value = elite_value.max(0).values
        score = torch.exp(cfg.temperature * (elite_value - max_value))
        score = score / (score.sum(0) + 1e-9)
        mean = (score.unsqueeze(0) * elite_actions).sum(1) / (score.sum(0) + 1e-9)
        var = (score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(1) / (
            score.sum(0) + 1e-9
        )
        std = var.sqrt().clamp(cfg.min_std, cfg.max_std)
    pick = torch.multinomial(score.squeeze(1), 1).item()
    plan_actions = elite_actions[:, pick]
    agent._prev_mean = mean
    a, a_std = plan_actions[0], std[0]
    a = a + a_std * noise.draw(A, gen).to(agent.device)
    _plan_explore.last_std = a_std.clone()
    return a.clamp(-1, 1).cpu()


def roll(agent, env, base, noise, gen, seed, dt):
    obs, _ = env.reset(seed=seed)
    agent._prev_mean.zero_()
    noise.reset()
    t0, event = True, None
    A_e, A_g, STD, RNGm, FIRED = [], [], [], [], []
    while True:
        a_e, a_g = paired_step(agent, obs, t0, noise, gen)
        t0 = False
        me = base._states[base.learner_id]
        opp = base._states[base.opponent_id]
        rg = relative(me, opp)
        A_e.append(a_e)
        A_g.append(a_g)
        STD.append(_plan_explore.last_std.cpu().numpy().copy())
        RNGm.append(rg.rng)
        FIRED.append(1.0 if a_e[4] >= 0.0 else 0.0)  # env fires when AGENT trigger >= 0
        obs, _r, term, trunc, info = env.step(a_e)
        if info.get("event"):
            event = info["event"]
        if term or trunc:
            break
    me = base._states[base.learner_id]
    opp = base._states[base.opponent_id]
    return dict(
        event=event,
        A_e=np.asarray(A_e),
        A_g=np.asarray(A_g),
        STD=np.asarray(STD),
        rng=np.asarray(RNGm),
        fired=np.asarray(FIRED),
        dealt=C.HP_MAX - float(opp.health),
        taken=C.HP_MAX - float(me.health),
        spent=float(C.AMMO_MAX - me.ammo),
        n=len(A_e),
    )


def roll_authority(env, base, dt, steps=24):
    """Open loop: hold full right roll + mid throttle, log bank angle per control step."""
    env.reset(seed=7)
    phis = []
    a = np.array([1.0, 0.0, 0.0, 0.8, -1.0], np.float32)
    for _ in range(steps):
        env.step(a)
        phis.append(math.degrees(float(base._states[base.learner_id].phi)))
    env.reset(seed=7)
    nzs = []
    a = np.array([0.0, -1.0, 0.0, 1.0, -1.0], np.float32)
    for _ in range(steps):
        env.step(a)
        st = base._states[base.learner_id]
        nzs.append(math.degrees(float(st.theta)))
    return phis, nzs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--arms", default="white,ou,pink")
    ap.add_argument("--tau-s", type=float, default=1.5)
    ap.add_argument("--separation", type=float, default=3000.0)
    ap.add_argument("--max-time-s", type=float, default=60.0)
    ap.add_argument("--rate-hz", type=float, default=120.0)
    ap.add_argument("--action-repeat", type=int, default=5)
    ap.add_argument("--me-dalt-ft", type=float, default=2000.0)
    ap.add_argument("--lateral-m", type=float, default=1400.0)
    ap.add_argument("--jitter", type=float, default=0.5)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, args.threads))
    device = resolve_device(args.device)
    dt = args.action_repeat / args.rate_hz

    sd = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = TDMPC2Config(**sd["cfg"])
    print(
        f"ckpt {args.checkpoint} collected={sd.get('collected', -1):,} stage={sd.get('stage_idx', -1)}\n"
        f"cfg: mpc={cfg.mpc} horizon={cfg.horizon} num_samples={cfg.num_samples} "
        f"num_elites={cfg.num_elites} num_pi_trajs={cfg.num_pi_trajs} "
        f"min_std={cfg.min_std} max_std={cfg.max_std} action_repeat={cfg.action_repeat} "
        f"obs_dim={cfg.obs_dim} discount={cfg.resolved_discount():.4f}\n"
        f"control dt = {dt * 1000:.1f} ms ({1 / dt:.1f} Hz)"
    )
    agent = TDMPC2(cfg, device)
    agent.load_state_dict(sd)
    agent.model.eval()

    def mk_env():
        return make_tdmpc_env(
            3,
            posture="offensive",
            aspect_deg=0.0,
            opp_heading_deg=180.0,
            separation_m=args.separation,
            action_repeat=args.action_repeat,
            frame_stack=1,
            max_time_s=args.max_time_s,
            rate_hz=args.rate_hz,
            outcome_mode=True,
            jitter=args.jitter,
            me_dalt_ft=args.me_dalt_ft,
            lateral_m_max=args.lateral_m,
        )

    env = mk_env()
    base = env.unwrapped

    phis, thetas = roll_authority(env, base, dt)
    print("\n=== roll / pitch authority per CONTROL step (open loop, full command) ===")
    print("  bank phi (deg) after k control steps: "
          + " ".join(f"k={k + 1}:{phis[k]:+.0f}" for k in (0, 1, 2, 4, 7, 11, 23)))
    print("  pitch theta (deg), elev=-1:           "
          + " ".join(f"k={k + 1}:{thetas[k]:+.0f}" for k in (0, 1, 2, 4, 7, 11, 23)))
    print(f"  -> one control step ({dt * 1000:.0f} ms) of full roll command moves bank "
          f"{phis[0]:+.0f} deg; 1.5 s (36 steps) of held roll would be many wraps.")

    tau_steps = args.tau_s / dt
    arms = {}
    for name in [a.strip() for a in args.arms.split(",") if a.strip()]:
        if name == "white":
            arms[name] = WhiteNoise()
        elif name == "ou":
            arms[name] = OUNoise(tau_steps)
        elif name == "pink":
            arms[name] = PinkNoise(int(args.max_time_s / dt) + 8, 1.0, seed=args.seed)
        else:
            raise SystemExit(f"unknown arm {name}")

    gen = torch.Generator()
    gen.manual_seed(args.seed + 4242)

    for name, noise in arms.items():
        print(f"\n================ ARM: {name}"
              + (f" (tau={args.tau_s:.2f} s = {tau_steps:.0f} steps)" if name == "ou" else "")
              + " ================")
        rows = [roll(agent, env, base, noise, gen, args.seed + 100 * i, dt)
                for i in range(args.episodes)]
        Ae = np.concatenate([r["A_e"] for r in rows])
        Ag = np.concatenate([r["A_g"] for r in rows])
        SD = np.concatenate([r["STD"] for r in rows])
        NZ = Ae - Ag  # censored by the [-1,1] clamp; see clip% below
        print(f"  episodes {len(rows)}  steps {len(Ae)}  events "
              f"{[r['event'] for r in rows]}  dealt {[round(r['dealt'], 1) for r in rows]}  "
              f"spent {[int(r['spent']) for r in rows]}")
        print(f"  {'chan':>5} {'MPPI std[0]':>11} {'|noise| rms':>11} {'clip%':>6} "
              f"{'a_expl std':>10} {'a_greedy std':>12} {'a_greedy mean':>13}")
        for i, ch in enumerate(CH):
            clip = 100.0 * float(np.mean(np.abs(Ae[:, i]) >= 0.999))
            print(f"  {ch:>5} {SD[:, i].mean():>11.3f} {np.sqrt((NZ[:, i] ** 2).mean()):>11.3f} "
                  f"{clip:>6.1f} {Ae[:, i].std():>10.3f} {Ag[:, i].std():>12.3f} "
                  f"{Ag[:, i].mean():>+13.3f}")

        print(f"  --- autocorrelation of the COLLECTED action sequence (lag in control steps) ---")
        print(f"  {'chan':>5} {'seq':>8} {'rho1':>6} {'rho6':>6} {'rho12':>6} {'rho36':>6} "
              f"{'rho72':>6} {'tau_int(s)':>10}")
        for i, ch in enumerate(CH):
            for tag, M in (("explore", Ae), ("greedy", Ag), ("noise", NZ)):
                r = acf(M[:, i], 120)
                s = 0.0
                for k in range(1, len(r)):
                    if r[k] <= 0:
                        break
                    s += r[k]
                print(f"  {ch:>5} {tag:>8} {r[1]:>6.3f} {r[6]:>6.3f} {r[12]:>6.3f} {r[36]:>6.3f} "
                      f"{r[72]:>6.3f} {(1 + 2 * s) * dt:>10.2f}")

        print("  --- sustained-command run lengths (per episode) ---")
        for i, ch in ((0, "ail"), (1, "elev")):
            for thr in (0.5,):
                for tag, key in (("explore", "A_e"), ("greedy", "A_g")):
                    L = np.concatenate([runs_above(r[key][:, i], thr) for r in rows])
                    print(f"  {ch:>5} {tag:>8} a>{thr}: n_runs {len(L)}  max {L.max()} steps "
                          f"({L.max() * dt:.2f} s)  mean {L.mean():.1f}  frac>=36 steps "
                          f"{float(np.mean(L >= 36)):.3f}")
        print("  --- window-mean amplitude of the injected noise (deviation from greedy) ---")
        for i, ch in ((0, "ail"), (1, "elev")):
            print(f"  {ch:>5} " + " ".join(
                f"T={T}({T * dt:.2f}s):{window_mean_std(NZ[:, i], T):.3f}" for T in (12, 36, 72)))
        print("  --- trigger ---")
        fired_e = np.concatenate([r["fired"] for r in rows])
        rng_all = np.concatenate([r["rng"] for r in rows])
        fired_g = (Ag[:, 4] >= 0.0).astype(float)
        print(f"  P(trigger down | explore) = {fired_e.mean():.3f}   "
              f"P(trigger down | greedy) = {fired_g.mean():.3f}")
        print(f"  agent a4: greedy mean {Ag[:, 4].mean():+.3f} std {Ag[:, 4].std():.3f}; "
              f"explore mean {Ae[:, 4].mean():+.3f} std {Ae[:, 4].std():.3f}")
        print(f"  P(trigger down | a_greedy4 < 0) [pure noise-induced fire] = "
              f"{float(np.mean(fired_e[Ag[:, 4] < 0.0])):.3f} over "
              f"{int((Ag[:, 4] < 0.0).sum())} steps")
        if fired_e.sum():
            print(f"  fire-range median {np.median(rng_all[fired_e > 0.5]):.0f} m, "
                  f"frac of trigger-down steps inside 600 m "
                  f"{float(np.mean(rng_all[fired_e > 0.5] < 600.0)):.3f}, inside R_EFF(1200) "
                  f"{float(np.mean(rng_all[fired_e > 0.5] < C.R_EFF)):.3f}")
        print(f"  rounds/ep {np.mean([r['spent'] for r in rows]):.0f} of {C.AMMO_MAX:.0f} "
              f"(trigger-down steps x {args.action_repeat} ticks x "
              f"{100.0 / args.rate_hz:.2f} rds/tick)")

    env.close()


if __name__ == "__main__":
    main()
