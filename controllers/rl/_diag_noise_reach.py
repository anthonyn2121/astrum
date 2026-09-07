"""PROBE-2 (a)/(b)/(c): reachability of a multi-second coherent maneuver under
i.i.d. vs temporally-correlated exploration noise. Pure math, no sim, no torch.

Models the exploration DEVIATION on one continuous channel as
    d[t] = sigma * e[t],  e[t] unit-marginal-variance,
with e[t] either white (current code, agent.py:121), OU(tau), or pink (1/f).
All arms are variance-matched (marginal std == 1) so the only difference is the
autocorrelation.

Reports:
  1. window-mean amplitude  std( mean_{t<T} e[t] )  -- the "maneuver-space" amplitude
  2. P(run of >=T steps with e >= thresh) per 1440-step (60 s) episode and per 5M steps
  3. P(sustained roll >=1.5 s THEN sustained pull >=1.0 s) on two independent channels
"""

from __future__ import annotations

import math

import numpy as np

DT = 1.0 / 24.0  # control-step period (action_repeat 5 @ 120 Hz)
RNG = np.random.default_rng(12345)


# ---------------------------------------------------------------- generators
def white(n, m, rng):
    return rng.standard_normal((m, n))


def ou(n, m, rng, tau_steps):
    """Discrete OU / AR(1) with rho = exp(-1/tau), unit marginal variance."""
    rho = math.exp(-1.0 / tau_steps)
    s = math.sqrt(1.0 - rho * rho)
    x = np.empty((m, n))
    x[:, 0] = rng.standard_normal(m)
    for t in range(1, n):
        x[:, t] = rho * x[:, t - 1] + s * rng.standard_normal(m)
    return x


def pink(n, m, rng, beta=1.0):
    """Timmer & Koenig (1995) colored noise: shape the FFT of white noise by
    f^{-beta/2}, then renormalize to unit variance. beta=1 -> pink (1/f power)."""
    nfft = 1 << (int(n * 2 - 1).bit_length())
    f = np.fft.rfftfreq(nfft, d=1.0)
    scale = np.zeros_like(f)
    scale[1:] = f[1:] ** (-beta / 2.0)
    scale[0] = 0.0  # drop DC so the mean is 0 (a DC offset is not exploration)
    spec = (rng.standard_normal((m, f.size)) + 1j * rng.standard_normal((m, f.size))) * scale
    x = np.fft.irfft(spec, n=nfft, axis=1)[:, :n]
    x /= x.std(axis=1, keepdims=True)
    return x


# ---------------------------------------------------------------- statistics
def acf(x, maxlag):
    """Mean autocorrelation over the m realizations, lags 0..maxlag."""
    m, n = x.shape
    xc = x - x.mean(axis=1, keepdims=True)
    v = (xc * xc).mean(axis=1, keepdims=True)
    out = []
    for k in range(maxlag + 1):
        out.append(float(((xc[:, : n - k] * xc[:, k:]).mean(axis=1, keepdims=True) / v).mean()))
    return np.array(out)


def integral_time(rho):
    """sum_{k>=1} rho_k  (truncated at first negative) -> correlation time in steps."""
    s = 0.0
    for k in range(1, len(rho)):
        if rho[k] <= 0:
            break
        s += rho[k]
    return 1.0 + 2.0 * s  # variance-inflation factor of a long-window mean


def window_mean_std(x, T):
    """std of the T-step running mean (the amplitude of exploration that survives
    the aircraft's own low-pass / the reward's multi-second credit window)."""
    m, n = x.shape
    c = np.cumsum(x, axis=1)
    wm = (c[:, T:] - c[:, :-T]) / T
    return float(wm.std())


def p_run(x, T, thresh):
    """Fraction of realizations containing >=1 run of >=T consecutive samples > thresh."""
    above = x > thresh
    m, n = above.shape
    hit = np.zeros(m, bool)
    for i in range(m):
        run = 0
        for t in range(n):
            run = run + 1 if above[i, t] else 0
            if run >= T:
                hit[i] = True
                break
    return hit.mean()


def p_run_analytic_white(T, thresh, n):
    """Exact-ish for white noise: p = P(z>thresh); expected # of run starts in n
    samples ~= (n-T+1) * p^T * (1-p)  -> use as an upper bound on P(>=1 run)."""
    from statistics import NormalDist

    p = 1.0 - NormalDist().cdf(thresh)
    return (n - T + 1) * (p**T) * (1.0 - p), p


# ---------------------------------------------------------------- main
def main() -> None:
    N = 1440  # control steps in a 60 s episode @24 Hz
    M = 4000  # realizations
    TAU = 36  # OU correlation time = 1.5 s (36 steps) -- the target maneuver scale

    arms = {
        "white (CURRENT)": white(N, M, RNG),
        f"OU tau={TAU} ({TAU * DT:.2f}s)": ou(N, M, RNG, TAU),
        "OU tau=12 (0.50s)": ou(N, M, RNG, 12),
        "pink beta=1": pink(N, M, RNG, 1.0),
        "brown beta=2": pink(N, M, RNG, 2.0),
    }

    print("=== (b) autocorrelation of the exploration sequence (control steps @24 Hz) ===")
    print(f"{'arm':>20} {'rho(1)':>7} {'rho(12)':>8} {'rho(36)':>8} {'rho(96)':>8} "
          f"{'tau_int(steps)':>14} {'tau_int(s)':>10}")
    for name, x in arms.items():
        r = acf(x, 200)
        ti = integral_time(r)
        print(f"{name:>20} {r[1]:>7.3f} {r[12]:>8.3f} {r[36]:>8.3f} {r[96]:>8.3f} "
              f"{ti:>14.1f} {ti * DT:>10.2f}")
    print("  NB action_repeat=5 is a ZERO-ORDER HOLD *below* this sampling rate: it sets the "
          "control period (208 ms) and contributes NO correlation between successive control "
          "steps. rho(1) above is the correlation between consecutive 208 ms commands.")

    print()
    print("=== (c1) maneuver-space amplitude: std of the T-step window mean (unit-variance arms) ===")
    print(f"{'arm':>20} " + " ".join(f"{f'T={T}({T * DT:.2f}s)':>15}" for T in (12, 36, 72, 96)))
    for name, x in arms.items():
        print(f"{name:>20} " + " ".join(f"{window_mean_std(x, T):>15.3f}" for T in (12, 36, 72, 96)))
    print("  white: 1/sqrt(T) exactly -> " + " ".join(
        f"T={T}:{1 / math.sqrt(T):.3f}" for T in (12, 36, 72, 96)))

    print()
    print("=== (c2) P(sustained same-sign command) per 60 s episode (1440 steps) ===")
    for thresh, T in ((0.5, 36), (1.0, 36), (0.5, 72), (0.0, 36)):
        ub, p1 = p_run_analytic_white(T, thresh, N)
        print(f"  target: e > {thresh:+.1f} held for {T} steps ({T * DT:.2f} s)   "
              f"[white per-step p={p1:.4f}]")
        for name, x in arms.items():
            if name.startswith("white"):
                print(f"{name:>22}: MC {p_run(x, T, thresh):.4g}   analytic E[#runs] = {ub:.3g}")
            else:
                print(f"{name:>22}: MC {p_run(x, T, thresh):.4g}")
        print()

    print("=== (c3) two-phase maneuver: roll-hold 1.5 s (36 steps, e>+0.5) THEN pull-hold 1.0 s")
    print("    (24 steps, e<-0.5) on the OTHER channel, phases independent ===")
    x_roll = arms
    rng2 = np.random.default_rng(999)
    second = {
        "white (CURRENT)": white(N, M, rng2),
        f"OU tau={TAU} ({TAU * DT:.2f}s)": ou(N, M, rng2, TAU),
        "OU tau=12 (0.50s)": ou(N, M, rng2, 12),
        "pink beta=1": pink(N, M, rng2, 1.0),
        "brown beta=2": pink(N, M, rng2, 2.0),
    }
    for name in arms:
        pa = p_run(x_roll[name], 36, 0.5)
        pb = p_run(-second[name], 24, 0.5)
        print(f"{name:>22}: P(roll phase)={pa:.4g}  P(pull phase)={pb:.4g}  "
              f"joint(indep)={pa * pb:.4g}  -> 1 in {1 / max(pa * pb, 1e-300):.3g} episodes")

    print()
    print("=== (c4) white-noise joint probability, analytic (no MC floor) ===")
    from statistics import NormalDist
    nd = NormalDist()
    for thresh, T in ((0.5, 36), (0.5, 24)):
        p = 1 - nd.cdf(thresh)
        print(f"  P(single run of {T} steps at e>{thresh}) = p^T = {p:.4f}^{T} = {p ** T:.3g}; "
              f"E[# in 1440 steps] = {(N - T + 1) * p ** T * (1 - p):.3g}")
    p36 = (1 - nd.cdf(0.5)) ** 36
    p24 = (1 - nd.cdf(0.5)) ** 24
    e36 = (N - 35) * p36
    e24 = (N - 23) * p24
    print(f"  joint per episode ~ {e36 * e24:.3g}  -> episodes needed ~ {1 / (e36 * e24):.3g}")
    print(f"  a 5M-control-step run is {5e6 / N:.0f} episodes of 60 s.")

    # ---------------------------------------------------------------- (c5)
    # Grounded on the MEASURED closed-loop numbers from _diag_explore_noise.py on the
    # c95 checkpoint (aileron channel, agent units where the box is [-1,1]):
    #   window-mean std of the INJECTED deviation, T = 12 / 36 / 72 control steps
    #     white 0.079 / 0.049 / 0.038      ou(tau=36) 0.166 / 0.137 / 0.108
    #     pink  0.160 / 0.136 / 0.118
    #   measured noise correlation time tau_int: white 0.05 s, ou 0.93 s, pink 1.25 s
    print()
    print("=== (c5) MEASURED closed-loop reachability (c95 ckpt, aileron, agent units) ===")
    meas = {  # arm -> (wm_std at T=36, wm_std at T=72, tau_int seconds)
        "white (CURRENT)": (0.049, 0.038, 0.05),
        "ou tau=36": (0.137, 0.108, 0.93),
        "pink beta=1": (0.136, 0.118, 1.25),
    }
    eps_per_run = 5e6 / N
    for T, key in ((36, 0), (72, 1)):
        print(f"  -- sustained MEAN aileron deviation held {T} steps ({T * DT:.2f} s) --")
        for D in (0.25, 0.50, 0.75):
            row = []
            for name, (w36, w72, tint) in meas.items():
                w = (w36, w72)[key]
                z = D / w
                p_win = 0.5 * math.erfc(z / math.sqrt(2.0))  # one-sided tail
                # independent windows per episode: decorrelate at max(T, tau_int/DT)
                stride = max(T, tint / DT)
                n_win = N / stride
                p_ep = 1.0 - math.exp(-p_win * n_win)  # Poisson-ish
                n_run = p_win * n_win * eps_per_run
                row.append(f"{name}: z={z:5.2f} p_win={p_win:8.2e} p_ep={p_ep:8.2e} "
                           f"E[#/5M]={n_run:8.2e}")
            print(f"    D={D:.2f} agent units (~{D * 90:.0f} deg/s of roll):")
            for r in row:
                print(f"      {r}")
    print("  NB full aileron command (D=1.0) = ~90 deg of bank per second (measured open loop);")
    print("     a displacement roll needs ~90-135 deg of bank held ~1.0-1.5 s => D ~ 0.5-1.0 if")
    print("     the greedy plan contributes nothing (greedy ail mean was -0.007, std 0.63).")


if __name__ == "__main__":
    main()
