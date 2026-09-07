"""Run `_verdict_tmp.py` in a PRE-curriculum-86 environment: `lead_error` back to the naive
nose→aim-point angle. (read-only forensics; throwaway probe)

Why this exists. curriculum-86 corrected `RelGeom.lead_error` to the velocity-inheritance-corrected
gun lay ([[gun-lay-velocity-inheritance]]). That was intended as a REWARD fix, but `build_observation`
feeds `rg.gun_quality` (obs idx 29) and `cos(rg.lead_error)` (idx 30) — so it is also an OBSERVATION
change. Every checkpoint trained before c86 therefore sees those two columns out of distribution, and
scoring one in today's env is not a fair baseline for it: c84 measured 2 kills / NET +9.8 here against
its recorded 10 kills / +26.9. Without this harness there is no way to tell "c86 is better" from
"c84 was handed shifted inputs".

What it patches, and nothing else:
  lead_error         -> naive_lead_error                       (obs idx 30, and K_AIM/K_SOLUTION)
  gun_quality        -> recomputed on naive_lead_error          (obs idx 29)
  gun_quality_honest -> recomputed on naive_lead_error
  lay_miss_m         -> rng·sin(naive_lead_error), lay_feasible True
                        (reward.py's `in_solution` gate reads `lay_miss_m`; pre-c86 it was exactly
                         this expression, so this restores the old gate bit-for-bit)

The scripted bots need no patch: `wez_fire` and `PursuitExpert._fire` were deliberately pinned to
`naive_lead_error` when the fix landed, so the benchmark is already frozen in both worlds.

`reward.py` does `from .geometry import relative`, a direct binding, so BOTH module attributes have to
be replaced — patching `geometry.relative` alone would fix the obs and silently leave the reward
corrected. `build_observation` resolves `relative` as a module global, so it follows the geometry patch.
"""

from __future__ import annotations

import dataclasses
import math
import sys

from astrum_rl import _paths


def _install_naive_lay() -> None:
    _paths.require_astrum_env()
    from astrum_dogfight import constants as C
    from astrum_dogfight import geometry, reward

    real_relative = geometry.relative

    def naive_relative(me, bandit):
        rg = real_relative(me, bandit)
        naive = rg.naive_lead_error
        # B1: read production's `range_factor` instead of re-deriving the old linear R_EFF ramp.
        # This shim exists to change EXACTLY ONE thing — the lay — so that a re-measurement isolates
        # curriculum-86. Keeping a hardcoded copy of the range formula would have made it differ from
        # production in two ways at once the moment B1 landed, silently confounding the comparison
        # it exists to make (the [[curriculum-92-lateral-separation]] mis-set-window lesson).
        rf = rg.range_factor
        track = math.exp(-0.5 * (naive / C.LEAD_SIGMA) ** 2)
        honest = math.exp(-0.5 * (naive / geometry.sigma_track(rg.rng)) ** 2)
        return dataclasses.replace(
            rg,
            lead_error=naive,
            gun_quality=(rf * track) if rg.in_range else 0.0,
            gun_quality_honest=(rf * honest) if rg.in_range else 0.0,
            lay_miss_m=rg.rng * math.sin(naive),
            lay_feasible=True,
        )

    geometry.relative = naive_relative
    reward.relative = naive_relative
    print("[naive-lay] PRE-c86 environment active: lead_error/gun_quality/lay_miss_m from the "
          "naive nose→aim-point angle. Scripted bots unchanged (already pinned naive).", flush=True)


if __name__ == "__main__":
    _install_naive_lay()
    import _verdict_tmp

    sys.exit(_verdict_tmp.main())
