"""F2 addendum — is the POSITIVE CONTROL genuine, or is it another FLY-BY?

`merge:lead_turn` and `merge:coach` both score D1 = 8/8 with heavy load and a long chi hold, which
would validate the detector.  But `merge:lead_turn` also closes to 228 m at up to +424 m/s and has 8
holds rejected by clause (e) "the bandit crossed the nose", which is the fly-by signature.  If its
PASSING event is also a pass -- clause (d) satisfied by the chi = 0.0 SENTINEL that
`out_of_plane_angle` returns when the LOS is within `CHI_VALID_ATA_DEG` of the velocity vector -- then
the detector has only ever been shown to accept passes, and F2's positive control does not stand.

So: dump every PASSING D1 event in full, and classify it.

  SENTINEL-EXIT : clause (d) is met by an invalid chi (bandit dead ahead) => a pass, NOT a roll back in.
  ROLLED-IN     : clause (d) is met by a chi_valid |chi| < CHI_RETURN_DEG => a genuine roll back in.

The fly-by arm is carried as the known-positive reference for the sentinel signature.
"""

from __future__ import annotations

import math

import _diag_f2_poscontrol as F
import numpy as np
from astrum_dogfight import bfm_detect as B

_LOG: list[str] = []


def log(s: str = "") -> None:
    print(s, flush=True)
    _LOG.append(s)


def _d(x: float) -> float:
    return math.degrees(x)


def classify(tr: list[B.Sample], ev) -> dict:
    """Everything needed to tell a roll-back-in from a fly-through, for one detected D1 event."""
    i, j = ev.i0, ev.i1  # i0 = hold start, i1 = the clause-(d) exit index
    hold = [(abs(s.chi) > math.radians(B.CHI_OUT_DEG) and s.nz > B.NZ_LOADED) for s in tr]
    hold_end = i
    for a, b in B._runs(hold):
        if a == i:
            hold_end = b
    valid_exit = None
    for k in range(hold_end + 1, len(tr)):
        if tr[k].t - tr[hold_end].t > B.CHI_RETURN_S:
            break
        if tr[k].chi_valid and abs(tr[k].chi) < math.radians(B.CHI_RETURN_DEG):
            valid_exit = k
            break
    return dict(
        i0=i,
        hold_end=hold_end,
        exit=j,
        hold_s=ev.detail.get("hold_s", tr[hold_end].t - tr[i].t),
        exit_valid=bool(tr[j].chi_valid),
        exit_chi=abs(_d(tr[j].chi)),
        kind=("SENTINEL-EXIT" if not tr[j].chi_valid else "ROLLED-IN"),
        valid_exit=valid_exit,
        asp0=_d(tr[i].aspect),
        asp1=_d(tr[j].aspect),
        d_aspect=ev.d_aspect_deg,
        rng0=tr[i].rng,
        rng1=tr[j].rng,
        rng_min=min(s.rng for s in tr[i : j + 1]),
        clo0=tr[i].closure,
        clo_max=max(s.closure for s in tr[i : j + 1]),
        ata0=_d(tr[i].ata),
        ata1=_d(tr[j].ata),
        nz_peak=ev.detail.get("nz_peak", max(s.nz for s in tr[i : hold_end + 1])),
        chi_peak=ev.detail.get("chi_peak_deg", 0.0),
        crossed=len({1 if tr[k].ata_az >= 0.0 else -1 for k in range(i, j + 1)}) > 1,
    )


ARMS = ("merge:lead_turn", "merge:coach", "merge:FLYBY(d1exists)", "merge:d3b_perch0")


def main() -> None:
    log("#" * 100)
    log("#  F2 ADDENDUM — is the positive control a MANEUVER or a FLY-BY?")
    log("#" * 100)
    log(
        f"\n  CHI_VALID_ATA_DEG = {B.CHI_VALID_ATA_DEG:.1f} deg (below this ata, out_of_plane_angle "
        f"returns chi = 0.0 with chi_valid = False)\n  CHI_RETURN_DEG = {B.CHI_RETURN_DEG:.0f} deg   "
        f"CHI_RETURN_S = {B.CHI_RETURN_S:.1f} s"
    )

    d3b = np.load("/workspace/controllers/rl/runs/d3b/d3b_best.npz")["genome"]
    flyby = np.array(
        [
            1.0,
            1.0,
            0.2,
            0.582792,
            1.0,
            0.2,
            1.0,
            1.0,
            0.2,
            0.872769,
            1.0,
            0.2,
            -0.007301,
            1.0,
            0.2,
            0.312102,
            1.0,
            0.2,
        ],
        float,
    )
    by = {a["name"]: a for a in F.arms(d3b, flyby)}

    tally: dict[str, dict[str, int]] = {}
    for nm in ARMS:
        arm = by[nm]
        log("\n" + "=" * 100)
        log(f"{nm}")
        log("=" * 100)
        tally[nm] = {"SENTINEL-EXIT": 0, "ROLLED-IN": 0}
        for sd in F.SEEDS:
            r, _st = F.fly(arm, sd)
            evs = B.detect_displacement_rolls(r.trace)
            for ev in evs:
                c = classify(r.trace, ev)
                tally[nm][c["kind"]] += 1
                log(
                    f"  seed {sd:>3}  {c['kind']:<14} hold {c['hold_s']:.3f} s  "
                    f"chi_peak {c['chi_peak']:.1f}  nz_peak {c['nz_peak']:.2f} g"
                )
                log(
                    f"            aspect {c['asp0']:.1f} -> {c['asp1']:.1f} "
                    f"(d_aspect {c['d_aspect']:+.1f})   ata {c['ata0']:.1f} -> {c['ata1']:.1f}"
                )
                log(
                    f"            rng {c['rng0']:.0f} -> {c['rng1']:.0f} m (min {c['rng_min']:.0f})  "
                    f"closure {c['clo0']:+.0f} -> max {c['clo_max']:+.0f} m/s  crossed_nose "
                    f"{c['crossed']}"
                )
                log(
                    f"            clause (d) exit: |chi| {c['exit_chi']:.1f} deg  chi_valid "
                    f"{c['exit_valid']}   first VALID roll-in in window: "
                    + (f"step {c['valid_exit']}" if c["valid_exit"] is not None else "NONE")
                )
            if not evs:
                log(f"  seed {sd:>3}  (no D1 event)")

    log("\n" + "=" * 100)
    log("VERDICT — how each arm's detected D1 events were satisfied")
    log("=" * 100)
    log(f"  {'arm':<26} {'ROLLED-IN':>10} {'SENTINEL-EXIT':>14}")
    for nm in ARMS:
        log(f"  {nm:<26} {tally[nm]['ROLLED-IN']:>10} {tally[nm]['SENTINEL-EXIT']:>14}")
    log(
        "\n  A positive control REQUIRES ROLLED-IN events. If every arm is SENTINEL-EXIT then the\n"
        "  detector has only been shown to accept fly-throughs and F2's outcome (A) does NOT hold."
    )

    path = f"{F.OUT}/f2_verify.txt"
    with open(path, "w") as fh:
        fh.write("\n".join(_LOG) + "\n")
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
