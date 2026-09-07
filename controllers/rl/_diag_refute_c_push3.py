"""Audit the two D2 events the PUSH arm scored, and price the roll INSIDE the yo-yo it belongs to.

Three things the verdict needs and the coarse sweep did not print:
  (1) the full Event.detail of the push arm's D2 (is it `clean`? `failed_exit`? `failed_aspect`,
      i.e. the c70 zoom-away the detector exists to catch?), plus its altitude price and nz peak;
  (2) the measured cost of the return ROLL in the pull-only arm at the same T_up -- wall time from
      the command edge to phi >= 110 deg, and the gamma the jet gains/loses while it rolls (the roll
      is not dead time, so pricing it as pure lost window is an overcharge);
  (3) how many rolls the detector's exit clause actually requires (it reads gamma only:
      bfm_detect.py:427-433), checked by printing phi at the exit index.
"""

from __future__ import annotations

import math

import _diag_d3b_express as E
import _diag_refute_c_push as P
from astrum_dogfight import bfm_detect as B

_LOG: list[str] = []


def log(s: str = "") -> None:
    print(s, flush=True)
    _LOG.append(s)


def main() -> None:
    n_steps = int(round(9.0 / E.DT))
    log("#" * 100)
    log("#  CAUSE C -- audit of the PUSH arm's D2 events and the roll's true price")
    log("#" * 100)

    for t_up in (2.6, 2.7):
        for name, phases, gate in (
            ("B PUSH0", [(t_up, 0.0, 1.0, 1.0), (30.0, 0.0, -1.0, 1.0)], True),
            ("A PULL115", [(t_up, 0.0, 1.0, 1.0), (30.0, 115.0, 1.0, 1.0)], True),
        ):
            ctl = P.Sched2(phases, gate=gate)
            r, rows = E.fly(ctl, n_steps=n_steps)
            det = B.detect_all(r.trace)
            log(
                f"\n  {name}  T_up {t_up:.2f}s  steps {r.steps}  event {r.event}  dalt {r.dalt_m:+.0f} m"
            )
            for ev in det["yoyos"]:
                log(
                    f"      D2 {ev.kind}  i0 {ev.i0} i1 {ev.i1}  t {r.trace[ev.i0].t:.3f}->{r.trace[ev.i1].t:.3f}"
                )
                log(
                    f"         d_aspect_deg {ev.d_aspect_deg:+.1f}  d_rng_m {ev.d_rng_m:+.0f}  "
                    f"d_es_m {ev.d_es_m:+.0f}  duration_s {ev.duration_s:.3f}"
                )
                log(
                    f"         alt {rows[ev.i0]['alt']:.0f} -> {rows[ev.i1]['alt']:.0f} m "
                    f"({rows[ev.i1]['alt'] - rows[ev.i0]['alt']:+.0f} m across the event); "
                    f"nz peak entry->exit {max(rows[k]['nz'] for k in range(ev.i0, ev.i1 + 1)):.2f} g"
                )
                for k, v in sorted(ev.detail.items()):
                    log(f"         detail {k:<16} {v}")
                log(
                    f"         phi at entry {rows[ev.i0]['phi']:+.1f} deg, at exit {rows[ev.i1]['phi']:+.1f} deg "
                    f"-> rolls inside the window: {'0' if abs(rows[ev.i1]['phi']) < 45 and abs(rows[ev.i0]['phi']) < 45 else 'see phi'}"
                )
            if not det["yoyos"]:
                log("      D2 0 events")
            # (2) price the return roll for the pull arm
            if name.startswith("A"):
                k0 = min(range(len(rows)), key=lambda k: abs(rows[k]["t"] - t_up))
                k110 = None
                for k in range(k0, len(rows)):
                    if abs(rows[k]["phi"]) >= 110.0:
                        k110 = k
                        break
                if k110 is not None:
                    dg = math.degrees(r.trace[k110].gamma - r.trace[k0].gamma)
                    log(
                        f"      RETURN ROLL: cmd edge t {rows[k0]['t']:.3f} phi {rows[k0]['phi']:+.1f} -> "
                        f"phi>=110 at t {rows[k110]['t']:.3f} (roll took {rows[k110]['t'] - rows[k0]['t']:.3f} s); "
                        f"gamma {math.degrees(r.trace[k0].gamma):+.2f} -> {math.degrees(r.trace[k110].gamma):+.2f} "
                        f"({dg:+.2f} deg WHILE rolling)"
                    )
                    seq = " ".join(f"{rows[k]['elev']:+.2f}" for k in range(k0, k110 + 1, 3))
                    log(f"      elev over the roll: {seq}")
    path = f"{E.OUT}/refute_c_push3.txt"
    with open(path, "w") as fh:
        fh.write("\n".join(_LOG) + "\n")
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
