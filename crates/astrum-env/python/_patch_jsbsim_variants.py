#!/usr/bin/env python3
"""Apply the A3 / defect-D-C edits to the copied JSBSim root. Driven by `make jsbsim-variants`.

D-C's claim is that `fcs/g-load-pid` — the F-16 FLCS loop that turns a commanded Nz into elevator —
never integrates in normal flight, because its `<trigger>` is `fcs/elevator-pid-trigger`, a switch
that DEFAULTS TO 1 and only drops to 0 below vc 5 kts. JSBSim's own header is ambiguous about the
threshold ("greater than zero (true)" in one sentence, "still true (> 1)" in the next), so whether a
trigger of exactly 1 blocks the ki = 0.025 integrator cannot be settled by reading. Hence two
variants that bracket the question by measurement:

  f16nopid   the `<trigger>` line removed. If sustained Nz rises, the trigger WAS gating the
             integrator and D-C is right. If nothing moves, D-C's premise is wrong and the
             historical 4.7-5.8 g figures are the airframe, not a config bug.
  f16kp10    kp 0.3 -> 1.0, integrator untouched. Separates "the loop needs integral action" from
             "the loop is simply low-gain" — a proportional-only loop with a droop can be closed
             either way, and only the first is a bug.

Both edits are scoped to the `fcs/g-load-pid` block by regex on that exact `<pid name=...>` element,
so nothing else in the 61 KB file can move. Every substitution is asserted, because a silent no-op
here would look exactly like a null result.
"""

import re
import sys

ROOT = "/workspace/.jsbsim-variants/aircraft"
BLOCK = r"<pid name=\"fcs/g-load-pid\">.*?</pid>"


def patch(variant: str, old: str, new: str) -> None:
    path = f"{ROOT}/{variant}/{variant}.xml"
    src = open(path).read()
    m = re.search(BLOCK, src, re.S)
    if not m:
        sys.exit(f"{variant}: fcs/g-load-pid block not found — the stock f16.xml changed upstream")
    edited = m.group(0).replace(old, new)
    if edited == m.group(0):
        sys.exit(f"{variant}: {old!r} not present in the g-load-pid block — nothing patched")
    open(path, "w").write(src[: m.start()] + edited + src[m.end() :])
    print(f"{variant}: {old.strip()!r} -> {new.strip()!r}")


patch("f16nopid", "<trigger>fcs/elevator-pid-trigger</trigger>\n", "")
patch("f16kp10", "<kp> 0.3000 </kp>", "<kp> 1.0000 </kp>")
