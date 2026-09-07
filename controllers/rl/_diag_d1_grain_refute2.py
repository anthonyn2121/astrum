"""Addendum: segment of the validity-gated clause (d) exit, and the load floor over the hold."""

from __future__ import annotations

import math

from _diag_d1_grain_refute import DT, NS, cmd_bank_deg, fly, seg_of
from astrum_dogfight import bfm_detect as B

r, acts = fly(62)
tr = r.trace
hi, hj, j = 127, 189, 190
nzs = [tr[k].nz for k in range(hi, hj + 1)]
chis = [abs(math.degrees(tr[k].chi)) for k in range(hi, hj + 1)]
print(
    "HOLD steps %d-%d: nz min %.2f max %.2f g ; |chi| min %.2f max %.2f deg"
    % (hi, hj, min(nzs), max(nzs), min(chis), max(chis))
)
print(
    "  every tick nz > NZ_LOADED(%.1f)? %s ; every tick |chi| > CHI_OUT(%.0f)? %s"
    % (
        B.NZ_LOADED,
        all(n > B.NZ_LOADED for n in nzs),
        B.CHI_OUT_DEG,
        all(c > B.CHI_OUT_DEG for c in chis),
    )
)
print(
    "\nvalidity-gated clause (d): first k>%d with chi_valid AND |chi|<%.0f deg, "
    "within CHI_RETURN_S=%.1f s of t[%d]=%.3f"
    % (hj, B.CHI_RETURN_DEG, B.CHI_RETURN_S, hj, tr[hj].t)
)
for k in range(hj + 1, len(tr)):
    if tr[k].t - tr[hj].t > B.CHI_RETURN_S:
        break
    if tr[k].chi_valid and abs(tr[k].chi) < math.radians(B.CHI_RETURN_DEG):
        print(
            "  -> step %d t %.3f SEGMENT %d  |chi| %.2f deg  cmd_bank %+.2f  nz %.2f"
            % (k, tr[k].t, seg_of(k), abs(math.degrees(tr[k].chi)), cmd_bank_deg(k), tr[k].nz)
        )
        break
print("  seg1/seg2 boundary is step %d, t %.3f s" % (2 * NS // 6, 2 * NS // 6 * DT))
print("\nsteps 188-204: step seg cmd_bank phi chi chi_valid nz")
for k in range(188, 205):
    print(
        "  %4d  %d  %+8.2f  %+7.2f  %+8.2f  %d  %.2f"
        % (
            k,
            seg_of(k),
            cmd_bank_deg(k),
            math.degrees(tr[k].phi),
            math.degrees(tr[k].chi),
            int(tr[k].chi_valid),
            tr[k].nz,
        )
    )
