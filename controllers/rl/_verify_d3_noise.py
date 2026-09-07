"""Is the +12.4 m 'build' above the tick-to-tick noise floor of cpa_miss? And what IS the
rejected 305-326 hold, in closure terms?  Pure stdlib."""

import csv

P = "/home/anthony_nguyen/aca/astrum/controllers/rl/runs/d3b/replay_nokill_seed28.csv"
with open(P) as fh:
    R = list(csv.DictReader(fh))


def g(k, c):
    return float(R[k][c])


print("=== cpa_miss tick-by-tick THROUGH the accepted hold 499-518 (+ 4 ticks either side) ===")
print(
    f"{'step':>4} {'t':>7} {'seg':>3} {'cpa':>9} {'dcpa':>7} {'nz':>5} {'|chi|':>7} "
    f"{'ata_az':>7} {'rng':>7} {'clos':>7} {'alt':>7} {'lethal':>6}"
)
for k in range(495, 524):
    d = g(k, "cpa_miss_m") - g(k - 1, "cpa_miss_m") if k else 0.0
    mark = " <" if k in (499, 518) else ""
    print(
        f"{k:>4} {g(k, 't_s'):>7.3f} {int(g(k, 'seg_idx')):>3} {g(k, 'cpa_miss_m'):>9.2f} "
        f"{d:>+7.2f} {g(k, 'nz_g'):>5.2f} {abs(g(k, 'chi_deg')):>7.1f} {g(k, 'ata_az_deg'):>+7.1f} "
        f"{g(k, 'rng_m'):>7.0f} {g(k, 'closure_ms'):>+7.1f} {g(k, 'alt_enu_m'):>7.0f} "
        f"{int(g(k, 'lethal')):>6}{mark}"
    )


# noise floor: |d cpa_miss| per tick over a quiet stretch and over the hold
def stats(lo, hi, lab):
    ds = [abs(g(k, "cpa_miss_m") - g(k - 1, "cpa_miss_m")) for k in range(lo + 1, hi + 1)]
    ds_s = sorted(ds)
    net = g(hi, "cpa_miss_m") - g(lo, "cpa_miss_m")
    tot = sum(ds)
    print(
        f"  {lab:<26} ticks={hi - lo:>3}  net={net:>+8.2f} m  "
        f"sum|d|={tot:>8.2f} m  mean|d|={tot / len(ds):>6.3f}  "
        f"median|d|={ds_s[len(ds_s) // 2]:>6.3f}  max|d|={ds_s[-1]:>6.3f}"
    )
    return net, tot


print("\n=== per-tick |d cpa_miss| : the build vs the path length that produced it ===")
stats(499, 518, "accepted hold 499-518")
stats(305, 326, "rejected hold 305-326")
stats(470, 499, "30 ticks just before")
stats(540, 575, "35 ticks near the end")

print("\n  monotonicity inside the accepted hold:")
seq = [g(k, "cpa_miss_m") for k in range(499, 519)]
ups = sum(1 for a, b in zip(seq, seq[1:]) if b > a)
print(
    f"    rising ticks {ups}/{len(seq) - 1};  min {min(seq):.2f} at "
    f"{499 + seq.index(min(seq))}, max {max(seq):.2f} at {499 + seq.index(max(seq))}, "
    f"endpoints {seq[0]:.2f} -> {seq[-1]:.2f}"
)
print(f"    is the endpoint the max? {abs(max(seq) - seq[-1]) < 1e-9}")
print("  monotonicity inside the rejected hold 305-326:")
seq2 = [g(k, "cpa_miss_m") for k in range(305, 327)]
ups2 = sum(1 for a, b in zip(seq2, seq2[1:]) if b > a)
print(f"    rising ticks {ups2}/{len(seq2) - 1};  endpoints {seq2[0]:.2f} -> {seq2[-1]:.2f}")

print("\n=== what the REJECTED hold 305-326 is, in closure/range terms ===")
for k in (305, 315, 326, 336, 350, 375, 402):
    print(
        f"  step {k:>3} t {g(k, 't_s'):>6.3f} seg {int(g(k, 'seg_idx'))} "
        f"rng {g(k, 'rng_m'):>6.0f} closure {g(k, 'closure_ms'):>+8.1f} "
        f"cpa {g(k, 'cpa_miss_m'):>8.1f} aspect {g(k, 'aspect_deg'):>6.1f} "
        f"nz {g(k, 'nz_g'):>5.2f} alt {g(k, 'alt_enu_m'):>6.0f}"
    )

print("\n=== nz / |chi| ranges: HOLD span vs EVENT span (audit A labels these 'over the hold') ===")
for lab, lo, hi in (("hold 499-518", 499, 518), ("event 499-575", 499, 575)):
    nz = [g(k, "nz_g") for k in range(lo, hi + 1)]
    ch = [abs(g(k, "chi_deg")) for k in range(lo, hi + 1)]
    print(
        f"  {lab}: nz {min(nz):.2f}-{max(nz):.2f} g   |chi| {min(ch):.1f}-{max(ch):.1f} deg   "
        f"alt {g(lo, 'alt_enu_m'):.0f}->{g(hi, 'alt_enu_m'):.0f} ({g(hi, 'alt_enu_m') - g(lo, 'alt_enu_m'):+.0f} m)"
    )

print("\n=== is the CASH truncated by the episode end? ===")
print(
    f"  last row step {int(g(len(R) - 1, 'step'))} t {g(len(R) - 1, 't_s'):.3f} s; exit_idx=575; rows={len(R)}"
)
print(
    f"  CHI_RETURN_S window from t(518)={g(518, 't_s'):.3f} would run to "
    f"{g(518, 't_s') + 4.0:.3f} s, i.e. step {518 + 96} -- the trace ends at step 575"
)
print(
    f"  so the 'cash' minimum is the FINAL tick of the episode: cpa {g(575, 'cpa_miss_m'):.1f} m, "
    f"rng {g(575, 'rng_m'):.0f} m, lethal={int(g(575, 'lethal'))}"
)

print("\n=== segment membership (seg_of = (step*6)//576) ===")
for k in (88, 108, 305, 326, 499, 518, 575):
    print(f"  step {k:>3}: seg {int(g(k, 'seg_idx'))}  (kill seed 62 ended at step 172, seg 1)")
print(
    f"\n  cmd at the accepted hold: bank {g(499, 'cmd_bank_deg'):.2f} deg  "
    f"load {g(499, 'cmd_load'):.4f}  thr {g(499, 'cmd_thr'):.4f}"
)
print(
    f"  cmd at the rejected hold: bank {g(305, 'cmd_bank_deg'):.2f} deg  "
    f"load {g(305, 'cmd_load'):.4f}  thr {g(305, 'cmd_thr'):.4f}"
)
