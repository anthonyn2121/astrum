import numpy as np
from loadcsv import load

K, hdr = load("/workspace/controllers/rl/runs/d3b/replay_kill_seed62.csv")
N, _ = load("/workspace/controllers/rl/runs/d3b/replay_nokill_seed28.csv")
print("rows kill", len(K["step"]), "rows nokill", len(N["step"]))
DT = 5.0 / 120.0
print("DT %.9f" % DT)

print("\n=== A. CITED POINT VALUES (kill seed 62) ===")
for s in (0, 84, 156, 171, 172):
    i = s
    print(f"step {s:4d} t={K['t_s'][i]:.3f} rng={K['rng_m'][i]:8.1f} ata={K['ata_deg'][i]:7.2f} "
          f"ata_az={K['ata_az_deg'][i]:8.2f} aspect={K['aspect_deg'][i]:7.2f} "
          f"lead_err={K['lead_error_deg'][i]:7.2f} alpha_req={K['alpha_req_deg'][i]:7.2f} "
          f"cpa={K['cpa_miss_m'][i]:9.2f} lay={K['lay_miss_m'][i]:8.2f} nz={K['nz_g'][i]:5.2f} "
          f"phi={K['phi_deg'][i]:7.1f} alt={K['alt_m'][i]:7.1f} gam={K['gamma_deg'][i]:7.2f} fire={K['fire'][i]:.0f}")

ata = K["ata_deg"]
first11 = int(np.argmax(ata <= 11.0))
print("\nfirst step with ata<=11: %d (t=%.3f rng=%.1f ata=%.3f)" % (first11, K['t_s'][first11], K['rng_m'][first11], ata[first11]))
print("min ata %.3f at step %d ; ata at 84 = %.4f" % (ata.min(), int(np.argmin(ata)), ata[84]))

print("\n=== B. cpa_miss MONOTONICITY (kill) ===")
c = K["cpa_miss_m"]
d = np.diff(c)
inc = np.where(d > 0)[0]
print("transitions %d ; increasing %d ; indices %s" % (len(d), len(inc), inc.tolist()))
for i in inc:
    print("   %d->%d : %.4f -> %.4f (%+.4f)" % (i, i+1, c[i], c[i+1], d[i]))
print("cpa first %.2f last %.2f" % (c[0], c[-1]))

print("\n=== C. LOS BEARING (inertial, from pos_e/pos_n) ===")
def bearing(D):
    dE = D["opp_pos_e_m"] - D["pos_e_m"]
    dN = D["opp_pos_n_m"] - D["pos_n_m"]
    return np.degrees(np.unwrap(np.arctan2(dE, dN)))
lb = bearing(K)
print("LOS brg step0 %.4f  step84 %.4f  step172 %.4f  NET %+.4f" % (lb[0], lb[84], lb[-1], lb[-1]-lb[0]))
print("LOS brg min %.3f max %.3f" % (lb.min(), lb.max()))
psi = np.degrees(np.unwrap(np.radians(K["psi_deg"])))
crs = np.degrees(np.unwrap(np.arctan2(K["vel_e_ms"], K["vel_n_ms"])))
print("own psi    step0 %.4f step172 %.4f NET %+.4f" % (psi[0], psi[-1], psi[-1]-psi[0]))
print("own course step0 %.4f step172 %.4f NET %+.4f" % (crs[0], crs[-1], crs[-1]-crs[0]))
print("total |d(LOS brg)| summed = %.3f ; total |d psi| summed = %.3f" % (np.abs(np.diff(lb)).sum(), np.abs(np.diff(psi)).sum()))

print("\n=== D. RATES (central diff) t=0.5..2.0 ===")
def rate(a):
    r = np.full(len(a), np.nan)
    r[1:-1] = (a[2:] - a[:-2]) / (2*DT)
    return r
pr, cr, lr = rate(psi), rate(crs), rate(lb)
t = K["t_s"]
print(f"{'t':>6}{'step':>6}{'psiRate':>9}{'crsRate':>9}{'LOSbrgR':>9}{'losAz':>8}{'losEl':>8}{'losTot':>8}{'ata':>8}{'gamma':>8}{'nz':>6}{'phi':>8}")
for s in list(range(8, 52, 2)):
    print(f"{t[s]:6.3f}{s:6d}{pr[s]:9.2f}{cr[s]:9.2f}{lr[s]:9.2f}{K['los_rate_az_dps'][s]:8.2f}"
          f"{K['los_rate_el_dps'][s]:8.2f}{K['los_rate_dps'][s]:8.2f}{K['ata_deg'][s]:8.2f}"
          f"{K['gamma_deg'][s]:8.2f}{K['nz_g'][s]:6.2f}{K['phi_deg'][s]:8.1f}")
m = (t >= 0.5) & (t <= 2.0)
print("\nwindow t in [0.5,2.0]: n=%d steps %d..%d" % (m.sum(), int(np.argmax(m)), int(len(t)-1-np.argmax(m[::-1]))))
for nm, a in (("psiRate", pr), ("courseRate", cr), ("LOSbrgRate", lr)):
    v = a[m]; v = v[~np.isnan(v)]
    print("  %-11s min %8.3f max %8.3f mean %8.3f  first %8.3f last %8.3f" % (nm, v.min(), v.max(), v.mean(), v[0], v[-1]))
for nm in ("los_rate_az_dps", "los_rate_el_dps", "los_rate_dps"):
    v = K[nm][m]
    print("  %-16s min %8.3f max %8.3f mean %8.3f" % (nm, v.min(), v.max(), v.mean()))
V = np.stack([K["vel_e_ms"], K["vel_n_ms"], K["vel_u_ms"]], 1)
Vh = V / np.linalg.norm(V, axis=1, keepdims=True)
fp = np.full(len(t), np.nan)
fp[1:-1] = np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i', Vh[2:], Vh[:-2]), -1, 1)))/(2*DT)
print("  TOTAL flight-path turn rate window: min %.2f max %.2f | whole ep max %.2f" % (np.nanmin(fp[m]), np.nanmax(fp[m]), np.nanmax(fp)))
print("  los_rate_dps whole ep: min %.2f max %.2f" % (K['los_rate_dps'].min(), K['los_rate_dps'].max()))
