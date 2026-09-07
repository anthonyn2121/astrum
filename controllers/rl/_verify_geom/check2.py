import numpy as np
from loadcsv import load

K, hdr = load("/workspace/controllers/rl/runs/d3b/replay_kill_seed62.csv")
N, _ = load("/workspace/controllers/rl/runs/d3b/replay_nokill_seed28.csv")
DT = 5.0/120.0
t = K["t_s"]
psi = np.degrees(np.unwrap(np.radians(K["psi_deg"])))
crs = np.degrees(np.unwrap(np.arctan2(K["vel_e_ms"], K["vel_n_ms"])))
dE = K["opp_pos_e_m"]-K["pos_e_m"]; dN = K["opp_pos_n_m"]-K["pos_n_m"]
lb = np.degrees(np.unwrap(np.arctan2(dE, dN)))
# Euler yaw rate from body rates
p = np.radians(K["roll_rate_dps"]); q = np.radians(K["pitch_rate_dps"]); r = np.radians(K["yaw_rate_dps"])
ph = np.radians(K["phi_deg"]); th = np.radians(K["theta_deg"])
psidot_body = np.degrees((q*np.sin(ph)+r*np.cos(ph))/np.cos(th))

print("=== hunting for 9.2 / 19.8 / 13.5 and 5.35 / 5.79 ===")
def fwd(a): 
    o=np.full(len(a),np.nan); o[:-1]=(a[1:]-a[:-1])/DT; return o
def ctr(a):
    o=np.full(len(a),np.nan); o[1:-1]=(a[2:]-a[:-2])/(2*DT); return o
ests = {"psi_fwd":fwd(psi), "psi_ctr":ctr(psi), "crs_fwd":fwd(crs), "crs_ctr":ctr(crs),
        "psidot_body":psidot_body, "lb_fwd":fwd(lb), "lb_ctr":ctr(lb)}
for s in (12, 18, 20, 24, 32, 36, 48):
    print(f"t={t[s]:.3f} step {s:3d} " + " ".join(f"{k}={v[s]:7.2f}" for k,v in ests.items()))
# where does psi rate hit 9.2 ?
for k in ("psi_fwd","psi_ctr","psidot_body","crs_ctr"):
    v = ests[k]
    for tgt in (9.2, 19.8, 13.5):
        idx = np.nanargmin(np.abs(v[:60]-tgt))
        print(f"  {k}: closest to {tgt} = {v[idx]:.3f} at step {idx} t={t[idx]:.3f}")
print("\n LOS brg rate range over candidate windows:")
for lo,hi in ((0.5,2.0),(0.75,2.0),(0.5,1.5),(1.0,2.0)):
    m=(t>=lo)&(t<=hi); vv=ests["lb_ctr"][m]; vv=vv[~np.isnan(vv)]
    vf=ests["lb_fwd"][m]; vf=vf[~np.isnan(vf)]
    print(f"  [{lo},{hi}] ctr {vv.min():.3f}..{vv.max():.3f}  fwd {vf.min():.3f}..{vf.max():.3f}")
# chunk-mean estimator
print("\n chunk-mean psi rate over 0.5 s blocks:")
for lo in (0.0,0.5,1.0,1.5,2.0):
    i0=int(round(lo/DT)); i1=int(round((lo+0.5)/DT))
    print(f"  [{lo},{lo+0.5}] psi {(psi[i1]-psi[i0])/0.5:7.2f}  crs {(crs[i1]-crs[i0])/0.5:7.2f}  lb {(lb[i1]-lb[i0])/0.5:7.2f}")

print("\n=== WHERE IS 71.88 ? ===")
for nm, a in (("lb", lb), ("psi", psi), ("crs", crs)):
    idx = np.nanargmin(np.abs(a-71.88)); print(f"  {nm}: closest to 71.88 = {a[idx]:.4f} at step {idx} t={t[idx]:.3f}; final {a[-1]:.4f}")
print("  psi_deg RAW final %.4f ; psi_deg raw max %.4f" % (K["psi_deg"][-1], K["psi_deg"].max()))

print("\n=== ata / lead_error / alpha_req EXTREMES vs cited endpoints ===")
for nm in ("ata_deg","lead_error_deg","alpha_req_deg","lay_miss_m","cpa_miss_m","aspect_deg"):
    a=K[nm]; print(f"  {nm:15s} first {a[0]:8.2f} min {a.min():8.2f}@{int(np.argmin(a)):3d} last {a[-1]:8.2f}")
print("  ata rises from step0 to max %.3f at step %d (t=%.3f) BEFORE falling" % (K["ata_deg"][:40].max(), int(np.argmax(K["ata_deg"][:40])), t[int(np.argmax(K["ata_deg"][:40]))]))
print("  steps where ata increased: %d of 172" % int((np.diff(K["ata_deg"])>0).sum()))
lo = np.nanargmin(np.abs(K["alpha_req_deg"]-10.1)); print("  alpha_req closest to 10.1 = %.3f at step %d t=%.3f" % (K["alpha_req_deg"][lo], lo, t[lo]))
lo = np.nanargmin(np.abs(K["lead_error_deg"]-2.6)); print("  lead_error closest to 2.6 = %.3f at step %d t=%.3f" % (K["lead_error_deg"][lo], lo, t[lo]))

print("\n=== NON-KILL SEED 28: identity through step 172, then what? ===")
cols = ["phi_deg","theta_deg","psi_deg","nz_g","alt_m","rng_m","ata_deg","cpa_miss_m","pos_e_m","pos_n_m","pos_u_m","opp_pos_e_m","opp_pos_n_m"]
mx = {c: float(np.abs(K[c]-N[c][:173]).max()) for c in cols}
print("  max|K-N| over steps 0..172:", {k: f"{v:.3e}" for k,v in mx.items()})
cN = N["cpa_miss_m"]
print("  seed28 cpa_miss: step172 %.2f ; min %.2f @step %d ; max after 172 %.1f @step %d ; final %.1f"
      % (cN[172], cN.min(), int(np.argmin(cN)), cN[173:].max(), 173+int(np.argmax(cN[173:])), cN[-1]))
dN2 = np.diff(cN); print("  seed28 increasing transitions: %d of %d (%.1f%%)" % ((dN2>0).sum(), len(dN2), 100*(dN2>0).mean()))
print("  seed28 increasing transitions BEFORE step 172: %d of 172" % (dN2[:172]>0).sum())
print("  seed28 rng: min %.1f @step %d ; final %.1f ; alt first %.1f last %.1f (dalt %.1f)"
      % (N["rng_m"].min(), int(np.argmin(N["rng_m"])), N["rng_m"][-1], N["alt_m"][0], N["alt_m"][-1], N["alt_m"][-1]-N["alt_m"][0]))
print("  seed28 opp_health final %.1f ; own health final %.1f" % (N["opp_health"][-1], N["health"][-1]))
# LOS bearing on seed 28 full
dE2=N["opp_pos_e_m"]-N["pos_e_m"]; dN3=N["opp_pos_n_m"]-N["pos_n_m"]
lb2=np.degrees(np.unwrap(np.arctan2(dE2,dN3)))
print("  seed28 LOS brg: step0 %.2f step172 %.2f final %.2f ; NET %+.2f" % (lb2[0], lb2[172], lb2[-1], lb2[-1]-lb2[0]))
print("\n=== ALTITUDE PRICE of the hold (kill seed) ===")
print("  alt %.1f -> %.1f  (dalt %.1f m) over 7.208 s" % (K["alt_m"][0], K["alt_m"][-1], K["alt_m"][-1]-K["alt_m"][0]))
print("  seg0 (0..95): alt %.1f -> %.1f (%.1f m) ; seg1 (96..172): %.1f -> %.1f (%.1f m)"
      % (K["alt_m"][0], K["alt_m"][95], K["alt_m"][95]-K["alt_m"][0], K["alt_m"][96], K["alt_m"][-1], K["alt_m"][-1]-K["alt_m"][96]))
print("  Es %.1f -> %.1f ; es_rel %.1f -> %.1f" % (K["es_m"][0], K["es_m"][-1], K["es_rel_m"][0], K["es_rel_m"][-1]))
nzl = K["nz_g"]>3.0
print("  loaded (nz>3) steps: %d of 173 = %.3f s ; longest run:" % (nzl.sum(), nzl.sum()*DT), end=" ")
best=cur=0
for b in nzl:
    cur = cur+1 if b else 0
    best=max(best,cur)
print("%d steps = %.3f s" % (best, best*DT))
