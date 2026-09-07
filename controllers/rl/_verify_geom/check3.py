import numpy as np
from loadcsv import load
K, _ = load("/workspace/controllers/rl/runs/d3b/replay_kill_seed62.csv")
N, _ = load("/workspace/controllers/rl/runs/d3b/replay_nokill_seed28.csv")
DT = 5/120; t = K["t_s"]
psi = np.degrees(np.unwrap(np.radians(K["psi_deg"])))
crs = np.degrees(np.unwrap(np.arctan2(K["vel_e_ms"], K["vel_n_ms"])))
lb = np.degrees(np.unwrap(np.arctan2(K["opp_pos_e_m"]-K["pos_e_m"], K["opp_pos_n_m"]-K["pos_n_m"])))

print("=== when does the HEADING (not the roll command) actually start moving? ===")
print(" ail[0]=%.3f elev[0]=%.3f ; first step |psi-psi0|>1deg: %d (t=%.3f); >5deg: %d (t=%.3f)"
      % (K["ail"][0], K["elev"][0], int(np.argmax(np.abs(psi-psi[0])>1)), t[int(np.argmax(np.abs(psi-psi[0])>1))],
         int(np.argmax(np.abs(psi-psi[0])>5)), t[int(np.argmax(np.abs(psi-psi[0])>5))]))
print(" first step |crs-crs0|>1deg: %d (t=%.3f) ; first nz>3: %d (t=%.3f) ; first |phi|>90: %d (t=%.3f)"
      % (int(np.argmax(np.abs(crs-crs[0])>1)), t[int(np.argmax(np.abs(crs-crs[0])>1))],
         int(np.argmax(K["nz_g"]>3)), t[int(np.argmax(K["nz_g"]>3))],
         int(np.argmax(np.abs(K["phi_deg"])>90)), t[int(np.argmax(np.abs(K["phi_deg"])>90))]))
print(" cumulative to t=0.5s: dpsi %+.2f  dcrs %+.2f  dLOSbrg %+.2f  -> ata %.2f->%.2f"
      % (psi[12]-psi[0], crs[12]-crs[0], lb[12]-lb[0], K["ata_deg"][0], K["ata_deg"][12]))
print(" cumulative to t=1.0s: dpsi %+.2f  dcrs %+.2f  dLOSbrg %+.2f  -> ata %.2f"
      % (psi[24]-psi[0], crs[24]-crs[0], lb[24]-lb[0], K["ata_deg"][24]))
print(" WHOLE EP:            dpsi %+.2f  dcrs %+.2f  dLOSbrg %+.2f  => nose DEFICIT vs LOS: psi %+.2f, course %+.2f"
      % (psi[-1]-psi[0], crs[-1]-crs[0], lb[-1]-lb[0], (psi[-1]-psi[0])-(lb[-1]-lb[0]), (crs[-1]-crs[0])-(lb[-1]-lb[0])))

print("\n=== LEAD vs LAG in the horizontal plane (ata_az sign; + = bandit right of nose = LAG while turning right) ===")
az = K["ata_az_deg"]
print(" ata_az: step0 %+.2f max %+.2f@%d min %+.2f@%d last %+.2f" % (az[0], az.max(), int(np.argmax(az)), az.min(), int(np.argmin(az)), az[-1]))
zc = np.where(np.diff(np.sign(az))!=0)[0]
print(" sign changes at steps:", zc.tolist(), "-> t:", [round(float(t[i]),3) for i in zc])
print(" steps with ata_az<0 (nose AHEAD of LOS = lead): %d of 173 = %.1f%%" % ((az<0).sum(), 100*(az<0).mean()))
laz = K["lead_az_deg"]
print(" lead_az (nose vs REQUIRED boresight): step0 %+.2f min %+.2f max %+.2f last %+.2f ; steps with |lead_az|<5: %d"
      % (laz[0], laz.min(), laz.max(), laz[-1], (np.abs(laz)<5).sum()))
print("\n  step   t     ata    ata_az  lead_az  lead_el alpha_req  aspect   rng    nz   phi   gamma  cpa")
for s in (0,12,24,48,72,84,96,120,140,156,164,165,168,172):
    print("  %4d %6.3f %7.2f %+8.2f %+8.2f %+8.2f %8.2f %8.2f %7.1f %5.2f %6.1f %7.2f %8.1f"
          % (s, t[s], K["ata_deg"][s], az[s], laz[s], K["lead_el_deg"][s], K["alpha_req_deg"][s],
             K["aspect_deg"][s], K["rng_m"][s], K["nz_g"][s], K["phi_deg"][s], K["gamma_deg"][s], K["cpa_miss_m"][s]))

print("\n=== is the BANDIT doing the converging? (aspect = his nose onto me) ===")
print(" aspect: %.2f -> %.2f (min %.2f, max %.2f@%d)" % (K["aspect_deg"][0], K["aspect_deg"][-1], K["aspect_deg"].min(), K["aspect_deg"].max(), int(np.argmax(K["aspect_deg"]))))
opb = np.degrees(np.unwrap(np.radians(K["opp_psi_deg"])))
print(" opp psi: %.2f -> %.2f NET %+.2f ; opp nz max %.2f ; opp |nz|>3 steps %d"
      % (opb[0], opb[-1], opb[-1]-opb[0], K["opp_nz_g"].max(), (K["opp_nz_g"]>3).sum()))
print(" reciprocal LOS bearing (bandit->me) net: %+.2f" % ((lb[-1]-lb[0])))
print(" opp heading deficit vs LOS: %+.2f" % ((opb[-1]-opb[0])-(lb[-1]-lb[0])))

print("\n=== hold price: longest nz>3 run ===")
nzl = K["nz_g"]>3
runs=[]; s=None
for i,b in enumerate(nzl):
    if b and s is None: s=i
    if not b and s is not None: runs.append((s,i-1)); s=None
if s is not None: runs.append((s,len(nzl)-1))
for a,b in runs:
    print("  steps %d-%d  %.3f s  alt %.1f->%.1f (%+.1f m)  Es %+.1f  cpa %.1f->%.1f  |chi|max %.1f"
          % (a,b,(b-a+1)*DT, K["alt_m"][a], K["alt_m"][b], K["alt_m"][b]-K["alt_m"][a],
             K["es_m"][b]-K["es_m"][a], K["cpa_miss_m"][a], K["cpa_miss_m"][b], np.abs(K["chi_deg"][a:b+1]).max()))
