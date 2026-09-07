import numpy as np
from loadcsv import load
K, _ = load("/workspace/controllers/rl/runs/d3b/replay_kill_seed62.csv")
DT=5/120; t=K["t_s"]
def wrap(a): return (a+180.0)%360.0-180.0
psi = K["psi_deg"]; 
crs = np.degrees(np.arctan2(K["vel_e_ms"], K["vel_n_ms"]))
lb  = np.degrees(np.arctan2(K["opp_pos_e_m"]-K["pos_e_m"], K["opp_pos_n_m"]-K["pos_n_m"]))
h_psi = wrap(lb-psi); h_crs = wrap(lb-crs)
print("HORIZONTAL nose-to-LOS (deg, + = bandit right of nose = LAG for a right turn)")
print("  lb-psi   : step0 %+.2f min %+.2f@%d max %+.2f@%d last %+.2f ; steps<0: %d of 173"
      % (h_psi[0], h_psi.min(), int(np.argmin(h_psi)), h_psi.max(), int(np.argmax(h_psi)), h_psi[-1], (h_psi<0).sum()))
print("  lb-course: step0 %+.2f min %+.2f@%d max %+.2f@%d last %+.2f ; steps<0: %d of 173"
      % (h_crs[0], h_crs.min(), int(np.argmin(h_crs)), h_crs.max(), int(np.argmax(h_crs)), h_crs[-1], (h_crs<0).sum()))
print("\n  step    t    lb-psi  lb-crs   ata   alpha_req  rng   nz  phi   gamma")
for s in (0,12,24,36,48,72,84,96,120,140,156,165,172):
    print("  %4d %6.3f %+8.2f %+8.2f %6.2f %8.2f %7.1f %5.2f %6.1f %7.2f"
          % (s,t[s],h_psi[s],h_crs[s],K["ata_deg"][s],K["alpha_req_deg"][s],K["rng_m"][s],K["nz_g"][s],K["phi_deg"][s],K["gamma_deg"][s]))
print("\nfirst step alpha_req <= 16.7 (measured F-16 alpha ceiling): ", end="")
i=int(np.argmax(K["alpha_req_deg"]<=16.7)); print("%d (t=%.3f rng=%.1f)"%(i,t[i],K["rng_m"][i]))
print("alpha_req at step 84 = %.2f  (nose 'inside 11 deg' but shot needs %.1f deg of alpha)"%(K["alpha_req_deg"][84],K["alpha_req_deg"][84]))
print("\n0.5 s BLOCK MEANS (deg/s), all three own-motion estimators vs LOS bearing:")
psiu=np.degrees(np.unwrap(np.radians(K["psi_deg"]))); crsu=np.degrees(np.unwrap(np.arctan2(K["vel_e_ms"],K["vel_n_ms"]))); lbu=np.degrees(np.unwrap(np.radians(lb)))
print("  block        psi_dot  course_dot   LOSbrg_dot   verdict")
for lo in (0.0,0.5,1.0,1.5,2.0,2.5,3.0):
    i0=int(round(lo/DT)); i1=int(round((lo+0.5)/DT))
    if i1>=len(t): break
    a=(psiu[i1]-psiu[i0])/0.5; b=(crsu[i1]-crsu[i0])/0.5; c=(lbu[i1]-lbu[i0])/0.5
    print("  [%.1f,%.1f]  %8.2f %11.2f %12.2f   psi %s LOS ; course %s LOS"
          %(lo,lo+0.5,a,b,c,">" if a>c else "<", ">" if b>c else "<"))
