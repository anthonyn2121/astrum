import sys

import numpy as np

sys.path.insert(0, "/workspace/controllers/rl")
import _diag_true_mpc as M
from astrum_dogfight import bfm_detect as B

d = np.load("/workspace/controllers/rl/runs/d3b/d3b_best.npz", allow_pickle=True)
g = np.asarray(d["genome"], float)
gm = M.Genome(6, 3, "bank", True)
env, base = M.build_env(M.CELLS["merge1500"], 62, 576 * M.CONTROL_DT_S)
r = M.roll_once(env, base, lambda s, me, bd, rel: gm.act(g, s, 576, me), 576, keep=False)
env.close()
s, comp = M.score(r)
det = B.detect_all(r.trace)
print(
    f"INCUMBENT (npz genome) seed 62: steps {r.steps} event {r.event} kill {r.kill} "
    f"hp {r.hp_dealt:.0f} tleth {r.tleth_s:.3f} score {s:+.3f}"
)
print(
    f"  D1 {len(det['displacement_rolls'])} D2 {len(det['yoyos'])} "
    f"D2c {len(det['yoyos_clean'])} D3 {len(det['turning_room'])} "
    f"dalt {r.trace[-1].alt - r.trace[0].alt:+.0f} m"
)
print(f"  components {comp}")
