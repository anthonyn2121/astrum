"""CONTROL: does the 'pulse then |ail|~0 hold' signature discriminate the d3b kill genome
from RANDOM genomes in the same parameterisation? Same cell (merge1500), same sim seed (62)."""

import sys

sys.path.insert(0, "/workspace/controllers/rl/vendor")
sys.path.insert(0, "/workspace/crates/astrum-env/python")
sys.path.insert(0, "/workspace/controllers/rl")
import _diag_true_mpc as M
import numpy as np
from astrum_dogfight import bfm_detect as B

DT = M.CONTROL_DT_S
N = 576
cell = M.CELLS["merge1500"]
gm = M.Genome(6, 3, "bank", True)
env, base = M.build_env(cell, 62, 24.0)

best = np.load("/workspace/controllers/rl/runs/d3b/d3b_best.npz", allow_pickle=True)
genomes = [("d3b_best", best["genome"].astype(float))]
rng = np.random.default_rng(4242)
for i in range(6):
    genomes.append((f"random{i}", rng.uniform(-1.0, 1.0, 18)))


def runs_of(mask):
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


print(
    f"{'genome':10s} {'window':9s} {'steps':>5} {'event':>12} {'nPuls':>6} {'lPuls_s':>9} {'nQHold':>8} "
    f"{'lQHold_s':>9} {'bankHold_s':>10} {'bankHold_dalt':>13} {'D1':>3} {'D2':>3} {'D3':>3} {'kill':>4}"
)
for nm, g in genomes:

    def ctl(step, me, bd, rel, g=g):
        return gm.act(g, step, N, me)

    r = M.roll_once(env, base, ctl, N, keep=True)
    for lbl, tr in (("full", r.trace), ("first173", r.trace[:173])):
        ail = np.array([s.ail for s in tr])
        nz = np.array([s.nz for s in tr])
        pr = runs_of(list(np.abs(ail) > 0.999))
        qr = runs_of([bool(a and b) for a, b in zip(np.abs(ail) < 0.10, nz > 3.0)])
        c = B.commitment(tr)
        d = B.detect_all(tr)
        lp = max(((j - i + 1) * DT for i, j in pr), default=0.0)
        lq = max(((j - i + 1) * DT for i, j in qr), default=0.0)
        print(
            f"{nm:10s} {lbl:9s} {len(tr):5d} {str(r.event):>12} {len(pr):6d} {lp:9.3f} {len(qr):8d} "
            f"{lq:9.3f} {c['longest_bank_hold_s']:10.3f} {c['bank_hold_dalt_m']:13.1f} "
            f"{len(d['displacement_rolls']):3d} {len(d['yoyos']):3d} {len(d['turning_room']):3d} {r.kill:4d}"
        )
