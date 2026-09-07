import csv
import numpy as np

def load(path):
    with open(path) as f:
        rd = csv.reader(f)
        hdr = next(rd)
        rows = [r for r in rd]
    arr = np.array([[float(x) for x in r] for r in rows], dtype=float)
    return {h: arr[:, i] for i, h in enumerate(hdr)}, hdr
