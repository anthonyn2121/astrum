"""Run the REAL bfm_detect.detect_all on traces rebuilt from the smp_* CSV block.

Verified on the host: smp_chi_rad == radians(chi_deg) to 2.2e-16, smp_nz == nz_g exactly,
smp_alt == alt_enu_m exactly, smp_cpa_miss == cpa_miss_m exactly, smp_chi_valid == chi_valid.
So the rebuilt trace is the same object detect_all read during the audit.

Container:
  docker compose -f /home/anthony_nguyen/aca/astrum/docker-compose.yml --profile rl run --rm \
    --no-deps dogfight-train bash -lc \
    'cd /workspace/controllers/rl && python _diag_refute_detectall.py'
"""

from __future__ import annotations

import csv
import importlib
import sys
import types

PKG = "/workspace/crates/astrum-env/python/astrum_dogfight"
sys.path.insert(0, "/workspace/crates/astrum-env/python")

# `astrum_dogfight/__init__.py` imports `env`, which needs the compiled `astrum_env` extension.
# bfm_detect needs only `constants` and `geometry`, neither of which touches the sim, so install a
# namespace shim for the package rather than dragging the extension in. This changes nothing about
# bfm_detect's own code path.
_shim = types.ModuleType("astrum_dogfight")
_shim.__path__ = [PKG]
sys.modules["astrum_dogfight"] = _shim
B = importlib.import_module("astrum_dogfight.bfm_detect")

RUNS = "/workspace/controllers/rl/runs/d3b"
# Sample field -> CSV column in the smp_* block (radians/native units, dumped verbatim).
FIELDS = {
    "t": "smp_t",
    "chi": "smp_chi_rad",
    "chi_valid": "smp_chi_valid",
    "gamma": "smp_gamma_rad",
    "nz": "smp_nz",
    "phi": "smp_phi_rad",
    "alt": "smp_alt",
    "ail": "smp_ail",
    "rng": "smp_rng",
    "ata": "smp_ata",
    "ata_az": "smp_ata_az",
    "aspect": "smp_aspect",
    "closure": "smp_closure",
    "cpa_miss": "smp_cpa_miss",
    "es_rel": "smp_es_rel",
    "lay_miss_m": "smp_lay_miss_m",
    "lethal": "smp_lethal",
}


def rebuild(path):
    rows = list(csv.DictReader(open(path)))
    tr = []
    for r in rows:
        kw = {}
        for f, col in FIELDS.items():
            v = r[col]
            if f in ("chi_valid", "lethal"):
                kw[f] = bool(int(float(v)))
            else:
                kw[f] = float(v)
        tr.append(B.Sample(**kw))
    return tr, rows


def show(tag, path):
    tr, rows = rebuild(path)
    r = B.detect_all(tr)
    c = r["commitment"]
    print("=" * 96)
    print(
        f"{tag}  ticks {r['ticks']}  duration {r['duration_s']:.3f} s  lethal_ticks {r['lethal_ticks']}"
    )
    print(f"  D1 displacement_rolls {len(r['displacement_rolls'])}")
    print(f"  D2 yoyos {len(r['yoyos'])}  yoyos_clean {len(r['yoyos_clean'])}")
    print(f"  D3 turning_room      {len(r['turning_room'])}")
    for e in r["turning_room"]:
        print(f"     D3 event i0={e.i0} i1={e.i1} detail={e.detail}")
    print(
        f"  D4 bank hold {c['longest_bank_hold_s']:.3f} s  dalt {c['bank_hold_dalt_m']:+.0f} m"
        f"  peak bank {c['bank_max_deg']:.1f} deg"
    )
    print(f"  D4 chi  hold {c['longest_chi_hold_s']:.3f} s  dalt {c['chi_hold_dalt_m']:+.0f} m")
    print(f"  D4 chi hist {c['chi_hist']}  valid_frac {c['valid_frac']:.3f}")
    print(f"  D5 lethality {r['lethality']}")

    # --- is the chi excursion a ROLL displacement, or the LOS plane rotating under a fixed
    #     lift vector? The controller is open-loop wrt the bandit (Genome.act reads own attitude
    #     only), so this is the discriminating question for the words "flies ... an out-of-plane hold".
    print(
        "\n  chi growth vs phi: over the loaded run, step / phi / chi / roll_rate / los_rate_az / ata_az"
    )
    for k in (28, 40, 52, 64, 76, 88, 94, 100, 104, 108):
        if k >= len(rows):
            break
        q = rows[k]
        print(
            f"    {k:3d} t {float(q['t_s']):5.3f}  phi {float(q['phi_deg']):+7.1f}  chi {float(q['chi_deg']):+6.1f}"
            f"  rollrate {float(q['roll_rate_dps']):+7.2f} dps  losrate_az {float(q['los_rate_az_dps']):+7.2f} dps"
            f"  ata_az {float(q['ata_az_deg']):+6.1f}  rng {float(q['rng_m']):7.1f}"
        )
    sl = [rows[k] for k in range(28, min(109, len(rows)))]
    dphi = max(float(q["phi_deg"]) for q in sl) - min(float(q["phi_deg"]) for q in sl)
    dchi = max(float(q["chi_deg"]) for q in sl) - min(float(q["chi_deg"]) for q in sl)
    print(
        f"    steps 28-108: phi span {dphi:.1f} deg   chi span {dchi:.1f} deg"
        f"   max|roll_rate| {max(abs(float(q['roll_rate_dps'])) for q in sl):.2f} dps"
    )

    # --- Es across the cited chi hold (price the hold in ENERGY as well as altitude)
    if len(rows) > 108:
        a, b = rows[88], rows[108]
        print(
            f"\n  hold 88-108 priced: alt_enu {float(a['alt_enu_m']):.0f} -> {float(b['alt_enu_m']):.0f} m"
            f"  Es {float(a['es_m']):.0f} -> {float(b['es_m']):.0f} m"
            f"  es_rel {float(a['es_rel_m']):.0f} -> {float(b['es_rel_m']):.0f} m"
            f"  vt {float(a['vt_ms']):.1f} -> {float(b['vt_ms']):.1f} m/s"
            f"  gamma {float(a['gamma_deg']):.1f} -> {float(b['gamma_deg']):.1f} deg"
        )


if __name__ == "__main__":
    show("KILL seed 62", RUNS + "/replay_kill_seed62.csv")
    show("NO-KILL seed 28", RUNS + "/replay_nokill_seed28.csv")
