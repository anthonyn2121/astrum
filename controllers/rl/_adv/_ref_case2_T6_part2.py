"""Part 2: decompose the 76.5 deg, and build the zero-altitude-loss counterfactual."""

from __future__ import annotations

import csv
import math

KILL = "/home/anthony_nguyen/aca/astrum/controllers/rl/runs/d3b/replay_kill_seed62.csv"
NOKILL = "/home/anthony_nguyen/aca/astrum/controllers/rl/runs/d3b/replay_nokill_seed28.csv"
DT = 5.0 / 120.0
G = 9.80665


def load(path):
    with open(path) as fh:
        rd = csv.DictReader(fh)
        rows = list(rd)
        names = rd.fieldnames or []
    d = {n: [float(r[n]) for r in rows] for n in names}
    d["_n"] = len(rows)
    return d


def hdr(s):
    print("\n" + "=" * 100)
    print("  " + s)
    print("=" * 100)


def unwrap(deg):
    out = [deg[0]]
    for k in range(1, len(deg)):
        dv = deg[k] - deg[k - 1]
        while dv > 180.0:
            dv -= 360.0
        while dv < -180.0:
            dv += 360.0
        out.append(out[-1] + dv)
    return out


def vhat(d, k):
    e, n, u = d["vel_e_ms"][k], d["vel_n_ms"][k], d["vel_u_ms"][k]
    m = math.sqrt(e * e + n * n + u * u)
    return (e / m, n / m, u / m)


def ang(a, b):
    dp = max(-1.0, min(1.0, a[0] * b[0] + a[1] * b[1] + a[2] * b[2]))
    return math.degrees(math.acos(dp))


def main():
    d = load(KILL)
    n = d["_n"]
    course = unwrap(
        [math.degrees(math.atan2(d["vel_e_ms"][k], d["vel_n_ms"][k])) for k in range(n)]
    )

    hdr("A  DECOMPOSE THE 76.51 deg: how much of the 'turn' is the nose going DOWN?")
    print(
        "  d(3D rotation) per step splits into a horizontal-track part and a flight-path-angle part:"
    )
    print("    d_psi_h = |d course| * cos(gamma)   (the component of track change on the velocity)")
    print("    d_gamma = |d gamma|")
    tot3 = totH = totV = 0.0
    for k in range(1, n):
        tot3 += ang(vhat(d, k - 1), vhat(d, k))
        dc = course[k] - course[k - 1]
        gm = math.radians(0.5 * (d["gamma_deg"][k] + d["gamma_deg"][k - 1]))
        totH += abs(dc) * math.cos(gm)
        totV += abs(d["gamma_deg"][k] - d["gamma_deg"][k - 1])
    print(f"  cumulative 3-D velocity-vector rotation      = {tot3:7.2f} deg   <- the claim's 76.5")
    print(f"    horizontal (track) component               = {totH:7.2f} deg")
    print(f"    vertical (flight-path-angle) component     = {totV:7.2f} deg")
    print(f"    quadrature check sqrt(H^2+V^2)             = {math.hypot(totH, totV):7.2f} deg")
    print(
        f"  gamma: {d['gamma_deg'][0]:+6.2f} -> min {min(d['gamma_deg']):+6.2f} (step "
        f"{d['gamma_deg'].index(min(d['gamma_deg']))}) -> {d['gamma_deg'][172]:+6.2f} deg"
    )
    print(f"  => {totV / tot3 * 100.0:.1f}% of the 'turn' the altitude allegedly BOUGHT is the")
    print("     flight-path angle rotating downward, i.e. it IS the descent. Circular.")

    hdr("B  ZERO-ALTITUDE-LOSS COUNTERFACTUAL: rotation the same nz gives in a LEVEL turn")
    print(
        f"  {'window':<18} {'steps':>10} {'dur s':>7} {'achieved 3D':>12} {'level-turn':>11} "
        f"{'surplus':>9} {'alt m':>9} {'m/deg surplus':>14}"
    )
    rows = (
        ("roll-in", 0, 27),
        ("slice pull", 28, 115),
        ("unload", 108, 122),
        ("tracking run-in", 123, 172),
        ("WHOLE", 0, 172),
    )
    for nm, i0, i1 in rows:
        ach = sum(ang(vhat(d, k - 1), vhat(d, k)) for k in range(i0 + 1, i1 + 1))
        lvl = 0.0
        for k in range(i0 + 1, i1 + 1):
            nz = d["nz_g"][k]
            v = d["vt_ms"][k]
            if nz > 1.0:
                lvl += math.degrees(G * math.sqrt(nz * nz - 1.0) / v) * DT
        dalt = d["alt_m"][i1] - d["alt_m"][i0]
        sur = ach - lvl
        cost = (abs(dalt) / sur) if abs(sur) > 1e-6 else float("nan")
        print(
            f"  {nm:<18} {i0:4d}-{i1:<5d} {(i1 - i0) * DT:7.3f} {ach:12.2f} {lvl:11.2f} "
            f"{sur:+9.2f} {dalt:+9.1f} {cost:14.1f}"
        )
    print(
        "  level-turn integral = sum over steps of g*sqrt(nz^2-1)/V * dt using the MEASURED nz and V"
    )
    print("  of that very step. It is what the airframe's own load factor delivers with the lift")
    print("  vector fully supporting weight, i.e. paying ZERO altitude.")

    hdr("C  WHERE THE ALTITUDE WENT vs WHERE THE ROTATION CAME FROM")
    print(f"  {'window':<18} {'alt share %':>12} {'rot share %':>12}")
    tot_alt = abs(d["alt_m"][172] - d["alt_m"][0])
    for nm, i0, i1 in (
        ("roll-in 0-27", 0, 27),
        ("pull 28-115", 28, 115),
        ("116-122 unload", 116, 122),
        ("run-in 123-172", 123, 172),
    ):
        dalt = abs(d["alt_m"][i1] - d["alt_m"][i0])
        ach = sum(ang(vhat(d, k - 1), vhat(d, k)) for k in range(i0 + 1, i1 + 1))
        print(f"  {nm:<18} {dalt / tot_alt * 100.0:12.1f} {ach / tot3 * 100.0:12.1f}")

    hdr("D  THE PERCH ITSELF: dalt_rel, and what the aspect did while it was spent")
    for k in (0, 28, 56, 86, 115, 140, 156, 172):
        print(
            f"    step {k:3d} t {d['t_s'][k]:6.3f}  dalt_rel {d['dalt_rel_m'][k]:+8.1f} m  "
            f"es_rel {d['es_rel_m'][k]:8.1f}  aspect {d['aspect_deg'][k]:7.2f}  "
            f"ata {d['ata_deg'][k]:6.2f}  hca {d['hca_deg'][k]:7.2f}  gamma {d['gamma_deg'][k]:+7.2f}"
        )
    print(
        f"  dalt_rel [0] {d['dalt_rel_m'][0]:+.1f} m (= the +2000 ft spawn, {2000 * 0.3048:.1f} m) "
        f"-> [172] {d['dalt_rel_m'][172]:+.1f} m: the perch was spent to ZERO."
    )
    print(
        f"  aspect [0] {d['aspect_deg'][0]:.2f} -> [172] {d['aspect_deg'][172]:.2f} deg, "
        f"max {max(d['aspect_deg'][:173]):.2f} at step {d['aspect_deg'][:173].index(max(d['aspect_deg'][:173]))}"
    )
    print(
        "  a perch cash-in in the textbook sense buys a REAR-QUARTER / offensive position. aspect"
    )
    print("  180 deg = reciprocal. This geometry got MORE head-on as the perch was spent.")

    hdr("E  es_rel THROUGH THE PULL, and the full-horizon picture on the non-kill seed")
    er = d["es_rel_m"]
    print(
        f"  es_rel [28] {er[28]:8.1f} -> [115] {er[115]:8.1f} = {er[115] - er[28]:+7.1f} m "
        f"over the pull"
    )
    print(
        f"  es_rel [35] {er[35]:8.1f} (peak) -> [112] {er[112]:8.1f} (min) = "
        f"{er[112] - er[35]:+7.1f} m"
    )
    print(
        f"  es_rel [123] {er[123]:8.1f} -> [172] {er[172]:8.1f} = {er[172] - er[123]:+7.1f} m "
        f"over the full-throttle run-in"
    )
    k28 = load(NOKILL)
    print(
        "\n  seed 28, steps 0-575 (SAME commanded genome; states bit-identical to seed 62 to 172):"
    )
    print(
        f"    my  Es [0] {k28['es_m'][0]:8.1f} -> [575] {k28['es_m'][575]:8.1f} "
        f"({k28['es_m'][575] - k28['es_m'][0]:+7.1f} m)   min {min(k28['es_m']):8.1f} at step "
        f"{k28['es_m'].index(min(k28['es_m']))}"
    )
    print(
        f"    opp Es [0] {k28['opp_es_m'][0]:8.1f} -> [575] {k28['opp_es_m'][575]:8.1f} "
        f"({k28['opp_es_m'][575] - k28['opp_es_m'][0]:+7.1f} m)"
    )
    print(
        f"    es_rel [575] {k28['es_rel_m'][575]:8.1f}  -> the rising es_rel is the BANDIT bleeding "
        f"{k28['opp_es_m'][0] - k28['opp_es_m'][575]:+.1f} m, not my energy management"
    )
    print(
        f"    my alt [575] {k28['alt_m'][575]:8.1f} m (started {k28['alt_m'][0]:.1f}); "
        f"opp alt [575] {k28['opp_alt_m'][575]:8.1f} m"
    )
    print(
        f"    opp_health [575] {k28['opp_health'][575]:.1f}   my health [575] {k28['health'][575]:.1f}"
    )

    hdr("F  DISCRIMINATIVE POWER: can the energy ledger tell a kill from a non-kill?")
    cols = ("alt_m", "vt_ms", "es_m", "es_rel_m", "nz_g", "gamma_deg", "phi_deg", "opp_es_m")
    worst = max(max(abs(d[c][k] - k28[c][k]) for k in range(173)) for c in cols)
    print(f"  max |kill seed 62 - NON-kill seed 28| over steps 0-172 across {cols} = {worst:.3e}")
    print("  the five kill seeds and the three non-kill seeds fly the SAME energy trajectory over")
    print("  this window (the genome is open-loop w.r.t. the bandit; only the gun PRNG differs).")
    print("  => the energy ledger has EXACTLY ZERO power to explain why this rollout killed.")


if __name__ == "__main__":
    main()
