"""Pin the honest gun-lay math against the sim's own projectile physics.

Three separate claims are tested, in increasing strength:

1. **Algebra.** The closed-form `intercept()` really is an intercept — propagating a
   drag-free round along its boresight for its own `tof` lands on the target to
   floating-point precision. No tolerance to argue about.
2. **The bias formula.** For a static aim point the closed form reduces to
   `θ_bias = asin(V·sin α / 1030)`, and that is 0.97°/2.31°/3.81° at α = 5/12/20°
   with V = 200 m/s — the numbers the reward is currently off by.
3. **Physics.** Scored against a tick-for-tick mirror of
   `crates/astrum-sim/src/combat/ballistics.rs`, the closed form must agree with the
   full drag/gravity/Coriolis solution to better than 0.1°, and the lay `geometry.py`
   flies today must miss the 8 m sphere in the regimes the run history stalled in.

Run directly (`python test_gunlay.py`) for the measurement tables behind the
assertions; `pytest` runs the assertions alone.
"""

from __future__ import annotations

from math import cos, degrees, radians, sin

import numpy as np
from astrum_dogfight import gunlay as G

LAT = radians(37.0)
ALT = 4572.0  # 15 kft — the altitude the run history flies at


def _shot(range_m: float, off_vel_deg: float, own_speed: float, crossing_mps: float = 0.0):
    """A shot geometry in ENU. The shooter sits at the origin at `ALT` flying due
    North at `own_speed`; the target sits `range_m` away, `off_vel_deg` off the
    shooter's velocity vector (rotated in the vertical plane, i.e. a pure pull), and
    crosses to the East at `crossing_mps`."""
    shooter_pos = np.array([0.0, 0.0, ALT])
    shooter_vel = np.array([0.0, own_speed, 0.0])
    a = radians(off_vel_deg)
    los = np.array([0.0, cos(a), sin(a)])  # North, pitched up by `a`
    target_pos = shooter_pos + los * range_m
    target_vel = np.array([crossing_mps, 0.0, 0.0])
    return shooter_pos, shooter_vel, target_pos, target_vel


# ---------------------------------------------------------------------------
# 1. The closed form is algebraically an intercept
# ---------------------------------------------------------------------------


def test_closed_form_is_exact_intercept():
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(400):
        sp = np.array([0.0, 0.0, ALT])
        sv = rng.normal(0.0, 150.0, 3)
        tp = sp + rng.normal(0.0, 400.0, 3)
        tv = rng.normal(0.0, 150.0, 3)
        sol = G.intercept(sp, sv, tp, tv)
        assert sol.feasible
        assert abs(float(np.linalg.norm(sol.boresight)) - 1.0) < 1e-12
        assert sol.tof_s > 0.0
        round_pos = sp + (sv + sol.boresight * G.MUZZLE_SPEED_MPS) * sol.tof_s
        target_pos = tp + tv * sol.tof_s
        worst = max(worst, float(np.linalg.norm(round_pos - target_pos)))
    assert worst < 1e-6, f"closed form is not an intercept: {worst:.3e} m"


def test_discriminant_is_always_positive_in_this_sim():
    """Both aircraft are subsonic, so the relative closing speed can never approach
    1030 m/s and the quadratic always has one positive root — no branch, no guard."""
    for own in (80.0, 200.0, 300.0, 500.0):
        for tgt in (80.0, 200.0, 300.0, 500.0):
            for sign in (+1.0, -1.0):  # head-on and tail-chase
                sp = np.array([0.0, 0.0, ALT])
                sv = np.array([0.0, own, 0.0])
                tp = np.array([0.0, 500.0, ALT])
                tv = np.array([0.0, sign * tgt, 0.0])
                assert G.intercept(sp, sv, tp, tv).feasible


# ---------------------------------------------------------------------------
# 2. The bias formula, and the numbers the reward is off by
# ---------------------------------------------------------------------------


def test_theta_bias_matches_closed_form_intercept():
    """For a static target, the angle between the naive line of sight and the correct
    boresight is exactly `asin(V·sin α / 1030)`."""
    for own in (150.0, 200.0, 260.0):
        for off in (0.0, 5.0, 12.0, 20.0, 25.0):
            sp, sv, tp, tv = _shot(300.0, off, own)
            sol = G.intercept(sp, sv, tp, tv)
            los = G._unit(tp - sp)
            measured = np.arccos(np.clip(np.dot(sol.boresight, los), -1.0, 1.0))
            predicted = G.theta_bias(own, radians(off))
            assert abs(measured - predicted) < 1e-9, (own, off, degrees(measured), degrees(predicted))


def test_theta_bias_reference_values():
    """The three numbers that size the problem: at 200 m/s the nose must sit ~1-4°
    further off the velocity vector than `geometry.py` believes."""
    for off, expect_deg in ((5.0, 0.969688), (12.0, 2.313728), (20.0, 3.807912)):
        got = degrees(G.theta_bias(200.0, radians(off)))
        assert abs(got - expect_deg) < 1e-5, f"α={off}°: {got:.6f}° != {expect_deg}°"


def test_bias_is_range_independent():
    """It depends only on own speed and off-velocity angle — so it cannot be tuned
    away by a range gate, and it is *worst* exactly where a gun kill happens."""
    ref = None
    for rng_m in (150.0, 250.0, 400.0, 600.0, 1000.0):
        sp, sv, tp, tv = _shot(rng_m, 12.0, 200.0)
        sol = G.intercept(sp, sv, tp, tv)
        ang = float(np.arccos(np.clip(np.dot(sol.boresight, G._unit(tp - sp)), -1.0, 1.0)))
        if ref is None:
            ref = ang
        assert abs(ang - ref) < 1e-12


def test_corrected_solution_hits_the_alpha_ceiling():
    """The correction rotates the nose *away* from the velocity vector, so an honest
    solution needs MORE α than the target's off-velocity angle. Against the measured
    10.7-15.0° usable α of this F-16 model, anything past ~12-13° off-velocity is
    physically unreachable — honest gunnery requires a low-crossing, near-in-plane
    shot."""
    alpha_ceiling_deg = 15.0
    nose = {off: off + degrees(G.theta_bias(200.0, radians(off))) for off in (5, 12, 20, 25)}
    assert abs(nose[12] - 14.3138) < 5e-3
    assert abs(nose[20] - 23.8072) < 5e-3
    assert nose[12] < alpha_ceiling_deg  # reachable, barely
    assert nose[20] > alpha_ceiling_deg  # not reachable at all


# ---------------------------------------------------------------------------
# 3. Scored against the sim's real projectile physics
# ---------------------------------------------------------------------------


def test_matches_the_rust_golden_trajectory():
    """The load-bearing test: this module's projectile physics must be the SAME
    physics the sim flies rounds with, not a plausible re-derivation of it.

    The identical case and expected values are asserted from the Rust side in
    `crates/astrum-sim/tests/ballistics.rs::golden_trajectory_pins_the_python_mirror`,
    so drift in either implementation breaks one of the two suites. Case: a 20 mm
    round at 15 kft fired due North at 1230 m/s (1030 muzzle + 200 m/s inherited),
    120 steps of 1/120 s, latitude 37° N."""
    omega = G.earth_rate_enu(radians(37.0))
    assert abs(omega[1] - 5.823_741_982_348_514e-5) < 1e-18
    assert abs(omega[2] - 4.388_504_357_552_398e-5) < 1e-18
    assert abs(G.DRAG_FACTOR - 9.424_777_960_769_38e-4) < 1e-15
    assert abs(G.air_density(4572.0) - 0.770_815_982_717_405) < 1e-12

    pos = np.array([0.0, 0.0, 4572.0])
    vel = np.array([0.0, 1230.0, 0.0])
    for _ in range(120):
        pos, vel = G.step_projectile(pos, vel, omega, 1.0 / 120.0)
    for label, got, want in (
        ("pos.x", pos[0], 0.042_459_845_373),
        ("pos.y", pos[1], 1014.656_328_395_519),
        ("pos.z", pos[2], 4567.605_753_963_890),
        ("vel.x", vel[0], 0.075_225_316_681),
        ("vel.y", vel[1], 849.305_371_114_120),
        ("vel.z", vel[2], -8.301_833_271_160),
    ):
        assert abs(float(got) - want) < 1e-9, f"{label}: {float(got):.12f} != Rust {want:.12f}"

    # 380 m/s of the 1230 bled off in one second — drag, not gravity, is what makes
    # `tof = rng/1030` unfixable by a constant.
    assert 370.0 < 1230.0 - float(vel[1]) < 390.0


def test_air_density_matches_isa():
    assert abs(G.air_density(0.0) - 1.225) < 1e-9
    assert abs(G.air_density(4572.0) - 0.7708) < 2e-3  # ISA 15 kft ≈ 0.771 kg/m³
    assert G.air_density(11_000.0) > G.air_density(12_000.0) > 0.0


def test_drag_is_the_dominant_projectile_force():
    """~500 m/s² of deceleration at the muzzle: the reason `tof = rng/1030` cannot be
    salvaged by any constant fudge factor."""
    a = G.projectile_accel(np.array([0.0, 1200.0, 0.0]), ALT, G.earth_rate_enu(LAT))
    assert 400.0 < float(np.linalg.norm(a[:2])) < 700.0


def test_numeric_intercept_converges():
    """The Gauss-Newton reference must actually land on the target under full
    physics, or it is not a reference."""
    for rng_m in (150.0, 250.0, 400.0, 600.0):
        for off in (0.0, 12.0):
            for cross in (0.0, 150.0):
                sp, sv, tp, tv = _shot(rng_m, off, 200.0, cross)
                sol = G.numeric_intercept(sp, sv, tp, tv, lat_rad=LAT)
                launch = sv + sol.boresight * G.MUZZLE_SPEED_MPS
                _, miss, _ = G._closest_approach(sp, launch, tp, tv, G.earth_rate_enu(LAT))
                assert miss < 0.5, f"rng={rng_m} off={off} cross={cross}: {miss:.3f} m"


def test_closed_form_lands_inside_the_kill_sphere():
    """The acceptance gate for using the cheap closed form in the reward, stated in
    the only unit that matters: propagated under FULL physics, its lay must land well
    inside the 8 m kill sphere across the gun envelope.

    Angular agreement with the drag-inclusive truth is deliberately *not* the gate —
    it degrades with range (0.037° at 150 m to 0.158° at 600 m) because gravity drop
    grows with time of flight, yet the corresponding miss only grows 0.08 m → 1.4 m.
    Judging a lay by degrees rather than metres is the same category error that made
    `lead_error → 0` look like a solved aim problem."""
    worst = 0.0
    worst_at = None
    omega = G.earth_rate_enu(LAT)
    for rng_m in (150.0, 250.0, 400.0, 600.0):
        for off in (0.0, 5.0, 12.0, 20.0):
            for own in (150.0, 200.0, 260.0):
                sp, sv, tp, tv = _shot(rng_m, off, own)
                lay = G.intercept(sp, sv, tp, tv).boresight
                _, miss, _ = G._closest_approach(sp, sv + lay * G.MUZZLE_SPEED_MPS, tp, tv, omega)
                if miss > worst:
                    worst, worst_at = miss, (rng_m, off, own)
    assert worst < 2.0, f"closed form misses by {worst:.2f} m at {worst_at}"
    assert worst < 0.25 * G.HIT_RADIUS_M


def test_closed_form_degrades_gracefully_on_a_crossing_target():
    """Where the drag-free closed form is weakest: a fast crosser. Drag lengthens the
    real time of flight, so a drag-free lead under-leads — 5.3 m of miss at 400 m
    against a 200 m/s crosser, still inside the sphere but no longer comfortable,
    against 15.8 m for the lay flown today. The cheap fix recovers ~96% of the error;
    the last ~4% needs a propagated time of flight."""
    omega = G.earth_rate_enu(LAT)
    for cross, max_closed in ((0.0, 1.0), (100.0, 4.0), (200.0, 6.5)):
        sp, sv, tp, tv = _shot(400.0, 12.0, 200.0, cross)
        closed = G.intercept(sp, sv, tp, tv).boresight
        naive = G._unit(G.naive_lead_point_from_parts(sp, tp, tv) - sp)
        _, closed_miss, _ = G._closest_approach(sp, sv + closed * G.MUZZLE_SPEED_MPS, tp, tv, omega)
        _, naive_miss, _ = G._closest_approach(sp, sv + naive * G.MUZZLE_SPEED_MPS, tp, tv, omega)
        assert closed_miss < max_closed, f"cross={cross}: closed form miss {closed_miss:.2f} m"
        assert closed_miss < G.HIT_RADIUS_M < naive_miss


def test_the_naive_error_is_inheritance_not_time_of_flight():
    """Which of the two bugs to fix first. Holding range and α fixed and sweeping the
    crossing rate barely moves the naive miss (14.15 m → 15.80 m over 0-200 m/s of
    crossing), so ~90% of the error is the missing velocity-inheritance term and only
    ~10% is the wrong `tof = rng/1030`. A tof-only fix would not have helped."""
    omega = G.earth_rate_enu(LAT)
    misses = []
    for cross in (0.0, 200.0):
        sp, sv, tp, tv = _shot(400.0, 12.0, 200.0, cross)
        naive = G._unit(G.naive_lead_point_from_parts(sp, tp, tv) - sp)
        _, miss, _ = G._closest_approach(sp, sv + naive * G.MUZZLE_SPEED_MPS, tp, tv, omega)
        misses.append(miss)
    tof_share = (misses[1] - misses[0]) / misses[1]
    assert tof_share < 0.15, f"tof error accounts for {tof_share:.1%} of the naive miss"


def test_naive_lay_misses_the_kill_sphere_where_the_runs_stalled():
    """The whole point. With `geometry.lead_error` EXACTLY zero — nose perfectly on
    the aim point the reward optimises — the round still misses the 8 m sphere at the
    250-600 m standoff every checkpoint fires from, while the corrected lay hits."""
    omega = G.earth_rate_enu(LAT)
    for rng_m, off, min_naive_miss in ((250.0, 12.0, 8.0), (400.0, 12.0, 12.0), (250.0, 20.0, 13.0)):
        sp, sv, tp, tv = _shot(rng_m, off, 200.0)
        naive_dir = G._unit(G.naive_lead_point_from_parts(sp, tp, tv) - sp)
        _, naive_miss, _ = G._closest_approach(sp, sv + naive_dir * G.MUZZLE_SPEED_MPS, tp, tv, omega)
        good = G.intercept(sp, sv, tp, tv).boresight
        _, good_miss, _ = G._closest_approach(sp, sv + good * G.MUZZLE_SPEED_MPS, tp, tv, omega)
        assert naive_miss > min_naive_miss, f"rng={rng_m} off={off}: naive miss only {naive_miss:.2f} m"
        assert good_miss < G.HIT_RADIUS_M, f"rng={rng_m} off={off}: corrected miss {good_miss:.2f} m"


def test_naive_lay_is_fine_close_and_low_alpha():
    """The one corner where the wrong model accidentally agrees with the physics —
    a settled, low-α, close, rear-quarter shot. This is the only thing ~85 runs ever
    learned, and this test says why."""
    omega = G.earth_rate_enu(LAT)
    sp, sv, tp, tv = _shot(150.0, 3.0, 200.0)
    naive_dir = G._unit(G.naive_lead_point_from_parts(sp, tp, tv) - sp)
    _, miss, _ = G._closest_approach(sp, sv + naive_dir * G.MUZZLE_SPEED_MPS, tp, tv, omega)
    assert miss < G.HIT_RADIUS_M


def test_round_ground_speed_exceeds_muzzle_speed_on_a_forward_shot():
    """Inheritance makes the round *faster* than its muzzle speed when fired forward:
    ~1180/1230/1290 m/s at V = 150/200/260. This is why `tof = rng/1030` is wrong even
    before drag enters."""
    for own, expect_gs in ((150.0, 1180.0), (200.0, 1230.0), (260.0, 1290.0)):
        sp, sv, tp, tv = _shot(400.0, 0.0, own)
        sol = G.intercept(sp, sv, tp, tv)
        assert abs(sol.ground_speed_mps - expect_gs) < 1.5, f"V={own}: {sol.ground_speed_mps:.1f}"


def test_naive_tof_error_changes_SIGN_with_range():
    """The naive `tof = rng/1030` cannot be repaired by a constant factor, because
    inheritance and drag push it in opposite directions and cross over inside the gun
    envelope. Measured at V=200 against the full propagator:

        150 m  +9.2%   250 m +12.0%   400 m  +8.4%   (too LONG ⇒ over-lead)
        900 m  -0.1%  1200 m  -5.5%                  (too SHORT ⇒ under-lead)

    Inside the 150-600 m band where every kill happens it over-leads by 6-12%. Only a
    propagated time of flight is right at both ends."""
    sp, sv, tp, tv = _shot(400.0, 0.0, 200.0)
    close = G.numeric_intercept(sp, sv, tp, tv, lat_rad=LAT).tof_s
    assert 400.0 / G.MUZZLE_SPEED_MPS > close, "naive tof should over-estimate at 400 m"
    sp, sv, tp, tv = _shot(1200.0, 0.0, 200.0)
    far = G.numeric_intercept(sp, sv, tp, tv, lat_rad=LAT).tof_s
    assert 1200.0 / G.MUZZLE_SPEED_MPS < far, "naive tof should under-estimate at 1200 m"


def test_dispersion_alone_is_not_the_wall():
    """8 mrad of dispersion is survivable out to ~600 m *if* the mean lay is right —
    so the wall is bias, not spread. (Dispersion is applied to the boresight before
    the shooter's velocity is added, which shrinks the round's angular spread.)"""
    sp, sv, tp, tv = _shot(400.0, 12.0, 200.0)
    good = G.intercept(sp, sv, tp, tv).boresight
    p, _ = G.p_hit(good, sp, sv, tp, tv, samples=600, lat_rad=LAT, seed=1)
    assert p > 0.6, f"P(hit) with a correct lay at 400 m is only {p:.3f}"


# ---------------------------------------------------------------------------
# 4. The batched propagator used by the rollout-scoring diagnostics
# ---------------------------------------------------------------------------


def test_air_density_vec_matches_scalar():
    alts = np.array([0.0, 1000.0, 4572.0, 10_999.0, 11_000.0, 11_001.0, 15_000.0])
    got = G.air_density_vec(alts)
    for a, g in zip(alts, got):
        assert abs(float(g) - G.air_density(float(a))) < 1e-12, a


def test_step_projectile_vec_matches_scalar():
    """The batched tick must be the SAME physics, not an approximation of it — the
    counterfactual diagnostics score real rounds with it."""
    omega = G.earth_rate_enu(LAT)
    rng = np.random.default_rng(11)
    pos = np.column_stack(
        [rng.normal(0, 300, 8), rng.normal(0, 300, 8), rng.uniform(3000, 6000, 8)]
    )
    vel = rng.normal(0, 700, (8, 3))
    bp, bv = pos.copy(), vel.copy()
    for _ in range(40):
        bp, bv = G.step_projectile_vec(bp, bv, omega)
    for i in range(8):
        sp, sv = pos[i].copy(), vel[i].copy()
        for _ in range(40):
            sp, sv = G.step_projectile(sp, sv, omega)
        assert float(np.linalg.norm(bp[i] - sp)) < 1e-9
        assert float(np.linalg.norm(bv[i] - sv)) < 1e-9


def test_p_hit_vec_agrees_with_scalar():
    """Same hit model, ~200× faster. The batched form takes the global closest approach
    over its horizon where the scalar breaks at the first receding tick, so it may read
    marginally higher; anything beyond Monte-Carlo noise is a real disagreement."""
    for rng_m, off in ((250.0, 12.0), (400.0, 0.0), (600.0, 5.0)):
        sp, sv, tp, tv = _shot(rng_m, off, 200.0, 80.0)
        good = G.intercept(sp, sv, tp, tv).boresight
        p_s, m_s = G.p_hit(good, sp, sv, tp, tv, samples=300, lat_rad=LAT, seed=5)
        p_v, m_v = G.p_hit_vec(good, sp, sv, tp, tv, samples=3000, lat_rad=LAT, seed=5)
        assert abs(p_s - p_v) < 0.08, f"rng={rng_m} off={off}: scalar {p_s:.3f} vs vec {p_v:.3f}"
        assert abs(m_s - m_v) < 1.5, f"rng={rng_m} off={off}: miss {m_s:.2f} vs {m_v:.2f}"


def test_p_hit_vec_reproduces_the_naive_vs_corrected_gap():
    """The load-bearing use: the batched scorer must show the same wall the scalar one
    does — a perfectly-flown naive lay is un-hittable at 400 m, the corrected lay is not."""
    sp, sv, tp, tv = _shot(400.0, 12.0, 200.0)
    naive = G._unit(G.naive_lead_point_from_parts(sp, tp, tv) - sp)
    good = G.intercept(sp, sv, tp, tv).boresight
    p_naive, _ = G.p_hit_vec(naive, sp, sv, tp, tv, samples=4000, lat_rad=LAT, seed=2)
    p_good, _ = G.p_hit_vec(good, sp, sv, tp, tv, samples=4000, lat_rad=LAT, seed=2)
    assert p_naive < 0.05, f"naive lay P(hit) {p_naive:.3f}"
    assert p_good > 0.9, f"corrected lay P(hit) {p_good:.3f}"


def _straight_path(target_pos, target_vel, horizon_s=1.0, dt=1.0 / 24.0):
    ts = np.arange(0.0, horizon_s + dt, dt)
    xyz = np.asarray(target_pos, float)[None, :] + np.asarray(target_vel, float)[None, :] * ts[:, None]
    return np.column_stack([ts, xyz])


def test_intercept_path_reduces_to_intercept_on_a_straight_target():
    """The clairvoyant solver must agree with the closed form when there is nothing to
    be clairvoyant about."""
    for rng_m, off, cross in ((250.0, 12.0, 0.0), (400.0, 5.0, 120.0), (600.0, 0.0, 200.0)):
        sp, sv, tp, tv = _shot(rng_m, off, 200.0, cross)
        a = G.intercept(sp, sv, tp, tv)
        b = G.intercept_path(sp, sv, _straight_path(tp, tv, 2.0))
        assert b.feasible
        assert abs(a.tof_s - b.tof_s) < 1e-6, (rng_m, off, cross, a.tof_s, b.tof_s)
        ang = degrees(float(np.arccos(np.clip(np.dot(a.boresight, b.boresight), -1.0, 1.0))))
        assert ang < 1e-4, f"{ang:.6f}°"


def test_refine_lay_removes_the_drag_free_residual():
    """The Newton refinement must beat the raw closed form under full physics and land
    where the expensive Gauss-Newton reference lands.

    The floor is not zero: `resolve_hits` samples the target's position once per tick and
    sweeps the round's segment against that point, so a 200 m/s crosser moves 1.67 m
    between samples and ~1 m of residual is the SIM's own measurement granularity, which
    both solvers inherit. Hence a 1.5 m gate, not a 0.1 m one."""
    omega = G.earth_rate_enu(LAT)
    dt = G.SIM_DT
    for rng_m, off, cross in ((250.0, 12.0, 0.0), (400.0, 12.0, 200.0), (600.0, 5.0, 100.0)):
        sp, sv, tp, tv = _shot(rng_m, off, 200.0, cross)
        n = int(1.8 / dt)
        table = tp[None, :] + tv[None, :] * ((1 + np.arange(n)) * dt)[:, None]
        raw = G.intercept(sp, sv, tp, tv).boresight
        ref = G.refine_lay(raw, sp, sv, table, omega, dt)
        num = G.numeric_intercept(sp, sv, tp, tv, lat_rad=LAT).boresight
        _, miss_raw, _ = G._closest_approach_table(sp, sv + raw * G.MUZZLE_SPEED_MPS, table, omega, dt)
        _, miss_ref, _ = G._closest_approach_table(sp, sv + ref * G.MUZZLE_SPEED_MPS, table, omega, dt)
        assert miss_ref < 1.5, f"rng={rng_m} cross={cross}: refined miss {miss_ref:.3f} m"
        assert miss_ref < miss_raw
        ang = degrees(float(np.arccos(np.clip(np.dot(ref, num), -1.0, 1.0))))
        assert ang < 0.03, f"refined vs Gauss-Newton disagree by {ang:.4f}°"


def _break_path(range_m: float, g: float, speed: float = 200.0, horizon_s: float = 2.0):
    """Shooter at the origin flying North at `speed`; bandit `range_m` dead ahead, co-speed,
    in a sustained flat `g` break to the right (turn rate ω = n·g₀/V). Returns
    `(shooter_pos, shooter_vel, bandit_pos0, bandit_vel0, path)`."""
    turn = g * 9.80665 / speed
    ts = np.arange(0.0, horizon_s + G.SIM_DT, G.SIM_DT)
    hdg = turn * ts
    tp0 = np.array([0.0, range_m, ALT])
    path = np.column_stack(
        [
            ts,
            tp0[0] + speed / turn * (1.0 - np.cos(hdg)),
            tp0[1] + speed / turn * np.sin(hdg),
            np.full_like(ts, ALT),
        ]
    )
    return (
        np.array([0.0, 0.0, ALT]),
        np.array([0.0, speed, 0.0]),
        tp0,
        np.array([0.0, speed, 0.0]),
        path,
    )


def _tables(path, tv0, tp0, horizon_s=1.8):
    ticks = (1 + np.arange(int(horizon_s / G.SIM_DT))) * G.SIM_DT
    cv = tp0[None, :] + tv0[None, :] * ticks[:, None]
    real = np.column_stack([np.interp(ticks, path[:, 0], path[:, k]) for k in (1, 2, 3)])
    return cv, real


def test_prediction_error_is_exactly_half_a_t_squared():
    """The size of the OTHER aim error — assuming the bandit holds its velocity.

    A lay computed from instantaneous state and refined against that same belief (which is
    all a real fire-control computer can do) misses a manoeuvring bandit by ½·a·tof², to
    within a few percent. That is 0.98 m at 150 m and 18.5 m at 600 m against an 8 g break:
    inside 250 m the bandit cannot manoeuvre out of the 8 m sphere in the time of flight,
    and past ~400 m it always can. This is a property of the WEAPON, not of the policy —
    no reward shaping moves it."""
    omega = G.earth_rate_enu(LAT)
    for rng_m, g in ((150.0, 8.0), (250.0, 8.0), (400.0, 8.0), (600.0, 8.0), (400.0, 5.0)):
        sp, sv, tp0, tv0, path = _break_path(rng_m, g)
        cv, real = _tables(path, tv0, tp0)
        pred = G.refine_lay(G.intercept(sp, sv, tp0, tv0).boresight, sp, sv, cv, omega)
        clair = G.refine_lay(G.intercept_path(sp, sv, path).boresight, sp, sv, real, omega)
        _, miss_pred, tof = G._closest_approach_table(
            sp, sv + pred * G.MUZZLE_SPEED_MPS, real, omega
        )
        _, miss_clair, _ = G._closest_approach_table(
            sp, sv + clair * G.MUZZLE_SPEED_MPS, real, omega
        )
        expect = 0.5 * g * 9.80665 * tof * tof
        assert abs(miss_pred - expect) < 0.05 * expect + 0.1, (
            f"rng={rng_m} {g}g: pred miss {miss_pred:.2f} m vs ½at² {expect:.2f} m"
        )
        assert miss_clair < 0.5, f"rng={rng_m} {g}g: clairvoyant miss {miss_clair:.2f} m"


def test_honest_gun_range_is_bounded_by_the_break_not_the_aim():
    """The doctrine this whole program has been circling, derived rather than tuned.

    Even with a PERFECT lay computed from perfect instantaneous state, P(hit) per round
    against an 8 g break collapses with range purely from ½·a·tof² — ~0.99 at 250 m, ~0.50
    at 400 m, ~0.00 at 600 m. At 100 rounds/s, 50 hits (one kill) is HALF A SECOND of
    correctly-laid fire inside 250 m, and unreachable at 600 m however good the tracking.
    Every checkpoint to date fires from 250-600 m."""
    p_by_range = {}
    for rng_m in (250.0, 400.0, 600.0):
        sp, sv, tp0, tv0, path = _break_path(rng_m, 8.0)
        cv, _ = _tables(path, tv0, tp0)
        pred = G.refine_lay(
            G.intercept(sp, sv, tp0, tv0).boresight, sp, sv, cv, G.earth_rate_enu(LAT)
        )
        p, _ = G.p_hit_vec(
            pred, sp, sv, tp0, tv0, samples=2000, lat_rad=LAT, seed=4, target_path=path
        )
        p_by_range[rng_m] = p
    assert p_by_range[250.0] > 0.95, p_by_range
    assert 0.35 < p_by_range[400.0] < 0.65, p_by_range
    assert p_by_range[600.0] < 0.05, p_by_range
    # …and 50 hits at 100 rounds/s is 0.5 s of fire at 250 m but nine minutes at 600 m.
    assert G.HITS_TO_KILL / (G.GUN_RATE_HZ * p_by_range[250.0]) < 0.6


def test_clamp_to_alpha_cone():
    v = np.array([0.0, 200.0, 0.0])
    inside = G._unit(np.array([0.0, cos(radians(5.0)), sin(radians(5.0))]))
    assert np.allclose(G.clamp_to_alpha_cone(inside, v, radians(15.0)), inside)
    outside = G._unit(np.array([0.0, cos(radians(40.0)), sin(radians(40.0))]))
    got = G.clamp_to_alpha_cone(outside, v, radians(15.0))
    ang = degrees(float(np.arccos(np.clip(np.dot(G._unit(v), got), -1.0, 1.0))))
    assert abs(ang - 15.0) < 1e-9
    # …and it clamps along the great circle toward the target, not to some other bearing.
    assert float(np.dot(np.cross(G._unit(v), outside), np.cross(G._unit(v), got))) > 0.0


def test_target_path_changes_the_scored_hit_probability():
    """`p_hit_vec(target_path=…)` must actually USE the path — the plumbing check for the
    counterfactual, where every lay is scored against the track the bandit really flew.

    Scored at 600 m, where an 8 g break moves the bandit ~18 m in the time of flight and
    the same lay goes from a near-certain hit to a clean miss. (At 400 m the two scores
    differ by only ~2×, which would not distinguish "uses the path" from "off by a bit" —
    that regime is pinned down by `test_honest_gun_range_is_bounded_by_the_break_not_the_aim`.)
    """
    sp, sv, tp0, tv0, path = _break_path(600.0, 8.0)
    cv, _ = _tables(path, tv0, tp0)
    lay = G.refine_lay(
        G.intercept(sp, sv, tp0, tv0).boresight, sp, sv, cv, G.earth_rate_enu(LAT)
    )
    p_straight, _ = G.p_hit_vec(lay, sp, sv, tp0, tv0, samples=1500, lat_rad=LAT, seed=4)
    p_real, _ = G.p_hit_vec(
        lay, sp, sv, tp0, tv0, samples=1500, lat_rad=LAT, seed=4, target_path=path
    )
    assert p_straight > 0.6, p_straight
    assert p_real < 0.05, p_real


if __name__ == "__main__":  # measurement tables behind the assertions
    omega = G.earth_rate_enu(LAT)

    print(f"\nDRAG_FACTOR = {G.DRAG_FACTOR:.6e} m²/kg   rho({ALT:.0f} m) = {G.air_density(ALT):.4f}")
    print(f"HITS_TO_KILL = {G.HITS_TO_KILL}   hit sphere = {G.HIT_RADIUS_M} m")

    print("\nθ_bias (deg) — boresight correction, RANGE-INDEPENDENT")
    print("  off-vel |" + "".join(f"  V={v:>5.0f}" for v in (150, 200, 260, 300)))
    for off in (0, 5, 10, 12, 15, 20, 25, 30):
        row = "".join(f"  {degrees(G.theta_bias(v, radians(off))):>7.2f}" for v in (150, 200, 260, 300))
        print(f"  {off:>6}° |{row}")

    print("\nnose-off-velocity required (deg) at V=200 vs measured usable α 10.7-15.0°")
    for off in (5, 10, 12, 15, 20, 25):
        need = off + degrees(G.theta_bias(200.0, radians(off)))
        print(f"  target {off:>3}° off velocity -> nose {need:>6.2f}°  {'OK' if need <= 15.0 else 'UNREACHABLE'}")

    print("\nmiss distance (m) / P(hit), lead_error EXACTLY 0, V=200, static target")
    print("  off-vel |" + "".join(f"       {r:>4.0f} m" for r in (150, 250, 400, 600)))
    for off in (0, 5, 12, 20):
        cells = []
        for rng_m in (150.0, 250.0, 400.0, 600.0):
            sp, sv, tp, tv = _shot(rng_m, off, 200.0)
            nd = G._unit(G.naive_lead_point_from_parts(sp, tp, tv) - sp)
            _, miss, _ = G._closest_approach(sp, sv + nd * G.MUZZLE_SPEED_MPS, tp, tv, omega)
            p, _ = G.p_hit(nd, sp, sv, tp, tv, samples=400, lat_rad=LAT, seed=3)
            cells.append(f"  {miss:>6.2f}/{p:.2f}")
        print(f"  {off:>6}° |" + "".join(cells))
    cells = []
    for rng_m in (150.0, 250.0, 400.0, 600.0):
        sp, sv, tp, tv = _shot(rng_m, 12.0, 200.0)
        good = G.intercept(sp, sv, tp, tv).boresight
        _, miss, _ = G._closest_approach(sp, sv + good * G.MUZZLE_SPEED_MPS, tp, tv, omega)
        p, _ = G.p_hit(good, sp, sv, tp, tv, samples=400, lat_rad=LAT, seed=3)
        cells.append(f"  {miss:>6.2f}/{p:.2f}")
    print("  CORRECT |" + "".join(cells))

    print("\nclosed form vs full-physics truth (deg of boresight disagreement)")
    print("  off-vel |" + "".join(f"   {r:>4.0f} m" for r in (150, 250, 400, 600)))
    for off in (0, 5, 12, 20):
        cells = []
        for rng_m in (150.0, 250.0, 400.0, 600.0):
            sp, sv, tp, tv = _shot(rng_m, off, 200.0)
            c = G.intercept(sp, sv, tp, tv).boresight
            t = G.numeric_intercept(sp, sv, tp, tv, lat_rad=LAT).boresight
            cells.append(f"  {degrees(float(np.arccos(np.clip(np.dot(c, t), -1, 1)))):>7.4f}")
        print(f"  {off:>6}° |" + "".join(cells))

    print("\ncrossing target, 400 m, V=200 — miss (m) by lay")
    print("  cross |   naive   closed  numeric")
    for cross in (0.0, 50.0, 100.0, 150.0, 200.0):
        sp, sv, tp, tv = _shot(400.0, 12.0, 200.0, cross)
        row = []
        for lay in (
            G._unit(G.naive_lead_point_from_parts(sp, tp, tv) - sp),
            G.intercept(sp, sv, tp, tv).boresight,
            G.numeric_intercept(sp, sv, tp, tv, lat_rad=LAT).boresight,
        ):
            _, miss, _ = G._closest_approach(sp, sv + lay * G.MUZZLE_SPEED_MPS, tp, tv, omega)
            row.append(f"  {miss:>7.2f}")
        print(f"  {cross:>5.0f} |" + "".join(row))

    print("\nrefine_lay convergence — miss (m) by iteration count")
    print("  geometry                |" + "".join(f"  it{i}" for i in range(6)) + "     vs G-N")
    for rng_m, off, cross in ((250.0, 12.0, 0.0), (400.0, 12.0, 200.0), (600.0, 5.0, 100.0), (900.0, 12.0, 200.0)):
        sp, sv, tp, tv = _shot(rng_m, off, 200.0, cross)
        n = int(2.5 / G.SIM_DT)
        table = tp[None, :] + tv[None, :] * ((1 + np.arange(n)) * G.SIM_DT)[:, None]
        seed = G.intercept(sp, sv, tp, tv).boresight
        cells, ref = [], None
        for it in range(6):
            d = G.refine_lay(seed, sp, sv, table, omega, iters=it)
            _, miss, _ = G._closest_approach_table(sp, sv + d * G.MUZZLE_SPEED_MPS, table, omega)
            cells.append(f"  {miss:>6.3f}")
            if it == 4:
                ref = d
        num = G.numeric_intercept(sp, sv, tp, tv, lat_rad=LAT).boresight
        ang = degrees(float(np.arccos(np.clip(np.dot(ref, num), -1.0, 1.0))))
        print(f"  rng{rng_m:>4.0f} off{off:>3.0f} cross{cross:>4.0f} |" + "".join(cells) + f"   {ang:>7.4f}°")

    print("\nPREDICTION error — a const-velocity lay vs a bandit that BREAKS (the second wall)")
    print("  rng    g |  pred miss   ½at²    P(pred)  |  clair miss   P(clair)")
    for rng_m, g in ((150.0, 8.0), (250.0, 8.0), (400.0, 5.0), (400.0, 8.0), (600.0, 5.0), (600.0, 8.0)):
        sp, sv, tp0, tv0, path = _break_path(rng_m, g)
        cv, real = _tables(path, tv0, tp0)
        pred = G.refine_lay(G.intercept(sp, sv, tp0, tv0).boresight, sp, sv, cv, omega)
        clair = G.refine_lay(G.intercept_path(sp, sv, path).boresight, sp, sv, real, omega)
        _, m_pred, tof = G._closest_approach_table(sp, sv + pred * G.MUZZLE_SPEED_MPS, real, omega)
        _, m_clair, _ = G._closest_approach_table(sp, sv + clair * G.MUZZLE_SPEED_MPS, real, omega)
        p_pred, _ = G.p_hit_vec(pred, sp, sv, tp0, tv0, samples=2000, lat_rad=LAT, seed=4, target_path=path)
        p_clair, _ = G.p_hit_vec(clair, sp, sv, tp0, tv0, samples=2000, lat_rad=LAT, seed=4, target_path=path)
        print(
            f"  {rng_m:>4.0f} {g:>3.0f}g |  {m_pred:>7.2f}m {0.5 * g * 9.80665 * tof * tof:>7.2f}m"
            f"   {p_pred:>6.3f}   |  {m_clair:>8.2f}m   {p_clair:>6.3f}"
        )
