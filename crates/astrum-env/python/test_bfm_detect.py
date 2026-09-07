#!/usr/bin/env python3
"""Tests for `astrum_dogfight.bfm_detect` — the BFM event detector.

Two parts:
  1. The scalar — fabricated `AState`s pin `chi` at the four geometries whose answer is known by
     inspection (in-plane pull, lift square across the LOS, rolled onto an abeam bandit, and the
     degenerate dead-ahead case where `chi` is undefined rather than zero).
  2. The detectors — synthetic `Sample` traces, built directly, exercise each clause. Every detector
     gets at least one POSITIVE and one NEAR-MISS trace that differs in exactly the clause under
     test, because a detector that fires on everything is worse than none at all: it would launder a
     passive policy as a manoeuvring one, which is the specific failure this module exists to catch.

Run via:  make env-test
"""

from math import degrees, radians

import numpy as np
from astrum_dogfight import bfm_detect as B
from astrum_dogfight.geometry import AState


def _state(*, pos, vel, phi=0.0, theta=0.0, psi=0.0, nz=1.0):
    pos = np.asarray(pos, float)
    return AState(
        id=1,
        team=0,
        alive=True,
        pos=pos,
        vel=np.asarray(vel, float),
        phi=phi,
        theta=theta,
        psi=psi,
        p=0.0,
        q=0.0,
        r=0.0,
        alpha=0.0,
        beta=0.0,
        nz=nz,
        airspeed=float(np.linalg.norm(vel)),
        alt=float(pos[2]),
        health=100.0,
        ammo=511,
    )


# ======================================================================================
# 1. The scalar
# ======================================================================================
def test_chi_zero_when_lift_points_at_bandit():
    """Wings level, bandit ahead and ABOVE: the lift vector is already on the LOS ⇒ in-plane."""
    me = _state(pos=[0, 0, 3000], vel=[0, 200, 0])  # heading north, level
    bandit = _state(pos=[0, 2000, 3600], vel=[0, 200, 0])
    chi, valid = B.out_of_plane_angle(me, bandit)
    assert valid
    assert abs(degrees(chi)) < 1.0, degrees(chi)


def test_chi_ninety_when_bandit_abeam_and_wings_level():
    """Bandit square on the right beam with wings level: the lift is 90° off the LOS plane.

    This is the canonical out-of-plane state — to pull toward him you must first roll 90°.
    """
    me = _state(pos=[0, 0, 3000], vel=[0, 200, 0])
    bandit = _state(pos=[2000, 0, 3000], vel=[0, 200, 0])
    chi, valid = B.out_of_plane_angle(me, bandit)
    assert valid
    assert abs(abs(degrees(chi)) - 90.0) < 1.0, degrees(chi)


def test_chi_returns_to_zero_once_rolled_onto_an_abeam_bandit():
    """Same abeam geometry, but rolled 90° right — now the pull is in-plane, so chi ≈ 0."""
    me = _state(pos=[0, 0, 3000], vel=[0, 200, 0], phi=radians(90.0))
    bandit = _state(pos=[2000, 0, 3000], vel=[0, 200, 0])
    chi, valid = B.out_of_plane_angle(me, bandit)
    assert valid
    assert abs(degrees(chi)) < 2.0, degrees(chi)


def test_chi_invalid_when_bandit_is_dead_ahead():
    """LOS along the velocity vector ⇒ no LOS plane exists ⇒ undefined, NOT an in-plane pull.

    Reporting 0.0 here without the flag would let a nose-on tail chase read as continuous perfect
    in-plane tracking, which is exactly the kind of false signal this program has been burned by.
    """
    me = _state(pos=[0, 0, 3000], vel=[0, 200, 0])
    bandit = _state(pos=[0, 2000, 3000], vel=[0, 200, 0])
    chi, valid = B.out_of_plane_angle(me, bandit)
    assert not valid
    assert chi == 0.0


def test_lethal_is_range_gated():
    """A PERFECT lay at 1519 m (the policy's measured median firing range) is not lethal.

    p_hit per round at 1200 m is 0.000, so `tHit` (ungated) inflates and `tLeth` must not.
    """
    near = _state(pos=[0, 0, 3000], vel=[0, 200, 0])
    far_bandit = _state(pos=[0, 1519, 3000], vel=[0, 200, 0])
    s = B.make_sample(0.0, near, far_bandit, [0, 0, 0, 1])
    assert s.lay_miss_m < 8.0, s.lay_miss_m  # the lay itself is perfect (co-speed, dead ahead)
    assert not s.lethal  # ...and it still must not count

    close_bandit = _state(pos=[0, 400, 3000], vel=[0, 200, 0])
    assert B.make_sample(0.0, near, close_bandit, [0, 0, 0, 1]).lethal


def test_cpa_miss_collapses_past_the_pass():
    """Already diverging ⇒ report the present separation, not a stale prediction behind the jet."""
    # co-altitude, 500 m apart laterally, flying directly apart
    assert abs(B.cpa_miss_m([0, 0, 0], [0, -100, 0], [500, 0, 0], [0, 100, 0]) - 500.0) < 1e-6
    # head-on, offset 300 m laterally ⇒ CPA miss is the 300 m offset
    assert abs(B.cpa_miss_m([0, 0, 0], [0, 100, 0], [300, 2000, 0], [0, -100, 0]) - 300.0) < 1e-6


# ======================================================================================
# 2. Detectors — synthetic traces
# ======================================================================================
DT = 1.0 / 24.0  # the trainer's control rate


def _tr(specs: list[dict]) -> list[B.Sample]:
    """Build a trace from per-tick overrides on a benign in-plane closing default."""
    out = []
    for i, sp in enumerate(specs):
        d = dict(
            t=i * DT,
            chi=0.0,
            chi_valid=True,
            gamma=0.0,
            nz=1.0,
            phi=0.0,
            alt=3000.0,
            ail=0.0,
            rng=1000.0,
            ata=radians(30.0),
            ata_az=radians(30.0),
            aspect=radians(90.0),
            closure=50.0,
            cpa_miss=500.0,
            es_rel=0.0,
            lay_miss_m=500.0,
            lethal=False,
        )
        d.update(sp)
        out.append(B.Sample(**d))
    return out


def _hold(n, **kw):
    return [dict(**kw) for _ in range(n)]


def _displacement_specs(*, nz=6.0, aspect_exit_deg=60.0, roll_back=True, side_flip=False):
    """In-plane → 75° out of plane held ~1 s under load → back in-plane, aspect improving."""
    pre = _hold(12, chi=radians(5.0))
    az = radians(-30.0) if side_flip else radians(30.0)
    out = _hold(30, chi=radians(75.0), nz=nz, ata_az=az)
    post_chi = radians(10.0) if roll_back else radians(75.0)
    post = _hold(24, chi=post_chi, nz=nz, aspect=radians(aspect_exit_deg))
    return pre + out + post


def _displacement_trace(**kw):
    return _tr(_displacement_specs(**kw))


def test_d1_detects_a_loaded_displacement_roll():
    ev = B.detect_displacement_rolls(_displacement_trace())
    assert len(ev) == 1, ev
    assert ev[0].kind == "displacement_roll"
    assert ev[0].d_aspect_deg < -B.ASPECT_GAIN_DEG
    assert ev[0].detail["chi_peak_deg"] > B.CHI_OUT_DEG


def test_d1_rejects_an_unloaded_drift():
    """The nz clause: a wings-level drift with the lift nowhere near the bandit is not a manoeuvre.

    This is the passivity failure mode that several curricula in this program mistook for progress.
    """
    assert B.detect_displacement_rolls(_displacement_trace(nz=1.0)) == []


def test_d1_rejects_a_roll_that_never_comes_back():
    """Rolling out and staying out is a departure, not a displacement."""
    assert B.detect_displacement_rolls(_displacement_trace(roll_back=False)) == []


def test_d1_rejects_a_side_flip():
    """`sign(ata_az)` flipping means the bandit crossed the nose = an overshoot, not a displacement."""
    assert B.detect_displacement_rolls(_displacement_trace(side_flip=True)) == []


def test_d1_rejects_a_roll_that_buys_no_angles():
    """Clause (f): the shape can be perfect and still have accomplished nothing."""
    assert B.detect_displacement_rolls(_displacement_trace(aspect_exit_deg=90.0)) == []


def _yoyo_trace(*, up=True, aspect_exit_deg=90.0, closure_exit=60.0, nz=6.0, rng_exit=800.0):
    """Hot entry (closure 180) → 40° flight-path excursion under load → back to the horizon.

    The flight path must RAMP from the horizon, not start displaced: the detector measures the swing
    relative to the flight path at the overshoot entry, so a fixture that begins at 20° would only
    ever show a 20° swing.
    """
    sgn = 1.0 if up else -1.0
    cold = _hold(6, closure=50.0)
    entry = [
        dict(closure=180.0, gamma=sgn * radians(40.0 * i / 23.0), nz=nz) for i in range(24)
    ]
    peak = _hold(12, closure=140.0, gamma=sgn * radians(40.0), nz=nz)
    back = _hold(24, closure=closure_exit, gamma=0.0, rng=rng_exit, aspect=radians(aspect_exit_deg))
    return _tr(cold + entry + peak + back)


def test_d2_detects_a_clean_high_yoyo():
    ev = B.detect_yoyos(_yoyo_trace())
    assert len(ev) == 1, ev
    assert ev[0].kind == "high_yoyo"
    assert ev[0].detail["clean"]
    assert not ev[0].detail["failed_aspect"]
    assert ev[0].detail["closure_entry"] > B.CLOSURE_OVERSHOOT
    assert ev[0].detail["closure_exit"] < B.CLOSURE_OVERSHOOT


def test_d2_classifies_a_low_yoyo():
    ev = B.detect_yoyos(_yoyo_trace(up=False))
    assert len(ev) == 1 and ev[0].kind == "low_yoyo", ev


def test_d2_flags_the_c70_zoom_away():
    """The load-bearing clause: aspect degrading past the slip bar is a zoom-away, not a yo-yo.

    c70's recorded pathology (mean ata 86°→158°, dealt 27→0) is exactly this. The event is still
    RETURNED — tagged `failed_aspect` — so the regression stays visible instead of being filtered
    into silence, and `clean` is False so it cannot inflate a headline count.
    """
    ev = B.detect_yoyos(_yoyo_trace(aspect_exit_deg=140.0))
    assert len(ev) == 1
    assert ev[0].detail["failed_aspect"]
    assert not ev[0].detail["clean"]


def test_d2_rejects_a_zoom_coast():
    """The nz clause again: an unloaded ballistic zoom has the right flight-path shape."""
    assert B.detect_yoyos(_yoyo_trace(nz=1.0)) == []


def test_d2_marks_a_failed_exit_when_closure_is_not_bled():
    ev = B.detect_yoyos(_yoyo_trace(closure_exit=200.0))
    assert len(ev) == 1 and ev[0].detail["failed_exit"] and not ev[0].detail["clean"]


def test_d3_requires_cpa_built_then_cashed():
    """Turning room: the predicted CPA must OPEN during the hold and CLOSE after."""
    pre = _hold(6, cpa_miss=200.0)
    build = [dict(chi=radians(60.0), nz=6.0, cpa_miss=200.0 + 8.0 * i) for i in range(24)]
    cash = _hold(24, chi=radians(10.0), nz=6.0, cpa_miss=150.0)
    assert len(B.detect_turning_room(_tr(pre + build + cash))) == 1

    # same hold, but the CPA keeps growing — that is flying away, and it must not count
    away = _hold(24, chi=radians(10.0), nz=6.0, cpa_miss=900.0)
    assert B.detect_turning_room(_tr(pre + build + away)) == []


def test_d4_measures_the_held_state_not_the_rate_command():
    """D4's CORRECTION: `a[0]` is a roll-RATE command, so a held roll shows as a PULSE, not a hold.

    `Control::Raw` writes `a[0]` to `fcs/aileron-cmd-norm`, and with `fcs/fbw-override` at its initial
    0 — nothing in this repo ever sets it — `f16.xml:364-396` feeds that into a roll-rate PID. So the
    command shape of a correctly flown displacement roll is: pulse the rate to establish bank, then
    command ~0 rate to HOLD it. The original D4 ("longest run with `sign(ail)` constant and
    `abs(ail) > 0.5`") therefore scores a correctly flown roll near ZERO and scores a jet spinning
    about its velocity vector highly — anti-informative, not merely noisy. Measured on c95 it read
    0.17 s, which would have been reported as "the policy cannot commit to a roll".

    The corrected statistics read the STATE the rate command integrates to.
    """
    bank, out = radians(70.0), radians(70.0)
    roll_in = _hold(7, phi=0.2, chi=0.2, ail=0.9, nz=5.0)
    hold = _hold(60, phi=bank, chi=out, ail=0.05, nz=5.0)  # banked, out of plane, rate cmd ~0
    roll_out = _hold(7, phi=0.2, chi=0.2, ail=-0.9, nz=5.0)
    c = B.commitment(_tr(roll_in + hold + roll_out))

    hold_s = 59 * DT  # 60 samples ⇒ 59 intervals
    assert c["longest_bank_hold_s"] >= hold_s - 1e-9, c
    assert c["longest_chi_hold_s"] >= hold_s - 1e-9, c
    assert abs(c["bank_max_deg"] - 70.0) < 1e-6, c
    # The OLD statistic on the SAME correctly flown roll sees only the 0.3 s pulses.
    assert c["longest_rate_cmd_run_s"] <= 7 * DT, c
    assert c["longest_rate_cmd_run_s"] < c["longest_bank_hold_s"] / 5.0, c

    # Sign constancy is load-bearing on the state too: a jet rocking ±70° is banked hard every tick
    # but committed to nothing.
    rock = _tr([
        dict(phi=bank if i % 2 == 0 else -bank, chi=out if i % 2 == 0 else -out, nz=5.0)
        for i in range(60)
    ])
    cr = B.commitment(rock)
    assert cr["longest_bank_hold_s"] == 0.0, ("sign flips must break the hold", cr)
    assert cr["longest_chi_hold_s"] == 0.0, cr

    # An invalid `chi` tick (LOS along the velocity vector ⇒ chi undefined, defaulted to 0.0) must
    # BREAK a chi hold rather than continue it or read as an in-plane pull. Bank, always well defined,
    # is deliberately unaffected — that asymmetry is the point.
    gapped = roll_in + _hold(30, phi=bank, chi=out, ail=0.05, nz=5.0) \
        + [dict(phi=bank, chi=0.0, chi_valid=False, ail=0.05, nz=5.0)] \
        + _hold(29, phi=bank, chi=out, ail=0.05, nz=5.0) + roll_out
    cg = B.commitment(_tr(gapped))
    assert cg["longest_chi_hold_s"] < c["longest_chi_hold_s"], cg
    assert abs(cg["longest_bank_hold_s"] - c["longest_bank_hold_s"]) < 1e-12, cg


def test_d4_holds_require_load_and_report_the_altitude_they_spend():
    """A hold must be a LOADED turn, and it must be priced in altitude.

    Shipped without the `nz > NZ_LOADED` clause that D1/D2/D3 all carry, D4 read c87 at a 44.6 s mean
    bank hold and a 13.0 s mean chi hold — higher than c95 on every out-of-plane statistic — while
    c87 decked 5/5 with zero lethal ticks. An unloaded wings-over descent is not commitment. And
    because a LOADED hold can still be a high-g spiral into the ground, duration alone is not enough
    either: the altitude spent is what separates a BFM turn from a death spiral.
    """
    bank, out = radians(70.0), radians(70.0)
    unloaded = _tr(_hold(60, phi=bank, chi=out, nz=1.2))  # steeply banked, barely pulling
    cu = B.commitment(unloaded)
    assert cu["longest_bank_hold_s"] == 0.0, ("an unloaded bank is not a hold", cu)
    assert cu["longest_chi_hold_s"] == 0.0, cu
    # Same geometry, now actually pulling ⇒ both holds appear. Load is the only difference.
    cl = B.commitment(_tr(_hold(60, phi=bank, chi=out, nz=5.0)))
    assert cl["longest_bank_hold_s"] >= 59 * DT - 1e-9, cl
    assert cl["longest_chi_hold_s"] >= 59 * DT - 1e-9, cl

    # dalt is SIGNED and measured across the hold: a descending hold must report negative.
    diving = _tr([dict(phi=bank, chi=out, nz=5.0, alt=3000.0 - 20.0 * i) for i in range(60)])
    cd = B.commitment(diving)
    assert cd["longest_bank_hold_s"] >= 59 * DT - 1e-9, cd
    assert abs(cd["bank_hold_dalt_m"] - (-20.0 * 59)) < 1e-6, cd
    assert abs(cd["chi_hold_dalt_m"] - (-20.0 * 59)) < 1e-6, cd
    # A level hold of the same duration is indistinguishable on duration and separated only by dalt.
    assert cl["bank_hold_dalt_m"] == 0.0, cl
    assert cd["longest_bank_hold_s"] == cl["longest_bank_hold_s"], (cd, cl)

    # No hold at all ⇒ no altitude to attribute, not a stale or partial number.
    assert B.commitment(_tr(_hold(10, phi=0.0, nz=1.0)))["bank_hold_dalt_m"] == 0.0


def test_the_rate_command_dither_statistic_still_needs_magnitude_and_sign():
    """The rate-command run is KEPT as a DITHER measure — on this interface a LOW value is correct.

    It is no longer read as commitment (see the test above), but its internal logic is still pinned
    against both historical mistakes: a stick sawing hard left/right every 4 ticks has
    `abs(ail) > 0.5` throughout, so the sign-agnostic form reports a long "sustained roll"; and a
    sustained LEFT roll is invisible to the one-sided `ail > 0.5` form. Only the conjunction gets both
    right. What changed is the CONCLUSION drawn from the number, not the number.
    """
    saw = _tr([dict(ail=0.9 if (i // 4) % 2 == 0 else -0.9) for i in range(48)])
    held_left = _tr([dict(ail=-0.9) for _ in range(48)])

    assert B.commitment(saw)["longest_rate_cmd_run_s"] < 0.25  # 4 ticks ≈ 0.17 s
    assert B.commitment(saw)["roll_runs"] == 12
    assert B.commitment(held_left)["longest_rate_cmd_run_s"] > 1.9  # 48 ticks ≈ 2.0 s
    assert B.commitment(held_left)["roll_runs"] == 1
    # the signed window mean separates them even more sharply than the run length
    assert B.commitment(saw)["max_abs_window_mean_ail"] < 0.3
    assert B.commitment(held_left)["max_abs_window_mean_ail"] > 0.85
    # ...and neither says anything about commitment: the SAW is the one that holds no bank, but the
    # sustained rate command is ALSO uncommitted — it is still rolling, never settling on a bank.
    assert B.commitment(held_left)["longest_bank_hold_s"] == 0.0


def test_d4_ignores_trim_sized_input():
    assert B.commitment(_tr([dict(ail=0.2) for _ in range(48)]))["longest_rate_cmd_run_s"] == 0.0


def test_d5_reports_both_base_rates_and_is_finite_only_when_grounded():
    """The ratio is meaningless without its two rates, so they are always returned.

    With no lethal tick anywhere the ratio must be NaN — not 0, not 1 — because "no evidence" and
    "evidence of no effect" are different claims.
    """
    # A baseline window must be event-free AND a full 8 s, so the episode has to be long enough to
    # contain one — 10 s of quiet, the manoeuvre, then a tail. A 3 s synthetic trace yields NO
    # baseline at all, which is the correct answer for a 3 s trace and not something to paper over.
    tr = _tr([{}] * 240 + _displacement_specs() + [{}] * 240)
    ev = B.detect_displacement_rolls(tr)
    assert len(ev) == 1

    dead = B.lethality_lift(tr, ev)
    assert dead["n_event_windows"] == 1 and dead["n_base_windows"] > 0
    assert dead["event_hits"] == 0 and np.isnan(dead["ratio"])

    # make the tick right after the event lethal ⇒ event window hits, baseline does not
    hot = list(tr)
    hot[ev[0].i0 + 2] = B.Sample(**{**hot[ev[0].i0 + 2].__dict__, "lethal": True})
    lift = B.lethality_lift(hot, ev)
    assert lift["p_event"] == 1.0
    assert lift["ratio"] == float("inf")  # honest: undefined-but-favourable, not a big finite number


def test_detect_all_and_summarize_run_clean_on_a_flat_trace():
    """A passive episode must produce zeros everywhere and crash nowhere."""
    flat = _tr([{} for _ in range(240)])
    r = B.detect_all(flat)
    assert r["events"] == [] and r["lethal_ticks"] == 0
    s = B.summarize([flat, flat])
    assert s["episodes"] == 2 and s["d1_per_ep"] == 0.0 and np.isnan(s["ratio"])
    assert "D5" in B.format_summary(s)
