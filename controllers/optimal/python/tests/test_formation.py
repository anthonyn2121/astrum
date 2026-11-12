import math

import pytest
from astrum_client.runtime import AttitudeLimits, Reference
from astrum_client.wire import AutopilotTargets
from astrum_optimal.laws._guidance import true_heading_for_grid_track
from astrum_optimal.laws.figure8 import Figure8Law
from astrum_optimal.laws.formation import SLOTS, FormationLaw, slot_state

DT = 1.0 / 120.0
REF = Reference(0.0, 4572.0, 100.0)


def flying(make_state, *, e, n, ve, vn, t, aircraft_id, **kw):
    return make_state(
        aircraft_id=aircraft_id,
        e=e,
        n=n,
        vel_enu_e=ve,
        vel_enu_n=vn,
        vel_ned_e=ve,
        vel_ned_n=vn,
        tas_mps=math.hypot(ve, vn),
        yaw=math.atan2(ve, vn),
        sim_time_s=t,
        tick=round(t / DT),
        alt_hae_m=4572.0,
        **kw,
    )


def formation(max_bank_deg=25.0, **kw):
    law = FormationLaw(
        REF, AttitudeLimits(max_bank_deg, 10.0), leader_id=1, slot=SLOTS["right"], **kw
    )
    law.configure(DT)
    return law


def targets(law, make_input, state, fleet) -> AutopilotTargets:
    action = law.step(make_input(state, fleet))
    assert isinstance(action.control, AutopilotTargets)
    return action.control


def test_slot_motion_includes_rotation_of_the_offset(make_state):
    # Leader flying north at 100 m/s, turning right at 0.1 rad/s (wings level
    # so psi_dot = r); slot 70 m behind and 70 m right.
    lead = flying(make_state, e=0.0, n=0.0, ve=0.0, vn=100.0, t=10.0, aircraft_id=1, r=0.1)
    e, n, alt, ve, vn, ae, an = slot_state(lead, -70.0, 70.0, 0.0, lead.sim_time_s)
    assert (e, n, alt) == pytest.approx((70.0, -70.0, 4572.0))
    assert (ve, vn) == pytest.approx((0.1 * -70.0, 100.0 - 0.1 * 70.0))
    # Steady turn: the slot's acceleration is the turn rate x its velocity.
    assert (ae, an) == pytest.approx((0.1 * vn, -0.1 * ve))


def test_follower_in_slot_of_a_turning_leader_commands_the_turn(make_state, make_input):
    # In the slot with the slot's velocity, the follower must keep turning with
    # it: the coordinated-turn bank for w*V (~44 deg, inside the f16's 60), as the
    # heading offset the sim's heading loop (2.5 rad bank per rad) turns into it.
    law = formation(60.0)
    lead = flying(make_state, e=0.0, n=0.0, ve=0.0, vn=100.0, t=10.0, aircraft_id=1, r=0.1)
    _, _, _, sve, svn, _, _ = slot_state(lead, -70.0, 70.0, 0.0, lead.sim_time_s)
    me = flying(make_state, e=70.0, n=-70.0, ve=sve, vn=svn, t=10.0, aircraft_id=2)
    cmd = targets(law, make_input, me, {1: lead})
    v = math.hypot(sve, svn)
    bank = math.atan(0.1 * v / 9.80665)
    assert cmd.tas_mps == pytest.approx(v, abs=1e-6)
    assert cmd.heading_rad == pytest.approx(me.yaw + bank / 2.5, abs=1e-9)
    assert cmd.alt_hae_m == pytest.approx(4572.0)


def test_repeated_leader_sample_causes_no_spike(make_state, make_input):
    law = formation()
    lead = flying(make_state, e=0.0, n=0.0, ve=0.0, vn=100.0, t=10.0, aircraft_id=1)
    cmds = []
    # The follower keeps flying its slot while the leader's latest sample is
    # repeated (dropped/late telemetry) for a few ticks.
    for k in range(6):
        t = 10.0 + k * DT
        me = flying(
            make_state, e=70.0, n=-70.0 + 100.0 * k * DT, ve=0.0, vn=100.0, t=t, aircraft_id=2
        )
        cmds.append(targets(law, make_input, me, {1: lead}))
    for c in cmds:
        assert c.tas_mps == pytest.approx(100.0, abs=1e-6)
        assert c.heading_rad == pytest.approx(0.0, abs=1e-9)


def test_stale_leader_is_rejected_and_follower_holds(make_state, make_input):
    law = formation(leader_max_age_s=0.5)
    lead = flying(make_state, e=0.0, n=0.0, ve=0.0, vn=100.0, t=10.0, aircraft_id=1)
    me = flying(make_state, e=500.0, n=-900.0, ve=30.0, vn=95.0, t=10.6, aircraft_id=2)
    cmd = targets(law, make_input, me, {1: lead})
    assert cmd.heading_rad == pytest.approx(me.yaw)
    assert cmd.tas_mps == pytest.approx(me.tas_mps)
    # No leader at all: same hold.
    assert targets(law, make_input, me, {}).heading_rad == pytest.approx(me.yaw)


def test_commanded_bank_is_limited(make_state, make_input):
    # 5 km right of the slot: a hard left turn at exactly the bank limit, as the
    # heading offset the sim's heading loop turns into that bank.
    lead = flying(make_state, e=0.0, n=0.0, ve=0.0, vn=100.0, t=10.0, aircraft_id=1)
    me = flying(make_state, e=5000.0, n=-70.0, ve=0.0, vn=100.0, t=10.0, aircraft_id=2)
    for bank, gain in ((25.0, 2.5), (60.0, 2.5), (25.0, 1.2)):
        law = formation(bank, ap_heading_gain=gain)
        cmd = targets(law, make_input, me, {1: lead})
        assert cmd.heading_rad == pytest.approx(-math.radians(bank) / gain, rel=1e-9)
        assert cmd.tas_mps == pytest.approx(100.0, abs=1e-9)


def test_non_positive_heading_gain_is_rejected():
    with pytest.raises(ValueError):
        formation(ap_heading_gain=0.0)


def test_grid_track_is_converted_to_true_heading(make_state):
    conv = 0.01
    s = make_state(
        vel_enu_e=0.0,
        vel_enu_n=100.0,
        vel_ned_n=100.0 * math.cos(conv),
        vel_ned_e=100.0 * math.sin(conv),
    )
    assert true_heading_for_grid_track(s, 0.5) == pytest.approx(0.5 + conv)


def test_figure8_flies_the_tangent_and_paces_by_ground_speed(make_state, make_input):
    law = Figure8Law(REF, rx=8000.0, ry=4000.0)
    law.configure(DT)
    # On the path at tau = 0 (the centre), moving along the tangent (8000, 4000).
    v = 100.0
    ve, vn = v * 8000.0 / math.hypot(8000, 4000), v * 4000.0 / math.hypot(8000, 4000)
    s0 = flying(make_state, e=0.0, n=0.0, ve=ve, vn=vn, t=1.0, aircraft_id=1)
    cmd = targets(law, make_input, s0, {})
    assert cmd.heading_rad == pytest.approx(math.atan2(8000.0, 4000.0), abs=1e-9)
    assert (cmd.alt_hae_m, cmd.tas_mps) == (REF.alt_hae_m, REF.tas_mps)
    s1 = flying(make_state, e=ve * 0.5, n=vn * 0.5, ve=ve, vn=vn, t=1.5, aircraft_id=1)
    targets(law, make_input, s1, {})
    assert law.tau == pytest.approx(v * 0.5 / math.hypot(8000.0, 4000.0))
