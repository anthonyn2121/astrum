"""formation — decentralized formation keeping via a small per-follower MPC.

Each follower runs this law independently. It reads the leader's latest
telemetry (``--leader``) from the fleet-wide latest-value subscription, places
its slot in the leader's *track* frame (a named ``--slot`` preset or explicit
``--slot-fwd/-right/-up`` offsets), and drives its slot error to zero with a
receding-horizon MPC over point-mass kinematics. The slot's own acceleration
plus the MPC's correction becomes a high-level ``autopilot`` target
(heading/alt/airspeed) for the sim's fixed-wing inner loop. No follower talks
to another and there is no central optimizer: the only coupling is the
leader's broadcast state.

**Slot state from one leader sample.** The leader's sample is used *at its own
sim time*: the slot position is the leader position plus the offset rotated by
the leader's ground track ``chi = atan2(v_e, v_n)``, and the slot velocity is
the leader's velocity plus ``chi_dot x offset``, where ``chi_dot`` is the
leader's heading rate from its body rates,
``psi_dot = (q sin(phi) + r cos(phi)) / cos(theta)`` (equal to the track rate
with no wind and no sideslip), and the slot acceleration is ``chi_dot x v_slot``
(exact for a steady turn). The slot is then propagated to the follower's
sample time with that velocity. Nothing is finite-differenced, so a repeated or
late leader sample cannot produce a velocity spike; a leader sample older than
``--leader-max-age-s`` (sim time) is rejected and the follower holds its
current heading, the last slot altitude and its airspeed until fresh data
arrives.

**The model.** Per horizontal grid axis (East, North; decoupled, identical)
the relative motion to the slot is a double integrator::

    x = [pos_err, vel_err]   (self - slot),  u = acceleration relative to the slot

discretized (ZOH) at the control period over ``--horizon-s``, cost
``x'Qx + u'Ru``. The follower's acceleration is the slot's (feedforward) plus
``u``, limited to ``g * tan(--max-bank-deg)`` (what a coordinated turn at the
bank limit can produce).

**Acceleration -> autopilot targets.** The acceleration is split along and
across the follower's ground track. The cross-track part is a coordinated-turn
bank ``phi = atan(a_lat / g)``; the sim's heading loop banks
``--ap-heading-gain`` rad per rad of heading error, so the heading target is
the current heading plus ``phi / gain`` — an exact inversion of that loop at
steady state. Leading the velocity by the acceleration instead (the obvious
``v + a*T``) realises only ``g * gain * T / V`` of the asked-for turn (half at
120 m/s), which a 40-deg lobe-end turn turns into a standing slot error that
grows without bound. The along-track part leads the speed:
``ground speed + a_along * --cmd-lead-s``.

**No-wind assumption.** The speed command is the desired *ground speed*,
equal to true airspeed only with no wind (the sim has none today), and the
bank is computed from the ground-track turn. With wind, work in the air-mass
frame (subtract the wind from both velocities first).
"""

from __future__ import annotations

import logging
import math

from astrum_client import cli
from astrum_client.mathutil import clamp, wrap_pi
from astrum_client.runtime import AttitudeLimits, Law, Reference, StepInput
from astrum_client.wire import Action, AutopilotTargets, VehicleKind, VehicleState

from astrum_optimal.laws._guidance import grid_track
from astrum_optimal.laws.mpc import horizon_steps
from astrum_optimal.linear import ChannelMPC, discretize_zoh

G = 9.80665

# Named slots: (forward, right, up) offset in the leader's track frame, metres.
SLOTS = {
    "left": (-70.0, -70.0, 0.0),
    "right": (-70.0, 70.0, 0.0),
    "trail": (-140.0, 0.0, -30.0),
}

log = logging.getLogger("formation")


def leader_turn_rate(lead: VehicleState) -> float:
    """Heading rate (rad/s) from the leader's Euler angles and body rates."""
    cos_theta = math.cos(lead.pitch)
    if abs(cos_theta) < 1e-3:  # vertical: heading rate undefined
        return 0.0
    return (lead.q * math.sin(lead.roll) + lead.r * math.cos(lead.roll)) / cos_theta


def slot_state(
    lead: VehicleState, fwd: float, right: float, up: float, t: float
) -> tuple[float, float, float, float, float, float, float]:
    """Slot ``(e, n, alt, v_e, v_n, a_e, a_n)`` at sim time ``t`` from one leader
    sample; the acceleration is that of a steady turn at the leader's turn rate."""
    chi = grid_track(lead)
    sc, cc = math.sin(chi), math.cos(chi)
    # Offset in the grid: forward = (sin chi, cos chi), right = (cos chi, -sin chi).
    r_e = fwd * sc + right * cc
    r_n = fwd * cc - right * sc
    w = leader_turn_rate(lead)  # +clockwise seen from above
    v_e = lead.vel_enu_e + w * r_n
    v_n = lead.vel_enu_n - w * r_e
    a_e, a_n = w * v_n, -w * v_e
    age = t - lead.sim_time_s
    e = lead.e + r_e + v_e * age
    n = lead.n + r_n + v_n * age
    alt = lead.alt_hae_m + up + lead.vel_enu_u * age
    return e, n, alt, v_e, v_n, a_e, a_n


class FormationLaw(Law):
    """Per-follower formation MPC (fixed-wing only)."""

    vehicle_kinds = frozenset({VehicleKind.FIXED_WING})
    needs_fleet = True

    @classmethod
    def add_args(cls, p) -> None:
        f, pos = cli.finite_float, cli.positive_float
        p.add_argument(
            "--leader", type=cli.aircraft_id, default=1, help="aircraft id of the leader to follow"
        )
        p.add_argument("--slot", choices=sorted(SLOTS), default="trail", help="named slot")
        p.add_argument("--slot-fwd", type=f, help="override slot forward offset (m, +ahead)")
        p.add_argument("--slot-right", type=f, help="override slot right offset (m, +starboard)")
        p.add_argument("--slot-up", type=f, help="override slot up offset (m, +above)")
        p.add_argument("--horizon-s", type=pos, default=4.0, help="MPC prediction horizon (s)")
        p.add_argument("--q-pos", type=f, default=0.8, help="MPC weight on slot position error")
        p.add_argument("--q-vel", type=f, default=3.0, help="MPC weight on slot velocity error")
        p.add_argument(
            "--r-acc", type=pos, default=2.0, help="MPC weight on commanded acceleration"
        )
        p.add_argument(
            "--cmd-lead-s",
            type=f,
            default=2.5,
            help="lead time turning the along-track acceleration into a speed command (s)",
        )
        p.add_argument(
            "--ap-heading-gain",
            type=pos,
            default=2.5,
            help="the sim autopilot's bank per rad of heading error, inverted to command a "
            "bank (fighter profile 2.5, trainer 1.2)",
        )
        p.add_argument("--v-min", type=f, default=80.0, help="min commanded airspeed (m/s)")
        p.add_argument("--v-max", type=f, default=250.0, help="max commanded airspeed (m/s)")
        p.add_argument(
            "--leader-max-age-s",
            type=pos,
            default=0.5,
            help="reject leader telemetry older than this (sim time) and hold",
        )

    @classmethod
    def from_args(cls, args):
        if args.leader == args.id:
            raise ValueError("--leader must differ from --id")
        fwd, right, up = SLOTS[args.slot]
        return cls(
            args.reference,
            args.limits,
            leader_id=args.leader,
            slot=(
                fwd if args.slot_fwd is None else args.slot_fwd,
                right if args.slot_right is None else args.slot_right,
                up if args.slot_up is None else args.slot_up,
            ),
            horizon_s=args.horizon_s,
            q_pos=args.q_pos,
            q_vel=args.q_vel,
            r_acc=args.r_acc,
            cmd_lead_s=args.cmd_lead_s,
            ap_heading_gain=args.ap_heading_gain,
            v_min=args.v_min,
            v_max=args.v_max,
            leader_max_age_s=args.leader_max_age_s,
        )

    def __init__(
        self,
        ref: Reference,
        limits: AttitudeLimits,
        *,
        leader_id: int,
        slot: tuple[float, float, float] = SLOTS["trail"],
        horizon_s: float = 4.0,
        q_pos: float = 0.8,
        q_vel: float = 3.0,
        r_acc: float = 2.0,
        cmd_lead_s: float = 2.5,
        ap_heading_gain: float = 2.5,
        v_min: float = 80.0,
        v_max: float = 250.0,
        leader_max_age_s: float = 0.5,
    ):
        if not ap_heading_gain > 0.0:
            raise ValueError("--ap-heading-gain must be > 0")
        self.leader_id = leader_id
        self.off_fwd, self.off_right, self.off_up = slot
        self.horizon_s = horizon_s
        self.q_pos, self.q_vel, self.r_acc = q_pos, q_vel, r_acc
        self.cmd_lead = cmd_lead_s
        self.ap_heading_gain = ap_heading_gain
        self.v_min, self.v_max = v_min, v_max
        self.max_age = leader_max_age_s
        self.max_bank = math.radians(limits.max_bank_deg)
        self.max_acc = G * math.tan(self.max_bank)
        self.hold_alt = ref.alt_hae_m
        self.mpc: ChannelMPC | None = None
        self._holding: str | None = None

    def configure(self, control_period_s: float) -> None:
        a, b = discretize_zoh([[0.0, 1.0], [0.0, 0.0]], [[0.0], [1.0]], control_period_s)
        q = [[self.q_pos, 0.0], [0.0, self.q_vel]]
        n = horizon_steps(self.horizon_s, control_period_s)
        self.mpc = ChannelMPC(a, b, q, [[self.r_acc]], n)

    def _hold(self, s: VehicleState, why: str) -> Action:
        if self._holding != why:
            log.warning("holding: %s", why)
            self._holding = why
        return Action(
            AutopilotTargets(s.yaw, self.hold_alt, clamp(s.tas_mps, self.v_min, self.v_max))
        )

    def step(self, inp: StepInput) -> Action:
        s = inp.state
        lead = inp.fleet.get(self.leader_id)
        if lead is None:
            return self._hold(s, f"no telemetry from leader {self.leader_id} yet")
        age = s.sim_time_s - lead.sim_time_s
        if abs(age) > self.max_age:
            return self._hold(s, f"leader telemetry is {age:.2f} s old (> {self.max_age} s)")
        if self._holding is not None:
            log.info("leader telemetry fresh again; formation keeping")
            self._holding = None

        slot_e, slot_n, slot_alt, slot_ve, slot_vn, slot_ae, slot_an = slot_state(
            lead, self.off_fwd, self.off_right, self.off_up, s.sim_time_s
        )
        self.hold_alt = slot_alt

        acc_e = slot_ae + float(self.mpc.move([s.e - slot_e, s.vel_enu_e - slot_ve])[0])
        acc_n = slot_an + float(self.mpc.move([s.n - slot_n, s.vel_enu_n - slot_vn])[0])
        acc = math.hypot(acc_e, acc_n)
        if acc > self.max_acc:
            acc_e, acc_n = acc_e * self.max_acc / acc, acc_n * self.max_acc / acc

        # Split along / across the ground track (right of track = (t_n, -t_e)).
        speed = math.hypot(s.vel_enu_e, s.vel_enu_n)
        if speed > 1.0:
            t_e, t_n = s.vel_enu_e / speed, s.vel_enu_n / speed
        else:
            t_e, t_n = math.sin(s.yaw), math.cos(s.yaw)
        a_along = acc_e * t_e + acc_n * t_n
        a_right = acc_e * t_n - acc_n * t_e
        bank = clamp(math.atan2(a_right, G), -self.max_bank, self.max_bank)
        heading = wrap_pi(s.yaw + bank / self.ap_heading_gain)
        tas = clamp(speed + a_along * self.cmd_lead, self.v_min, self.v_max)
        return Action(AutopilotTargets(heading, slot_alt, tas))
