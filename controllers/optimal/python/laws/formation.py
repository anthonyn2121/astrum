"""formation — decentralized formation-keeping via a small per-follower MPC.

Each follower runs this law *independently*. It watches the leader's telemetry
off the Zenoh bus (``--leader``), places its slot in the leader's heading frame
(a named ``--slot`` preset, or explicit ``--slot-fwd/-right/-up`` offsets), and
drives its slot error to zero with a receding-horizon MPC over point-mass
kinematics. The MPC output — a desired velocity vector — becomes a high-level
``autopilot`` target (heading/alt/airspeed) that the sim's fixed-wing inner loop
tracks.

**Decentralized:** no follower talks to any other follower and there is no
central optimizer. The only coupling is the leader's broadcast state (which the
runtime hands over via ``self.world`` because this law sets ``needs_world``), so
each aircraft solves its own QP from telemetry alone.

**The model.** Per horizontal axis (East, North, decoupled and identical) the
relative motion to the slot is a double integrator::

    x = [pos_err, vel_err]     pos_err = self - slot,  vel_err = self - slot vel
    x[k+1] = A x[k] + B u[k]   A = [[1, dt],[0, 1]],  B = [[dt^2/2],[dt]]

with stage cost ``x'Qx + u'Ru`` (u = commanded acceleration). :class:`ChannelMPC`
(reused from ``laws/mpc.py``) precomputes the first-move gain, so each frame is
two matrix-vector products. The accel is turned into a velocity command by
leading the current velocity along it (``--cmd-lead-s``); at steady state
(errors -> 0) that is exactly the slot's velocity, i.e. formation lock. The
horizon look-ahead (``--horizon-s``) must outlast the follower's turn/accel
response or the optimizer under-uses the input — see ``laws/mpc.py``.

Hard limits (min/max airspeed) are applied as clamps on the command; the full
per-horizon QP with bank/accel bounds is the documented next step in
:meth:`ChannelMPC.move`.
"""

import math

import numpy as np
from astrum_control import Command, Reference, VehicleState
from astrum_control import Law as BaseLaw

from laws.mpc import ChannelMPC

# Named slots: (forward, right, up) offset in the leader's heading frame, metres.
# forward = along the leader's nose, right = starboard wing, up = above.
SLOTS = {
    "left": (-70.0, -70.0, 0.0),
    "right": (-70.0, 70.0, 0.0),
    "trail": (-140.0, 0.0, -30.0),
}


def add_args(p):
    """formation slot + MPC flags."""
    p.add_argument("--leader", type=int, default=1, help="aircraft id of the leader to follow")
    p.add_argument(
        "--slot",
        choices=sorted(SLOTS),
        default="trail",
        help="named formation slot (leader-frame offset)",
    )
    p.add_argument("--slot-fwd", type=float, help="override slot forward offset (m, +ahead)")
    p.add_argument("--slot-right", type=float, help="override slot right offset (m, +starboard)")
    p.add_argument("--slot-up", type=float, help="override slot up offset (m, +above)")
    p.add_argument("--horizon", type=int, default=20, help="MPC prediction horizon (steps)")
    p.add_argument(
        "--horizon-s",
        type=float,
        default=4.0,
        help="total look-ahead (s); model step = horizon-s / horizon",
    )
    # Defaults tuned offline against a point-mass model of the sim's fixed-wing
    # inner loop (bank = 1.2*hdg_err, clamped +/-25 deg, ~1.5 s roll lag) and then
    # confirmed in a full JSBSim run: they capture the slot in ~5 s and hold it to
    # ~20 m over the gentle sections, degrading to ~35 m only at the figure-eight's
    # tightest lobe tip (where the leader decelerates hardest) — well clear of the
    # gains where the inner-loop lag drives instability.
    p.add_argument("--q-pos", type=float, default=0.8, help="MPC weight on slot position error")
    p.add_argument("--q-vel", type=float, default=3.0, help="MPC weight on slot velocity error")
    p.add_argument("--r-acc", type=float, default=2.0, help="MPC weight on commanded acceleration")
    p.add_argument(
        "--cmd-lead-s",
        type=float,
        default=2.5,
        help="lead time turning the MPC accel into a velocity command (s)",
    )
    p.add_argument("--v-min", type=float, default=80.0, help="min commanded airspeed (m/s)")
    p.add_argument("--v-max", type=float, default=250.0, help="max commanded airspeed (m/s)")


class Law(BaseLaw):
    """Per-follower formation MPC. See module docstring."""

    needs_world = True  # ask the runtime for the whole fleet's telemetry

    def __init__(self, args):
        super().__init__(args)
        self.world = {}  # runtime overwrites with its live {id: VehicleState}
        self.leader_id = args.leader

        fwd, right, up = SLOTS[args.slot]
        if args.slot_fwd is not None:
            fwd = args.slot_fwd
        if args.slot_right is not None:
            right = args.slot_right
        if args.slot_up is not None:
            up = args.slot_up
        self.off_fwd, self.off_right, self.off_up = fwd, right, up

        n = args.horizon
        dt = args.horizon_s / n
        # Per-axis double integrator: state [pos_err, vel_err], input = accel.
        # East and North share identical dynamics, so one ChannelMPC serves both.
        a = np.array([[1.0, dt], [0.0, 1.0]])
        b = np.array([[0.5 * dt * dt], [dt]])
        q = np.diag([args.q_pos, args.q_vel])
        self.mpc = ChannelMPC(a, b, q, [[args.r_acc]], n)

        self.cmd_lead = args.cmd_lead_s
        self.v_min, self.v_max = args.v_min, args.v_max
        self.alt_fallback = args.alt
        self._prev_slot = None  # (t, slot_e, slot_n) for the slot-velocity estimate

    def control(self, s: VehicleState, ref: Reference) -> Command:
        lead = self.world.get(self.leader_id)
        if lead is None or lead.aircraft_id == s.aircraft_id:
            # Leader not seen yet (or --leader == our own id): hold what we have.
            return Command.autopilot(s.yaw, self.alt_fallback, max(self.v_min, s.airspeed))

        # Slot position in ENU = leader position + offset rotated by leader
        # heading psi. Forward unit = (sin psi, cos psi); right = (cos psi, -sin psi).
        psi = lead.yaw
        sp, cp = math.sin(psi), math.cos(psi)
        slot_e = lead.e + self.off_fwd * sp + self.off_right * cp
        slot_n = lead.n + self.off_fwd * cp - self.off_right * sp
        slot_alt = lead.alt_m + self.off_up

        # Slot velocity by finite-differencing the slot position — this captures
        # the leader's turn, which sweeps the offset around. Fall back to the
        # leader's own velocity on the first frame.
        slot_ve, slot_vn = lead.v_e, lead.v_n
        if self._prev_slot is not None:
            pt, pe, pn = self._prev_slot
            d = s.t - pt
            if d > 1e-6:
                slot_ve = (slot_e - pe) / d
                slot_vn = (slot_n - pn) / d
        self._prev_slot = (s.t, slot_e, slot_n)

        # MPC per axis on [pos_err, vel_err] -> desired acceleration.
        acc_e = self.mpc.move([s.e - slot_e, s.v_e - slot_ve])
        acc_n = self.mpc.move([s.n - slot_n, s.v_n - slot_vn])

        # Turn the accel into a velocity command by leading the current velocity
        # along it; at steady state this collapses to the slot's velocity.
        v_e = s.v_e + acc_e * self.cmd_lead
        v_n = s.v_n + acc_n * self.cmd_lead
        speed = min(self.v_max, max(self.v_min, math.hypot(v_e, v_n)))
        heading = math.atan2(v_e, v_n)
        return Command.autopilot(heading, slot_alt, speed)
