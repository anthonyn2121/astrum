//! The F450's physical size, and the hull a collision test uses.
//!
//! # Every number here comes out of the JSBSim model
//!
//! The platform is `aircraft/F450/`, and its own definition files are the authority
//! on how big it is — not a datasheet, and not an estimate:
//!
//! | quantity | file | value |
//! |---|---|---|
//! | motor arm offsets | `Propulsion.xml` | four thrusters at `(∓0.1651, ±0.1651, +0.025)` m |
//! | rotor diameter | `engine/DJI_9450.xml` | `9.4 IN` |
//! | gear footprint | `Gear.xml` | contacts at `\|x\| = \|y\| = 0.1016` m |
//! | gear contact depth | `Gear.xml` | all three contacts at `z = −0.1524` m |
//!
//! Those offsets are relative to the CG, because `Mass.xml` puts the CG at the
//! structural origin. **JSBSim's structural axes are `x` aft, `y` right, `z` up**, so
//! a thruster at structural `z = +0.025` sits 2.5 cm *above* the CG and a gear
//! contact at `z = −0.1524` hangs 15.2 cm *below* it. Nothing in the model declares
//! anything higher than the rotor plane, so that is where the hull stops.
//!
//! `Metrics.xml` is deliberately **not** used. Its `wingspan`/`chord` of 0.127 m are
//! aerodynamic reference lengths chosen to make `wingarea` come out at 0.016129 m²
//! (`= 0.127²`); they are dimensionless bookkeeping for the coefficient build-up, not
//! a statement about how wide the airframe is. Taking them literally would model an
//! F450 as a 13 cm object that sails through any gate.
//!
//! # Why the hull is five parts and not one box
//!
//! A gate's opening is 0.508 m from centre to edge and the F450 is 0.28448 m from
//! CG to blade tip, so a square pass has **22.4 cm of clearance per side** — and at
//! the worst heading, where a corner reaches 0.35287 m, **15.5 cm**. Neither obvious
//! single shape is accurate to a small fraction of that:
//!
//! * A **single box** across the tip circle is exact head-on, but its corners are
//!   empty air on a real quad. It is wrong vertically when the drone banks: rolled 45°
//!   and pitched 30°, an ordinary racing attitude, it claims 0.371 m of reach where
//!   the airframe reaches 0.318 m above the CG — 5.3 cm of fiction, a third of the
//!   worst-heading margin. (Its crabbed diagonal, `HALF_SPAN_M·√2 = 0.40232 m`, no
//!   longer exceeds the half-opening as it did in a 31" gate, so it would stop making
//!   the course impassable; it would still be a false crash a few centimetres wide.)
//! * A **sphere** on the CG needs the tip circle, `0.1651·√2 + 0.11938 = 0.35287 m`.
//!   It is round and the airframe is not: it reaches 0.35287 m *below* the CG where the
//!   gear reaches 0.1524 m, so a drone hovering 0.30 m up would be called a ground
//!   crash (20 cm of error against a 15.5 cm budget); and it cuts the headroom under a
//!   top bar from 36.4 cm to 15.5 cm.
//!
//! So the hull is what the model actually describes: **four rotor discs and a centre
//! section**. Each rotor is a sphere of blade radius on its motor — exact in the
//! lateral direction that binds, and fat only in body `z`, where a gate has 1.016 m
//! of height to give. The centre section is a box over the gear footprint, from the
//! feet up to the rotor plane, which is the only structure the model declares away
//! from the rotors. Between them, along the ±x and ±y body directions, the hull has
//! notches — and so does an F450, whose arms run out on the diagonals.
//!
//! The one place this is still conservative is a prop modelled as a ball rather than
//! a disc: at compound attitudes that reads up to 2.5 cm fat, against a 15.5 cm
//! worst-heading budget.

use bevy::math::DVec3;

use crate::gate::{Gate, BAR_M, IN_TO_M};
use crate::geom::{obb_overlap, sphere_overlaps_obb, Obb};

/// Offset of each motor from the CG along both horizontal body axes
/// (`Propulsion.xml`: four thrusters at `x = ∓0.1651`, `y = ±0.1651`). That is a
/// diagonal wheelbase of `0.1651·2·√2 = 0.467 m`, against the 450 mm the platform's
/// name advertises — the model's figure is the one flown, so it is the one used.
pub const ARM_M: f64 = 0.1651;

/// Rotor diameter (`engine/DJI_9450.xml`: `<diameter unit="IN"> 9.4 </diameter>`).
pub const PROP_DIAMETER_M: f64 = 9.4 * IN_TO_M;
/// Rotor radius — how far a blade tip reaches past its motor.
pub const PROP_RADIUS_M: f64 = PROP_DIAMETER_M / 2.0;

/// Height of the rotor plane above the CG (`Propulsion.xml`, structural `z = +0.025`).
pub const ROTOR_PLANE_M: f64 = 0.025;
/// Depth of the landing-gear contacts below the CG (`Gear.xml`, structural
/// `z = −0.1524`) — the lowest declared point on the airframe.
pub const GEAR_DEPTH_M: f64 = 0.1524;
/// Half-width of the gear footprint (`Gear.xml`: contacts at `|x| = |y| = 0.1016`).
///
/// The shipped model has three contacts, not four, and says why: *"This is a fake
/// point, JSBSim struggles with a 4-point ground trim"*. The real gear is the four
/// corners this half-width describes, and the box below covers them either way.
pub const GEAR_SPAN_M: f64 = 0.1016;

/// Half the airframe's width, fore-aft and left-right: motor arm plus blade tip.
pub const HALF_SPAN_M: f64 = ARM_M + PROP_RADIUS_M;
/// Width across the prop tips — the dimension a gate's opening constrains.
pub const WIDTH_M: f64 = 2.0 * HALF_SPAN_M;
/// Overall height: gear feet to rotor plane.
pub const HEIGHT_M: f64 = GEAR_DEPTH_M + ROTOR_PLANE_M;

/// Longest gap allowed between collision samples inside one tick, in metres.
///
/// A once-per-tick test must not let the drone teleport through a gate: at 20 m/s a
/// 120 Hz tick covers 0.17 m, which is more than the 0.0762 m of frame stock. Rather
/// than inflate the hull to cover the whole step — 8 cm at that speed, half the
/// worst-heading clearance budget — [`crate::systems::detect_crash`] subdivides the
/// step and tests the true hull at each sample.
///
/// Tunnelling needs the hull to be clear on both sides of a piece of stock at
/// consecutive samples, so it takes a gap wider than the stock plus the hull part's
/// own thickness. The thinnest thing on a course is a `BAR_M` = 0.0762 m bar and the
/// smallest part of the hull is a 0.2388 m rotor sphere, so anything under 0.31 m is
/// safe; this is half of that. What a discrete sample *can* miss is a graze that
/// exists for less than 0.15 m of travel, which is a fair thing to let go.
pub const SAMPLE_STEP_M: f64 = 0.15;

/// The drone's body axes — forward, right, up — as world ENU unit vectors.
///
/// These are the columns of the standard `Rz(psi)·Ry(theta)·Rx(phi)` body→NED
/// rotation, read out in ENU (`(E, N, U) = (ned.y, ned.x, −ned.z)`) and with the
/// third negated so it points up rather than down. `(forward, right, up)` is
/// therefore **left**-handed: the right-handed aero triad is `(forward, right, down)`.
pub fn body_axes(phi_rad: f64, theta_rad: f64, psi_rad: f64) -> [DVec3; 3] {
    let (sp, cp) = phi_rad.sin_cos();
    let (st, ct) = theta_rad.sin_cos();
    let (ss, cs) = psi_rad.sin_cos();
    let forward = DVec3::new(ct * ss, ct * cs, st);
    let right = DVec3::new(sp * st * ss + cp * cs, sp * st * cs - cp * ss, -sp * ct);
    let up = DVec3::new(sp * cs - cp * st * ss, -(cp * st * cs + sp * ss), cp * ct);
    [forward, right, up]
}

/// The airframe's collision volume at one pose, in world ENU metres: four rotor
/// spheres and the centre section. See the module docs for why it is not one box.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hull {
    /// Centres of the four rotor discs — the motor positions.
    pub rotors: [DVec3; 4],
    /// Radius of each rotor sphere: one blade length.
    pub rotor_radius: f64,
    /// Frame plates and landing gear: a box over the gear footprint, running from the
    /// feet up to the rotor plane.
    pub core: Obb,
}

impl Hull {
    /// The hull at a pose. `pos_enu` is the **CG**, which is what JSBSim reports as
    /// the aircraft's position and what every body offset in this module is measured
    /// from.
    ///
    /// Note that the CG is not the middle of the airframe: 15.2 cm of leg hangs below
    /// it against 2.5 cm of rotor above, so the core box is carried down the body's
    /// own up axis and banking swings it out sideways.
    pub fn at(pos_enu: DVec3, phi_rad: f64, theta_rad: f64, psi_rad: f64) -> Self {
        let axes @ [fwd, right, up] = body_axes(phi_rad, theta_rad, psi_rad);
        let plane = pos_enu + up * ROTOR_PLANE_M;
        Self {
            rotors: [
                plane + (fwd + right) * ARM_M,
                plane + (fwd - right) * ARM_M,
                plane - (fwd - right) * ARM_M,
                plane - (fwd + right) * ARM_M,
            ],
            rotor_radius: PROP_RADIUS_M,
            core: Obb {
                center: pos_enu + up * (ROTOR_PLANE_M - GEAR_DEPTH_M) / 2.0,
                axes,
                half_extents: DVec3::new(GEAR_SPAN_M, GEAR_SPAN_M, HEIGHT_M / 2.0),
            },
        }
    }

    /// The lowest point of the airframe, as a world ENU altitude.
    pub fn min_z(&self) -> f64 {
        self.rotors.iter().map(|c| c.z - self.rotor_radius).fold(self.core.min_z(), f64::min)
    }

    /// Does the airframe intersect a gate's frame hardware?
    ///
    /// Everything is re-expressed in the gate's local frame, where all three pieces of
    /// stock are axis-aligned. Only the hardware counts: the opening is a hole and
    /// there is nothing at all below the sill, so flying through or underneath a gate
    /// touches nothing.
    pub fn hits_gate(&self, gate: &Gate) -> bool {
        let core = self.core.to_basis(gate.center, gate.basis());
        let rotors = self.rotors.map(|c| gate.to_local(c));

        // Cheap reject first: the stock is 7 cm deep, so a part that does not reach
        // the slab around the gate's plane cannot touch any of the three boxes, and
        // most ticks of a race are nowhere near a gate.
        let depth = BAR_M / 2.0;
        let core_near = core.center.z.abs() - core.projection_radius(DVec3::Z) <= depth;
        let rotor_near = rotors.iter().any(|c| c.z.abs() - self.rotor_radius <= depth);
        if !core_near && !rotor_near {
            return false;
        }

        gate.bars().iter().any(|bar| {
            let b = bar.obb();
            obb_overlap(&core, &b)
                || rotors.iter().any(|c| sphere_overlaps_obb(*c, self.rotor_radius, &b))
        })
    }
}
