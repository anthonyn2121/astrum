//! A racing gate: the physical frame, its pose in the world, and the geometry a
//! flythrough test needs.
//!
//! # Hardware
//!
//! One gate is three pieces of 3-inch stock: two 31-inch uprights carrying a
//! 37-inch top bar. That arithmetic is the whole gate —
//!
//! ```text
//!            <------------- 37" ------------->         top bar (37 x 3)
//!           +----------------------------------+  --+
//!           |###                            ###|    | 3"
//!           |###                            ###|  --+
//!           |###                            ###|
//!           |###          31" x 31"         ###|    upright (31 x 3)
//!           |###          opening           ###|    31" tall
//!           |###                            ###|
//!           |###                            ###|
//!           +###----------------------------###+  --- sill (open below)
//!            <3">        <-- 31" -->
//! ```
//!
//! — because `37" = 31" + 2 x 3"`: the top bar spans the opening plus both
//! uprights, so the opening comes out 31 inches square. Dimensions are converted
//! from inches, which is what makes that identity exact.
//!
//! Note the frame is **not** symmetric about the opening's centre: there is a bar
//! above but nothing below, so the lowest hardware is the sill, at the bottom of
//! the opening.
//!
//! # Why 31 inches and not the 25 the brief asked for
//!
//! The opening was widened from 25" (0.635 m) on 2026-09-08, because 25" makes the
//! course unflyable for this airframe rather than merely difficult. An F450 is
//! 0.56896 m across its prop tips but **0.7057 m corner to corner**, and it is the
//! corner that a gate sees whenever the drone is not square to the opening. In a
//! 0.635 m hole that capped the approach at ±13° of crab — while the sampler is
//! allowed a 120° turn, whose bisected normal leaves a crossing up to 60° off. The
//! geometry and the envelope contradicted each other.
//!
//! 27.78" is the exact threshold at which heading stops mattering (the worst case is
//! 45° of crab, reaching `ARM_M·√2 + PROP_RADIUS_M = 0.35287 m`). 31" is the first
//! whole inch above it whose *worst-heading* margin, 4.08 cm, beats the 3.30 cm the
//! 25" gate offered only when perfectly square. Square on, the margin is now 10.92 cm
//! and the headroom 24.93 cm. See `crates/astrum-race/tests/airframe.rs`, which holds
//! all of those numbers as assertions.
//!
//! # Pose
//!
//! A gate is a centre and a yaw. The centre is the middle of the *opening* (not of
//! the hardware), in world ENU metres; the yaw is the heading of the gate's
//! **normal** — the direction of flight through it — using the sim's convention of
//! `psi = 0` facing North, `+psi` turning East. Gates stand upright: no bank, no
//! tilt, so the opening is always a vertical square.
//!
//! Gate-local coordinates used throughout are `x` = right (starboard of the
//! flight direction), `y` = up, `z` = along the normal (positive downstream). Because
//! `x` is starboard, that triple is **left**-handed, like a camera's — the
//! right-handed cycle is `(normal, up, right)`. Nothing here takes a cross product in
//! local coordinates, but anything that starts to should know which it is.

use bevy::math::DVec3;
use bevy::prelude::{Component, Quat, Transform};

use crate::geom::Obb;

/// Inches to metres, exactly.
pub const IN_TO_M: f64 = 0.0254;

/// Side of the square opening: 31 in (~79 cm). See the module docs for why 31 and
/// not the brief's 25.
pub const OPENING_M: f64 = 31.0 * IN_TO_M;
/// Frame stock width and depth: 3 in (~7 cm).
pub const BAR_M: f64 = 3.0 * IN_TO_M;
/// Length of the top bar: 37 in (~94 cm) — the opening plus both uprights.
pub const TOP_BAR_M: f64 = 37.0 * IN_TO_M;
/// Length of each upright: 31 in (~79 cm) — it sets the opening's height.
pub const UPRIGHT_M: f64 = 31.0 * IN_TO_M;

/// Overall width of the assembled frame (the top bar spans it).
pub const OUTER_WIDTH_M: f64 = TOP_BAR_M;
/// Overall height of the assembled frame: upright plus the top bar above it.
pub const OUTER_HEIGHT_M: f64 = UPRIGHT_M + BAR_M;

/// Radius of the smallest sphere about the opening's centre that contains every
/// piece of hardware.
///
/// The course sampler uses this to keep a leg clear of a gate it is not flying
/// through: one sphere test instead of three box tests. The extreme point is a top
/// corner of the top bar, since the frame extends `OPENING_M/2 + BAR_M` upwards but
/// only `OPENING_M/2` down.
pub fn frame_radius_m() -> f64 {
    (OUTER_WIDTH_M / 2.0).hypot(OPENING_M / 2.0 + BAR_M)
}

/// One piece of the frame: a box, axis-aligned **in gate-local coordinates**.
///
/// Both fields are in gate-local metres (`x` right, `y` up, `z` along the normal);
/// use [`Gate::to_world`] on [`Bar::center`] to place it in the world.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bar {
    /// Centre of the box, gate-local.
    pub center: DVec3,
    /// Half-extents along the gate's right / up / normal axes.
    pub half_extents: DVec3,
}

impl Bar {
    /// Is a gate-local point inside this box?
    pub fn contains_local(&self, p: DVec3) -> bool {
        let d = p - self.center;
        d.x.abs() <= self.half_extents.x
            && d.y.abs() <= self.half_extents.y
            && d.z.abs() <= self.half_extents.z
    }

    /// This piece of stock as a box, in gate-local coordinates (where it is
    /// axis-aligned by construction).
    pub fn obb(&self) -> Obb {
        Obb::aabb(self.center, self.half_extents)
    }
}

/// A gate in the world: which one it is, where its opening is, and which way it
/// points. Also the ECS component carried by each gate entity.
#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct Gate {
    /// Position in the course order, `0` first.
    pub index: u32,
    /// Centre of the opening, world ENU metres.
    pub center: DVec3,
    /// Heading of the normal (radians; `0` = North, `+` = East).
    pub yaw_rad: f64,
}

impl Gate {
    /// A gate with its opening centred at `center`, normal on heading `yaw_rad`.
    pub fn new(index: u32, center: DVec3, yaw_rad: f64) -> Self {
        Self { index, center, yaw_rad }
    }

    /// Unit normal: the direction of flight through the opening, in ENU.
    pub fn normal(&self) -> DVec3 {
        let (s, c) = self.yaw_rad.sin_cos();
        DVec3::new(s, c, 0.0) // psi=0 → North(+Y), +psi → East(+X)
    }

    /// Unit vector to the right of the flight direction, in ENU.
    pub fn right(&self) -> DVec3 {
        let (s, c) = self.yaw_rad.sin_cos();
        DVec3::new(c, -s, 0.0)
    }

    /// Unit up vector. Gates stand upright, so this is always world up.
    pub fn up(&self) -> DVec3 {
        DVec3::Z
    }

    /// Half-width (and half-height) of the opening.
    pub fn half_opening_m(&self) -> f64 {
        OPENING_M / 2.0
    }

    /// Height of the sill — the bottom edge of the opening, the lowest hardware.
    pub fn sill_m(&self) -> f64 {
        self.center.z - self.half_opening_m()
    }

    /// World point → gate-local `(right, up, normal)`.
    pub fn to_local(&self, p: DVec3) -> DVec3 {
        let d = p - self.center;
        DVec3::new(d.dot(self.right()), d.dot(self.up()), d.dot(self.normal()))
    }

    /// Gate-local `(right, up, normal)` → world point.
    pub fn to_world(&self, p: DVec3) -> DVec3 {
        self.center + p.x * self.right() + p.y * self.up() + p.z * self.normal()
    }

    /// Does a gate-local point lie within the opening, ignoring its `z`?
    ///
    /// The caller owns the plane crossing (the sign change of local `z` between two
    /// ticks); this answers whether that crossing went *through the hole*.
    pub fn opening_contains(&self, local: DVec3) -> bool {
        let h = self.half_opening_m();
        local.x.abs() <= h && local.y.abs() <= h
    }

    /// The four corners of the opening in world coordinates: bottom-left,
    /// bottom-right, top-right, top-left, as seen looking downstream.
    pub fn opening_corners(&self) -> [DVec3; 4] {
        let h = self.half_opening_m();
        [
            self.to_world(DVec3::new(-h, -h, 0.0)),
            self.to_world(DVec3::new(h, -h, 0.0)),
            self.to_world(DVec3::new(h, h, 0.0)),
            self.to_world(DVec3::new(-h, h, 0.0)),
        ]
    }

    /// The three pieces of hardware, gate-local: left upright, right upright, top
    /// bar. Each is 3 in thick and 3 in deep.
    pub fn bars(&self) -> [Bar; 3] {
        let h = self.half_opening_m();
        let t = BAR_M / 2.0;
        // Uprights flank the opening and span its full height; the top bar sits on
        // them, overhanging by exactly one bar width on each side (37" = 31" + 2x3").
        let upright = DVec3::new(t, UPRIGHT_M / 2.0, t);
        let top = DVec3::new(TOP_BAR_M / 2.0, t, t);
        [
            Bar { center: DVec3::new(-(h + t), 0.0, 0.0), half_extents: upright },
            Bar { center: DVec3::new(h + t, 0.0, 0.0), half_extents: upright },
            Bar { center: DVec3::new(0.0, h + t, 0.0), half_extents: top },
        ]
    }

    /// The gate's local basis, as the axis triple [`Obb::to_basis`] takes:
    /// `(right, up, normal)`.
    pub fn basis(&self) -> [DVec3; 3] {
        [self.right(), self.up(), self.normal()]
    }

    /// Is a **world** point inside the frame hardware?
    ///
    /// A point test, so it is only meaningful for something that *is* a point. An
    /// aircraft is not: use [`crate::airframe::Hull::hits_gate`], which knows the shape
    /// of the F450 and lives with the rest of the airframe's dimensions.
    pub fn frame_contains(&self, p: DVec3) -> bool {
        let local = self.to_local(p);
        self.bars().iter().any(|b| b.contains_local(local))
    }

    /// The gate's pose as a Bevy transform, for renderers and replay export. The
    /// rotation matches the sim's yaw convention (`Quat::from_rotation_z(-psi)`).
    pub fn transform(&self) -> Transform {
        Transform::from_translation(self.center.as_vec3())
            .with_rotation(Quat::from_rotation_z(-self.yaw_rad as f32))
    }
}
