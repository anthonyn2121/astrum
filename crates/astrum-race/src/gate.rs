//! A racing gate: the physical frame, its pose in the world, and the geometry a
//! flythrough test needs.
//!
//! # Hardware
//!
//! One gate is three pieces of 3-inch stock: two uprights as tall as the opening
//! carrying a top bar as wide as the opening plus both uprights. Everything derives
//! from **one** number, [`OPENING_IN`] (40 in), and the stock width [`BAR_IN`]:
//!
//! ```text
//!            <------------ 46" ------------->         top bar (46 x 3)
//!           +----------------------------------+  --+
//!           |###                            ###|    | 3"
//!           |###                            ###|  --+
//!           |###                            ###|
//!           |###          40" x 40"         ###|    upright (40 x 3)
//!           |###          opening           ###|    40" tall
//!           |###                            ###|
//!           |###                            ###|
//!           +###----------------------------###+  --- sill (open below)
//!            <3">        <-- 40" -->
//! ```
//!
//! — the top bar is `OPENING + 2 x BAR` because it spans the opening plus both
//! uprights, and the uprights are `OPENING` long because they set its height. Those
//! are computed, never typed twice, so resizing the gate is a one-constant change.
//! Dimensions are converted from inches, which is what makes the identity exact.
//!
//! Note the frame is **not** symmetric about the opening's centre: there is a bar
//! above but nothing below, so the lowest hardware is the sill, at the bottom of
//! the opening.
//!
//! # Why 40 inches
//!
//! The opening has been resized twice, each time because of the airframe's corners.
//! An F450 is 0.56896 m across its prop tips but **0.7057 m corner to corner**, and
//! it is the corner that a gate sees whenever the drone is not square to the opening.
//! 25" (0.635 m) capped the approach at ±13° of crab, while the sampler is allowed a
//! 120° turn whose bisected normal leaves a crossing up to 60° off. 27.78" is the exact
//! threshold at which heading stops mattering (the worst case is 45° of crab, reaching
//! `ARM_M·√2 + PROP_RADIUS_M = 0.35287 m`); 31" cleared it, but with only 4.08 cm to
//! spare at the worst heading — too thin for a controller that is merely good, so a
//! decent run still scored as a crash.
//!
//! At 40" the worst-heading margin is 15.51 cm (22.35 cm square on) and the headroom
//! under the top bar is 36.36 cm. See `crates/astrum-race/tests/airframe.rs`, which
//! holds all of those numbers as assertions, and `docs/racing-environment-design.md`
//! §2 for the full table.
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
//! local coordinates, but anything that starts to should know which it is. The
//! world-ENU vectors of that basis are [`Gate::normal`], [`Gate::right`] and
//! [`Gate::up`]; consumers that need the basis on the wire (renderers, controllers)
//! should send those rather than re-derive the handedness from the yaw.

use bevy::math::DVec3;
use bevy::prelude::{Component, Quat, Transform};

use crate::geom::Obb;

/// Inches to metres, exactly.
pub const IN_TO_M: f64 = 0.0254;

/// Side of the square opening, in inches — **the** gate-size constant: every other
/// dimension below is derived from it and [`BAR_IN`]. See the module docs for why 40.
pub const OPENING_IN: f64 = 40.0;
/// Frame stock width and depth, in inches.
pub const BAR_IN: f64 = 3.0;

/// Side of the square opening: 40 in (1.016 m).
pub const OPENING_M: f64 = OPENING_IN * IN_TO_M;
/// Frame stock width and depth: 3 in (7.62 cm).
pub const BAR_M: f64 = BAR_IN * IN_TO_M;
/// Length of the top bar: the opening plus both uprights (46 in).
pub const TOP_BAR_M: f64 = OPENING_M + 2.0 * BAR_M;
/// Length of each upright: it sets the opening's height (40 in).
pub const UPRIGHT_M: f64 = OPENING_M;

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
    /// bar. Each is `BAR_IN` thick and deep.
    pub fn bars(&self) -> [Bar; 3] {
        let h = self.half_opening_m();
        let t = BAR_M / 2.0;
        // Uprights flank the opening and span its full height; the top bar sits on
        // them, overhanging by exactly one bar width on each side (top = opening + 2 bars).
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
