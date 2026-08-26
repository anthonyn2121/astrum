//! ECS components for gun combat (design doc §2.2). Shooters/targets get `Gun`,
//! `Health`, `Team`, `Trigger` attached in `spawn_fleet`; a `Dead` marker is
//! inserted on elimination; bullets are their own lightweight entities.

use bevy::prelude::*;

use super::ballistics::{Ballistics, DVec3};

/// A fixed, forward-firing gun. Attached to aircraft that can shoot. Defaults
/// model the M61A1 Vulcan (design doc §3.3); every field is tunable.
#[derive(Component, Clone, Copy, Debug)]
pub struct Gun {
    /// Muzzle position in body axes (m): nose = +X, starboard = +Y, down = +Z.
    pub muzzle_offset_body: DVec3,
    /// Cyclic fire rate (rounds/second).
    pub rate_hz: f64,
    /// Muzzle speed (m/s), added along the boresight at the trigger instant.
    pub muzzle_speed_mps: f64,
    /// Rounds remaining — a plain counter, no JSBSim mass/CG coupling (§3.3).
    pub ammo: u16,
    /// 1σ dispersion half-angle of the muzzle cone (radians).
    pub dispersion_rad: f64,
    /// Fractional-round accumulator: at 120 Hz the gun fires < 1 round/tick, so
    /// `fire_guns` accrues `rate_hz·dt` here and spawns a round each time it
    /// crosses 1, carrying the remainder between ticks.
    pub accum: f64,
}

impl Gun {
    /// M61A1 Vulcan reference configuration (~6000 rpm, 1030 m/s, 511 rounds,
    /// ~8 mrad dispersion), muzzle a few metres ahead on the boresight.
    pub fn m61a1() -> Self {
        Self {
            muzzle_offset_body: DVec3::new(4.0, 0.0, 0.0),
            rate_hz: 100.0,
            muzzle_speed_mps: 1030.0,
            ammo: 511,
            dispersion_rad: 0.008,
            accum: 0.0,
        }
    }
}

/// Hit-point pool. Reaching `hp <= 0` makes `apply_damage` insert a [`Dead`].
#[derive(Component, Clone, Copy, Debug)]
pub struct Health {
    pub hp: f32,
    pub hp_max: f32,
}

impl Health {
    /// A full pool of `hp_max` points.
    pub fn full(hp_max: f32) -> Self {
        Self { hp: hp_max, hp_max }
    }
    pub fn is_dead(&self) -> bool {
        self.hp <= 0.0
    }
}

/// Team / side id, used for owner-and-friendly exclusion in hit resolution.
/// Generalises 1v1 to NvN.
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct Team(pub u8);

/// Per-aircraft firing intent for this tick — set by command ingest (or directly
/// by tests in Phase 1, which has no wire), read by `fire_guns`.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct Trigger(pub bool);

/// Marker inserted when an aircraft's HP reaches zero. Its FDM is frozen and the
/// wreck stays observable (decision §12.5); `end_on_elimination` reads this to
/// decide episode termination.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct Dead;

/// One in-flight round — a lightweight entity, and **never** an FDM entity
/// (JSBSim is process-global and not thread-safe; §2.3). Its ENU position lives
/// on the entity's `Transform` (metres, `f32`, like every other object);
/// velocity and ballistics are `f64` here for integration accuracy.
#[derive(Component, Clone, Copy, Debug)]
pub struct Bullet {
    /// Velocity in the ENU world frame (m/s).
    pub vel_enu: DVec3,
    /// Drag properties handed to the integrator.
    pub ballistics: Ballistics,
    /// Firing aircraft id and team — excluded from this round's hit tests.
    pub owner_id: u32,
    pub team: u8,
    /// Remaining lifetime (s) before the round is culled.
    pub ttl_s: f64,
}
