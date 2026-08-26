//! Gun-combat mechanics (design doc `docs/dogfight-environment-design.md`).
//!
//! Phase 1 is pure ECS + physics with no wire changes. It starts from the
//! ground up: [`ballistics`] is the free-projectile integrator under every gun
//! round, kept as dependency-light pure functions so it can be unit-tested in
//! isolation before the `fire_guns`/`step_bullets`/`resolve_hits` systems wire
//! it into the simulation. [`components`] and [`resources`] are the ECS data the
//! systems read and write: `Gun`/`Health`/`Team`/`Trigger` on aircraft, `Bullet`
//! on rounds, plus the tunable [`resources::CombatParams`] and the seeded
//! [`resources::CombatRng`] that keeps dispersion reproducible (§7).

pub mod ballistics;
pub mod components;
pub mod geometry;
pub mod resources;
pub mod systems;

pub use components::{Bullet, Dead, Gun, Health, Team, Trigger};
pub use resources::{CombatParams, CombatRng, CombatState};
