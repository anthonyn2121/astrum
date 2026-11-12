//! Gun-combat mechanics (design doc `docs/dogfight-environment-design.md`):
//! [`ballistics`] is the free-projectile integrator under every round,
//! [`geometry`] the attitude/frame and swept-hit helpers, [`fire_control`] the
//! ballistic gun solution, [`components`]/[`resources`] the ECS data, and
//! [`systems`] the per-tick chain.

pub mod ballistics;
pub mod components;
pub mod fire_control;
pub mod geometry;
pub mod resources;
pub mod systems;

pub use components::{Bullet, Dead, Gun, Health, Team, Trigger};
pub use fire_control::{gun_solution, GunSolution};
pub use resources::{CombatParams, CombatRng, CombatState};
