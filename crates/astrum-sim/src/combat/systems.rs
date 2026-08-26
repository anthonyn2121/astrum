//! The gun-combat system chain, run each fixed step between `sync_geo_transform`
//! (fresh pose) and `publish_telemetry` (fresh health):
//!
//!   `fire_guns → step_bullets → resolve_hits → apply_damage → end_on_elimination`
//!
//! Rounds are ordinary lightweight entities carrying a [`Bullet`] + a `Transform`
//! (their ENU position); they never own an FDM (JSBSim is process-global and not
//! thread-safe). A dead aircraft is *frozen* rather than removed: `apply_damage`
//! inserts a [`Dead`] marker and the core `step_fdm` stops integrating anything
//! marked `Dead`, so the wreck holds its final pose (decision §12.5).

use bevy::prelude::*;

use crate::components::{Aircraft, GeoState, SimParams, WorldFrame};

use super::ballistics::{earth_rate_enu, step_projectile, DVec3};
use super::components::{Bullet, Dead, Gun, Health, Team, Trigger};
use super::geometry::{body_to_enu, boresight_enu, segment_point_dist_sq};
use super::resources::{CombatParams, CombatRng, CombatState};

/// ENU position of an entity's `Transform`, widened to `f64` for the physics.
fn enu_of(tf: &Transform) -> DVec3 {
    DVec3::new(tf.translation.x as f64, tf.translation.y as f64, tf.translation.z as f64)
}

/// Tilt unit-ish `dir` by a small random cone of 1σ half-angle `sigma` (rad),
/// applying two independent Gaussian offsets in the plane perpendicular to it.
/// Drawn from the shared seeded [`CombatRng`] so the whole sim is reproducible.
fn disperse(dir: DVec3, sigma: f64, rng: &mut CombatRng) -> DVec3 {
    let d = dir.normalize_or_zero();
    if sigma <= 0.0 || d == DVec3::ZERO {
        return d;
    }
    // Orthonormal basis perpendicular to d: pair it with world Up, unless d is
    // nearly vertical, in which case pair it with North instead.
    let helper = if d.z.abs() < 0.99 { DVec3::Z } else { DVec3::Y };
    let e1 = d.cross(helper).normalize_or_zero();
    let e2 = d.cross(e1);
    let a1 = sigma * rng.next_gaussian();
    let a2 = sigma * rng.next_gaussian();
    (d + a1 * e1 + a2 * e2).normalize_or_zero()
}

/// Spawn rounds for every armed, triggered, living aircraft this tick. Fractional
/// rounds-per-tick accumulate in [`Gun::accum`]; each whole round leaves the
/// muzzle carrying the aircraft's own velocity plus the muzzle speed along a
/// dispersed boresight.
// The shooter query genuinely needs all six components (id, attitude+velocity,
// ENU position, team, gun, trigger); `type_complexity` is the usual Bevy-query
// false positive here, so allow it rather than obscure a plain tuple.
#[allow(clippy::type_complexity)]
pub fn fire_guns(
    params: Res<SimParams>,
    combat: Res<CombatParams>,
    mut rng: ResMut<CombatRng>,
    mut commands: Commands,
    mut q: Query<(&Aircraft, &GeoState, &Transform, &Team, &mut Gun, &Trigger), Without<Dead>>,
) {
    let dt = params.dt;
    for (ac, gs, tf, team, mut gun, trigger) in &mut q {
        if !trigger.0 || gun.ammo == 0 {
            gun.accum = 0.0; // a released (or empty) trigger carries no partial round
            continue;
        }
        gun.accum += gun.rate_hz * dt;

        let boresight = boresight_enu(gs.phi_rad, gs.theta_rad, gs.psi_rad);
        let muzzle =
            enu_of(tf) + body_to_enu(gs.phi_rad, gs.theta_rad, gs.psi_rad, gun.muzzle_offset_body);
        let ac_vel = DVec3::new(gs.vel_east_mps, gs.vel_north_mps, -gs.vel_down_mps);

        while gun.accum >= 1.0 && gun.ammo > 0 {
            gun.accum -= 1.0;
            gun.ammo -= 1;
            let dir = disperse(boresight, gun.dispersion_rad, &mut rng);
            let vel = ac_vel + dir * gun.muzzle_speed_mps;
            commands.spawn((
                Bullet {
                    vel_enu: vel,
                    ballistics: combat.round,
                    owner_id: ac.id,
                    team: team.0,
                    ttl_s: combat.bullet_ttl_s,
                },
                Transform::from_translation(Vec3::new(
                    muzzle.x as f32,
                    muzzle.y as f32,
                    muzzle.z as f32,
                )),
            ));
        }
    }
}

/// Integrate every round one tick with the free-projectile physics, and cull
/// rounds whose lifetime has run out.
pub fn step_bullets(
    params: Res<SimParams>,
    frame: Res<WorldFrame>,
    mut commands: Commands,
    mut q: Query<(Entity, &mut Bullet, &mut Transform)>,
) {
    let dt = params.dt;
    let omega = earth_rate_enu(frame.0.origin().lat_rad);
    let base_alt = frame.0.origin().alt_m; // ENU-origin altitude for the density lookup

    for (ent, mut b, mut tf) in &mut q {
        b.ttl_s -= dt;
        if b.ttl_s <= 0.0 {
            commands.entity(ent).despawn();
            continue;
        }
        let pos = enu_of(&tf);
        let (new_pos, new_vel) =
            step_projectile(pos, b.vel_enu, base_alt + pos.z, &b.ballistics, omega, dt);
        b.vel_enu = new_vel;
        tf.translation = Vec3::new(new_pos.x as f32, new_pos.y as f32, new_pos.z as f32);
    }
}

/// Resolve bullet↔aircraft collisions with a swept segment-vs-sphere test so a
/// fast round can't tunnel through a target in a single tick. A round hits the
/// first living, non-friendly aircraft whose bounding sphere its per-tick segment
/// pierces; on a hit the round is consumed and the target loses HP. Same-team
/// rounds (which includes the shooter's own) are ignored — no friendly fire.
pub fn resolve_hits(
    params: Res<SimParams>,
    combat: Res<CombatParams>,
    mut commands: Commands,
    bullets: Query<(Entity, &Bullet, &Transform)>,
    mut targets: Query<(&Team, &Transform, &mut Health), Without<Dead>>,
) {
    let dt = params.dt;
    let r2 = combat.hit_radius_m * combat.hit_radius_m;

    for (bent, b, btf) in &bullets {
        let end = enu_of(btf);
        let start = end - b.vel_enu * dt; // the segment this round swept this tick
        let mut consumed = false;
        for (team, ttf, mut hp) in &mut targets {
            if team.0 == b.team {
                continue;
            }
            if segment_point_dist_sq(start, end, enu_of(ttf)) <= r2 {
                hp.hp -= combat.damage_per_hit;
                consumed = true;
                break;
            }
        }
        if consumed {
            commands.entity(bent).despawn();
        }
    }
}

/// Turn lethal damage into death: any living aircraft at ≤ 0 HP gets a [`Dead`]
/// marker. The core `step_fdm` stops integrating `Dead` entities, so the wreck
/// simply holds its final pose (decision §12.5).
pub fn apply_damage(mut commands: Commands, q: Query<(Entity, &Health), Without<Dead>>) {
    for (ent, hp) in &q {
        if hp.is_dead() {
            commands.entity(ent).insert(Dead);
        }
    }
}

/// End the episode once the engagement is decided — i.e. at most one team still
/// has a living aircraft. Guarded by [`CombatState::initial_living_teams`] so a
/// single-team or non-combat scenario never self-terminates.
pub fn end_on_elimination(
    state: Res<CombatState>,
    living: Query<&Team, (With<Health>, Without<Dead>)>,
    mut exit: MessageWriter<AppExit>,
) {
    if state.initial_living_teams < 2 {
        return; // not a fight — nothing to decide
    }
    let mut first: Option<u8> = None;
    for t in &living {
        match first {
            None => first = Some(t.0),
            Some(f) if f != t.0 => return, // ≥ 2 teams still in it
            _ => {}
        }
    }
    let remaining = usize::from(first.is_some());
    println!("[astrum-sim] engagement decided — {remaining} team(s) remain, exiting");
    exit.write(AppExit::Success);
}
