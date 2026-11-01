//! The gun-combat chain. Ordered so every hit test spans one consistent time
//! interval `[t_k, t_k+1]`:
//!
//!   `fire_guns` (rounds leave the muzzle at `t_k`, before physics)
//!   → *step_fdm* (aircraft advance to `t_k+1`)
//!   → `step_bullets` (rounds advance to `t_k+1`)
//!   → `resolve_hits` → `apply_damage` → `end_on_elimination`
//!
//! A dead aircraft is frozen rather than removed (decision §12.5).

use bevy::prelude::*;

use astrum_geo::{Enu, Geodetic};
use astrum_proto::{EndReason, EventKind};

use crate::components::{Aircraft, AircraftState, SimParams, WorldFrame};
use crate::events::EventLog;
use crate::RunMode;

use super::ballistics::{earth_rate_enu, grid_altitude, step_projectile, DVec3};
use super::components::{Bullet, Dead, Gun, Health, Team, Trigger};
use super::geometry::{body_to_grid, first_contact};
use super::resources::{CombatParams, CombatRng, CombatState};

pub(crate) fn dvec(v: astrum_geo::Enu) -> DVec3 {
    DVec3::new(v.e, v.n, v.u)
}

pub(crate) fn geodetic(s: &AircraftState) -> Geodetic {
    Geodetic::from_degrees(s.fdm.lat_deg, s.fdm.lon_deg, s.fdm.alt_hae_m)
}

/// Tilt `dir` by a small random cone of 1σ half-angle `sigma` (rad), drawn from
/// the shared seeded [`CombatRng`] so the whole sim is reproducible.
fn disperse(dir: DVec3, sigma: f64, rng: &mut CombatRng) -> DVec3 {
    let d = dir.normalize_or_zero();
    if sigma <= 0.0 || d == DVec3::ZERO {
        return d;
    }
    let helper = if d.z.abs() < 0.99 { DVec3::Z } else { DVec3::Y };
    let e1 = d.cross(helper).normalize_or_zero();
    let e2 = d.cross(e1);
    let a1 = sigma * rng.next_gaussian();
    let a2 = sigma * rng.next_gaussian();
    (d + a1 * e1 + a2 * e2).normalize_or_zero()
}

/// Muzzle position and boresight (unit) of a shooter, in the world grid.
pub fn muzzle_and_boresight(
    frame: &astrum_geo::LocalFrame,
    s: &AircraftState,
    gun: &Gun,
) -> (DVec3, DVec3) {
    let at = geodetic(s);
    let (phi, theta, psi) = (s.fdm.phi_rad, s.fdm.theta_rad, s.fdm.psi_rad);
    let muzzle =
        dvec(s.pos_enu) + body_to_grid(frame, &at, phi, theta, psi, gun.muzzle_offset_body);
    let bore = body_to_grid(frame, &at, phi, theta, psi, DVec3::X);
    (muzzle, bore)
}

/// Spawn rounds for every armed, triggered, living aircraft at the start of the
/// tick, reporting trigger presses and releases. Fractional rounds accumulate
/// in [`Gun::accum`]; each round carries the aircraft's own velocity plus the
/// muzzle speed along a dispersed boresight.
#[allow(clippy::type_complexity)] // a plain six-component query
pub fn fire_guns(
    params: Res<SimParams>,
    frame: Res<WorldFrame>,
    combat: Res<CombatParams>,
    mut rng: ResMut<CombatRng>,
    mut events: ResMut<EventLog>,
    mut commands: Commands,
    mut q: Query<(&Aircraft, &AircraftState, &Team, &mut Gun, &Trigger), Without<Dead>>,
) {
    let mut shooters: Vec<_> = q.iter_mut().collect();
    // RNG draws must not depend on ECS iteration order.
    shooters.sort_by_key(|(ac, ..)| ac.id);
    for (ac, state, team, mut gun, trigger) in shooters {
        match (trigger.0, gun.burst_from) {
            (true, None) => {
                gun.burst_from = Some(gun.ammo);
                events.push(EventKind::TriggerPressed { aircraft_id: ac.id, ammo: gun.ammo });
            }
            (false, Some(from)) => events.push(release(ac.id, &mut gun, from)),
            _ => {}
        }
        if !trigger.0 || gun.ammo == 0 {
            gun.accum = 0.0; // a released (or empty) trigger carries no partial round
            continue;
        }
        gun.accum += gun.rate_hz * params.dt;
        let (muzzle, bore) = muzzle_and_boresight(&frame.0, state, &gun);
        let own_vel = dvec(state.vel_enu);
        while gun.accum >= 1.0 && gun.ammo > 0 {
            gun.accum -= 1.0;
            gun.ammo -= 1;
            let dir = disperse(bore, gun.dispersion_rad, &mut rng);
            commands.spawn(Bullet {
                pos: muzzle,
                prev_pos: muzzle,
                vel: own_vel + dir * gun.muzzle_speed_mps,
                ballistics: combat.round,
                owner_id: ac.id,
                team: team.0,
                ttl_s: combat.bullet_ttl_s,
            });
        }
    }
}

fn release(aircraft_id: u32, gun: &mut Gun, from: u16) -> EventKind {
    gun.burst_from = None;
    EventKind::TriggerReleased { aircraft_id, rounds_fired: from - gun.ammo, ammo: gun.ammo }
}

fn enu(v: DVec3) -> Enu {
    Enu { e: v.x, n: v.y, u: v.z }
}

/// Integrate every round one tick and cull rounds whose lifetime ran out or
/// that reached the ground (the ellipsoid; there is no terrain model).
pub fn step_bullets(
    params: Res<SimParams>,
    frame: Res<WorldFrame>,
    mut commands: Commands,
    mut q: Query<(Entity, &mut Bullet)>,
) {
    let dt = params.dt;
    let origin = frame.0.origin();
    let omega = earth_rate_enu(origin.lat_rad);
    for (ent, mut b) in &mut q {
        b.ttl_s -= dt;
        let alt = grid_altitude(origin.alt_hae_m, b.pos);
        if b.ttl_s <= 0.0 || alt < 0.0 {
            commands.entity(ent).despawn();
            continue;
        }
        let (pos, vel) = step_projectile(b.pos, b.vel, alt, &b.ballistics, omega, dt);
        b.prev_pos = b.pos;
        b.pos = pos;
        b.vel = vel;
    }
}

/// Resolve bullet↔aircraft hits over the tick with a swept test in relative
/// motion (both the round and the target move during the step). A round hits
/// the living, non-friendly aircraft it reaches *first*; ties go to the lower id.
pub fn resolve_hits(
    combat: Res<CombatParams>,
    mut events: ResMut<EventLog>,
    mut commands: Commands,
    bullets: Query<(Entity, &Bullet)>,
    mut targets: Query<(&Aircraft, &Team, &AircraftState, &mut Health), Without<Dead>>,
) {
    let r = combat.hit_radius_m;
    for (bent, b) in &bullets {
        let mut best: Option<(f64, u32)> = None;
        for (ac, team, st, _) in &targets {
            if team.0 == b.team {
                continue;
            }
            if let Some(s) =
                first_contact(b.prev_pos, b.pos, dvec(st.prev_pos_enu), dvec(st.pos_enu), r)
            {
                if best.is_none_or(|(bs, bid)| s < bs || (s == bs && ac.id < bid)) {
                    best = Some((s, ac.id));
                }
            }
        }
        if let Some((s, id)) = best {
            for (ac, _, _, mut hp) in &mut targets {
                if ac.id == id {
                    hp.hp -= combat.damage_per_hit;
                    hp.last_hit_by = Some(b.owner_id);
                    events.push(EventKind::Hit {
                        shooter_id: b.owner_id,
                        target_id: id,
                        damage: combat.damage_per_hit,
                        hit_points: hp.hp,
                        pos_enu: enu(b.prev_pos + (b.pos - b.prev_pos) * s),
                    });
                }
            }
            commands.entity(bent).despawn();
        }
    }
}

/// Turn lethal damage into death: a living aircraft at ≤ 0 HP gets [`Dead`].
pub fn apply_damage(mut commands: Commands, q: Query<(Entity, &Health), Without<Dead>>) {
    for (ent, hp) in &q {
        if hp.is_dead() {
            commands.entity(ent).insert(Dead);
        }
    }
}

/// Report a kill whenever an aircraft is marked [`Dead`] (by `apply_damage`, or
/// by anyone else), crediting the shooter of the last round that hit it; a
/// burst it was firing ends with it.
pub fn on_kill(
    add: On<Add, Dead>,
    mut events: ResMut<EventLog>,
    mut q: Query<(&Aircraft, &AircraftState, &Health, Option<&mut Gun>)>,
) {
    let Ok((ac, state, hp, gun)) = q.get_mut(add.entity) else { return };
    if let Some(mut gun) = gun {
        if let Some(from) = gun.burst_from {
            events.push(release(ac.id, &mut gun, from));
        }
    }
    events.push(EventKind::Kill {
        victim_id: ac.id,
        killer_id: hp.last_hit_by,
        pos_enu: state.pos_enu,
    });
}

/// Number of teams that still have a living aircraft.
pub fn living_team_count<'a>(teams: impl Iterator<Item = &'a Team>) -> usize {
    let mut v: Vec<u8> = teams.map(|t| t.0).collect();
    v.sort_unstable();
    v.dedup();
    v.len()
}

/// The engagement is decided: ≥ 2 teams started and ≤ 1 still has a living aircraft.
pub fn engagement_decided<'a>(state: &CombatState, living: impl Iterator<Item = &'a Team>) -> bool {
    state.initial_living_teams >= 2 && living_team_count(living) <= 1
}

/// End the run once the engagement is decided.
pub fn end_on_elimination(
    state: Res<CombatState>,
    mode: Option<Res<RunMode>>,
    living: Query<&Team, (With<Health>, Without<Dead>)>,
    mut events: ResMut<EventLog>,
    mut exit: MessageWriter<AppExit>,
) {
    if engagement_decided(&state, living.iter()) {
        if mode.is_some_and(|m| *m == RunMode::Server) {
            println!("[astrum-sim] engagement decided — exiting");
        }
        events.end(EndReason::EngagementDecided, "");
        exit.write(AppExit::Success);
    }
}
