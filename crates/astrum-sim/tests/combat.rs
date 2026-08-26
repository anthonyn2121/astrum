//! End-to-end tests of the gun-combat system chain
//! (`fire_guns → step_bullets → resolve_hits → apply_damage → end_on_elimination`).
//!
//! These drive a *minimal* Bevy app that registers exactly the combat systems and
//! feeds them hand-placed entities. The combat systems are decoupled from JSBSim
//! (bullets never own an FDM, and death is a `Dead` marker rather than an FDM
//! call), so this needs no `FdmInstance`, no `JSBSIM_LOCK`, and gives full control
//! over geometry — letting each test assert a precise outcome. `App::new()` alone
//! registers `Messages<AppExit>` and runs the `Update` schedule under
//! `App::update()`, and exposes `App::should_exit()` to observe termination.
//!
//! Run:  `cargo test -p astrum-sim --test combat -- --nocapture`

use std::f64::consts::FRAC_PI_2;

use astrum_geo::{Geodetic, LocalFrame};
use astrum_sim::combat::ballistics::DVec3;
use astrum_sim::combat::components::{Bullet, Dead, Gun, Health, Team, Trigger};
use astrum_sim::combat::resources::{CombatParams, CombatRng, CombatState};
use astrum_sim::combat::systems::{
    apply_damage, end_on_elimination, fire_guns, resolve_hits, step_bullets,
};
use astrum_sim::components::{Aircraft, GeoState, SimParams, WorldFrame};
use bevy::prelude::*;

const DT: f64 = 1.0 / 120.0;

/// A minimal app running the whole combat chain over hand-placed entities.
fn combat_app(seed: u64, initial_living_teams: u8) -> App {
    let mut app = App::new();
    let origin = Geodetic::from_degrees(37.6188, -122.3756, 0.0);
    app.insert_resource(SimParams { rate_hz: 120.0, dt: DT, max_time_s: 0.0, realtime: false })
        .insert_resource(WorldFrame(LocalFrame::new(origin)))
        .insert_resource(CombatParams::default())
        .insert_resource(CombatRng::seeded(seed))
        .insert_resource(CombatState { initial_living_teams })
        .add_systems(
            Update,
            (fire_guns, step_bullets, resolve_hits, apply_damage, end_on_elimination).chain(),
        );
    app
}

fn vec3(p: DVec3) -> Vec3 {
    Vec3::new(p.x as f32, p.y as f32, p.z as f32)
}

/// Spawn an armed aircraft at `pos` (ENU, m) with ENU velocity `vel`, heading
/// `psi_rad` (0 = North, +E), wings level. Trigger starts off.
fn spawn_shooter(app: &mut App, id: u32, team: u8, pos: DVec3, vel: DVec3, psi_rad: f64, gun: Gun) {
    let geo = GeoState {
        psi_rad,
        vel_east_mps: vel.x,
        vel_north_mps: vel.y,
        vel_down_mps: -vel.z,
        ..Default::default()
    };
    app.world_mut().spawn((
        Aircraft { id },
        geo,
        Transform::from_translation(vec3(pos)),
        Team(team),
        Health::full(100.0),
        Trigger(false),
        gun,
    ));
}

/// Spawn an unarmed target (no `Gun`, so it never fires) at `pos`.
fn spawn_target(app: &mut App, id: u32, team: u8, pos: DVec3, health: f32) {
    app.world_mut().spawn((
        Aircraft { id },
        GeoState::default(),
        Transform::from_translation(vec3(pos)),
        Team(team),
        Health::full(health),
        Trigger(false),
    ));
}

fn set_trigger(app: &mut App, id: u32, on: bool) {
    let mut q = app.world_mut().query::<(&Aircraft, &mut Trigger)>();
    let world = app.world_mut();
    for (a, mut t) in q.iter_mut(world) {
        if a.id == id {
            t.0 = on;
        }
    }
}

fn hp_of(app: &mut App, id: u32) -> f32 {
    let mut q = app.world_mut().query::<(&Aircraft, &Health)>();
    let world = app.world();
    q.iter(world).find(|(a, _)| a.id == id).map(|(_, h)| h.hp).expect("aircraft present")
}

fn is_dead(app: &mut App, id: u32) -> bool {
    let mut q = app.world_mut().query::<(&Aircraft, Option<&Dead>)>();
    let world = app.world();
    q.iter(world).find(|(a, _)| a.id == id).map(|(_, d)| d.is_some()).expect("aircraft present")
}

fn bullet_count(app: &mut App) -> usize {
    let mut q = app.world_mut().query::<&Bullet>();
    let world = app.world();
    q.iter(world).count()
}

/// Bit-exact, sorted bullet positions — for comparing two runs byte for byte.
fn bullet_pos_bits(app: &mut App) -> Vec<[u32; 3]> {
    let mut q = app.world_mut().query::<(&Bullet, &Transform)>();
    let world = app.world();
    let mut v: Vec<[u32; 3]> = q
        .iter(world)
        .map(|(_, t)| {
            [t.translation.x.to_bits(), t.translation.y.to_bits(), t.translation.z.to_bits()]
        })
        .collect();
    v.sort_unstable();
    v
}

#[test]
fn swept_test_prevents_tunneling() {
    // A very fast round whose single-tick step (~20 m) straddles an 8 m-radius
    // target: both segment endpoints end up ~10 m from the target centre, so a
    // point test at either tick would MISS — only the swept segment test catches
    // it. We spawn the round directly to control the geometry exactly.
    let mut app = combat_app(1, 2);
    spawn_target(&mut app, 2, 1, DVec3::new(100.0, 0.0, 0.0), 100.0);
    let round = CombatParams::default().round;
    app.world_mut().spawn((
        Bullet {
            vel_enu: DVec3::new(2500.0, 0.0, 0.0),
            ballistics: round,
            owner_id: 1,
            team: 0,
            ttl_s: 3.0,
        },
        Transform::from_translation(Vec3::new(90.0, 0.0, 0.0)),
    ));

    app.update();

    assert!(
        hp_of(&mut app, 2) < 100.0,
        "swept test must catch a round that steps clean past a target"
    );
    assert_eq!(bullet_count(&mut app), 0, "a round that connects is consumed");
}

#[test]
fn muzzle_velocity_is_boresight_plus_own_velocity() {
    // Fire-only app: no `step_bullets`, so the round's stored velocity is the
    // pristine muzzle velocity. Facing East (ψ=90°) while flying North at 200 m/s,
    // the round should leave at (muzzle_speed East) + (own North velocity) — the
    // physical lead the shooter's motion imparts.
    let mut app = App::new();
    app.insert_resource(SimParams { rate_hz: 120.0, dt: DT, max_time_s: 0.0, realtime: false })
        .insert_resource(CombatParams::default())
        .insert_resource(CombatRng::seeded(1))
        .add_systems(Update, fire_guns);

    let mut gun = Gun::m61a1();
    gun.dispersion_rad = 0.0; // deterministic boresight
    spawn_shooter(&mut app, 1, 0, DVec3::ZERO, DVec3::new(0.0, 200.0, 0.0), FRAC_PI_2, gun);
    set_trigger(&mut app, 1, true);

    // 100 Hz gun on a 120 Hz step fires < 1 round/tick, so two ticks land exactly
    // one round.
    app.update();
    app.update();

    let mut q = app.world_mut().query::<&Bullet>();
    let world = app.world();
    let bullets: Vec<&Bullet> = q.iter(world).collect();
    assert_eq!(bullets.len(), 1, "two ticks at 100 Hz should fire exactly one round");
    let v = bullets[0].vel_enu;
    assert!((v.x - 1030.0).abs() < 1e-6, "East component should be the muzzle speed, got {}", v.x);
    assert!((v.y - 200.0).abs() < 1e-6, "North component should inherit own velocity, got {}", v.y);
    assert!(v.z.abs() < 1e-9, "level shot has no vertical component, got {}", v.z);
}

#[test]
fn sustained_burst_kills_target_and_ends_episode() {
    // Shooter at the origin facing East, a stationary target 80 m ahead on the
    // boresight. With zero dispersion every round connects; a 20 HP target dies
    // after ten 2 HP hits, and the two-team engagement then terminates.
    let mut app = combat_app(42, 2);
    let mut gun = Gun::m61a1();
    gun.dispersion_rad = 0.0;
    spawn_shooter(&mut app, 1, 0, DVec3::ZERO, DVec3::ZERO, FRAC_PI_2, gun);
    spawn_target(&mut app, 2, 1, DVec3::new(80.0, 0.0, 0.0), 20.0);
    set_trigger(&mut app, 1, true);

    let mut exited = false;
    for _ in 0..400 {
        app.update();
        if app.should_exit().is_some() {
            exited = true;
            break;
        }
    }

    assert!(exited, "the engagement should end once one team is eliminated");
    assert!(is_dead(&mut app, 2), "the target should be marked Dead");
    assert!(
        hp_of(&mut app, 2) <= 0.0,
        "the target's HP should be depleted, got {}",
        hp_of(&mut app, 2)
    );
    assert!(!is_dead(&mut app, 1), "the shooter should survive");
}

#[test]
fn friendly_fire_does_no_damage() {
    // Same team, single-team scenario: rounds pass through the friendly and the
    // episode never self-terminates.
    let mut app = combat_app(3, 1);
    let mut gun = Gun::m61a1();
    gun.dispersion_rad = 0.0;
    spawn_shooter(&mut app, 1, 0, DVec3::ZERO, DVec3::ZERO, FRAC_PI_2, gun);
    spawn_target(&mut app, 2, 0, DVec3::new(80.0, 0.0, 0.0), 20.0);
    set_trigger(&mut app, 1, true);

    for _ in 0..200 {
        app.update();
    }

    assert!((hp_of(&mut app, 2) - 20.0).abs() < f32::EPSILON, "a teammate takes no damage");
    assert!(app.should_exit().is_none(), "a single-team scenario never self-terminates");
}

#[test]
fn engagement_is_deterministic() {
    // Two identical runs (same seed, dispersion on so the RNG is exercised) must
    // produce byte-identical state: HP, round count, and every round's position.
    let run = || {
        let mut app = combat_app(7, 2);
        spawn_shooter(&mut app, 1, 0, DVec3::ZERO, DVec3::ZERO, FRAC_PI_2, Gun::m61a1());
        spawn_target(&mut app, 2, 1, DVec3::new(200.0, 0.0, 0.0), 1000.0);
        set_trigger(&mut app, 1, true);
        for _ in 0..120 {
            app.update();
        }
        (hp_of(&mut app, 2), bullet_count(&mut app), bullet_pos_bits(&mut app))
    };

    let (hp_a, n_a, pos_a) = run();
    let (hp_b, n_b, pos_b) = run();
    assert_eq!(hp_a.to_bits(), hp_b.to_bits(), "target HP must be identical across runs");
    assert_eq!(n_a, n_b, "round count must be identical across runs");
    assert_eq!(pos_a, pos_b, "every round's position must be identical across runs");
    assert!(n_a > 0, "the run should actually fire rounds");
    assert!(hp_a < 1000.0, "the run should actually land hits");
}
