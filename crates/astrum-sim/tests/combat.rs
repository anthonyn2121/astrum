//! The gun-combat chain over hand-placed entities (no JSBSim): swept hit tests
//! in relative motion, earliest-hit attribution, velocity inheritance, damage,
//! termination, the events it reports, and determinism.

use std::f64::consts::FRAC_PI_2;

use astrum_fdm::FdmState;
use astrum_geo::{Enu, Geodetic, LocalFrame};
use astrum_proto::{EndReason, EventKind};
use astrum_sim::combat::ballistics::DVec3;
use astrum_sim::combat::systems::{
    apply_damage, end_on_elimination, fire_guns, on_kill, resolve_hits, step_bullets,
};
use astrum_sim::combat::{
    Bullet, CombatParams, CombatRng, CombatState, Dead, Gun, Health, Team, Trigger,
};
use astrum_sim::components::{Aircraft, AircraftState, SimClock, SimParams, SimStatus, WorldFrame};
use astrum_sim::events::{publish_events, EventLog};
use bevy::prelude::*;

const DT: f64 = 1.0 / 120.0;
const ORIGIN: (f64, f64) = (37.6188, -122.3756);

fn params() -> SimParams {
    SimParams {
        rate_hz: 120.0,
        dt: DT,
        max_time_s: 0.0,
        speed: 0.0,
        lease_ticks: None,
        lockstep: false,
        lockstep_timeout_s: 1.0,
    }
}

fn base_app(seed: u64, teams: u8) -> App {
    let mut app = App::new();
    app.insert_resource(params())
        // The grid sits 3 km up, so level rounds never reach the h = 0 cut-off.
        .insert_resource(WorldFrame(LocalFrame::new(Geodetic::from_degrees(
            ORIGIN.0, ORIGIN.1, 3000.0,
        ))))
        .insert_resource(CombatParams::default())
        .insert_resource(CombatRng::seeded(seed))
        .insert_resource(CombatState { initial_living_teams: teams })
        .insert_resource(SimClock::default())
        .insert_resource(SimStatus::default())
        .insert_resource(EventLog::default())
        .add_observer(on_kill);
    app
}

/// The full chain (aircraft are stationary here, so no FDM step is needed).
fn combat_app(seed: u64, teams: u8) -> App {
    let mut app = base_app(seed, teams);
    app.add_systems(
        Update,
        (fire_guns, step_bullets, resolve_hits, apply_damage, end_on_elimination, publish_events)
            .chain(),
    );
    app
}

/// Every event of the latest update.
fn events(app: &App) -> Vec<EventKind> {
    app.world().resource::<EventLog>().last().iter().map(|e| e.kind.clone()).collect()
}

fn state(pos: DVec3, prev: DVec3, vel: DVec3, psi_rad: f64) -> AircraftState {
    AircraftState {
        // Near the grid origin, so local level == grid.
        fdm: FdmState { lat_deg: ORIGIN.0, lon_deg: ORIGIN.1, psi_rad, ..Default::default() },
        pos_enu: Enu { e: pos.x, n: pos.y, u: pos.z },
        prev_pos_enu: Enu { e: prev.x, n: prev.y, u: prev.z },
        vel_enu: Enu { e: vel.x, n: vel.y, u: vel.z },
    }
}

fn spawn_shooter(app: &mut App, id: u32, team: u8, vel: DVec3, psi_rad: f64, gun: Gun) {
    app.world_mut().spawn((
        Aircraft { id },
        state(DVec3::ZERO, DVec3::ZERO, vel, psi_rad),
        Team(team),
        Health::full(100.0),
        Trigger(true),
        gun,
    ));
}

fn spawn_target(app: &mut App, id: u32, team: u8, pos: DVec3, prev: DVec3, hp: f32) {
    app.world_mut().spawn((
        Aircraft { id },
        state(pos, prev, DVec3::ZERO, 0.0),
        Team(team),
        Health::full(hp),
    ));
}

fn spawn_round(app: &mut App, prev: DVec3, pos: DVec3) {
    let round = CombatParams::default().round;
    app.world_mut().spawn(Bullet {
        pos,
        prev_pos: prev,
        vel: (pos - prev) / DT,
        ballistics: round,
        owner_id: 99,
        team: 0,
        ttl_s: 3.0,
    });
}

fn hp_of(app: &mut App, id: u32) -> f32 {
    let mut q = app.world_mut().query::<(&Aircraft, &Health)>();
    q.iter(app.world()).find(|(a, _)| a.id == id).map(|(_, h)| h.hp).unwrap()
}

fn is_dead(app: &mut App, id: u32) -> bool {
    let mut q = app.world_mut().query::<(&Aircraft, Has<Dead>)>();
    q.iter(app.world()).find(|(a, _)| a.id == id).map(|(_, d)| d).unwrap()
}

fn bullets(app: &mut App) -> Vec<Bullet> {
    let mut q = app.world_mut().query::<&Bullet>();
    q.iter(app.world()).copied().collect()
}

fn hits_only(app: &mut App) {
    app.add_systems(Update, (resolve_hits, apply_damage).chain());
}

#[test]
fn swept_test_catches_a_round_that_steps_past_a_target() {
    // A 2500 m/s round steps 20.8 m in one tick, straddling an 8 m-radius target:
    // both endpoints are ~10 m away, only the swept test sees the pass.
    let mut app = base_app(1, 2);
    hits_only(&mut app);
    spawn_target(&mut app, 2, 1, DVec3::new(100.0, 0.0, 0.0), DVec3::new(100.0, 0.0, 0.0), 100.0);
    spawn_round(&mut app, DVec3::new(90.0, 0.0, 0.0), DVec3::new(110.8, 0.0, 0.0));
    app.update();
    assert!(hp_of(&mut app, 2) < 100.0);
    assert!(bullets(&mut app).is_empty(), "a round that connects is consumed");
}

#[test]
fn a_target_moving_during_the_tick_is_tested_where_it_was() {
    // A 1000 m/s round crosses in front of a target retreating south at 300 m/s.
    // Its end-of-tick position is 9.5 m off the round's path (a miss for an
    // end-point test), but in relative motion they pass 7.9 m apart: a hit.
    let mut app = base_app(1, 2);
    hits_only(&mut app);
    spawn_target(&mut app, 2, 1, DVec3::new(4.17, -9.5, 0.0), DVec3::new(4.17, -7.0, 0.0), 100.0);
    spawn_round(&mut app, DVec3::ZERO, DVec3::new(1000.0 * DT, 0.0, 0.0));
    app.update();
    assert!(hp_of(&mut app, 2) < 100.0, "the relative-motion sweep must register the pass");
}

#[test]
fn a_round_hits_the_first_target_it_reaches() {
    let mut app = base_app(1, 2);
    hits_only(&mut app);
    // Both targets lie within the hit radius of the round's 20 m segment.
    spawn_target(&mut app, 7, 1, DVec3::new(17.0, 0.0, 0.0), DVec3::new(17.0, 0.0, 0.0), 100.0);
    spawn_target(&mut app, 8, 1, DVec3::new(3.0, 0.0, 0.0), DVec3::new(3.0, 0.0, 0.0), 100.0);
    spawn_round(&mut app, DVec3::new(-6.0, 0.0, 0.0), DVec3::new(14.0, 0.0, 0.0));
    app.update();
    assert!(hp_of(&mut app, 8) < 100.0, "the nearer target along the path takes the hit");
    assert_eq!(hp_of(&mut app, 7), 100.0, "one round damages one target");
}

#[test]
fn muzzle_velocity_is_boresight_plus_own_velocity() {
    let mut app = base_app(1, 1);
    app.add_systems(Update, fire_guns);
    let mut gun = Gun::m61a1();
    gun.dispersion_rad = 0.0;
    // Facing east while flying north at 200 m/s.
    spawn_shooter(&mut app, 1, 0, DVec3::new(0.0, 200.0, 0.0), FRAC_PI_2, gun);
    // 100 rounds/s at 120 Hz: two ticks fire exactly one round.
    app.update();
    app.update();
    let b = bullets(&mut app);
    assert_eq!(b.len(), 1);
    assert!((b[0].vel - DVec3::new(1030.0, 200.0, 0.0)).length() < 1e-6, "{:?}", b[0].vel);
    assert!((b[0].pos - DVec3::new(4.0, 0.0, 0.0)).length() < 1e-9, "leaves 4 m ahead on the nose");
}

#[test]
fn sustained_burst_kills_and_decides_the_engagement() {
    let mut app = combat_app(42, 2);
    let mut gun = Gun::m61a1();
    gun.dispersion_rad = 0.0;
    spawn_shooter(&mut app, 1, 0, DVec3::ZERO, FRAC_PI_2, gun);
    spawn_target(&mut app, 2, 1, DVec3::new(80.0, 0.0, 0.0), DVec3::new(80.0, 0.0, 0.0), 20.0);
    let mut exited = false;
    let mut seen = Vec::new();
    for _ in 0..400 {
        app.update();
        seen.extend(events(&app));
        if app.should_exit().is_some() {
            exited = true;
            break;
        }
    }
    assert!(exited, "one team eliminated ends the run");
    assert!(is_dead(&mut app, 2) && hp_of(&mut app, 2) <= 0.0);
    assert!(!is_dead(&mut app, 1));

    // Reported: the press, every hit (the shooter's, on the target, where the
    // round struck), the kill credited to the shooter, then the run's end.
    assert_eq!(seen[0], EventKind::TriggerPressed { aircraft_id: 1, ammo: 511 });
    let n = seen.len();
    let hits = &seen[1..n - 2];
    let damage = CombatParams::default().damage_per_hit;
    for (i, hit) in hits.iter().enumerate() {
        let EventKind::Hit { shooter_id, target_id, hit_points, pos_enu, .. } = *hit else {
            panic!("event {i} between the press and the kill: {hit:?}");
        };
        assert_eq!((shooter_id, target_id), (1, 2));
        assert_eq!(hit_points, 20.0 - damage * (i + 1) as f32);
        let miss = ((pos_enu.e - 80.0).powi(2) + pos_enu.n.powi(2) + pos_enu.u.powi(2)).sqrt();
        assert!(miss <= CombatParams::default().hit_radius_m + 1e-9, "struck {miss} m off");
    }
    assert!(matches!(seen[n - 2], EventKind::Kill { victim_id: 2, killer_id: Some(1), .. }));
    assert!(matches!(
        seen[n - 1],
        EventKind::RunEnded { reason: EndReason::EngagementDecided, .. }
    ));
}

#[test]
fn a_shooter_destroyed_mid_burst_reports_its_release_then_its_death() {
    let mut app = combat_app(5, 1);
    spawn_shooter(&mut app, 1, 0, DVec3::ZERO, FRAC_PI_2, Gun::m61a1());
    for _ in 0..30 {
        app.update();
    }
    let mut q = app.world_mut().query::<(Entity, &Gun)>();
    let (shooter, ammo) = q.iter(app.world()).map(|(e, g)| (e, g.ammo)).next().unwrap();
    assert!(ammo < 511, "the burst fired");
    // Destroyed by something other than a round: nobody to credit.
    app.world_mut().entity_mut(shooter).insert(Dead);
    app.update();
    assert_eq!(
        events(&app),
        [
            EventKind::TriggerReleased { aircraft_id: 1, rounds_fired: 511 - ammo, ammo },
            EventKind::Kill {
                victim_id: 1,
                killer_id: None,
                pos_enu: Enu { e: 0.0, n: 0.0, u: 0.0 }
            },
        ]
    );
}

#[test]
fn friendly_fire_does_no_damage() {
    let mut app = combat_app(3, 1);
    let mut gun = Gun::m61a1();
    gun.dispersion_rad = 0.0;
    spawn_shooter(&mut app, 1, 0, DVec3::ZERO, FRAC_PI_2, gun);
    spawn_target(&mut app, 2, 0, DVec3::new(80.0, 0.0, 0.0), DVec3::new(80.0, 0.0, 0.0), 20.0);
    for _ in 0..200 {
        app.update();
    }
    assert_eq!(hp_of(&mut app, 2), 20.0);
    assert!(app.should_exit().is_none(), "a single-team scenario never self-terminates");
}

#[test]
fn rounds_below_the_ellipsoid_are_culled() {
    let mut app = combat_app(1, 1);
    // 3001 m below a grid origin 3000 m above the ellipsoid.
    spawn_round(&mut app, DVec3::new(0.0, 0.0, -3000.5), DVec3::new(0.0, 0.0, -3001.0));
    app.update();
    assert!(bullets(&mut app).is_empty());
}

#[test]
fn engagement_is_deterministic() {
    let run = || {
        let mut app = combat_app(7, 2);
        spawn_shooter(&mut app, 1, 0, DVec3::ZERO, FRAC_PI_2, Gun::m61a1());
        spawn_shooter(&mut app, 3, 0, DVec3::new(0.0, 10.0, 0.0), FRAC_PI_2, Gun::m61a1());
        spawn_target(
            &mut app,
            2,
            1,
            DVec3::new(200.0, 0.0, 0.0),
            DVec3::new(200.0, 0.0, 0.0),
            1000.0,
        );
        for _ in 0..120 {
            app.update();
        }
        let mut pos: Vec<[u64; 3]> =
            bullets(&mut app).iter().map(|b| b.pos.to_array().map(f64::to_bits)).collect();
        pos.sort_unstable();
        (hp_of(&mut app, 2).to_bits(), pos)
    };
    let (hp_a, pos_a) = run();
    assert!(!pos_a.is_empty() && f32::from_bits(hp_a) < 1000.0, "the run fires and hits");
    assert_eq!(run(), (hp_a, pos_a));
}
