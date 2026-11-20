//! Several drones on one course: the start grid, independent scoring and crashes, the
//! end of the run, the results — and, on JSBSim, two scripted controllers in lockstep.
//!
//! Drones are ghosts to each other (no collisions); each is judged on its own against the
//! gates and the ground.

mod common;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use astrum_net::Session;
use astrum_proto::{EndReason, EventKind, RaceOutcome, Telemetry};
use astrum_race::build_race_app;
use astrum_race::components::{Race, RaceProgress};
use astrum_race::gate::Gate;
use astrum_race::scenario::RaceScenario;
use astrum_sim::components::{SimClock, SimParams, SimStatus};
use astrum_sim::events::EventLog;
use astrum_sim::net::LastPublished;
use astrum_sim::RunMode;
use bevy::math::DVec3;
use bevy::prelude::*;
use common::{
    exited, fdm_guard, flat_course, fly_id_to, frozen, progress_of, rig_n, scenario_toml_n,
    straight_course, Target,
};

// --- the start grid ----------------------------------------------------------------

/// Where each drone starts relative to gate 0, as `(id, along normal, along right, up)`.
fn grid(ids: &[u32], course_body: &str) -> (Vec<(u32, f64, f64, f64)>, Gate) {
    let toml = scenario_toml_n(ids, "", "", course_body);
    let race = RaceScenario::from_toml_str(&toml).unwrap().resolve(None).unwrap();
    let g0 = race.course.gates[0];
    let slots = race
        .grid()
        .into_iter()
        .map(|(id, at)| {
            let local = g0.to_local(at);
            (id, local.z, local.x, local.y)
        })
        .collect();
    for a in &race.scenario.aircraft {
        assert!((a.heading_deg - g0.yaw_rad.to_degrees()).abs() < 1e-9, "faces along gate 0");
        assert_eq!(a.tas_mps, 0.0);
    }
    (slots, g0)
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-3
}

/// One drone starts where a single-drone race always has: 10 m up gate 0's centreline.
#[test]
fn one_drone_starts_on_the_centreline() {
    let (slots, _) = grid(&[1], "gates = 3");
    let [(1, along, right, up)] = slots[..] else { panic!("{slots:?}") };
    assert!(close(along, -10.0) && close(right, 0.0) && close(up, 0.0), "{slots:?}");
}

/// Two drones start 2 m apart, centred on the centreline, the lower id on the left.
#[test]
fn two_drones_start_side_by_side() {
    let (slots, _) = grid(&[1, 2], "gates = 3");
    assert_eq!(slots.iter().map(|s| s.0).collect::<Vec<_>>(), [1, 2]);
    assert!(slots.iter().all(|s| close(s.1, -10.0) && close(s.3, 0.0)), "{slots:?}");
    assert!(close(slots[0].2, -1.0) && close(slots[1].2, 1.0), "{slots:?}");
}

/// Three drones: -2, 0, +2 m.
#[test]
fn three_drones_fill_the_grid() {
    let (slots, _) = grid(&[1, 2, 3], "gates = 3");
    let rights: Vec<f64> = slots.iter().map(|s| s.2).collect();
    assert!(rights.iter().zip([-2.0, 0.0, 2.0]).all(|(a, b)| close(*a, b)), "{rights:?}");
    assert!(slots.iter().all(|s| close(s.1, -10.0) && close(s.3, 0.0)), "{slots:?}");
}

/// Slots go by ascending id, whatever order the file lists them in, and the spacing is
/// the scenario's.
#[test]
fn slots_follow_the_ids_and_the_spacing() {
    let (slots, _) = grid(&[7, 3], "gates = 3\ngrid_spacing_m = 3.0");
    // `grid()` lists by id: 3 then 7.
    assert_eq!(slots.iter().map(|s| s.0).collect::<Vec<_>>(), [3, 7]);
    assert!(close(slots[0].2, -1.5) && close(slots[1].2, 1.5), "{slots:?}");

    let wide = scenario_toml_n(&[1, 2], "", "", "gates = 3\ngrid_spacing_m = -1.0");
    assert!(RaceScenario::from_toml_str(&wide).is_err());
}

/// The grid survives the resolved scenario (what the bag stores).
#[test]
fn the_grid_survives_the_resolved_scenario() {
    let toml = scenario_toml_n(&[1, 2, 3], "", "", "gates = 4\ngrid_spacing_m = 2.5");
    let race = RaceScenario::from_toml_str(&toml).unwrap().resolve(Some(3)).unwrap();
    let again = RaceScenario::from_toml_str(&race.to_toml()).unwrap().resolve(None).unwrap();
    assert_eq!(again.scenario, race.scenario);
    assert_eq!(again.grid_spacing_m(), 2.5);
}

// --- independent scoring, crashes, the end -----------------------------------------

fn passed(events: &[EventKind]) -> Vec<(u32, u32)> {
    events
        .iter()
        .filter_map(|e| match e {
            EventKind::GatePassed { aircraft_id, gate, .. } => Some((*aircraft_id, *gate)),
            _ => None,
        })
        .collect()
}

fn run_ended(events: &[EventKind]) -> Vec<EndReason> {
    events
        .iter()
        .filter_map(|e| match e {
            EventKind::RunEnded { reason, .. } => Some(*reason),
            _ => None,
        })
        .collect()
}

/// Each drone is scored on its own: one passes the gate, the other goes round it.
#[test]
fn drones_are_scored_independently() {
    let mut app =
        rig_n(straight_course(), &[DVec3::new(0.0, -5.0, 3.0), DVec3::new(1.5, -5.0, 3.0)]);
    let first = fly_id_to(&mut app, 1, DVec3::new(0.0, 5.0, 3.0));
    let second = fly_id_to(&mut app, 2, DVec3::new(1.5, 5.0, 3.0));
    assert_eq!(passed(&first), [(1, 0)]);
    assert!(
        second.contains(&EventKind::GateMissed { aircraft_id: 2, gate: 0 }),
        "the other went round: {second:?}"
    );
    let (a, b) = (progress_of(&app, 1), progress_of(&app, 2));
    assert_eq!((a.passed, a.missed, a.next_gate), (1, 0, 1));
    assert_eq!((b.passed, b.missed, b.next_gate), (0, 1, 0));
}

/// One drone crashes, the other flies on and completes: the crashed one is frozen and no
/// longer scored, the run goes on, and it ends `RaceComplete`.
#[test]
fn one_crashes_and_the_other_completes() {
    let mut app =
        rig_n(straight_course(), &[DVec3::new(0.0, -5.0, 2.0), DVec3::new(0.0, -5.0, 3.0)]);

    let events = fly_id_to(&mut app, 1, DVec3::new(0.0, -5.0, 0.1));
    assert!(events.contains(&EventKind::Crashed {
        aircraft_id: 1,
        outcome: RaceOutcome::CrashedGround,
        gate: 0
    }));
    assert_eq!(progress_of(&app, 1).outcome, RaceOutcome::CrashedGround);
    assert!(frozen(&mut app, 1), "a wreck is frozen where it fell");
    assert!(!frozen(&mut app, 2));
    assert!(!exited(&app), "the other drone is still racing");
    assert_eq!(progress_of(&app, 2).outcome, RaceOutcome::Running);

    // The wreck is not scored, even if something moves it through a gate.
    fly_id_to(&mut app, 1, DVec3::new(0.0, -2.0, 3.0));
    let ghost = fly_id_to(&mut app, 1, DVec3::new(0.0, 2.0, 3.0));
    assert!(passed(&ghost).is_empty(), "{ghost:?}");

    let mut events = Vec::new();
    for y in -5..=30 {
        events.extend(fly_id_to(&mut app, 2, DVec3::new(0.0, f64::from(y), 3.0)));
        if exited(&app) {
            break;
        }
    }
    assert_eq!(passed(&events), [(2, 0), (2, 1)]);
    assert_eq!(progress_of(&app, 2).outcome, RaceOutcome::Complete);
    assert_eq!(progress_of(&app, 1).passed, 0);
    assert_eq!(run_ended(&events), [EndReason::RaceComplete]);
    assert!(exited(&app));
}

/// A drone that hits a gate is out too (and frozen, while others race).
#[test]
fn a_gate_strike_ends_that_drones_race_only() {
    let mut app =
        rig_n(straight_course(), &[DVec3::new(0.30, -1.0, 3.0), DVec3::new(0.0, -5.0, 3.0)]);
    let events = fly_id_to(&mut app, 1, DVec3::new(0.30, 1.0, 3.0));
    assert!(events.contains(&EventKind::Crashed {
        aircraft_id: 1,
        outcome: RaceOutcome::CrashedGate,
        gate: 0
    }));
    assert!(frozen(&mut app, 1) && !exited(&app));
    assert_eq!(progress_of(&app, 2).outcome, RaceOutcome::Running);
}

/// Every drone crashed: the run ends `Crashed`, and nothing is frozen by the last crash.
#[test]
fn all_crashed_ends_the_run_crashed() {
    let mut app =
        rig_n(straight_course(), &[DVec3::new(0.0, -5.0, 2.0), DVec3::new(0.0, -6.0, 2.0)]);
    fly_id_to(&mut app, 1, DVec3::new(0.0, -5.0, 0.1));
    assert!(!exited(&app), "one is still flying");
    let events = fly_id_to(&mut app, 2, DVec3::new(0.0, -6.0, 0.1));
    assert!(exited(&app));
    assert_eq!(run_ended(&events), [EndReason::Crashed]);
    assert!(frozen(&mut app, 1) && !frozen(&mut app, 2), "the last crash ends the run first");
}

/// At the time cap the drones still racing are timed out; with one having completed the
/// run is a `RaceComplete`, and with none it is left to the sim's own `MaxTime`.
#[test]
fn the_time_cap_times_out_the_stragglers() {
    let mut app =
        rig_n(straight_course(), &[DVec3::new(0.0, -5.0, 3.0), DVec3::new(0.0, -5.0, 3.0)]);
    fly_id_to(&mut app, 1, DVec3::new(0.0, 5.0, 3.0));
    fly_id_to(&mut app, 1, DVec3::new(0.0, 25.0, 3.0));
    assert_eq!(progress_of(&app, 1).outcome, RaceOutcome::Complete);
    assert!(!exited(&app), "drone 2 is still racing");

    app.world_mut().resource_mut::<SimParams>().max_time_s = 1.0;
    app.world_mut().resource_mut::<SimClock>().tick = 120;
    let events = fly_id_to(&mut app, 2, DVec3::new(0.0, -4.0, 3.0));
    assert_eq!(progress_of(&app, 2).outcome, RaceOutcome::TimedOut);
    assert_eq!(progress_of(&app, 1).outcome, RaceOutcome::Complete, "a finished race stays");
    assert_eq!(run_ended(&events), [EndReason::RaceComplete]);

    // Nobody completed: the race layer leaves the end to the sim's time cap.
    let mut none = rig_n(straight_course(), &[DVec3::new(0.0, -5.0, 3.0)]);
    none.world_mut().resource_mut::<SimParams>().max_time_s = 1.0;
    none.world_mut().resource_mut::<SimClock>().tick = 120;
    let events = fly_id_to(&mut none, 1, DVec3::new(0.0, -4.0, 3.0));
    assert_eq!(progress_of(&none, 1).outcome, RaceOutcome::TimedOut);
    assert!(run_ended(&events).is_empty() && !exited(&none), "{events:?}");
}

// --- the results ----------------------------------------------------------------------

fn standing(id: u32, passed: u32, last_pass_s: f64, outcome: RaceOutcome) -> RaceProgress {
    RaceProgress { passed, last_pass_s, outcome, ..RaceProgress::new(id) }
}

fn race_of(drones: &[RaceProgress]) -> Race {
    let mut race = Race::new(drones.iter().map(|d| d.aircraft_id));
    for d in drones {
        *race.drone_mut(d.aircraft_id).unwrap() = *d;
    }
    race
}

/// Completers rank by finish time, the rest by gates passed then by when they passed
/// them, then by id; the winner is first place if it completed.
#[test]
fn places_rank_finish_time_then_progress() {
    let race = race_of(&[
        standing(1, 8, 41.5, RaceOutcome::Complete),
        standing(2, 8, 38.0, RaceOutcome::Complete),
        standing(3, 5, 20.0, RaceOutcome::CrashedGate),
        standing(4, 5, 18.0, RaceOutcome::TimedOut),
        standing(5, 6, 99.0, RaceOutcome::Running),
        standing(6, 5, 18.0, RaceOutcome::CrashedGround),
    ]);
    let results = race.results();
    let order: Vec<(u32, u32)> = results.iter().map(|r| (r.aircraft_id, r.place)).collect();
    assert_eq!(order, [(2, 1), (1, 2), (5, 3), (4, 4), (6, 5), (3, 6)]);
    assert_eq!(results[0].finish_time_s, Some(38.0));
    assert_eq!(results[2].finish_time_s, None);
    assert_eq!(race.winner(), Some(2));

    let none = race_of(&[
        standing(1, 3, 10.0, RaceOutcome::CrashedGround),
        standing(2, 4, 12.0, RaceOutcome::CrashedGate),
    ]);
    assert_eq!(none.results()[0].aircraft_id, 2, "more gates passed ranks first");
    assert_eq!(none.winner(), None, "nobody completed");
}

// --- real drones --------------------------------------------------------------------

/// What a two-drone run ended as, for comparing two of them.
#[derive(Debug, Clone, PartialEq)]
struct Flown {
    telemetry: Vec<Telemetry>,
    progress: Vec<RaceProgress>,
    events: Vec<EventKind>,
}

/// Two scripted controllers fly gate 0 in lockstep (drone 1 faster), until both passed it.
fn duel_to_gate_0() -> Flown {
    let toml = scenario_toml_n(
        &[1, 2],
        "lockstep = true\nlockstep_timeout_s = 20.0",
        "lockstep = true",
        &format!("{}\ngrid_spacing_m = 0.6", flat_course(3.0, 3)),
    );
    let race = RaceScenario::from_toml_str(&toml).unwrap().resolve(None).unwrap();
    let g0 = race.course.gates[0];
    // Both fly at the gate's centre, from 0.3 m either side of it (ghosts: they may overlap).
    let target = |speed_mps: f64| Target {
        east: g0.center.x,
        north: g0.center.y,
        up: g0.center.z,
        speed_mps,
    };
    let (fast, slow) = (target(2.5), target(1.5));

    let session = Session::in_process().unwrap();
    let mut app = build_race_app(race, RunMode::Embedded);
    astrum_sim::net::connect(&mut app, &session).unwrap();
    astrum_race::net::connect(&mut app, &session).unwrap();

    let stop = Arc::new(AtomicBool::new(false));
    let controllers = [
        common::spawn_controller(&session, &stop, 1, fast),
        common::spawn_controller(&session, &stop, 2, slow),
    ];

    let mut events = Vec::new();
    app.update(); // startup publishes tick 0, then waits for both answers
    let done = |app: &App| {
        app.world().resource::<Race>().drones().iter().all(|d| d.passed >= 1 || d.finished())
    };
    while app.world().resource::<SimClock>().tick < 120 * 30 && !done(&app) {
        app.update();
        events.extend(app.world().resource::<EventLog>().last().iter().map(|e| e.kind.clone()));
        assert!(app.world().resource::<SimStatus>().fatal.is_none());
    }
    stop.store(true, Ordering::Relaxed);
    for c in controllers {
        c.join().unwrap();
    }

    let mut telemetry = app.world().resource::<LastPublished>().0.clone();
    for t in &mut telemetry {
        // Controller epochs are random per sender.
        t.command.owner_epoch = 0;
        t.command.last_refused_epoch = 0;
    }
    let progress = app.world().resource::<Race>().drones().to_vec();
    Flown { telemetry, progress, events }
}

/// Two controllers, two drones, one lockstep run: each passes gate 0 (the faster one
/// first), the passes are announced per aircraft, and the whole run replays exactly.
#[test]
fn two_scripted_drones_race_in_lockstep_and_the_run_replays_exactly() {
    let _fdm = fdm_guard();
    let a = duel_to_gate_0();
    let b = duel_to_gate_0();

    assert_eq!(a.telemetry.len(), 2);
    for d in &a.progress {
        assert_eq!(
            (d.passed, d.outcome),
            (1, RaceOutcome::Running),
            "aircraft {} flew gate 0: {:?}",
            d.aircraft_id,
            a.events
        );
    }
    let order: Vec<u32> = passed(&a.events).iter().map(|(id, _)| *id).collect();
    assert_eq!(order, [1, 2], "the faster drone is through first");
    assert!(a.progress[0].last_pass_s < a.progress[1].last_pass_s);
    assert_eq!(a, b, "a lockstep run with deterministic controllers replays exactly");
}
