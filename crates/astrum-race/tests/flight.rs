//! The real thing: a scripted controller flies the F450 at gate 0 through the sim's
//! own lockstep, over an in-process Zenoh session, and the race is scored.
//!
//! The controller is a plain feedback law in another thread — a deterministic
//! function of the telemetry it receives, sending `Multirotor` body-rate demands and
//! collective like any multirotor controller — so the whole run, JSBSim included, is
//! reproducible: flown twice it ends in the identical state.

mod common;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use astrum_net::{CommandSender, LatestTelemetry, Selection, Session};
use astrum_proto::{Control, EventKind, MultirotorRaw, RaceOutcome, Telemetry};
use astrum_race::build_race_app;
use astrum_race::components::{RaceCourse, RaceProgress};
use astrum_race::scenario::RaceScenario;
use astrum_sim::components::{SimClock, SimStatus};
use astrum_sim::events::EventLog;
use astrum_sim::net::LastPublished;
use astrum_sim::RunMode;
use bevy::prelude::*;
use common::{fdm_guard, flat_course, scenario_toml};

/// Normalised rate demand per rad/s (the F450's FCS: ~0.75 rad/s per unit).
const RATE_PER_UNIT: f64 = 0.75;
/// Collective that roughly hovers the F450 (the altitude loop trims the rest).
const HOVER_COLLECTIVE: f64 = 0.45;

/// Where gate 0 is and how fast to fly at it: enough to be a real pass, slow enough
/// that an untuned attitude loop stays inside the opening.
struct Target {
    east: f64,
    north: f64,
    up: f64,
    speed_mps: f64,
}

/// Altitude hold, heading hold north, a forward-speed loop through pitch and a
/// lateral loop through roll, with an integrator on the altitude. All state is in
/// `integral`, so the law is a deterministic function of the telemetry sequence.
fn law(t: &Telemetry, target: &Target, integral: &mut f64) -> MultirotorRaw {
    let (dt, e) = (1.0 / 120.0, t.pos_enu);
    let v = t.vel_enu;
    let a = t.attitude;

    let up_err = target.up - e.u;
    *integral = (*integral + up_err * dt).clamp(-0.1, 0.1);
    let tilt = (a.phi_rad.cos() * a.theta_rad.cos()).max(0.5);
    let collective =
        ((HOVER_COLLECTIVE + 0.12 * up_err + 0.5 * *integral - 0.10 * v.u) / tilt).clamp(0.0, 1.0);

    let v_des = (0.5 * (target.north - e.n)).clamp(0.0, target.speed_mps);
    let pitch_des = (-0.10 * (v_des - v.n)).clamp(-0.2, 0.2); // nose down to go north
    let roll_des = (0.10 * (target.east - e.e) - 0.15 * v.e).clamp(-0.1, 0.1);
    let yaw_err = (-a.psi_rad + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU)
        - std::f64::consts::PI;

    let rate = |err: f64, kp: f64| (kp * err / RATE_PER_UNIT).clamp(-1.0, 1.0);
    MultirotorRaw {
        roll_rate: rate(roll_des - a.phi_rad, 4.0),
        pitch_rate: rate(pitch_des - a.theta_rad, 4.0),
        yaw_rate: rate(yaw_err, 2.0),
        collective,
    }
}

/// What a run ended as, for comparing two of them.
#[derive(Debug, Clone, PartialEq)]
struct Flown {
    telemetry: Vec<Telemetry>,
    passed: u32,
    outcome: RaceOutcome,
    events: Vec<EventKind>,
}

/// Fly the scenario in lockstep until gate 0 is passed (or the race ends, or
/// `max_ticks`), under [`law`] in a controller thread.
fn fly_to_gate_0(max_ticks: u64) -> Flown {
    let toml = scenario_toml(
        "lockstep = true\nlockstep_timeout_s = 20.0",
        "lockstep = true",
        &flat_course(3.0, 3),
    );
    let race = RaceScenario::from_toml_str(&toml).unwrap().resolve(None).unwrap();
    let g0 = race.course.gates[0].center;
    let target = Target { east: g0.x, north: g0.y, up: g0.z, speed_mps: 2.5 };

    let session = Session::in_process().unwrap();
    let mut app = build_race_app(race, RunMode::Embedded);
    astrum_sim::net::connect(&mut app, &session).unwrap();
    astrum_race::net::connect(&mut app, &session).unwrap();

    let stop = Arc::new(AtomicBool::new(false));
    let controller = {
        let (session, stop) = (session.clone(), Arc::clone(&stop));
        std::thread::spawn(move || {
            let mut sub = LatestTelemetry::subscribe(&session, Selection::Aircraft(1)).unwrap();
            let mut tx = CommandSender::new(&session, 1).unwrap();
            let (mut last, mut integral) = (None, 0.0);
            while !stop.load(Ordering::Relaxed) {
                if sub.wait(Duration::from_millis(20)).is_err() {
                    continue;
                }
                let t = sub.latest(1).unwrap();
                if last != Some(t.tick) {
                    let cmd = law(&t, &target, &mut integral);
                    tx.send(Control::Multirotor(cmd), false, t.tick).unwrap();
                    last = Some(t.tick);
                }
            }
        })
    };

    let mut events = Vec::new();
    app.update(); // startup publishes tick 0, then waits for its answer
    while app.world().resource::<SimClock>().tick < max_ticks
        && !app.world().resource::<RaceProgress>().finished()
        && app.world().resource::<RaceProgress>().passed < 1
    {
        app.update();
        events.extend(app.world().resource::<EventLog>().last().iter().map(|e| e.kind.clone()));
        assert!(app.world().resource::<SimStatus>().fatal.is_none());
    }
    stop.store(true, Ordering::Relaxed);
    controller.join().unwrap();

    let p = *app.world().resource::<RaceProgress>();
    let mut telemetry = app.world().resource::<LastPublished>().0.clone();
    for t in &mut telemetry {
        // Controller epochs are random per sender.
        t.command.owner_epoch = 0;
        t.command.last_refused_epoch = 0;
    }
    Flown { telemetry, passed: p.passed, outcome: p.outcome, events }
}

/// A scripted controller flies the F450 from 10 m upstream through gate 0 — scored,
/// announced as an event — and a lockstep run is reproducible to the bit.
#[test]
fn a_scripted_controller_flies_through_gate_0_and_the_run_replays_exactly() {
    let _fdm = fdm_guard();
    let a = fly_to_gate_0(120 * 20);
    let b = fly_to_gate_0(120 * 20);

    let t = a.telemetry[0];
    assert_eq!(a.outcome, RaceOutcome::Running, "it flew, it did not crash: {:?}", a.events);
    assert_eq!(a.passed, 1, "gate 0 passed (CG at {:?}, tick {})", t.pos_enu, t.tick);
    assert!(
        a.events.iter().any(|e| matches!(e, EventKind::GatePassed { gate: 0, .. })),
        "and announced: {:?}",
        a.events
    );
    assert!(t.sim_time_s > 3.0, "it takes a real flight, not a teleport: {:.2}s", t.sim_time_s);
    assert_eq!(a, b, "a lockstep run with a deterministic controller replays exactly");
}

/// The course the app holds is the one the scenario resolves to.
#[test]
fn the_app_flies_the_resolved_course() {
    let _fdm = fdm_guard();
    let toml = scenario_toml("", "", &flat_course(3.0, 3));
    let race = RaceScenario::from_toml_str(&toml).unwrap().resolve(None).unwrap();
    let gates = race.course.gates.clone();
    let app = build_race_app(race, RunMode::Embedded);
    assert_eq!(app.world().resource::<RaceCourse>().0.gates, gates);
}
