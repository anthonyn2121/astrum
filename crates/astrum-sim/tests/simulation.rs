//! Physics-level behaviour of the real app (JSBSim FDMs) through the embedded
//! env API: the fleet flies on the grid, the clock is the sim's own, the
//! autopilot tracks targets, and an abandoned controller triggers the failsafe.

mod common;

use astrum_proto::ControlMode;
use astrum_sim::combat::Dead;
use astrum_sim::components::{Aircraft, SimClock};
use astrum_sim::env::SimEnv;
use astrum_sim::net::LastPublished;
use astrum_sim::{build_app, RunMode};
use bevy::prelude::*;
use common::{autopilot, of, raw, scenario, single_ship, spec, two_ship, wrap, DT, ORIGIN};

#[test]
fn fleet_flies_on_the_grid_with_independent_fdms() {
    let mut env = SimEnv::new(two_ship()).unwrap();
    let early = env.observe();
    let mut late = early.clone();
    for _ in 0..240 {
        late = env.step(&[]).unwrap().0;
    }
    assert_eq!(late[0].tick, 241);
    assert!((late[0].sim_time_s - 241.0 * DT).abs() < 1e-12, "time is tick·dt");
    for (a, b) in early.iter().zip(&late) {
        let moved =
            ((b.pos_enu.e - a.pos_enu.e).powi(2) + (b.pos_enu.n - a.pos_enu.n).powi(2)).sqrt();
        // 2 s at ~55 m/s.
        assert!((moved - 110.0).abs() < 10.0, "aircraft {} moved {moved:.1} m", b.aircraft_id);
        // Grid velocity and NED velocity describe the same motion near the origin.
        assert!(
            (b.vel_enu.e - b.vel_ned.e).abs() < 0.05 && (b.vel_enu.n - b.vel_ned.n).abs() < 0.05
        );
    }
    let sep = (late[0].pos_enu.e - late[1].pos_enu.e).abs();
    assert!(sep > 200.0, "opposite headings separate, sep {sep:.1} m");
}

#[test]
fn trimmed_uncommanded_aircraft_holds_its_initial_condition() {
    let mut env = SimEnv::new(single_ship()).unwrap();
    let mut t = env.observe()[0];
    for _ in 0..1200 {
        t = env.step(&[]).unwrap().0[0];
    }
    assert_eq!(t.control_mode, ControlMode::Uncommanded);
    assert!((t.tas_mps - 55.0).abs() < 1.5, "TAS held, got {}", t.tas_mps);
    assert!((t.alt_hae_m - 1524.0).abs() < 20.0, "altitude held, got {}", t.alt_hae_m);
    assert!(wrap(t.attitude.psi_rad - 90f64.to_radians()).abs() < 0.05, "heading held");
}

#[test]
fn autopilot_turns_to_heading_and_holds_altitude_and_speed() {
    let mut env = SimEnv::new(single_ship()).unwrap();
    let target = autopilot(0.0, 1600.0, 52.0);
    let mut t = env.observe()[0];
    for _ in 0..(60.0 / DT) as usize {
        t = env.step(&[(1, target)]).unwrap().0[0];
    }
    assert_eq!(t.control_mode, ControlMode::Autopilot);
    let hdg_err = wrap(0.0 - t.attitude.psi_rad).abs();
    assert!(hdg_err < 0.03, "heading settles (PI loops), err {:.3} rad", hdg_err);
    assert!(
        (t.alt_hae_m - 1600.0).abs() < 10.0,
        "climbs to and holds 1600 m, got {:.1}",
        t.alt_hae_m
    );
    assert!((t.tas_mps - 52.0).abs() < 1.5, "holds 52 m/s, got {:.2}", t.tas_mps);
}

#[test]
fn fighter_autopilot_captures_an_altitude_step_without_porpoising() {
    // Without climb-rate damping the f16's altitude loop limit-cycled at ±1 g
    // (71 m / 21° α peak-to-peak at 140 m/s) instead of settling.
    let mut env =
        SimEnv::new(scenario(vec![spec(1, "f16", ORIGIN.0, 0.0, 4572.0, 140.0)])).unwrap();
    let target = autopilot(0.0, 4602.0, 140.0);
    let (mut alt, mut alpha) = (Vec::new(), Vec::new());
    for _ in 0..(60.0 / DT) as usize {
        let t = env.step(&[(1, target)]).unwrap().0[0];
        alt.push(t.alt_hae_m);
        alpha.push(t.aero.alpha_rad.to_degrees());
    }
    let p2p = |v: &[f64]| {
        v.iter().copied().fold(f64::MIN, f64::max) - v.iter().copied().fold(f64::MAX, f64::min)
    };
    let last_20s = alt.len() - (20.0 / DT) as usize;
    let overshoot = alt.iter().copied().fold(f64::MIN, f64::max) - 4602.0;
    assert!(overshoot < 10.0, "a 30 m step overshoots by {overshoot:.1} m");
    assert!(
        p2p(&alt[last_20s..]) < 3.0,
        "altitude settles, last 20 s p2p {:.1} m",
        p2p(&alt[last_20s..])
    );
    assert!(
        p2p(&alpha[last_20s..]) < 1.0,
        "α settles, last 20 s p2p {:.1}°",
        p2p(&alpha[last_20s..])
    );
    assert!(
        (alt[alt.len() - 1] - 4602.0).abs() < 3.0,
        "holds 4602 m, got {:.1}",
        alt[alt.len() - 1]
    );
}

#[test]
fn abandoned_raw_controller_falls_back_to_the_failsafe_hold() {
    let mut env = SimEnv::new(single_ship()).unwrap();
    // A controller sends a gentle right roll for 0.25 s, then disappears.
    let mut t = env.observe()[0];
    for _ in 0..30 {
        t = env.step(&[(1, raw(0.2, 0.0, 0.6))]).unwrap().0[0];
    }
    assert_eq!(t.control_mode, ControlMode::Raw);
    // Within the 0.5 s lease the last command is held …
    for _ in 0..(0.4 / DT) as usize {
        t = env.step(&[]).unwrap().0[0];
    }
    assert_eq!(t.control_mode, ControlMode::Raw);
    // … then the sim takes over and holds the state at expiry.
    for _ in 0..(0.2 / DT) as usize {
        t = env.step(&[]).unwrap().0[0];
    }
    assert_eq!(t.control_mode, ControlMode::Failsafe);
    let at_expiry = t;
    for _ in 0..(20.0 / DT) as usize {
        t = env.step(&[]).unwrap().0[0];
    }
    assert!(
        t.attitude.phi_rad.abs() < 0.05,
        "failsafe levels the wings, φ={:.3}",
        t.attitude.phi_rad
    );
    assert!((t.alt_hae_m - at_expiry.alt_hae_m).abs() < 25.0, "failsafe holds altitude");
    assert!(
        wrap(t.attitude.psi_rad - at_expiry.attitude.psi_rad).abs() < 0.1,
        "failsafe holds the heading at expiry"
    );
}

#[test]
fn sim_clock_is_independent_of_dead_aircraft() {
    let mut app = build_app(two_ship(), RunMode::Embedded);
    app.update();
    // Kill whichever aircraft the ECS iterates first.
    let first = app
        .world_mut()
        .query_filtered::<Entity, With<Aircraft>>()
        .iter(app.world())
        .next()
        .unwrap();
    app.world_mut().entity_mut(first).insert(Dead);
    let dead_id = app.world().get::<Aircraft>(first).unwrap().id;
    app.update();
    let frozen = of(&app.world().resource::<LastPublished>().0, dead_id);
    for _ in 0..120 {
        app.update();
    }
    let tick = app.world().resource::<SimClock>().tick;
    let obs = app.world().resource::<LastPublished>().0.clone();
    assert_eq!(tick, 122);
    for t in &obs {
        assert_eq!(t.tick, tick, "every sample carries the sim clock");
    }
    let dead = of(&obs, dead_id);
    assert!(!dead.combat.alive);
    assert_eq!(dead.pos_enu, frozen.pos_enu, "a wreck is frozen in place");
}
