//! The assembled racing app: a real headless Bevy app, driven by `update()` through
//! the same `Startup` + `Update` schedule the binary runs. These tests need JSBSim
//! (the F450 model), so each takes the process-wide FDM guard.

mod common;

use astrum_race::components::RaceCourse;
use astrum_race::course::{generate, CourseParams};
use astrum_race::gate::Gate;
use astrum_race::scenario::RaceScenario;
use astrum_race::{build_app, build_app_with_course};
use bevy::prelude::*;
use common::{drone_state, fdm_guard, progress, race_scenario};

/// Startup stands up the whole track: one entity per gate, at the gate's pose, plus
/// the drone on gate 0's approach.
#[test]
fn app_spawns_the_course_and_the_drone() {
    let _fdm = fdm_guard();
    let params = CourseParams { gates: 6, ..CourseParams::default() };
    let scenario = race_scenario(params);
    let course = generate(scenario.sim.seed, &params).unwrap();
    let first = course.gates[0];

    let mut app = build_app_with_course(scenario, course.clone());
    app.update(); // Startup: spawn_course then spawn_drone, plus one physics step

    // Every gate is an entity, in order, matching the resource.
    let mut gates: Vec<Gate> = {
        let mut q = app.world_mut().query::<&Gate>();
        let world = app.world();
        q.iter(world).copied().collect()
    };
    gates.sort_by_key(|g| g.index);
    assert_eq!(gates.len(), 6, "one entity per gate");
    assert_eq!(gates, app.world().resource::<RaceCourse>().0.gates, "entities match the course");

    // Each gate entity carries the transform its pose implies.
    let mut q = app.world_mut().query::<(&Gate, &Transform)>();
    let world = app.world();
    for (g, tf) in q.iter(world) {
        assert!((tf.translation - g.center.as_vec3()).length() < 1e-4, "gate {} pose", g.index);
    }

    // The drone spawned upstream of gate 0, on its centreline, at its height.
    let st = drone_state(&mut app).expect("the drone should have spawned");
    let offset = st.pos_enu - first.center;
    let along = offset.dot(first.normal());
    let lateral = (offset - along * first.normal() - offset.z * bevy::math::DVec3::Z).length();
    assert!(
        (along + 10.0).abs() < 1.0,
        "should start ~10 m upstream of gate 0, got {along:.2} m along the normal"
    );
    assert!(lateral < 1.0, "should start on the gate's centreline, {lateral:.2} m off");
    assert!((st.pos_enu.z - first.center.z).abs() < 1.0, "should start at the gate's height");
    assert!(st.lat_deg.is_finite() && st.lon_deg.is_finite() && st.alt_m.is_finite());
}

/// Physics runs, the pose keeps being derived, and the drone holds the station it
/// launched on: one step per `update()`, no drift, nothing scored.
///
/// The hold is what makes the environment a usable vessel — an F450 with motors
/// running and no commands falls, hits the ground, and JSBSim NaNs out.
#[test]
fn app_holds_station_and_scores_nothing_by_default() {
    let _fdm = fdm_guard();
    let scenario = race_scenario(CourseParams { gates: 4, ..CourseParams::default() });
    let mut app = build_app(scenario).expect("the default envelope is satisfiable");

    app.update();
    let start = drone_state(&mut app).unwrap();
    for _ in 0..1200 {
        app.update();
    } // 10 s @ 120 Hz
    let st = drone_state(&mut app).unwrap();

    // One fixed step per update, no wall-clock coupling.
    assert!(
        (st.sim_time_s - start.sim_time_s - 10.0).abs() < 0.05,
        "1200 steps at 120 Hz should be ~10 s of sim, got {:.3}",
        st.sim_time_s - start.sim_time_s
    );
    assert!(st.pos_enu.is_finite() && st.vel_enu.is_finite(), "state stayed finite");
    assert!(st.phi_rad.is_finite() && st.theta_rad.is_finite() && st.psi_rad.is_finite());

    // Held its launch point: altitude within a metre, drift under a few metres,
    // essentially level, and not still moving.
    let drift = (st.pos_enu - start.pos_enu).length();
    eprintln!(
        "[hold] drift={drift:.2} m dalt={:.2} m |v|={:.2} m/s phi={:.2} deg theta={:.2} deg",
        st.pos_enu.z - start.pos_enu.z,
        st.vel_enu.length(),
        st.phi_rad.to_degrees(),
        st.theta_rad.to_degrees()
    );
    assert!(
        (st.pos_enu.z - start.pos_enu.z).abs() < 1.0,
        "should hold altitude, drifted {drift:.2} m"
    );
    assert!(drift < 3.0, "should hold station, drifted {drift:.2} m");
    assert!(st.vel_enu.length() < 1.0, "should be settled, moving {:.2} m/s", st.vel_enu.length());
    assert!(st.phi_rad.abs() < 0.2 && st.theta_rad.abs() < 0.2, "should stay near level");

    // Nothing commands the drone through a gate, so nothing is scored.
    let p = progress(&app);
    assert_eq!((p.passed, p.next_gate, p.missed), (0, 0, 0), "an unflown course scores nothing");
    assert!(!p.finished);
    // ...but the gate tracker has seen the pending gate and holds a reference tick.
    assert!(p.last_local.is_some(), "the tracker should be watching gate 0");
}

/// `max_time_s` terminates the app, and the run is reproducible: the same scenario
/// twice gives the same trajectory.
#[test]
fn app_honours_the_time_limit_and_repeats_exactly() {
    let _fdm = fdm_guard();
    let mut scenario = race_scenario(CourseParams { gates: 4, ..CourseParams::default() });
    scenario.sim.max_time_s = 2.0;

    let fly = |s: RaceScenario| {
        let mut app = build_app(s).unwrap();
        let mut steps = 0;
        // 120 Hz x 2 s = 240 steps; the cap is a generous upper bound on that.
        while app.should_exit().is_none() && steps < 2000 {
            app.update();
            steps += 1;
        }
        (steps, drone_state(&mut app).unwrap())
    };

    let (steps_a, a) = fly(scenario.clone());
    let (steps_b, b) = fly(scenario);

    assert!(steps_a < 2000, "the time limit should have stopped the run, not the cap");
    assert!((a.sim_time_s - 2.0).abs() < 0.1, "should stop at ~2 s, got {:.3}", a.sim_time_s);
    assert_eq!(steps_a, steps_b, "the same scenario must take the same number of steps");
    assert_eq!(a.pos_enu, b.pos_enu, "and end in exactly the same place");
}

/// An unsatisfiable envelope fails before the app is built — the caller sees the
/// error instead of a course-less app logging its way through a run.
#[test]
fn unsatisfiable_envelope_fails_the_build() {
    let scenario = race_scenario(CourseParams {
        gates: 60,
        min_gate_gap_m: 8.0,
        course_radius_m: 20.0,
        tries_per_gate: 20,
        max_backtracks: 20,
        ..CourseParams::default()
    });
    assert!(build_app(scenario).is_err(), "an overpacked course should not build an app");
}

/// The shipped `config/race.toml` parses, validates, and generates.
#[test]
fn shipped_scenario_is_valid() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../config/race.toml");
    let scenario = RaceScenario::load(path).expect("config/race.toml should load");
    assert_eq!(scenario.drone.model, "F450", "the racing platform is the F450");
    let course = astrum_race::generate_course(&scenario).expect("its envelope is satisfiable");
    assert_eq!(course.len(), scenario.course.gates as usize);
}
