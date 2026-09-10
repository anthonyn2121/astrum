//! `astrum-race` — the ASTRUM drone-racing environment.
//!
//! A headless Bevy app (`MinimalPlugins`, no render/winit) in which a DJI F450
//! quadrotor, flown by JSBSim, has to fly through a course of racing gates that is
//! **randomized per seed** by the Monte-Carlo sampler in [`course`].
//!
//! This is a separate environment from the within-visual-range combat world in
//! `astrum-sim`: it shares only the JSBSim wrapper (`astrum-fdm`) and the
//! coordinate math (`astrum-geo`), and none of that crate's combat code.
//!
//! # Shape of the app
//!
//! * [`gate`] — the physical gate: a 37 x 3 in top bar on two 31 x 3 in uprights,
//!   for a 31 in square opening, plus the local-frame geometry a flythrough test
//!   needs.
//! * [`airframe`] — how big the F450 actually is, taken from its JSBSim model, and
//!   the five-part hull the crash test uses. There are **10.9 cm of clearance per
//!   side** square to a gate and **4.1 cm** at the worst heading, which is what
//!   decides how accurate that hull has to be.
//! * [`course`] — rejection sampling with backtracking that turns a seed into a
//!   sequence of gates no leg threads and no two gates overlap.
//! * [`scenario`] — the TOML document (`config/race.toml`): world origin, sim rate,
//!   the drone, and the randomization envelope.
//! * [`control`] — the F450's stabilising inner loop, and the
//!   [`FlightCommand`](control::FlightCommand) an outer controller writes. The drone
//!   launches holding station; **nothing here flies the course** — that is the next
//!   piece of work, and it plugs in by writing `FlightCommand`.
//! * [`systems`] — spawn, step, pose, gate scoring, crash detection, pacing, and
//!   termination. A run ends in exactly one way — some system marks an
//!   [`Outcome`](components::Outcome) on
//!   [`RaceProgress`](components::RaceProgress) — and one system turns that into an
//!   exit.
//! * [`net`] — the optional Zenoh bridge: the drone's telemetry and the race's task
//!   state out, actuator commands in, so a controller can be developed
//!   off-process in any language. Off by default; [`net::install`] switches it on.
//!
//! Course generation happens in [`build_app`], *before* the app is standing, so an
//! unsatisfiable envelope is an error the caller can see rather than a log line
//! from inside a startup system.
//!
//! ```no_run
//! let scenario = astrum_race::scenario::RaceScenario::load("config/race.toml")?;
//! let mut app = astrum_race::build_app(scenario)?;
//! app.run();
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod airframe;
pub mod components;
pub mod control;
pub mod course;
pub mod gate;
pub mod geom;
pub mod net;
pub mod rng;
pub mod scenario;
pub mod systems;

use bevy::app::ScheduleRunnerPlugin;
use bevy::prelude::*;
use std::time::Duration;

use astrum_geo::{Geodetic, LocalFrame};
use components::{RaceCourse, RaceProgress, SimParams, WorldFrame};
use control::stabilize;
use course::{Course, CourseError};
use net::{ingest_commands, publish_race, publish_telemetry};
use scenario::RaceScenario;
use systems::{
    detect_crash, enforce_time_limit, pace, report_progress, spawn_course, spawn_drone, step_fdm,
    stop_when_finished, sync_pose, track_gates, Diag, Pacer,
};

/// Draw the scenario's course: `sim.seed` against the `[course]` envelope.
pub fn generate_course(scenario: &RaceScenario) -> Result<Course, CourseError> {
    course::generate(scenario.sim.seed, &scenario.course)
}

/// Build the racing app from a scenario, randomizing the course from its seed.
pub fn build_app(scenario: RaceScenario) -> Result<App, CourseError> {
    let course = generate_course(&scenario)?;
    Ok(build_app_with_course(scenario, course))
}

/// Build the racing app on a course you already have (does not run it).
///
/// Shared by the binary and the integration tests, which drive it with manual
/// `update()` calls through the same [`Startup`] + [`Update`] schedule. Taking the
/// course as an argument lets a test pin an exact track instead of a seed.
pub fn build_app_with_course(scenario: RaceScenario, course: Course) -> App {
    let rate = scenario.sim.rate_hz;
    let dt = 1.0 / rate;
    let params = SimParams {
        rate_hz: rate,
        dt,
        max_time_s: scenario.sim.max_time_s,
        realtime: scenario.sim.realtime,
    };

    let origin = Geodetic::from_degrees(
        scenario.world.origin_lat_deg,
        scenario.world.origin_lon_deg,
        scenario.world.origin_alt_m,
    );
    let frame = WorldFrame(LocalFrame::new(origin));

    println!(
        "[astrum-race] ENU origin: {:.6}, {:.6} @ {:.1} m",
        scenario.world.origin_lat_deg, scenario.world.origin_lon_deg, scenario.world.origin_alt_m
    );

    // Physics steps exactly once per runner loop; `pace` throttles the loop (Bevy's
    // own sleep is a no-op in this headless configuration). Pacing changes
    // wall-clock timing only, never the number of steps or their dt.
    let mut app = App::new();
    app.add_plugins(MinimalPlugins.set(ScheduleRunnerPlugin::run_loop(Duration::ZERO)))
        .insert_resource(scenario)
        .insert_resource(params)
        .insert_resource(frame)
        .insert_resource(RaceCourse(course))
        .insert_resource(RaceProgress::default())
        .insert_resource(Diag::new())
        .insert_resource(Pacer::new(dt, params.realtime))
        // The course must exist before the drone, which starts on gate 0's approach.
        .add_systems(Startup, (spawn_course, spawn_drone).chain())
        .add_systems(
            Update,
            (
                // Commands off the wire arrive before the loop that services them,
                // the inner loop sets this tick's control inputs, then physics, then
                // the pose it produced.
                ingest_commands,
                stabilize,
                step_fdm,
                sync_pose,
                // Everything that can end the run, judged against that fresh pose, and
                // all of it ahead of publishing so the wire always carries the outcome
                // that ended the run. Crash detection is first: clipping a gate while
                // the CG crosses inside the opening is a crash, not a pass.
                detect_crash,
                track_gates,
                enforce_time_limit,
                publish_telemetry,
                publish_race,
                report_progress,
                // ...and one place the app actually exits.
                stop_when_finished,
                pace,
            )
                .chain(),
        );
    app
}
