//! `astrum-race` — the ASTRUM drone-racing environment, as a client of `astrum-sim`.
//!
//! A race is an ordinary `astrum-sim` scenario (one DJI F450, flown by JSBSim)
//! plus a `[course]` table: a randomized course of racing gates, or explicit gates.
//! [`build_race_app`] builds the sim's own app with [`astrum_sim::build_app`] — fleet
//! spawn, FDM stepping, command ownership and leases, lockstep, telemetry, events,
//! recording, pacing, the control plane — and adds the race systems to its tick
//! through [`astrum_sim::TickSet`]. A controller flies a race exactly as it flies
//! any multirotor: it sends `Multirotor` commands for the aircraft and reads the
//! race's course and progress from `astrum/race/*`.
//!
//! * [`gate`] — the physical gate and the local-frame geometry a flythrough test needs.
//! * [`airframe`] — how big the F450 actually is, and the five-part hull the crash
//!   test uses.
//! * [`course`] — rejection sampling with backtracking that turns a seed into a
//!   sequence of gates no leg threads and no two gates overlap.
//! * [`scenario`] — the race scenario document, its resolution to a fixed course
//!   ([`scenario::ResolvedRace`]) and the JSON course view the web API serves.
//! * [`systems`] — crash detection, gate scoring, the time limit, the end of the run
//!   and the publication of `RaceCourse`/`RaceState`.
//! * [`net`] — puts the race's publisher on a Zenoh session.
//!
//! ```no_run
//! let race = astrum_race::scenario::RaceScenario::load("config/scenario.race.toml")?.resolve(None)?;
//! let mut app = astrum_race::build_race_app(race, astrum_sim::RunMode::Server);
//! astrum_sim::run_to_exit(&mut app);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod airframe;
pub mod components;
pub mod course;
pub mod gate;
pub mod geom;
pub mod net;
pub mod rng;
pub mod scenario;
pub mod systems;

use astrum_sim::components::SimStatus;
use astrum_sim::{RunMode, TickSet};
use bevy::prelude::*;

use components::{RaceCourse, RaceProgress};
use scenario::ResolvedRace;
use systems::{fail_spawn, log_course, publish_race, tick_systems};

pub use scenario::{course_view, is_race_scenario, CourseView, RaceScenario, RaceScenarioError};

/// Build the sim's app for a resolved race, with the race systems added (does not
/// run it; put it on the bus with [`astrum_sim::net::connect`] and [`net::connect`]).
///
/// At startup, after the sim spawns the drone and before tick 0 is published, the
/// course and the initial state go out. Every tick, after the step and before the
/// telemetry: crash detection → gate scoring → the time limit → the end of the run →
/// the publication of the state.
pub fn build_race_app(race: ResolvedRace, mode: RunMode) -> App {
    let aircraft_id = race.scenario.aircraft[0].id;
    let mut app = astrum_sim::build_app(race.scenario, mode);
    app.insert_resource(RaceCourse(race.course))
        .insert_resource(RaceProgress::new(aircraft_id))
        .add_systems(
            Startup,
            (
                log_course,
                publish_race.run_if(SimStatus::running),
                fail_spawn.run_if(not(SimStatus::running)),
            )
                .chain()
                .in_set(TickSet::Spawned),
        )
        .add_systems(Update, tick_systems().in_set(TickSet::Stepped));
    app
}
