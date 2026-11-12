//! Resources the race systems share.

use astrum_proto::RaceOutcome;
use bevy::prelude::*;

use crate::course::Course;

/// The course being flown, fixed before the app is built.
#[derive(Resource, Debug, Clone)]
pub struct RaceCourse(pub Course);

/// How far round the course the drone has got, and how the race ended.
///
/// A gate counts when the drone's CG crosses its plane **inside the opening**,
/// downstream, and gates are taken **in order**. The outcome is first-writer-wins
/// ([`RaceProgress::finish`]), which is what decides terminators that fire on the
/// same tick: crash detection runs before gate scoring, so clipping an upright
/// while the CG crosses inside the opening is a crash, not a pass.
#[derive(Resource, Debug, Clone, Copy)]
pub struct RaceProgress {
    /// The aircraft that is racing.
    pub aircraft_id: u32,
    /// Index of the gate the drone is trying to fly through next; the gate count
    /// once every gate is passed.
    pub next_gate: u32,
    /// Gates passed cleanly, in order.
    pub passed: u32,
    /// Plane crossings that missed the opening (flew round the gate).
    pub missed: u32,
    /// Simulated time the last gate was passed (s); `0` before the first.
    pub last_pass_s: f64,
    /// Why the race ended; [`RaceOutcome::Running`] until it does. This is the value
    /// published on the wire.
    pub outcome: RaceOutcome,
}

impl RaceProgress {
    pub fn new(aircraft_id: u32) -> Self {
        Self {
            aircraft_id,
            next_gate: 0,
            passed: 0,
            missed: 0,
            last_pass_s: 0.0,
            outcome: RaceOutcome::Running,
        }
    }

    /// The race is over (for whatever reason `outcome` names).
    pub fn finished(&self) -> bool {
        self.outcome != RaceOutcome::Running
    }

    /// End the race with `outcome`; reports whether this call is what ended it.
    pub fn finish(&mut self, outcome: RaceOutcome) -> bool {
        if self.finished() {
            return false;
        }
        self.outcome = outcome;
        true
    }
}
