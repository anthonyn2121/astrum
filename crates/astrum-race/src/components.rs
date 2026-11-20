//! Resources the race systems share.

use astrum_proto::RaceOutcome;
use bevy::prelude::*;

use crate::course::Course;

/// The course being flown, fixed before the app is built. Every drone races it.
#[derive(Resource, Debug, Clone)]
pub struct RaceCourse(pub Course);

/// How far round the course one drone has got, and how its race ended.
///
/// A gate counts when the drone's CG crosses its plane **inside the opening**,
/// downstream, and gates are taken **in order**. The outcome is first-writer-wins
/// ([`RaceProgress::finish`]), which is what decides terminators that fire on the same
/// tick: crash detection runs before gate scoring, so clipping an upright while the CG
/// crosses inside the opening is a crash, not a pass.
#[derive(Debug, Clone, Copy, PartialEq)]
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
    /// Why this drone's race ended; [`RaceOutcome::Running`] until it does. This is
    /// the value published on the wire.
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

    /// This drone's race is over (for whatever reason `outcome` names).
    pub fn finished(&self) -> bool {
        self.outcome != RaceOutcome::Running
    }

    /// The drone crashed.
    pub fn crashed(&self) -> bool {
        matches!(self.outcome, RaceOutcome::CrashedGround | RaceOutcome::CrashedGate)
    }

    /// End this drone's race with `outcome`; reports whether this call is what ended it.
    pub fn finish(&mut self, outcome: RaceOutcome) -> bool {
        if self.finished() {
            return false;
        }
        self.outcome = outcome;
        true
    }
}

/// One drone's standing in the race: [`Race::results`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DroneResult {
    pub aircraft_id: u32,
    /// 1-based rank (see [`Race::results`]).
    pub place: u32,
    pub passed: u32,
    pub missed: u32,
    pub outcome: RaceOutcome,
    /// Sim time of the last gate passed, when the drone completed the course.
    pub finish_time_s: Option<f64>,
}

/// Every racing drone's progress, ascending by aircraft id.
#[derive(Resource, Debug, Clone)]
pub struct Race {
    drones: Vec<RaceProgress>,
}

impl Race {
    /// A race for these aircraft (any order; stored by ascending id).
    pub fn new(ids: impl IntoIterator<Item = u32>) -> Self {
        let mut drones: Vec<RaceProgress> = ids.into_iter().map(RaceProgress::new).collect();
        drones.sort_by_key(|d| d.aircraft_id);
        Self { drones }
    }

    pub fn drones(&self) -> &[RaceProgress] {
        &self.drones
    }

    pub fn drones_mut(&mut self) -> &mut [RaceProgress] {
        &mut self.drones
    }

    pub fn drone(&self, aircraft_id: u32) -> Option<&RaceProgress> {
        self.drones.iter().find(|d| d.aircraft_id == aircraft_id)
    }

    pub fn drone_mut(&mut self, aircraft_id: u32) -> Option<&mut RaceProgress> {
        self.drones.iter_mut().find(|d| d.aircraft_id == aircraft_id)
    }

    /// Every drone's race is over.
    pub fn all_finished(&self) -> bool {
        self.drones.iter().all(RaceProgress::finished)
    }

    /// Every drone, ranked. Drones that completed the course come first, by finish
    /// time (the time of their last gate) ascending; the rest follow by gates passed
    /// descending, then by the time of their last pass ascending. Ties go to the lower
    /// aircraft id.
    pub fn results(&self) -> Vec<DroneResult> {
        let mut ranked: Vec<&RaceProgress> = self.drones.iter().collect();
        ranked.sort_by(|a, b| {
            let (ca, cb) = (a.outcome == RaceOutcome::Complete, b.outcome == RaceOutcome::Complete);
            cb.cmp(&ca)
                .then_with(|| match ca {
                    true => std::cmp::Ordering::Equal,
                    false => b.passed.cmp(&a.passed),
                })
                .then(a.last_pass_s.total_cmp(&b.last_pass_s))
                .then(a.aircraft_id.cmp(&b.aircraft_id))
        });
        ranked
            .iter()
            .zip(1..)
            .map(|(d, place)| DroneResult {
                aircraft_id: d.aircraft_id,
                place,
                passed: d.passed,
                missed: d.missed,
                outcome: d.outcome,
                finish_time_s: (d.outcome == RaceOutcome::Complete).then_some(d.last_pass_s),
            })
            .collect()
    }

    /// The aircraft that won: first place, if it completed the course.
    pub fn winner(&self) -> Option<u32> {
        self.results().first().filter(|r| r.finish_time_s.is_some()).map(|r| r.aircraft_id)
    }
}
