//! Flight tracks: each aircraft's grid (ENU) position over sim time, thinned to
//! a fixed number of points, so a sweep overview can draw every run at once.
//! The whole recording is far too large for that; a track keeps only the shape.
//!
//! Samples are the bag's telemetry, read as [`crate::metrics`] reads them: a
//! payload that fails to decode, or whose aircraft id disagrees with its topic,
//! is skipped. An aircraft's samples are ordered by sim time, and of two sharing
//! a time the later one in the bag wins. An aircraft with more samples than
//! `max_points` keeps `max_points` of them, the ones nearest to evenly spaced sim
//! times from its first sample to its last. Numbers are rounded for the JSON:
//! times to milliseconds, positions to decimetres.

use std::collections::BTreeMap;

use astrum_proto::{decode_telemetry, Telemetry};
use serde::Serialize;

use crate::{aircraft_models, BagReader, Result, TopicKind};

/// The flight tracks of one recorded run.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TrackSet {
    /// Sim time of the last telemetry sample in the bag (s); 0 without telemetry.
    pub sim_time_s: f64,
    /// One per aircraft with telemetry, ascending id.
    pub aircraft: Vec<AircraftTrack>,
}

/// One aircraft's thinned track.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AircraftTrack {
    pub id: u32,
    /// From the bag's recorded scenario; None when unknown.
    pub model: Option<String>,
    /// Parallel arrays, time-ordered, equal length: sim time (s) and the grid (ENU) position (m).
    pub t: Vec<f64>,
    pub e: Vec<f64>,
    pub n: Vec<f64>,
    pub u: Vec<f64>,
    /// Sim time at which the aircraft was last seen alive; None if it was alive in its last sample.
    pub died_at_s: Option<f64>,
}

/// The parts of one telemetry sample a track keeps.
struct Sample {
    t_s: f64,
    alive: bool,
    enu: [f64; 3],
}

impl From<&Telemetry> for Sample {
    fn from(t: &Telemetry) -> Self {
        Self {
            t_s: t.sim_time_s,
            alive: t.combat.alive,
            enu: [t.pos_enu.e, t.pos_enu.n, t.pos_enu.u],
        }
    }
}

/// The flight tracks of the bag's aircraft, each thinned to at most `max_points`
/// samples (`max_points` below 2 is taken as 2: the first and the last sample).
/// Errors only if the bag's messages cannot be read.
pub fn tracks(bag: &BagReader, max_points: usize) -> Result<TrackSet> {
    let max_points = max_points.max(2);
    let mut samples: BTreeMap<u32, Vec<Sample>> = BTreeMap::new();
    for msg in bag.messages()? {
        let Some(TopicKind::Telemetry(topic_id)) = TopicKind::of(&msg.topic) else { continue };
        let Ok(t) = decode_telemetry(&msg.payload) else { continue };
        if t.aircraft_id == topic_id {
            samples.entry(t.aircraft_id).or_default().push(Sample::from(&t));
        }
    }

    let models: BTreeMap<u32, String> = bag
        .scenario()
        .map(aircraft_models)
        .unwrap_or_default()
        .into_iter()
        .map(|a| (a.id, a.model))
        .collect();

    let sim_time_s = samples.values().flat_map(|s| s.iter().map(|s| s.t_s)).fold(0.0, f64::max);
    let aircraft = samples
        .into_iter()
        .map(|(id, samples)| {
            let model = models.get(&id).cloned();
            track_of(id, model, samples, max_points)
        })
        .collect();
    Ok(TrackSet { sim_time_s: round(sim_time_s, 3), aircraft })
}

/// One aircraft's track from its unordered samples (at least one).
fn track_of(
    id: u32,
    model: Option<String>,
    mut samples: Vec<Sample>,
    max_points: usize,
) -> AircraftTrack {
    // A stable sort keeps bag order between equal times, so the dedup below keeps the later one.
    samples.sort_by(|a, b| a.t_s.total_cmp(&b.t_s));
    let mut ordered: Vec<Sample> = Vec::with_capacity(samples.len());
    for s in samples {
        match ordered.last_mut() {
            Some(last) if last.t_s == s.t_s => *last = s,
            _ => ordered.push(s),
        }
    }

    let died_at_s = ordered.iter().find(|s| !s.alive).map(|s| round(s.t_s, 3));
    let kept = thin(&ordered, max_points);
    AircraftTrack {
        id,
        model,
        t: kept.iter().map(|s| round(s.t_s, 3)).collect(),
        e: kept.iter().map(|s| round(s.enu[0], 1)).collect(),
        n: kept.iter().map(|s| round(s.enu[1], 1)).collect(),
        u: kept.iter().map(|s| round(s.enu[2], 1)).collect(),
        died_at_s,
    }
}

/// The samples of a time-ordered, non-empty track thinned to `max_points` (at
/// least 2): all of them when there are no more than that, else the sample
/// nearest each of `max_points` evenly spaced sim times from the first to the
/// last. A sample picked twice is kept once.
fn thin(samples: &[Sample], max_points: usize) -> Vec<&Sample> {
    if samples.len() <= max_points {
        return samples.iter().collect();
    }
    let first = samples[0].t_s;
    let last = samples[samples.len() - 1].t_s;
    let mut kept: Vec<&Sample> = Vec::with_capacity(max_points);
    let mut previous = None;
    for i in 0..max_points {
        let at = first + i as f64 * (last - first) / (max_points - 1) as f64;
        let index = nearest(samples, at);
        if previous != Some(index) {
            kept.push(&samples[index]);
            previous = Some(index);
        }
    }
    kept
}

/// Index of the sample whose time is nearest `at` (a tie picks the earlier one).
/// `samples` is non-empty and time-ordered; the lookup is a binary search.
fn nearest(samples: &[Sample], at: f64) -> usize {
    let after = samples.partition_point(|s| s.t_s < at);
    if after == 0 {
        return 0;
    }
    if after == samples.len() {
        return samples.len() - 1;
    }
    let before = after - 1;
    if at - samples[before].t_s <= samples[after].t_s - at {
        before
    } else {
        after
    }
}

/// `x` rounded to `decimals` decimal places.
fn round(x: f64, decimals: i32) -> f64 {
    let scale = 10f64.powi(decimals);
    (x * scale).round() / scale
}
