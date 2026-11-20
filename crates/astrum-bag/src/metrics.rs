//! Run metrics: one pass over a bag's telemetry and events, reduced to the
//! fleet-level numbers a run is judged by (and that sweeps tabulate) plus each
//! aircraft's extremes and final state.
//!
//! Fleet metrics are named by their metric ids (`end_reason`, `sim_time_s`,
//! `min_separation_m`, `min_alt_hae_m`, `max_bank_deg`, `max_nz_g`,
//! `failsafe_entries`, `controller_losses`, `command_refusals`, `hits`, `kills`,
//! and for race runs `gates_passed`, `gates_missed`, `race_time_s`, `race_outcome`,
//! `race_winner`). An extreme over no samples is `None`, a count over no events 0.
//!
//! **Race metrics** come from the `gate-passed`, `gate-missed` and `crashed`
//! events plus the `run-ended` event and the recorded scenario, and are `None`
//! for a run that is not a race (a bag whose recorded scenario has no `[course]`
//! table and that holds no race event). A race may have several aircraft on the
//! same gates; each is judged on its own ([`RaceResult`], `race`, ordered by
//! aircraft id; every `[[aircraft]]` of the scenario and every aircraft with a
//! race event has one):
//!
//! * `gates_passed` / `gates_missed` count its `gate-passed` / `gate-missed` events;
//! * `race_time_s` is the sim time of its last gate passed (its finish time when it
//!   completed the course), `None` if it passed none;
//! * `race_outcome` is its `crashed` event's outcome (`crashed-ground`, `crashed-gate`,
//!   `timed-out`, `spawn-failed`), else `complete` if it passed every gate of the course
//!   (the scenario's explicit `[[course.gate]]`, else its `course.gates`; for a lone
//!   aircraft a `race-complete` run end also counts), else `timed-out` if the run ended at
//!   `max-time`; `None` while it has not finished;
//! * `place` is the 1-based rank: aircraft that completed the course first, by
//!   `race_time_s` ascending; then the rest by `gates_passed` descending, then by
//!   `race_time_s` ascending (an aircraft that passed none ranks last); remaining ties
//!   go to the lower aircraft id, so places are distinct.
//!
//! `race_winner` is the aircraft id of place 1 if it completed the course, else `None`.
//! The fleet-level `gates_passed` / `gates_missed` are totals over all aircraft,
//! `race_time_s` the sim time of the last gate passed by any aircraft, and
//! `race_outcome` the outcome of the aircraft in place 1; for a one-aircraft race
//! all four are simply that aircraft's.
//!
//! **Samples.** Only samples of living aircraft (`combat.alive`) feed the
//! extremes and the separation; the final state is the aircraft's last sample
//! (highest tick), dead or alive. A bag holds every tick when the sim recorded
//! it in-process (`astrum run --record`), or only what a live subscriber
//! received (`astrum record`): each aircraft's control-rate samples, possibly
//! missing some a latest-value transport dropped.
//!
//! **Separation** is the 3-D ENU distance between two living aircraft sampled
//! *at the same tick*: positions are never interpolated, so aircraft whose
//! control rates differ are compared on the ticks they share (every tick of
//! the slower one when the rates divide each other), and a tick one of them is
//! missing from the bag is skipped for that pair. The minimum can therefore
//! miss a closest approach between shared ticks by up to the distance the pair
//! closes in one shared sample period.
//!
//! **One run per bag.** Metrics assume the bag holds a single run: a bag
//! recorded across a sim restart is not split, so its metrics mix both runs
//! (ticks restart, samples of the same tick number from either run are paired,
//! and the final state is whichever sample has the highest tick).
//!
//! The model labels come from the recorded scenario via [`aircraft_models`]; a
//! missing or unparseable scenario only leaves them `None`.

use std::collections::BTreeMap;

use astrum_proto::{
    decode_event, decode_telemetry, ControlMode, EndReason, EventKind, RaceOutcome, Telemetry,
};
use serde::Serialize;

use crate::{aircraft_models, BagReader, Result, TopicKind};

/// Fleet-level metrics of one recorded run (a bag spanning a sim restart mixes
/// the runs; see the [module docs](self)).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RunMetrics {
    /// The `run-ended` event's reason (`max-time`, `engagement-decided`,
    /// `fatal`, `stopped`, `race-complete`, `crashed`); `None` if the bag has no `run-ended` event.
    pub end_reason: Option<String>,
    /// The `run-ended` event's message (the fatal error; empty otherwise).
    pub end_message: Option<String>,
    /// Sim time of the last telemetry sample (s); 0 without telemetry.
    pub sim_time_s: f64,
    /// Smallest same-tick distance between two living aircraft (m).
    pub min_separation_m: Option<f64>,
    /// The two aircraft ids at [`Self::min_separation_m`], ascending.
    pub min_separation_pair: Option<[u32; 2]>,
    /// Sim time of [`Self::min_separation_m`] (s).
    pub min_separation_t_s: Option<f64>,
    /// Lowest height above the ellipsoid of any living aircraft (m).
    pub min_alt_hae_m: Option<f64>,
    /// Largest |bank angle| of any living aircraft (deg).
    pub max_bank_deg: Option<f64>,
    /// Largest normal load factor of any living aircraft (g).
    pub max_nz_g: Option<f64>,
    /// `mode-change` events into failsafe, fleet-wide.
    pub failsafe_entries: u32,
    /// `controller-down` events.
    pub controller_losses: u32,
    /// Commands the sim refused, summed over aircraft: each aircraft's running
    /// total since the sim started, from its last telemetry sample (so a live
    /// `astrum record` joined mid-run includes refusals from before it joined).
    pub command_refusals: u64,
    /// `hit` events.
    pub hits: u32,
    /// `kill` events.
    pub kills: u32,
    /// `gate-passed` events; `None` for a run that is not a race.
    pub gates_passed: Option<u32>,
    /// `gate-missed` events; `None` for a run that is not a race.
    pub gates_missed: Option<u32>,
    /// Sim time of the last `gate-passed` event of any aircraft (s).
    pub race_time_s: Option<f64>,
    /// How the race ended for the aircraft in place 1 (see the [module docs](self)).
    pub race_outcome: Option<String>,
    /// Each race aircraft's result, by aircraft id; `None` for a run that is not a race.
    pub race: Option<Vec<RaceResult>>,
    /// The aircraft that won the race: place 1, if it completed the course.
    pub race_winner: Option<u32>,
    /// Telemetry and event payloads that did not decode (or whose aircraft id
    /// disagrees with their topic), skipped.
    pub rejected: u64,
    /// Every aircraft with telemetry in the bag, by id.
    pub aircraft: Vec<AircraftMetrics>,
}

/// One race aircraft's result (see the [module docs](self)).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RaceResult {
    pub aircraft: u32,
    pub gates_passed: u32,
    pub gates_missed: u32,
    /// Sim time of its last gate passed (s): its finish time when it completed the course.
    pub race_time_s: Option<f64>,
    pub race_outcome: Option<String>,
    /// 1-based rank among the race's aircraft.
    pub place: u32,
}

/// One aircraft's metrics.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AircraftMetrics {
    pub id: u32,
    /// JSBSim model from the bag's recorded scenario ([`aircraft_models`]);
    /// `None` without a parseable scenario entry for this id.
    pub model: Option<String>,
    /// From the last sample.
    pub alive: bool,
    /// From the last sample.
    pub hit_points: f32,
    /// Over the living samples, as the fleet fields.
    pub min_alt_hae_m: Option<f64>,
    pub max_bank_deg: Option<f64>,
    pub max_nz_g: Option<f64>,
    /// `mode-change` events into failsafe for this aircraft.
    pub failsafe_entries: u32,
    pub final_state: FinalState,
}

/// An aircraft's last recorded state.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FinalState {
    pub sim_time_s: f64,
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub alt_hae_m: f64,
    /// True heading in [0, 360) (deg).
    pub heading_deg: f64,
    pub tas_mps: f64,
    /// World-grid position `[e, n, u]` (m).
    pub enu_m: [f64; 3],
}

#[derive(Default)]
struct Extremes {
    min_alt_hae_m: Option<f64>,
    max_bank_deg: Option<f64>,
    max_nz_g: Option<f64>,
}

impl Extremes {
    fn add(&mut self, t: &Telemetry) {
        self.min_alt_hae_m = Some(self.min_alt_hae_m.map_or(t.alt_hae_m, |m| m.min(t.alt_hae_m)));
        let bank = t.attitude.phi_rad.to_degrees().abs();
        self.max_bank_deg = Some(self.max_bank_deg.map_or(bank, |m| m.max(bank)));
        self.max_nz_g = Some(self.max_nz_g.map_or(t.aero.nz_g, |m| m.max(t.aero.nz_g)));
    }

    fn merge(&mut self, o: &Extremes) {
        let pick = |a: Option<f64>, b: Option<f64>, f: fn(f64, f64) -> f64| match (a, b) {
            (Some(a), Some(b)) => Some(f(a, b)),
            (a, b) => a.or(b),
        };
        self.min_alt_hae_m = pick(self.min_alt_hae_m, o.min_alt_hae_m, f64::min);
        self.max_bank_deg = pick(self.max_bank_deg, o.max_bank_deg, f64::max);
        self.max_nz_g = pick(self.max_nz_g, o.max_nz_g, f64::max);
    }
}

#[derive(Default)]
struct Track {
    last: Option<Telemetry>,
    extremes: Extremes,
    failsafe_entries: u32,
}

struct Closest {
    distance_m: f64,
    pair: [u32; 2],
    t_s: f64,
}

/// Living aircraft positions sampled at one tick.
struct TickSamples {
    t_s: f64,
    /// `(id, [e, n, u])`.
    living: Vec<(u32, [f64; 3])>,
}

/// Compute the run's metrics from `bag`.
///
/// Errors only if the bag's messages cannot be read.
pub fn compute(bag: &BagReader) -> Result<RunMetrics> {
    let models: BTreeMap<u32, String> = bag
        .scenario()
        .map(aircraft_models)
        .unwrap_or_default()
        .into_iter()
        .map(|a| (a.id, a.model))
        .collect();

    let mut tracks: BTreeMap<u32, Track> = BTreeMap::new();
    let mut ticks: BTreeMap<u64, TickSamples> = BTreeMap::new();
    let mut m = RunMetrics {
        end_reason: None,
        end_message: None,
        sim_time_s: 0.0,
        min_separation_m: None,
        min_separation_pair: None,
        min_separation_t_s: None,
        min_alt_hae_m: None,
        max_bank_deg: None,
        max_nz_g: None,
        failsafe_entries: 0,
        controller_losses: 0,
        command_refusals: 0,
        hits: 0,
        kills: 0,
        gates_passed: None,
        gates_missed: None,
        race_time_s: None,
        race_outcome: None,
        race: None,
        race_winner: None,
        rejected: 0,
        aircraft: Vec::new(),
    };

    // Race metrics are reported for a race scenario even when no gate event happened.
    let mut race = bag.scenario().is_some_and(has_course_table);
    let mut racers: BTreeMap<u32, Racer> = BTreeMap::new();
    let mut ended: Option<EndReason> = None;

    for msg in bag.messages()? {
        match TopicKind::of(&msg.topic) {
            Some(TopicKind::Telemetry(topic_id)) => match decode_telemetry(&msg.payload) {
                Ok(t) if t.aircraft_id == topic_id => {
                    let track = tracks.entry(t.aircraft_id).or_default();
                    if t.combat.alive {
                        track.extremes.add(&t);
                        let p = t.pos_enu;
                        let living = &mut ticks
                            .entry(t.tick)
                            .or_insert(TickSamples { t_s: t.sim_time_s, living: Vec::new() })
                            .living;
                        // A repeated (id, tick) sample replaces the earlier one.
                        living.retain(|(id, _)| *id != t.aircraft_id);
                        living.push((t.aircraft_id, [p.e, p.n, p.u]));
                    }
                    // Later in the bag wins a tie: the newest sample of the tick.
                    if track.last.is_none_or(|l| t.tick >= l.tick) {
                        track.last = Some(t);
                    }
                }
                _ => m.rejected += 1,
            },
            Some(TopicKind::Events) => match decode_event(&msg.payload) {
                Ok(e) => match e.kind {
                    EventKind::ModeChange { aircraft_id, from, to, .. }
                        if to == ControlMode::Failsafe && from != ControlMode::Failsafe =>
                    {
                        m.failsafe_entries += 1;
                        tracks.entry(aircraft_id).or_default().failsafe_entries += 1;
                    }
                    EventKind::ControllerDown { .. } => m.controller_losses += 1,
                    EventKind::Hit { .. } => m.hits += 1,
                    EventKind::Kill { .. } => m.kills += 1,
                    EventKind::RunEnded { reason, message } => {
                        m.end_reason = Some(reason.name().to_owned());
                        m.end_message = Some(message);
                        ended = Some(reason);
                    }
                    EventKind::GatePassed { aircraft_id, .. } => {
                        race = true;
                        let r = racers.entry(aircraft_id).or_default();
                        r.passed += 1;
                        r.last_pass_s = Some(e.sim_time_s);
                        let t = e.sim_time_s;
                        m.race_time_s = Some(m.race_time_s.map_or(t, |last| last.max(t)));
                    }
                    EventKind::GateMissed { aircraft_id, .. } => {
                        race = true;
                        racers.entry(aircraft_id).or_default().missed += 1;
                    }
                    EventKind::Crashed { aircraft_id, outcome, .. } => {
                        race = true;
                        racers.entry(aircraft_id).or_default().crash = Some(outcome);
                    }
                    _ => {}
                },
                Err(_) => m.rejected += 1,
            },
            Some(TopicKind::Command(_)) | None => {}
        }
    }

    if race {
        let scenario = bag.scenario();
        for a in scenario.map(aircraft_models).unwrap_or_default() {
            racers.entry(a.id).or_default();
        }
        let results = rank(&racers, scenario.and_then(gate_count), ended);
        m.gates_passed = Some(results.iter().map(|r| r.gates_passed).sum());
        m.gates_missed = Some(results.iter().map(|r| r.gates_missed).sum());
        if let Some(first) = results.iter().find(|r| r.place == 1) {
            m.race_outcome = first.race_outcome.clone();
            m.race_winner = (first.race_outcome.as_deref() == Some(RaceOutcome::Complete.name()))
                .then_some(first.aircraft);
        }
        m.race = Some(results);
    }

    let mut closest: Option<Closest> = None;
    for TickSamples { t_s, living } in ticks.values() {
        for (i, (a, pa)) in living.iter().enumerate() {
            for (b, pb) in &living[i + 1..] {
                let d =
                    ((pa[0] - pb[0]).powi(2) + (pa[1] - pb[1]).powi(2) + (pa[2] - pb[2]).powi(2))
                        .sqrt();
                if closest.as_ref().is_none_or(|c| d < c.distance_m) {
                    closest =
                        Some(Closest { distance_m: d, pair: [*a.min(b), *a.max(b)], t_s: *t_s });
                }
            }
        }
    }
    if let Some(c) = closest {
        m.min_separation_m = Some(c.distance_m);
        m.min_separation_pair = Some(c.pair);
        m.min_separation_t_s = Some(c.t_s);
    }

    let mut fleet = Extremes::default();
    for (id, track) in tracks {
        // An aircraft known only from events has no state to report.
        let Some(last) = track.last else { continue };
        fleet.merge(&track.extremes);
        m.sim_time_s = m.sim_time_s.max(last.sim_time_s);
        m.command_refusals += u64::from(last.command.refusals);
        m.aircraft.push(AircraftMetrics {
            id,
            model: models.get(&id).cloned(),
            alive: last.combat.alive,
            hit_points: last.combat.hit_points,
            min_alt_hae_m: track.extremes.min_alt_hae_m,
            max_bank_deg: track.extremes.max_bank_deg,
            max_nz_g: track.extremes.max_nz_g,
            failsafe_entries: track.failsafe_entries,
            final_state: FinalState {
                sim_time_s: last.sim_time_s,
                lat_deg: last.lat_deg,
                lon_deg: last.lon_deg,
                alt_hae_m: last.alt_hae_m,
                heading_deg: heading_deg(last.attitude.psi_rad),
                tas_mps: last.tas_mps,
                enu_m: [last.pos_enu.e, last.pos_enu.n, last.pos_enu.u],
            },
        });
    }
    m.min_alt_hae_m = fleet.min_alt_hae_m;
    m.max_bank_deg = fleet.max_bank_deg;
    m.max_nz_g = fleet.max_nz_g;
    Ok(m)
}

/// One race aircraft's tally from the events.
#[derive(Default)]
struct Racer {
    passed: u32,
    missed: u32,
    last_pass_s: Option<f64>,
    crash: Option<RaceOutcome>,
}

/// The gates of the recorded course: its explicit `[[course.gate]]`, else `course.gates`.
fn gate_count(scenario_toml: &str) -> Option<u32> {
    let table = scenario_toml.parse::<toml::Table>().ok()?;
    let course = table.get("course")?.as_table()?;
    let explicit = course.get("gate").and_then(toml::Value::as_array).map_or(0, Vec::len);
    let n = if explicit > 0 {
        explicit as i64
    } else {
        course.get("gates").and_then(toml::Value::as_integer)?
    };
    u32::try_from(n).ok().filter(|n| *n > 0)
}

/// Every aircraft's result with its place (see the [module docs](self)), by aircraft id.
fn rank(
    racers: &BTreeMap<u32, Racer>,
    gates: Option<u32>,
    ended: Option<EndReason>,
) -> Vec<RaceResult> {
    let lone = racers.len() == 1;
    let outcome = |r: &Racer| {
        r.crash.or_else(|| {
            let done = gates.is_some_and(|n| r.passed >= n)
                || (lone && ended == Some(EndReason::RaceComplete));
            if done {
                Some(RaceOutcome::Complete)
            } else if ended == Some(EndReason::MaxTime) {
                Some(RaceOutcome::TimedOut)
            } else {
                None
            }
        })
    };
    let mut results: Vec<(RaceResult, bool)> = racers
        .iter()
        .map(|(id, r)| {
            let outcome = outcome(r);
            let result = RaceResult {
                aircraft: *id,
                gates_passed: r.passed,
                gates_missed: r.missed,
                race_time_s: r.last_pass_s,
                race_outcome: outcome.map(|o| o.name().to_owned()),
                place: 0,
            };
            (result, outcome == Some(RaceOutcome::Complete))
        })
        .collect();
    // Completed first, then more gates, then the earlier last pass (none: last), then id.
    let mut order: Vec<usize> = (0..results.len()).collect();
    order.sort_by(|&a, &b| {
        let ((ra, done_a), (rb, done_b)) = (&results[a], &results[b]);
        let time = |r: &RaceResult| r.race_time_s.unwrap_or(f64::INFINITY);
        done_b
            .cmp(done_a)
            .then(rb.gates_passed.cmp(&ra.gates_passed).then(time(ra).total_cmp(&time(rb))))
            .then(ra.aircraft.cmp(&rb.aircraft))
    });
    for (place, i) in order.into_iter().enumerate() {
        results[i].0.place = place as u32 + 1;
    }
    results.into_iter().map(|(r, _)| r).collect()
}

/// Whether a scenario TOML has a `[course]` table (a race scenario).
fn has_course_table(scenario_toml: &str) -> bool {
    scenario_toml
        .parse::<toml::Table>()
        .is_ok_and(|t| t.get("course").is_some_and(toml::Value::is_table))
}

/// `psi` as a heading in [0, 360) deg (`rem_euclid` rounds a tiny negative up to 360).
fn heading_deg(psi_rad: f64) -> f64 {
    let h = psi_rad.to_degrees().rem_euclid(360.0);
    if h >= 360.0 {
        0.0
    } else {
        h
    }
}
