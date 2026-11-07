//! Run metrics: one pass over a bag's telemetry and events, reduced to the
//! fleet-level numbers a run is judged by (and that sweeps tabulate) plus each
//! aircraft's extremes and final state.
//!
//! Fleet metrics are named by their metric ids (`end_reason`, `sim_time_s`,
//! `min_separation_m`, `min_alt_hae_m`, `max_bank_deg`, `max_nz_g`,
//! `failsafe_entries`, `controller_losses`, `command_refusals`, `hits`, `kills`).
//! An extreme over no samples is `None`, a count over no events 0.
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

use std::collections::BTreeMap;

use astrum_proto::{decode_event, decode_telemetry, ControlMode, EventKind, Telemetry};
use serde::{Deserialize, Serialize};

use crate::{BagError, BagReader, Result, TopicKind};

/// Fleet-level metrics of one recorded run.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RunMetrics {
    /// The `run-ended` event's reason (`max-time`, `engagement-decided`,
    /// `fatal`, `stopped`); `None` if the bag has no `run-ended` event.
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
    /// Commands the sim refused, summed over aircraft (each aircraft's count
    /// from its last telemetry sample).
    pub command_refusals: u64,
    /// `hit` events.
    pub hits: u32,
    /// `kill` events.
    pub kills: u32,
    /// Telemetry and event payloads that did not decode (or whose aircraft id
    /// disagrees with their topic), skipped.
    pub rejected: u64,
    /// Every aircraft with telemetry in the bag, by id.
    pub aircraft: Vec<AircraftMetrics>,
}

/// One aircraft's metrics.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AircraftMetrics {
    pub id: u32,
    /// JSBSim model from the bag's recorded scenario; `None` without one.
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

/// The fleet as the recorded scenario declares it: only what metrics need.
#[derive(Deserialize)]
struct ScenarioFleet {
    #[serde(default)]
    aircraft: Vec<ScenarioAircraft>,
}

#[derive(Deserialize)]
struct ScenarioAircraft {
    id: u32,
    model: String,
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
/// Errors if the bag's messages cannot be read or its recorded scenario is not
/// a valid scenario TOML (`[[aircraft]]` entries need `id` and `model`).
pub fn compute(bag: &BagReader) -> Result<RunMetrics> {
    let models: BTreeMap<u32, String> = match bag.scenario() {
        Some(toml) => toml::from_str::<ScenarioFleet>(toml)
            .map_err(|e| BagError::Scenario(e.to_string()))?
            .aircraft
            .into_iter()
            .map(|a| (a.id, a.model))
            .collect(),
        None => BTreeMap::new(),
    };

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
        rejected: 0,
        aircraft: Vec::new(),
    };

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
                    }
                    _ => {}
                },
                Err(_) => m.rejected += 1,
            },
            Some(TopicKind::Command(_)) | None => {}
        }
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

/// `psi` as a heading in [0, 360) deg (`rem_euclid` rounds a tiny negative up to 360).
fn heading_deg(psi_rad: f64) -> f64 {
    let h = psi_rad.to_degrees().rem_euclid(360.0);
    if h >= 360.0 {
        0.0
    } else {
        h
    }
}
