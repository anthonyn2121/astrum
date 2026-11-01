//! The Zenoh bridge: commands in (with optional lockstep) and telemetry out.
//!
//! Telemetry is assembled once per tick by [`publish_telemetry`] into
//! [`LastPublished`] whether or not a publisher is installed — the RL env reads
//! its observations from the same resource, so both front doors see identical
//! state. The transport wrappers are `NonSend` resources; without them the
//! bridge systems only assemble telemetry, so the physics-only app (tests, the
//! embedded env) runs without a session.

use std::collections::{BTreeSet, HashMap};
use std::time::{Duration, Instant};

use astrum_bag::wall_clock_ns;
use astrum_net::{CommandReceiver, CommandReject, TelemetryPublisher};
use astrum_proto::{
    encode_command, encode_telemetry, Aero, BodyRates, CombatStatus, Command, Euler, Ned, Telemetry,
};
use bevy::prelude::*;

use crate::combat::{Dead, Gun, Health, Team, Trigger};
use crate::components::{
    Aircraft, AircraftIndex, AircraftState, SimClock, SimParams, SimStatus, VehicleKind,
};
use crate::pilot::{apply_command, Pilot, Refusal};
use crate::record::TickRecorder;
use crate::systems::StopRequested;

const RECORDING_FAILED: &str = "recording stopped: the bag writer failed";

/// The telemetry of the latest tick, one sample per aircraft, sorted by id.
#[derive(Resource, Debug, Clone, Default)]
pub struct LastPublished(pub Vec<Telemetry>);

/// Command refusals since the last report, by aircraft id.
#[derive(Resource, Debug, Default)]
pub struct RefusalLog {
    counts: HashMap<u32, u64>,
    rejected_payloads: u64,
    last_report: Option<Instant>,
}

impl RefusalLog {
    fn refuse(&mut self, id: u32, why: &Refusal) {
        let n = self.counts.entry(id).or_default();
        if *n == 0 {
            eprintln!("[astrum-sim] aircraft {id}: command refused: {why}");
        }
        *n += 1;
    }

    fn reject(&mut self, r: &CommandReject) {
        if self.rejected_payloads == 0 {
            eprintln!("[astrum-sim] {r}");
        }
        self.rejected_payloads += 1;
    }

    /// Summarise repeated refusals at most once per second of wall time.
    fn report(&mut self) {
        let now = Instant::now();
        if self.last_report.is_some_and(|t| now - t < Duration::from_secs(1)) {
            return;
        }
        self.last_report = Some(now);
        let mut ids: Vec<_> = self.counts.iter().filter(|(_, n)| **n > 1).collect();
        ids.sort();
        for (id, n) in ids {
            eprintln!("[astrum-sim] aircraft {id}: {n} commands refused in the last interval");
        }
        if self.rejected_payloads > 1 {
            eprintln!(
                "[astrum-sim] {} malformed command payloads rejected",
                self.rejected_payloads
            );
        }
        self.counts.clear();
        self.rejected_payloads = 0;
    }
}

/// Build one aircraft's telemetry sample.
#[allow(clippy::too_many_arguments)]
pub fn telemetry(
    tick: u64,
    sim_time_s: f64,
    ac: &Aircraft,
    st: &AircraftState,
    kind: &VehicleKind,
    pilot: &Pilot,
    team: &Team,
    hp: &Health,
    gun: Option<&Gun>,
    dead: bool,
) -> Telemetry {
    let f = &st.fdm;
    Telemetry {
        aircraft_id: ac.id,
        tick,
        sim_time_s,
        lat_deg: f.lat_deg,
        lon_deg: f.lon_deg,
        alt_hae_m: f.alt_hae_m,
        pos_enu: st.pos_enu,
        vel_enu: st.vel_enu,
        vel_ned: Ned { n: f.vel_north_mps, e: f.vel_east_mps, d: f.vel_down_mps },
        attitude: Euler { phi_rad: f.phi_rad, theta_rad: f.theta_rad, psi_rad: f.psi_rad },
        rates: BodyRates { p_rad_s: f.p_rad_s, q_rad_s: f.q_rad_s, r_rad_s: f.r_rad_s },
        aero: Aero { alpha_rad: f.alpha_rad, beta_rad: f.beta_rad, nz_g: f.nz_g, mach: f.mach },
        tas_mps: f.tas_mps,
        fault_flags: 0,
        vehicle_kind: (*kind).into(),
        control_mode: pilot.mode.control_mode(),
        combat: CombatStatus {
            team: team.0,
            hit_points: hp.hp,
            hit_points_max: hp.hp_max,
            alive: !dead,
            ammo: gun.map_or(0, |g| g.ammo),
        },
    }
}

/// The components a telemetry sample is built from.
pub type TelemetryQuery<'w, 's> = Query<
    'w,
    's,
    (
        &'static Aircraft,
        &'static AircraftState,
        &'static VehicleKind,
        &'static Pilot,
        &'static Team,
        &'static Health,
        Option<&'static Gun>,
        Has<Dead>,
    ),
>;

/// Assemble this tick's telemetry into [`LastPublished`] (every aircraft, every
/// tick), record all of it, and publish the samples of aircraft whose control
/// tick this is (`control_rate_hz`; default every tick). Each published or
/// recorded sample is encoded once.
#[allow(clippy::too_many_arguments)]
pub fn publish_telemetry(
    publisher: Option<NonSendMut<TelemetryPublisher>>,
    recorder: Option<Res<TickRecorder>>,
    params: Res<SimParams>,
    clock: Res<SimClock>,
    mut status: ResMut<SimStatus>,
    mut last: ResMut<LastPublished>,
    mut due: Local<Vec<(u32, bool)>>,
    q: TelemetryQuery,
) {
    let t = clock.time_s(&params);
    last.0.clear();
    due.clear();
    for (ac, st, kind, pilot, team, hp, gun, dead) in &q {
        last.0.push(telemetry(clock.tick, t, ac, st, kind, pilot, team, hp, gun, dead));
        due.push((ac.id, pilot.due(clock.tick)));
    }
    last.0.sort_unstable_by_key(|s| s.aircraft_id);
    due.sort_unstable_by_key(|(id, _)| *id);
    let mut publisher = publisher;
    if publisher.is_none() && recorder.is_none() {
        return;
    }
    let log_time_ns = if recorder.is_some() { wall_clock_ns() } else { 0 };
    for (s, &(_, due)) in last.0.iter().zip(due.iter()) {
        let publish = due && publisher.is_some();
        if !publish && recorder.is_none() {
            continue;
        }
        let payload = encode_telemetry(s);
        let to_publish = match &recorder {
            Some(r) => {
                let keep = publish.then(|| payload.clone());
                if !r.telemetry(s.aircraft_id, log_time_ns, s.sim_time_s, payload) {
                    status.fatal.get_or_insert_with(|| RECORDING_FAILED.to_owned());
                }
                keep
            }
            None => Some(payload),
        };
        if let (Some(p), Some(payload)) = (publisher.as_mut(), to_publish) {
            if let Err(e) = p.publish_raw(s.aircraft_id, payload) {
                eprintln!(
                    "[astrum-sim] telemetry publish failed for aircraft {}: {e}",
                    s.aircraft_id
                );
            }
        }
    }
}

/// Record a received command (as decoded) when recording.
fn record_received(
    recorder: Option<&TickRecorder>,
    received: &Result<Command, CommandReject>,
    status: &mut SimStatus,
) {
    if let (Some(r), Ok(cmd)) = (recorder, received) {
        if !r.command(cmd.target_id, wall_clock_ns(), encode_command(cmd)) {
            status.fatal.get_or_insert_with(|| RECORDING_FAILED.to_owned());
        }
    }
}

/// How often a lockstep wait re-sends the last telemetry, so a controller that
/// subscribed late (or lost a sample) still gets the tick it must answer.
const LOCKSTEP_REPUBLISH: Duration = Duration::from_millis(100);

/// Apply one received command; returns the aircraft id and the telemetry tick
/// it answered if it was applied.
fn apply_received(
    received: Result<Command, CommandReject>,
    now: u64,
    lease: Option<u64>,
    index: &AircraftIndex,
    q: &mut Query<(&mut Pilot, Option<&mut Trigger>, Has<Dead>)>,
    log: &mut RefusalLog,
) -> Option<(u32, u64)> {
    let cmd = match received {
        Ok(c) => c,
        Err(r) => {
            log.reject(&r);
            return None;
        }
    };
    let found = index.0.get(&cmd.target_id).and_then(|e| q.get_mut(*e).ok());
    let Some((mut pilot, trigger, dead)) = found else {
        log.refuse(cmd.target_id, &Refusal::UnknownAircraft { id: cmd.target_id });
        return None;
    };
    if dead {
        return None; // a wreck takes no commands
    }
    match apply_command(&mut pilot, trigger.map(|t| t.into_inner()), &cmd, now, lease) {
        Ok(()) => Some((cmd.target_id, cmd.telemetry_tick)),
        Err(why) => {
            log.refuse(cmd.target_id, &why);
            None
        }
    }
}

/// Drain queued commands and apply them in arrival order (the latest command
/// per aircraft wins). In a lockstep scenario, then block until every living
/// lockstep aircraft due a command this tick (see `control_rate_hz`) has one
/// answering it, re-sending that telemetry while waiting; too long a wait is
/// fatal, and a stop request ends the wait.
#[allow(clippy::too_many_arguments)]
pub fn ingest_commands(
    receiver: Option<NonSend<CommandReceiver>>,
    publisher: Option<NonSendMut<TelemetryPublisher>>,
    recorder: Option<Res<TickRecorder>>,
    stop: Option<Res<StopRequested>>,
    last: Res<LastPublished>,
    params: Res<SimParams>,
    clock: Res<SimClock>,
    index: Res<AircraftIndex>,
    mut status: ResMut<SimStatus>,
    mut log: ResMut<RefusalLog>,
    mut q: Query<(&mut Pilot, Option<&mut Trigger>, Has<Dead>)>,
) {
    let Some(rx) = receiver else { return };
    let recorder = recorder.as_deref();
    let now = clock.tick;
    let lease = params.lease_ticks;
    let mut answered: BTreeSet<u32> = BTreeSet::new();
    for r in rx.drain() {
        record_received(recorder, &r, &mut status);
        if let Some((id, tick)) = apply_received(r, now, lease, &index, &mut q, &mut log) {
            if tick == now {
                answered.insert(id);
            }
        }
    }

    if params.lockstep {
        let pending =
            |answered: &BTreeSet<u32>, q: &Query<(&mut Pilot, Option<&mut Trigger>, Has<Dead>)>| {
                index
                    .0
                    .iter()
                    .filter(|(id, e)| {
                        q.get(**e).is_ok_and(|(p, _, dead)| p.lockstep && !dead && p.due(now))
                            && !answered.contains(id)
                    })
                    .map(|(id, _)| *id)
                    .collect::<BTreeSet<u32>>()
            };
        let deadline = Instant::now() + Duration::from_secs_f64(params.lockstep_timeout_s);
        let mut publisher = publisher;
        let mut waiting = pending(&answered, &q);
        while !waiting.is_empty() {
            if stop.as_ref().is_some_and(|s| s.requested()) {
                return;
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                status.fatal = Some(format!(
                    "lockstep: no command answering tick {now} from aircraft {waiting:?} within {:.1} s",
                    params.lockstep_timeout_s
                ));
                return;
            }
            match rx.recv_timeout(left.min(LOCKSTEP_REPUBLISH)) {
                Ok(r) => {
                    record_received(recorder, &r, &mut status);
                    if let Some((id, tick)) =
                        apply_received(r, now, lease, &index, &mut q, &mut log)
                    {
                        if tick == now {
                            answered.insert(id);
                        }
                    }
                }
                Err(astrum_net::NetError::Timeout) => {
                    if let Some(p) = publisher.as_mut() {
                        for s in last.0.iter().filter(|s| waiting.contains(&s.aircraft_id)) {
                            let _ = p.publish(s);
                        }
                    }
                }
                Err(e) => {
                    status.fatal = Some(format!("lockstep: command intake failed: {e}"));
                    return;
                }
            }
            waiting = pending(&answered, &q);
        }
    }
    log.report();
}
