//! The race systems, added to `astrum-sim`'s tick at [`TickSet::Stepped`]: after the
//! sim has stepped every FDM and advanced the clock, before it publishes the tick's
//! telemetry — so a race decision is in the same tick's events, `RaceState` and exit.
//!
//! The order within the hook is the rules of the race: crash detection, gate
//! scoring, the time limit, then the end of the run and the publication.

use astrum_geo::Enu;
use astrum_net::RacePublisher;
use astrum_proto::{
    EndReason, EventKind, RaceCourse as WireCourse, RaceGate, RaceOutcome, RaceState,
};
use astrum_sim::components::{Aircraft, AircraftState, SimClock, SimParams};
use astrum_sim::events::EventLog;
use astrum_sim::pilot::Pilot;
use astrum_sim::RunMode;
use bevy::ecs::schedule::ScheduleConfigs;
use bevy::ecs::system::ScheduleSystem;
use bevy::math::DVec3;
use bevy::prelude::*;

use crate::airframe::{Hull, SAMPLE_STEP_M};
use crate::components::{RaceCourse, RaceProgress};
use crate::course::Course;
use crate::gate::{BAR_M, OPENING_M, TOP_BAR_M};

/// How close to a gate a plane crossing has to be before it counts as a *miss*
/// rather than as unrelated flying. A gate's plane is mathematically infinite;
/// clipping the tally to a few gate-widths keeps "went round the outside" (which is
/// what a pilot would call a miss) distinct from "flew past 30 m away".
const MISS_RADIUS_M: f64 = 3.0;

fn vec(p: Enu) -> DVec3 {
    DVec3::new(p.e, p.n, p.u)
}

fn enu(v: DVec3) -> Enu {
    Enu { e: v.x, n: v.y, u: v.z }
}

/// The course as it goes on the wire, with each gate's explicit basis.
pub fn wire_course(course: &Course) -> WireCourse {
    WireCourse {
        seed: course.seed,
        opening_m: OPENING_M,
        bar_m: BAR_M,
        top_bar_m: TOP_BAR_M,
        gates: course
            .gates
            .iter()
            .map(|g| RaceGate {
                index: g.index,
                center: enu(g.center),
                yaw_rad: g.yaw_rad,
                normal: enu(g.normal()),
                right: enu(g.right()),
                up: enu(g.up()),
            })
            .collect(),
    }
}

/// The race's tick, in rules order: crash detection, gate scoring, the time limit,
/// the end of the run, the publication of the state. [`crate::build_race_app`] puts
/// it at [`astrum_sim::TickSet::Stepped`]; tests put it behind a hand-placed drone.
pub fn tick_systems() -> ScheduleConfigs<ScheduleSystem> {
    (detect_crash, track_gates, time_limit, end_race, publish_race).chain()
}

/// Startup: say what is about to be flown.
pub fn log_course(course: Res<RaceCourse>, mode: Res<RunMode>) {
    if *mode == RunMode::Server {
        println!("[astrum-race] course: {}", course.0.summary());
    }
}

/// End the race if the drone hit something: the ground, or a gate's frame.
///
/// # What counts
///
/// The airframe is the five-part hull [`Hull`] builds out of the F450's own JSBSim
/// model — four 9.4 in rotor discs on 0.1651 m arms, over a centre section that runs
/// from the gear feet to the rotor plane.
///
/// * **Ground.** Any contact at all. The racing floor is the ENU origin plane, which
///   is also JSBSim's terrain (a race's `world.origin_alt_hae_m` is 0), so
///   `min_z() <= 0` fires exactly as the gear feet touch down and needs no tolerance
///   constant. Touching down gently is still a crash: this is a race.
/// * **Gate frame.** The two uprights and the top bar of *any* gate, not just the
///   pending one — you can very easily clip the gate you have already passed, or the
///   next one round a tight corner. The opening is a hole and nothing hangs below the
///   sill, so flying through or underneath a gate touches nothing.
///
/// # Why it samples the step rather than the tick
///
/// One test per tick is not enough: at 20 m/s a 120 Hz step covers 0.17 m, more than
/// the 0.0762 m of frame stock, so the drone would teleport through gates. The usual
/// fix is to inflate the hull to cover the whole step, but that trades away exactly
/// what this environment cannot spare — the clearance is a few cm at the worst
/// heading. So the step (`prev_pos_enu` → `pos_enu`) is subdivided and the true hull
/// tested at each sample, at most [`SAMPLE_STEP_M`] apart. Attitude is held across the
/// samples; at 120 Hz it moves a degree or two.
pub fn detect_crash(
    course: Res<RaceCourse>,
    mut progress: ResMut<RaceProgress>,
    mode: Res<RunMode>,
    clock: Res<SimClock>,
    params: Res<SimParams>,
    mut events: ResMut<EventLog>,
    q: Query<&AircraftState, With<Aircraft>>,
) {
    if progress.finished() {
        return;
    }
    let Some(st) = q.iter().next() else { return };
    let (from, to) = (vec(st.prev_pos_enu), vec(st.pos_enu));
    let travel = to - from;
    let samples = (travel.length() / SAMPLE_STEP_M).ceil().max(1.0) as u32;
    let f = &st.fdm;

    for k in 1..=samples {
        let at = from + travel * (f64::from(k) / f64::from(samples));
        let hull = Hull::at(at, f.phi_rad, f.theta_rad, f.psi_rad);

        let (outcome, gate, what) = if hull.min_z() <= 0.0 {
            (RaceOutcome::CrashedGround, progress.next_gate, "ground contact".to_owned())
        } else if let Some(g) = course.0.gates.iter().find(|g| hull.hits_gate(g)) {
            (RaceOutcome::CrashedGate, g.index, format!("struck gate {}'s frame", g.index))
        } else {
            continue;
        };
        progress.finish(outcome);
        events.push(EventKind::Crashed { aircraft_id: progress.aircraft_id, outcome, gate });
        if *mode == RunMode::Server {
            println!(
                "[astrum-race] CRASH: {what} at sim {:.2}s, {:.1} m/s ({:.1} m/s vertical), \
                 {} gates passed",
                clock.time_s(&params),
                st.vel_enu.e.hypot(st.vel_enu.n).hypot(st.vel_enu.u),
                st.vel_enu.u,
                progress.passed,
            );
        }
        return;
    }
}

/// Score the pending gate: did this tick's motion carry the drone through it?
///
/// Works on the *segment* from `prev_pos_enu` to `pos_enu`, expressed in the gate's
/// local frame. At racing speed a 120 Hz tick covers metres, so a point-inside test
/// against a 7 cm-thick frame would step clean over the gate; instead we look for the
/// sign change of the along-normal coordinate and check where the crossing landed.
/// Gates must be taken in order, downstream.
pub fn track_gates(
    course: Res<RaceCourse>,
    mut progress: ResMut<RaceProgress>,
    mode: Res<RunMode>,
    clock: Res<SimClock>,
    params: Res<SimParams>,
    mut events: ResMut<EventLog>,
    q: Query<&AircraftState, With<Aircraft>>,
) {
    if progress.finished() {
        return;
    }
    let Some(st) = q.iter().next() else { return };
    let Some(gate) = course.0.gates.get(progress.next_gate as usize) else {
        return;
    };

    let prev = gate.to_local(vec(st.prev_pos_enu));
    let local = gate.to_local(vec(st.pos_enu));
    // Downstream crossing of the gate plane only.
    if !(prev.z < 0.0 && local.z >= 0.0) {
        return;
    }
    // Where in the plane the straight-line path pierced it.
    let t = (-prev.z / (local.z - prev.z)).clamp(0.0, 1.0);
    let hit = prev + t * (local - prev);
    let now = clock.time_s(&params);
    let verbose = *mode == RunMode::Server;

    if gate.opening_contains(hit) {
        progress.passed += 1;
        progress.last_pass_s = now;
        progress.next_gate += 1;
        events.push(EventKind::GatePassed {
            aircraft_id: progress.aircraft_id,
            gate: gate.index,
            offset_x_m: hit.x,
            offset_y_m: hit.y,
        });
        if verbose {
            let speed = st.vel_enu.e.hypot(st.vel_enu.n).hypot(st.vel_enu.u);
            println!(
                "[astrum-race] gate {}/{} passed at sim {now:.2}s ({speed:.2} m/s, {:.2} m from centre)",
                progress.passed,
                course.0.len(),
                hit.x.hypot(hit.y),
            );
        }
        if progress.next_gate as usize >= course.0.len() {
            progress.finish(RaceOutcome::Complete);
            if verbose {
                println!(
                    "[astrum-race] course complete: {} gates in {now:.2}s ({} missed crossings)",
                    progress.passed, progress.missed
                );
            }
        }
    } else if hit.x.hypot(hit.y) <= MISS_RADIUS_M {
        // Crossed the gate's plane beside the hole: went round the outside.
        progress.missed += 1;
        events.push(EventKind::GateMissed { aircraft_id: progress.aircraft_id, gate: gate.index });
        if verbose {
            println!(
                "[astrum-race] gate {} missed at sim {now:.2}s ({:.2} m off centre)",
                gate.index,
                hit.x.hypot(hit.y),
            );
        }
    }
}

/// Mark the race timed out once `sim.max_time_s` is reached with gates left. The
/// sim's own `stop_at_max_time` ends the run (reason `MaxTime`) later in the same
/// tick; this only makes sure the tick's `RaceState` says why.
pub fn time_limit(
    mut progress: ResMut<RaceProgress>,
    clock: Res<SimClock>,
    params: Res<SimParams>,
) {
    if params.max_time_s > 0.0 && clock.time_s(&params) >= params.max_time_s {
        progress.finish(RaceOutcome::TimedOut);
    }
}

/// End the run when the race is won or lost (the sim ends it itself at its time
/// cap, and in an [`RunMode::Embedded`] run the caller decides).
pub fn end_race(
    progress: Res<RaceProgress>,
    mode: Res<RunMode>,
    mut events: ResMut<EventLog>,
    mut exit: MessageWriter<AppExit>,
) {
    if *mode == RunMode::Embedded {
        return;
    }
    let reason = match progress.outcome {
        RaceOutcome::Complete => EndReason::RaceComplete,
        RaceOutcome::CrashedGround | RaceOutcome::CrashedGate => EndReason::Crashed,
        _ => return,
    };
    events.end(reason, "");
    exit.write(AppExit::Success);
}

/// Publish `astrum/race/course/<id>` on the first tick and about once per simulated
/// second, and `astrum/race/state/<id>` on each of the aircraft's control ticks.
/// Without a session this does nothing.
pub fn publish_race(
    publisher: Option<NonSend<RacePublisher>>,
    course: Res<RaceCourse>,
    progress: Res<RaceProgress>,
    clock: Res<SimClock>,
    params: Res<SimParams>,
    q: Query<&Pilot, With<Aircraft>>,
) {
    let Some(publisher) = publisher else { return };
    if clock.tick.is_multiple_of(params.rate_hz.round().max(1.0) as u64) {
        if let Err(e) = publisher.publish_course(&wire_course(&course.0)) {
            eprintln!("[astrum-race] course publish failed: {e}");
        }
    }
    if q.iter().next().is_some_and(|p| p.due(clock.tick)) {
        publish_state(&publisher, &course.0, &progress, clock.time_s(&params));
    }
}

fn publish_state(publisher: &RacePublisher, course: &Course, progress: &RaceProgress, t: f64) {
    let state = RaceState {
        aircraft_id: progress.aircraft_id,
        pending_gate: progress.next_gate,
        passed: progress.passed,
        missed: progress.missed,
        gate_count: course.len() as u32,
        last_pass_s: progress.last_pass_s,
        sim_time_s: t,
        finished: progress.finished(),
        outcome: progress.outcome,
    };
    if let Err(e) = publisher.publish_state(&state) {
        eprintln!("[astrum-race] state publish failed: {e}");
    }
}

/// Startup, when the sim could not spawn the drone (its fatal error ends the run as
/// `EndReason::Fatal`): the race says so too.
pub fn fail_spawn(
    mut progress: ResMut<RaceProgress>,
    publisher: Option<NonSend<RacePublisher>>,
    course: Res<RaceCourse>,
) {
    progress.finish(RaceOutcome::SpawnFailed);
    if let Some(p) = publisher {
        publish_state(&p, &course.0, &progress, 0.0);
    }
}
