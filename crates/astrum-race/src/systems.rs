//! The race systems, added to `astrum-sim`'s tick at [`TickSet::Stepped`]: after the
//! sim has stepped every FDM and advanced the clock, before it publishes the tick's
//! telemetry — so a race decision is in the same tick's events, `RaceState` and exit.
//!
//! The order within the hook is the rules of the race: crash detection, gate
//! scoring, the time limit, then the end of the run and the publication. Every drone
//! is judged on its own, against the shared gates and the ground, and never against
//! another drone (drones are ghosts to each other).

use astrum_geo::Enu;
use astrum_proto::{
    EndReason, EventKind, RaceCourse as WireCourse, RaceGate, RaceOutcome, RaceState,
};
use astrum_sim::combat::Dead;
use astrum_sim::components::{Aircraft, AircraftState, SimClock, SimParams};
use astrum_sim::events::EventLog;
use astrum_sim::pilot::Pilot;
use astrum_sim::RunMode;
use bevy::ecs::schedule::ScheduleConfigs;
use bevy::ecs::system::ScheduleSystem;
use bevy::math::DVec3;
use bevy::prelude::*;

use crate::airframe::{Hull, SAMPLE_STEP_M};
use crate::components::{Race, RaceCourse};
use crate::course::Course;
use crate::gate::{BAR_M, OPENING_M, TOP_BAR_M};
use crate::net::RaceBus;

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
/// the end of the run, the freezing of crashed drones, the publication of the state.
/// [`crate::build_race_app`] puts it at [`astrum_sim::TickSet::Stepped`]; tests put it
/// behind hand-placed drones.
pub fn tick_systems() -> ScheduleConfigs<ScheduleSystem> {
    (detect_crash, track_gates, time_limit, end_race, freeze_crashed, publish_race).chain()
}

/// Startup: say what is about to be flown.
pub fn log_course(course: Res<RaceCourse>, race: Res<Race>, mode: Res<RunMode>) {
    if *mode == RunMode::Server {
        println!("[astrum-race] course: {}", course.0.summary());
        if race.drones().len() > 1 {
            let ids: Vec<String> =
                race.drones().iter().map(|d| d.aircraft_id.to_string()).collect();
            println!("[astrum-race] {} drones race it: aircraft {}", ids.len(), ids.join(", "));
        }
    }
}

/// End a drone's race if it hit something: the ground, or a gate's frame.
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
/// Drones are not tested against each other. A drone whose race is over is not
/// tested at all.
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
    mut race: ResMut<Race>,
    mode: Res<RunMode>,
    clock: Res<SimClock>,
    params: Res<SimParams>,
    mut events: ResMut<EventLog>,
    q: Query<(&Aircraft, &AircraftState)>,
) {
    for (ac, st) in &q {
        let Some(progress) = race.drone_mut(ac.id).filter(|d| !d.finished()) else { continue };
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
            events.push(EventKind::Crashed { aircraft_id: ac.id, outcome, gate });
            if *mode == RunMode::Server {
                println!(
                    "[astrum-race] aircraft {}: CRASH: {what} at sim {:.2}s, {:.1} m/s \
                     ({:.1} m/s vertical), {} gates passed",
                    ac.id,
                    clock.time_s(&params),
                    st.vel_enu.e.hypot(st.vel_enu.n).hypot(st.vel_enu.u),
                    st.vel_enu.u,
                    progress.passed,
                );
            }
            break;
        }
    }
}

/// Score each drone's pending gate: did this tick's motion carry it through?
///
/// Works on the *segment* from `prev_pos_enu` to `pos_enu`, expressed in the gate's
/// local frame. At racing speed a 120 Hz tick covers metres, so a point-inside test
/// against a 7 cm-thick frame would step clean over the gate; instead we look for the
/// sign change of the along-normal coordinate and check where the crossing landed.
/// Gates must be taken in order, downstream, by each drone on its own.
pub fn track_gates(
    course: Res<RaceCourse>,
    mut race: ResMut<Race>,
    mode: Res<RunMode>,
    clock: Res<SimClock>,
    params: Res<SimParams>,
    mut events: ResMut<EventLog>,
    q: Query<(&Aircraft, &AircraftState)>,
) {
    for (ac, st) in &q {
        let Some(progress) = race.drone_mut(ac.id).filter(|d| !d.finished()) else { continue };
        let Some(gate) = course.0.gates.get(progress.next_gate as usize) else { continue };

        let prev = gate.to_local(vec(st.prev_pos_enu));
        let local = gate.to_local(vec(st.pos_enu));
        // Downstream crossing of the gate plane only.
        if !(prev.z < 0.0 && local.z >= 0.0) {
            continue;
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
                aircraft_id: ac.id,
                gate: gate.index,
                offset_x_m: hit.x,
                offset_y_m: hit.y,
            });
            if verbose {
                let speed = st.vel_enu.e.hypot(st.vel_enu.n).hypot(st.vel_enu.u);
                println!(
                    "[astrum-race] aircraft {}: gate {}/{} passed at sim {now:.2}s ({speed:.2} m/s, {:.2} m from centre)",
                    ac.id,
                    progress.passed,
                    course.0.len(),
                    hit.x.hypot(hit.y),
                );
            }
            if progress.next_gate as usize >= course.0.len() {
                progress.finish(RaceOutcome::Complete);
                if verbose {
                    println!(
                        "[astrum-race] aircraft {}: course complete: {} gates in {now:.2}s ({} missed crossings)",
                        ac.id, progress.passed, progress.missed
                    );
                }
            }
        } else if hit.x.hypot(hit.y) <= MISS_RADIUS_M {
            // Crossed the gate's plane beside the hole: went round the outside.
            progress.missed += 1;
            events.push(EventKind::GateMissed { aircraft_id: ac.id, gate: gate.index });
            if verbose {
                println!(
                    "[astrum-race] aircraft {}: gate {} missed at sim {now:.2}s ({:.2} m off centre)",
                    ac.id,
                    gate.index,
                    hit.x.hypot(hit.y),
                );
            }
        }
    }
}

/// Mark every drone still racing timed out once `sim.max_time_s` is reached. The
/// sim's own `stop_at_max_time` ends the run (reason `MaxTime`) later in the same tick;
/// this only makes sure the tick's `RaceState` says why.
pub fn time_limit(mut race: ResMut<Race>, clock: Res<SimClock>, params: Res<SimParams>) {
    if params.max_time_s > 0.0 && clock.time_s(&params) >= params.max_time_s {
        for d in race.drones_mut() {
            d.finish(RaceOutcome::TimedOut);
        }
    }
}

/// End the run once every drone's race is over: `RaceComplete` if any drone completed
/// the course, `Crashed` if every drone crashed. (Drones that ran out of time end the
/// run through the sim's own time cap, as `MaxTime` — unless one completed.) In an
/// [`RunMode::Embedded`] run the caller decides.
pub fn end_race(
    race: Res<Race>,
    mode: Res<RunMode>,
    mut events: ResMut<EventLog>,
    mut exit: MessageWriter<AppExit>,
) {
    if *mode == RunMode::Embedded || !race.all_finished() {
        return;
    }
    let any = |outcome| race.drones().iter().any(|d| d.outcome == outcome);
    let reason = if any(RaceOutcome::Complete) {
        EndReason::RaceComplete
    } else if race.drones().iter().all(|d| d.crashed()) {
        EndReason::Crashed
    } else {
        return; // out of time: the sim's own cap ends it
    };
    events.end(reason, "");
    exit.write(AppExit::Success);
}

/// A drone that crashed while others still race is frozen where it is — marked
/// [`Dead`], which the sim already freezes (no FDM step, no commands, no lockstep
/// wait; its telemetry says `alive = false`) — so its wreck stays observable and its
/// controller cannot stall the others. When the crash ended the run nothing is frozen.
pub fn freeze_crashed(
    race: Res<Race>,
    mut commands: Commands,
    q: Query<(Entity, &Aircraft), Without<Dead>>,
) {
    if race.all_finished() {
        return;
    }
    for (ent, ac) in &q {
        if race.drone(ac.id).is_some_and(|d| d.crashed()) {
            commands.entity(ent).insert(Dead);
        }
    }
}

/// Publish each drone's `astrum/race/course/<id>` on the first tick and about once per
/// simulated second, and its `astrum/race/state/<id>` on each of the aircraft's control
/// ticks. Without a session this does nothing.
pub fn publish_race(
    bus: Option<NonSend<RaceBus>>,
    course: Res<RaceCourse>,
    race: Res<Race>,
    clock: Res<SimClock>,
    params: Res<SimParams>,
    q: Query<(&Aircraft, &Pilot)>,
) {
    let Some(bus) = bus else { return };
    let t = clock.time_s(&params);
    let course_due = clock.tick.is_multiple_of(params.rate_hz.round().max(1.0) as u64);
    let wire = course_due.then(|| wire_course(&course.0));
    for (ac, pilot) in &q {
        if let (Some(p), Some(wire)) = (bus.publisher(ac.id), &wire) {
            if let Err(e) = p.publish_course(wire) {
                eprintln!("[astrum-race] aircraft {}: course publish failed: {e}", ac.id);
            }
        }
        if pilot.due(clock.tick) {
            publish_state(&bus, &course.0, &race, ac.id, t);
        }
    }
}

fn publish_state(bus: &RaceBus, course: &Course, race: &Race, id: u32, t: f64) {
    let (Some(publisher), Some(progress)) = (bus.publisher(id), race.drone(id)) else { return };
    let state = RaceState {
        aircraft_id: id,
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
        eprintln!("[astrum-race] aircraft {id}: state publish failed: {e}");
    }
}

/// Startup, when the sim could not spawn the fleet (its fatal error ends the run as
/// `EndReason::Fatal`): the race says so too, for every drone.
pub fn fail_spawn(mut race: ResMut<Race>, bus: Option<NonSend<RaceBus>>, course: Res<RaceCourse>) {
    for d in race.drones_mut() {
        d.finish(RaceOutcome::SpawnFailed);
    }
    if let Some(bus) = bus {
        for d in race.drones() {
            publish_state(&bus, &course.0, &race, d.aircraft_id, 0.0);
        }
    }
}
