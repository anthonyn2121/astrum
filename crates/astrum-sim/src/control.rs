//! Run control (`astrum/control/*`: pause, resume, step, speed) and the run
//! status the sim answers on `astrum/status`.
//!
//! Control decides only *when* the next tick starts, never what it computes: a
//! lockstep run that is paused, stepped and resumed ends bit-identical to one
//! flown straight through. Requests are applied between ticks. While paused the
//! sim re-sends the last telemetry every [`PAUSED_POLL`], so controllers don't
//! take the silence for the end of the run, and leases count ticks, so a pause
//! never trips the failsafe.
//!
//! Without a [`ControlServer`] (the RL env, in-process sweep runs, most tests)
//! the run never pauses and these systems return at once.

use std::collections::BTreeSet;
use std::sync::PoisonError;
use std::time::{Duration, Instant};

use astrum_net::{ControlRequest, ControlServer, RunStatus, StatusBoard, TelemetryPublisher};
use bevy::prelude::*;

use crate::components::{Aircraft, SimClock, SimParams};
use crate::net::{Controllers, LastPublished};
use crate::systems::{Pacer, StopRequested};
use crate::RunMode;

/// How long a paused sim waits for a request before re-sending telemetry.
pub const PAUSED_POLL: Duration = Duration::from_millis(100);

/// How often a running sim refreshes its status (and on every control change).
const STATUS_EVERY: Duration = Duration::from_millis(100);

/// Whether ticks run.
#[derive(Resource, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RunControl {
    paused: bool,
    /// Ticks left of a `step`; the run pauses when it reaches 0.
    steps_left: Option<u64>,
}

impl RunControl {
    /// Run condition of the tick.
    pub fn ticking(control: Res<RunControl>) -> bool {
        !control.paused
    }

    pub fn paused(&self) -> bool {
        self.paused
    }
}

/// The status board the control server answers `astrum/status` from.
#[derive(Resource, Clone)]
pub struct Board(pub StatusBoard);

impl Board {
    /// Show that the current tick is waiting on these lockstep aircraft.
    pub fn show_waiting(&self, waiting: &BTreeSet<u32>) {
        let mut s = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        s.state = "waiting".into();
        s.lockstep_waiting = waiting.iter().copied().collect();
    }
}

/// Apply the control requests received since the last tick. While paused,
/// block (up to [`PAUSED_POLL`]) for the next one, re-sending the last
/// telemetry each time.
#[allow(clippy::too_many_arguments)]
pub fn serve_control(
    server: Option<NonSend<ControlServer>>,
    publisher: Option<NonSendMut<TelemetryPublisher>>,
    stop: Option<Res<StopRequested>>,
    last: Res<LastPublished>,
    mode: Res<RunMode>,
    clock: Res<SimClock>,
    mut control: ResMut<RunControl>,
    mut params: ResMut<SimParams>,
    mut pacer: ResMut<Pacer>,
) {
    let Some(server) = server else { return };
    let requests = if control.paused && !stop.is_some_and(|s| s.requested()) {
        if let Some(mut p) = publisher {
            for s in &last.0 {
                let _ = p.publish(s);
            }
        }
        server.wait(PAUSED_POLL)
    } else {
        server.drain()
    };
    for request in requests {
        let was_paused = control.paused;
        let what = match request {
            ControlRequest::Pause => {
                *control = RunControl { paused: true, steps_left: None };
                "paused".to_owned()
            }
            ControlRequest::Resume => {
                *control = RunControl { paused: false, steps_left: None };
                "resumed".to_owned()
            }
            ControlRequest::Step(n) => {
                *control = RunControl { paused: false, steps_left: Some(n) };
                format!("stepping {n} ticks")
            }
            ControlRequest::Speed(x) => {
                params.speed = x;
                pacer.set_speed(params.dt, x);
                if x > 0.0 {
                    format!("speed {x}× real time")
                } else {
                    "unpaced".to_owned()
                }
            }
        };
        if was_paused && !control.paused {
            pacer.restart(); // no sprint to make up for the pause
        }
        if *mode == RunMode::Server {
            println!("[astrum-sim] tick {}: {what}", clock.tick);
        }
    }
}

/// Count down a `step`, pausing after its last tick (the end of the tick chain).
pub fn count_steps(mut control: ResMut<RunControl>, clock: Res<SimClock>, mode: Res<RunMode>) {
    let Some(n) = control.steps_left else { return };
    if n > 1 {
        // Not a state change: keeps the status refresh at its normal rate.
        control.bypass_change_detection().steps_left = Some(n - 1);
        return;
    }
    *control = RunControl { paused: true, steps_left: None };
    if *mode == RunMode::Server {
        println!("[astrum-sim] tick {}: paused (step done)", clock.tick);
    }
}

/// Refresh the status board: every [`STATUS_EVERY`] and on any control change.
pub fn publish_status(
    board: Option<Res<Board>>,
    control: Res<RunControl>,
    clock: Res<SimClock>,
    params: Res<SimParams>,
    controllers: Res<Controllers>,
    q: Query<(), With<Aircraft>>,
    mut window: Local<Option<(Instant, u64)>>,
) {
    let Some(board) = board else { return };
    let now = Instant::now();
    if !control.is_changed() && window.is_some_and(|(t, _)| now.duration_since(t) < STATUS_EVERY) {
        return;
    }
    let (since, tick0) = window.unwrap_or((now, clock.tick));
    let elapsed = now.duration_since(since).as_secs_f64();
    let achieved_speed =
        if elapsed > 0.0 { (clock.tick - tick0) as f64 * params.dt / elapsed } else { 0.0 };
    let status = RunStatus {
        tick: clock.tick,
        sim_time_s: clock.time_s(&params),
        state: if control.paused { "paused" } else { "running" }.into(),
        speed: params.speed,
        achieved_speed,
        aircraft: q.iter().count(),
        controllers: controllers.present(),
        lockstep_waiting: Vec::new(),
    };
    *board.0.lock().unwrap_or_else(PoisonError::into_inner) = status;
    *window = Some((now, clock.tick));
}
