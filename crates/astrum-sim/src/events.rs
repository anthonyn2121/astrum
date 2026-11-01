//! The event stream: the discrete things that happen during a run (trigger
//! presses, hits, kills, control-mode changes, controllers coming and going,
//! the run's end), as opposed to the per-tick state in telemetry.
//!
//! Systems [`EventLog::push`] what they see; at the end of every update
//! [`publish_events`] stamps them with the tick, publishes each on
//! `astrum/events`, records it when recording, and keeps the update's events
//! for the RL env ([`EventLog::last`]). An event's tick is the `Telemetry.tick`
//! that first shows it (a hit lands in the frame the round struck; while paused,
//! the current tick). Order within an update is the order the tick chain ran.

use astrum_bag::wall_clock_ns;
use astrum_net::EventPublisher;
use astrum_proto::{encode_event, EndReason, Event, EventKind};
use bevy::prelude::*;

use crate::components::{SimClock, SimParams, SimStatus};
use crate::net::RECORDING_FAILED;
use crate::record::TickRecorder;

/// Events of the current update, then of the latest one.
#[derive(Resource, Debug, Default)]
pub struct EventLog {
    pending: Vec<EventKind>,
    /// A `run-ended` was reported (the first reason wins).
    ended: bool,
    last: Vec<Event>,
}

impl EventLog {
    pub fn push(&mut self, kind: EventKind) {
        self.pending.push(kind);
    }

    /// Report why the run ended; only the first call counts.
    pub fn end(&mut self, reason: EndReason, message: &str) {
        if !self.ended {
            self.ended = true;
            self.push(EventKind::RunEnded { reason, message: message.to_owned() });
        }
    }

    /// The events of the latest update, in order.
    pub fn last(&self) -> &[Event] {
        &self.last
    }
}

/// Stamp this update's events and send them on (bus, recording, [`EventLog::last`]).
pub fn publish_events(
    mut log: ResMut<EventLog>,
    clock: Res<SimClock>,
    params: Res<SimParams>,
    publisher: Option<NonSend<EventPublisher>>,
    recorder: Option<Res<TickRecorder>>,
    mut status: ResMut<SimStatus>,
) {
    if log.pending.is_empty() {
        if !log.last.is_empty() {
            log.last.clear();
        }
        return;
    }
    let EventLog { pending, last, .. } = &mut *log;
    last.clear();
    let sim_time_s = clock.time_s(&params);
    let log_time_ns = if recorder.is_some() { wall_clock_ns() } else { 0 };
    for kind in pending.drain(..) {
        let event = Event { tick: clock.tick, sim_time_s, kind };
        if publisher.is_some() || recorder.is_some() {
            let payload = encode_event(&event);
            if let Some(r) = &recorder {
                if !r.event(log_time_ns, sim_time_s, payload.clone()) {
                    status.fatal.get_or_insert_with(|| RECORDING_FAILED.to_owned());
                }
            }
            if let Some(p) = &publisher {
                if let Err(e) = p.publish_raw(payload) {
                    eprintln!("[astrum-sim] event publish failed: {e}");
                }
            }
        }
        last.push(event);
    }
}
