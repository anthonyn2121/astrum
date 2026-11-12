//! The race's side of the Zenoh bridge: the course and the aircraft's progress.
//!
//! Everything else — telemetry, commands, events, tick ends, the control plane —
//! is the sim's own ([`astrum_sim::net::connect`]); a race adds
//! `astrum/race/course/<id>` and `astrum/race/state/<id>`.

use astrum_net::{RacePublisher, Session};
use bevy::prelude::*;

use crate::components::RaceProgress;

/// Publish the race's course and state on `session` (call after
/// [`astrum_sim::net::connect`], on an app from [`crate::build_race_app`]).
pub fn connect(app: &mut App, session: &Session) -> astrum_net::Result<()> {
    let id = app.world().resource::<RaceProgress>().aircraft_id;
    app.insert_non_send(RacePublisher::new(session, id)?);
    Ok(())
}
