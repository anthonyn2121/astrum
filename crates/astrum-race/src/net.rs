//! The race's side of the Zenoh bridge: each drone's course and progress.
//!
//! Everything else — telemetry, commands, events, tick ends, the control plane —
//! is the sim's own ([`astrum_sim::net::connect`]); a race adds, per aircraft,
//! `astrum/race/course/<id>` and `astrum/race/state/<id>`.

use astrum_net::{RacePublisher, Session};
use bevy::prelude::*;

use crate::components::Race;

/// One [`RacePublisher`] per racing aircraft.
pub struct RaceBus(Vec<(u32, RacePublisher)>);

impl RaceBus {
    /// The publisher of aircraft `id`'s race messages.
    pub fn publisher(&self, id: u32) -> Option<&RacePublisher> {
        self.0.iter().find(|(i, _)| *i == id).map(|(_, p)| p)
    }
}

/// Publish every drone's course and state on `session` (call after
/// [`astrum_sim::net::connect`], on an app from [`crate::build_race_app`]).
pub fn connect(app: &mut App, session: &Session) -> astrum_net::Result<()> {
    let ids: Vec<u32> =
        app.world().resource::<Race>().drones().iter().map(|d| d.aircraft_id).collect();
    let publishers = ids
        .into_iter()
        .map(|id| Ok((id, RacePublisher::new(session, id)?)))
        .collect::<astrum_net::Result<_>>()?;
    app.insert_non_send(RaceBus(publishers));
    Ok(())
}
