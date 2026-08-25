//! Zenoh bridge integration test. Drives the full ECS over a single in-process
//! Zenoh session — the same telemetry-out / commands-in path the binary uses over
//! TCP — publishing commands in and subscribing to telemetry out, then asserting
//! the commands reached the right aircraft and telemetry came back for the fleet.
//!
//! The test's pub/sub share the sim's [`Session`], so same-session delivery is
//! synchronous (no slow-joiner race like ZMQ PUB/SUB): a `publish` lands in the
//! subscriber's channel inline, and a `send` lands in the receiver's before the
//! next `app.update()` drains it.

mod common;

use astrum_net::{
    CommandReceiver, CommandSender, Session, TelemetryPublisher, TelemetrySubscriber,
};
use astrum_proto::{CommandMsg, Control};
use astrum_sim::build_app;
use common::{aircraft_probe, fdm_guard, two_ship};

#[test]
fn net_bridge_telemetry_out_and_commands_in() {
    let _fdm = fdm_guard();
    // One shared session so the test's sockets meet the sim's in-process.
    let session = Session::in_process().unwrap();
    let publisher = TelemetryPublisher::new(&session);
    let receiver = CommandReceiver::bind(&session).unwrap();

    let mut app = build_app(two_ship());
    app.insert_non_send(publisher);
    app.insert_non_send(receiver);

    // Subscriber + command sender on the same session.
    let mut sub = TelemetrySubscriber::connect(&session).unwrap();
    sub.subscribe_all().unwrap();
    let sender = CommandSender::connect(&session);

    // Spawn the fleet and pump a few ticks.
    for _ in 0..5 {
        app.update();
    }

    // Raw command → aircraft 1; Autopilot command → aircraft 2.
    sender
        .send(&CommandMsg {
            target_id: 1,
            control: Control::Raw { aileron: 0.5, elevator: 0.0, rudder: 0.0, throttle: 0.9 },
        })
        .unwrap();
    sender
        .send(&CommandMsg {
            target_id: 2,
            control: Control::Autopilot { heading_rad: 1.0, alt_msl_m: 2000.0, airspeed_mps: 55.0 },
        })
        .unwrap();

    // Collect telemetry across several ticks.
    let mut seen_ids = std::collections::HashSet::new();
    let mut last_ac1 = None;
    for _ in 0..60 {
        app.update();
        while let Some(s) = sub.try_recv().unwrap() {
            seen_ids.insert(s.aircraft_id);
            if s.aircraft_id == 1 {
                last_ac1 = Some(s);
            }
        }
    }

    // Telemetry reached the subscriber for both aircraft, with sane values.
    assert!(seen_ids.contains(&1) && seen_ids.contains(&2), "telemetry for both ids: {seen_ids:?}");
    let t = last_ac1.expect("aircraft 1 telemetry");
    assert!((t.lat_deg - 37.6188).abs() < 0.5, "plausible latitude, got {}", t.lat_deg);
    assert!(t.sim_time_s > 0.0 && t.alt_m.is_finite());

    // The raw command reached aircraft 1's FDM and disengaged its autopilot.
    let (aileron1, ap1) = aircraft_probe(&mut app, 1);
    assert!((aileron1 - 0.5).abs() < 1e-6, "raw aileron applied, got {aileron1}");
    assert!(!ap1.engaged, "raw command disengages autopilot");

    // The autopilot command engaged aircraft 2 with its targets.
    let (_, ap2) = aircraft_probe(&mut app, 2);
    assert!(ap2.engaged, "autopilot engaged for aircraft 2");
    assert!((ap2.heading_rad - 1.0).abs() < 1e-9);
    assert!((ap2.alt_msl_m - 2000.0).abs() < 1e-9);
    assert!((ap2.airspeed_mps - 55.0).abs() < 1e-9);
}
