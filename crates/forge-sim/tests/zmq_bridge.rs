//! ZMQ bridge integration test. Drives the full ECS over an inproc ZMQ context —
//! the same PUB (telemetry) / PULL (commands) path the binary uses over TCP —
//! pushing commands in and subscribing to telemetry out, then asserting the
//! commands reached the right aircraft and telemetry came back for the fleet.

mod common;

use common::{aircraft_probe, fdm_guard, two_ship};
use forge_net::{CommandReceiver, CommandSender, TelemetryPublisher, TelemetrySubscriber};
use forge_proto::{CommandMsg, Control};
use forge_sim::build_app;

#[test]
fn zmq_bridge_telemetry_out_and_commands_in() {
    let _fdm = fdm_guard();
    // Shared context so the test's sockets can meet the sim's over inproc.
    let ctx = forge_net::context();
    let publisher = TelemetryPublisher::bind(&ctx, "inproc://bridge-tlm").unwrap();
    let receiver = CommandReceiver::bind(&ctx, "inproc://bridge-cmd").unwrap();

    let mut app = build_app(two_ship());
    app.insert_non_send(publisher);
    app.insert_non_send(receiver);

    // Subscriber + command sender on the same context.
    let sub = TelemetrySubscriber::connect(&ctx, "inproc://bridge-tlm").unwrap();
    sub.subscribe_all().unwrap();
    let sender = CommandSender::connect(&ctx, "inproc://bridge-cmd").unwrap();

    // Spawn the fleet and pump a few ticks so the SUB finishes connecting.
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

    // Collect telemetry across several ticks (PUB/SUB has a slow-joiner race).
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
