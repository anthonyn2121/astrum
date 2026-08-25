//! Zenoh transport integration tests over a single in-process session: command
//! fan-in round-trips and telemetry delivery + per-aircraft key filtering.
//!
//! A publisher and subscriber on one [`Session`] deliver locally and
//! synchronously (Zenoh invokes same-session subscriber callbacks inline in the
//! `put`), so these are deterministic; the short retry loops just guard against
//! any residual scheduling lag.

use astrum_net::{
    CommandReceiver, CommandSender, Session, TelemetryPublisher, TelemetrySubscriber,
};
use astrum_proto::{CommandMsg, Control, TelemetrySample};

fn sample(id: u32) -> TelemetrySample {
    TelemetrySample {
        aircraft_id: id,
        sim_time_s: 1.0,
        lat_deg: 37.6188,
        lon_deg: -122.3756,
        alt_m: 1524.0,
        enu_e: 1.0,
        enu_n: 2.0,
        enu_u: 3.0,
        vel_north_mps: 50.0,
        vel_east_mps: 0.0,
        vel_down_mps: 0.0,
        phi_rad: 0.0,
        theta_rad: 0.0,
        psi_rad: 0.5,
        airspeed_mps: 51.0,
        health: 0,
        vehicle_kind: astrum_proto::VehicleKind::FixedWing,
    }
}

#[test]
fn command_fan_in_roundtrip() {
    let session = Session::in_process().unwrap();
    let rx = CommandReceiver::bind(&session).unwrap();
    let tx = CommandSender::connect(&session);

    let cmd = CommandMsg {
        target_id: 42,
        control: Control::Raw { aileron: 0.1, elevator: -0.2, rudder: 0.0, throttle: 0.8 },
    };
    tx.send(&cmd).unwrap();

    // Only one command was sent, so retry-draining (without resending) yields
    // exactly one copy once it lands.
    let mut drained = Vec::new();
    for _ in 0..1000 {
        drained = rx.drain();
        if !drained.is_empty() {
            break;
        }
    }
    assert_eq!(drained, vec![cmd]);
}

#[test]
fn telemetry_delivery_roundtrip() {
    let session = Session::in_process().unwrap();
    let pubr = TelemetryPublisher::new(&session);
    let mut sub = TelemetrySubscriber::connect(&session).unwrap();
    sub.subscribe(7).unwrap();

    let s = sample(7);
    let mut got = None;
    for _ in 0..1000 {
        pubr.publish(&s).unwrap();
        if let Some(r) = sub.try_recv().unwrap() {
            got = Some(r);
            break;
        }
    }
    assert_eq!(got, Some(s));
}

#[test]
fn telemetry_key_filter_excludes_others() {
    let session = Session::in_process().unwrap();
    let pubr = TelemetryPublisher::new(&session);
    let mut sub = TelemetrySubscriber::connect(&session).unwrap();
    sub.subscribe(7).unwrap();

    // Publish for a different id; the key filter must exclude it. Establish the
    // subscription first by confirming id 7 gets through.
    let mine = sample(7);
    let mut established = false;
    for _ in 0..1000 {
        pubr.publish(&mine).unwrap();
        if sub.try_recv().unwrap().is_some() {
            established = true;
            break;
        }
    }
    assert!(established, "subscription never established");

    // Drain any residual id-7 frames queued during establishment.
    while sub.try_recv().unwrap().is_some() {}

    // Now a burst for id 9 must not be delivered (key astrum/telemetry/9 does not
    // match the subscription astrum/telemetry/7).
    for _ in 0..50 {
        pubr.publish(&sample(9)).unwrap();
    }
    assert!(sub.try_recv().unwrap().is_none());
}
