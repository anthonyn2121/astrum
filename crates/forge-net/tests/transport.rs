//! ZMQ transport integration tests over an inproc context: command PUSH/PULL
//! round-trips and telemetry PUB/SUB delivery + per-aircraft topic filtering.

use forge_net::{context, CommandReceiver, CommandSender, TelemetryPublisher, TelemetrySubscriber};
use forge_proto::{CommandMsg, Control, TelemetrySample};

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
        vehicle_kind: forge_proto::VehicleKind::FixedWing,
    }
}

#[test]
fn command_push_pull_roundtrip() {
    // PUSH/PULL has no slow-joiner problem, so this is deterministic.
    let ctx = context();
    let rx = CommandReceiver::bind(&ctx, "inproc://cmd-test").unwrap();
    let tx = CommandSender::connect(&ctx, "inproc://cmd-test").unwrap();

    let cmd = CommandMsg {
        target_id: 42,
        control: Control::Raw { aileron: 0.1, elevator: -0.2, rudder: 0.0, throttle: 0.8 },
    };
    tx.send(&cmd).unwrap();

    // Give the transfer a moment, then drain (retry a few times to dodge
    // any inproc scheduling lag without a fixed sleep).
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
fn telemetry_pub_sub_roundtrip() {
    // PUB/SUB has a slow-joiner race: the subscriber must finish connecting
    // and subscribing before a published message is retained. We republish
    // in a bounded loop until the first sample arrives.
    let ctx = context();
    let pubr = TelemetryPublisher::bind(&ctx, "inproc://tlm-test").unwrap();
    let sub = TelemetrySubscriber::connect(&ctx, "inproc://tlm-test").unwrap();
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
fn telemetry_topic_filter_excludes_others() {
    let ctx = context();
    let pubr = TelemetryPublisher::bind(&ctx, "inproc://tlm-filter").unwrap();
    let sub = TelemetrySubscriber::connect(&ctx, "inproc://tlm-filter").unwrap();
    sub.subscribe(7).unwrap();

    // Publish for a different id; the filter must exclude it. Establish the
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

    // Now a burst for id 9 must not be delivered.
    for _ in 0..50 {
        pubr.publish(&sample(9)).unwrap();
    }
    assert!(sub.try_recv().unwrap().is_none());
}
