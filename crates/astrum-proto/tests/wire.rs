//! Wire-format integration tests: encode/decode round-trips for telemetry, both
//! command union arms, the racing course/progress messages, and the bag frame
//! stream, all through the public helpers.

use astrum_proto::{
    decode_command, decode_race_course, decode_race_state, decode_telemetry, encode_command,
    encode_race_course, encode_race_state, encode_telemetry, write_bag_frame, BagReader,
    CommandMsg, Control, RaceCourseMsg, RaceGateSpec, RaceOutcome, RaceStateMsg, TelemetrySample,
    VehicleKind,
};

fn sample(id: u32, t: f64) -> TelemetrySample {
    TelemetrySample {
        aircraft_id: id,
        sim_time_s: t,
        lat_deg: 37.6188,
        lon_deg: -122.3756,
        alt_m: 1524.0,
        enu_e: 12.5,
        enu_n: -34.0,
        enu_u: 3.25,
        vel_north_mps: 50.0,
        vel_east_mps: 1.0,
        vel_down_mps: -0.5,
        phi_rad: 0.1,
        theta_rad: 0.02,
        psi_rad: 1.57,
        airspeed_mps: 51.4,
        health: 0,
        vehicle_kind: VehicleKind::FixedWing,
    }
}

#[test]
fn telemetry_roundtrip() {
    let s = sample(7, 12.5);
    assert_eq!(decode_telemetry(&encode_telemetry(&s)).unwrap(), s);
}

#[test]
fn telemetry_vehicle_kind_roundtrip() {
    // Both classes survive the wire, and it defaults to fixed-wing.
    let mut s = sample(2, 3.0);
    s.vehicle_kind = VehicleKind::Multirotor;
    assert_eq!(
        decode_telemetry(&encode_telemetry(&s)).unwrap().vehicle_kind,
        VehicleKind::Multirotor
    );
    assert_eq!(VehicleKind::default(), VehicleKind::FixedWing);
}

#[test]
fn command_raw_roundtrip() {
    let c = CommandMsg {
        target_id: 3,
        control: Control::Raw { aileron: 0.1, elevator: -0.2, rudder: 0.0, throttle: 0.75 },
    };
    assert_eq!(decode_command(&encode_command(&c)).unwrap(), c);
}

#[test]
fn command_autopilot_roundtrip() {
    let c = CommandMsg {
        target_id: 9,
        control: Control::Autopilot { heading_rad: 1.2, alt_msl_m: 1500.0, airspeed_mps: 60.0 },
    };
    assert_eq!(decode_command(&encode_command(&c)).unwrap(), c);
}

fn course(gates: u32) -> RaceCourseMsg {
    RaceCourseMsg {
        drone_id: 1,
        seed: 42,
        opening_m: 0.635,
        bar_m: 0.0762,
        gates: (0..gates)
            .map(|i| RaceGateSpec {
                index: i,
                center_e: i as f64 * 1.5,
                center_n: i as f64 * -12.0,
                center_u: 2.0 + i as f64 * 0.25,
                yaw_rad: i as f64 * 0.3,
            })
            .collect(),
    }
}

#[test]
fn race_course_roundtrip() {
    let c = course(8);
    assert_eq!(decode_race_course(&encode_race_course(&c)).unwrap(), c);
}

#[test]
fn race_course_preserves_gate_order() {
    // Gate order *is* the course — a list that round-trips shuffled would be a
    // silently different track, so check the sequence, not just the set.
    let decoded = decode_race_course(&encode_race_course(&course(12))).unwrap();
    let indices: Vec<u32> = decoded.gates.iter().map(|g| g.index).collect();
    assert_eq!(indices, (0..12).collect::<Vec<u32>>());
    assert_eq!(decoded.gates[5].center_n, -60.0);
}

#[test]
fn race_course_with_no_gates_roundtrips() {
    // A degenerate course must not be an encode/decode error: an empty capnp list
    // is legal, and the sampler's own emptiness checks are elsewhere.
    let c = RaceCourseMsg { drone_id: 3, seed: 0, opening_m: 0.635, bar_m: 0.0762, gates: vec![] };
    assert_eq!(decode_race_course(&encode_race_course(&c)).unwrap(), c);
}

#[test]
fn race_state_roundtrip() {
    let s = RaceStateMsg {
        drone_id: 1,
        sim_time_s: 12.75,
        pending_gate: 3,
        passed: 3,
        missed: 1,
        gate_count: 8,
        last_pass_s: 11.5,
        finished: false,
        outcome: RaceOutcome::Running,
    };
    assert_eq!(decode_race_state(&encode_race_state(&s)).unwrap(), s);
}

#[test]
fn race_state_carries_the_finish() {
    let s = RaceStateMsg {
        drone_id: 1,
        sim_time_s: 40.0,
        pending_gate: 8,
        passed: 8,
        missed: 0,
        gate_count: 8,
        last_pass_s: 39.5,
        finished: true,
        outcome: RaceOutcome::Complete,
    };
    let got = decode_race_state(&encode_race_state(&s)).unwrap();
    assert!(got.finished);
    assert_eq!(got.outcome, RaceOutcome::Complete);
    // The documented invariant: a finished run's pending index is one past the end.
    assert_eq!(got.pending_gate, got.gate_count);
}

/// Every outcome survives the wire, and the default is the one an unset field reads
/// as. That second half is the compatibility guarantee: `running` is ordinal 0, so a
/// `RaceState` written before the field existed decodes as "still going" rather than
/// as a spurious crash.
#[test]
fn every_race_outcome_roundtrips() {
    for outcome in [
        RaceOutcome::Running,
        RaceOutcome::Complete,
        RaceOutcome::CrashedGround,
        RaceOutcome::CrashedGate,
        RaceOutcome::TimedOut,
        RaceOutcome::SpawnFailed,
    ] {
        let s = RaceStateMsg { drone_id: 1, outcome, finished: true, ..Default::default() };
        let got = decode_race_state(&encode_race_state(&s)).unwrap();
        assert_eq!(got.outcome, outcome, "{outcome:?} did not survive the wire");
    }
    assert_eq!(RaceOutcome::default(), RaceOutcome::Running);
    let bare = RaceStateMsg { drone_id: 1, ..Default::default() };
    assert_eq!(decode_race_state(&encode_race_state(&bare)).unwrap().outcome, RaceOutcome::Running);
}

#[test]
fn bag_stream_roundtrip() {
    let mut buf = Vec::new();
    for i in 0..5u32 {
        write_bag_frame(&mut buf, 1000 + i as u64, &sample(i, i as f64)).unwrap();
    }
    let mut rd = BagReader::new(std::io::Cursor::new(buf));
    let mut got = Vec::new();
    while let Some((ns, s)) = rd.next().unwrap() {
        got.push((ns, s));
    }
    assert_eq!(got.len(), 5);
    assert_eq!(got[3].0, 1003);
    assert_eq!(got[3].1.aircraft_id, 3);
    assert_eq!(got[4].1.sim_time_s, 4.0);
}
