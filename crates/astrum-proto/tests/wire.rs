//! Wire-format behaviour tests: round-trips for every payload and every command
//! arm, plus the decoder's rejection rules.

use astrum_proto::{
    decode_command, decode_event, decode_race_course, decode_race_state, decode_telemetry,
    decode_tick_end, encode_command, encode_event, encode_race_course, encode_race_state,
    encode_telemetry, encode_tick_end, AutopilotTargets, Command, CommandStatus, Control,
    ControlMode, EndReason, Enu, Event, EventKind, MultirotorRaw, RaceCourse, RaceGate,
    RaceOutcome, RaceState, RawActuators, Refusal, Telemetry, TickEnd, VehicleKind,
};

#[test]
fn telemetry_roundtrip_preserves_every_field() {
    let mut t = Telemetry {
        aircraft_id: 7,
        tick: 99,
        sim_time_s: 0.825,
        lat_deg: 37.6,
        lon_deg: -122.4,
        alt_hae_m: 1500.0,
        tas_mps: 120.0,
        vehicle_kind: VehicleKind::Multirotor,
        control_mode: ControlMode::Failsafe,
        control_period_s: 0.05,
        command: CommandStatus {
            owner_epoch: u64::MAX,
            last_seq: 12,
            refusals: 3,
            last_refusal: Refusal::WrongVehicleKind,
            last_refused_epoch: 5,
        },
        ..Default::default()
    };
    t.pos_enu.e = 1.0;
    t.vel_enu.u = -2.0;
    t.vel_ned.d = 2.0;
    t.attitude.phi_rad = 0.3;
    t.rates.q_rad_s = -0.1;
    t.aero.nz_g = 4.5;
    t.combat.alive = true;
    t.combat.ammo = 12;
    assert_eq!(decode_telemetry(&encode_telemetry(&t)).unwrap(), t);
}

#[test]
fn command_roundtrips_every_arm_with_header_fields() {
    for control in [
        Control::Raw(RawActuators { aileron: 0.1, elevator: -0.2, rudder: 0.0, throttle: 0.75 }),
        Control::Autopilot(AutopilotTargets { heading_rad: 1.2, alt_hae_m: 1500.0, tas_mps: 60.0 }),
        Control::Multirotor(MultirotorRaw {
            roll_rate: 0.5,
            pitch_rate: -0.25,
            yaw_rate: 0.125,
            collective: 0.6,
        }),
    ] {
        let c = Command {
            target_id: 3,
            seq: 41,
            telemetry_tick: 9000,
            trigger: true,
            epoch: 0xdead_beef,
            control,
        };
        assert_eq!(decode_command(&encode_command(&c)).unwrap(), c);
    }
}

#[test]
fn tick_end_roundtrips_and_rejects_garbage() {
    let t = TickEnd { tick: 1 << 40, sim_time_s: 12.5 };
    let bytes = encode_tick_end(&t);
    assert_eq!(decode_tick_end(&bytes).unwrap(), t);
    assert!(decode_tick_end(&bytes[..bytes.len() - 8]).is_err());
    let mut long = bytes.clone();
    long.extend_from_slice(&[0u8; 8]);
    assert!(decode_tick_end(&long).is_err());
}

#[test]
fn trailing_bytes_are_rejected() {
    let c = Command {
        target_id: 1,
        seq: 1,
        telemetry_tick: 0,
        trigger: false,
        epoch: 1,
        control: Control::Raw(RawActuators::default()),
    };
    let mut bytes = encode_command(&c);
    bytes.extend_from_slice(&[0u8; 8]);
    assert!(decode_command(&bytes).is_err());
}

#[test]
fn truncated_and_garbage_payloads_are_errors_not_panics() {
    let bytes = encode_telemetry(&Telemetry::default());
    assert!(decode_telemetry(&bytes[..bytes.len() - 8]).is_err());
    assert!(decode_telemetry(&[]).is_err());
    assert!(decode_command(b"definitely not capnp").is_err());
}

#[test]
fn decoding_reads_unaligned_payloads() {
    // Zenoh payload slices carry no alignment guarantee.
    let bytes = encode_telemetry(&Telemetry { aircraft_id: 5, ..Default::default() });
    let mut shifted = vec![0u8; bytes.len() + 1];
    shifted[1..].copy_from_slice(&bytes);
    assert_eq!(decode_telemetry(&shifted[1..]).unwrap().aircraft_id, 5);
}

fn gate(index: u32) -> RaceGate {
    let f = f64::from(index);
    RaceGate {
        index,
        center: Enu { e: f, n: 2.0 * f, u: 1.5 },
        yaw_rad: 0.1 * f,
        normal: Enu { e: 0.0, n: 1.0, u: 0.0 },
        right: Enu { e: 1.0, n: 0.0, u: 0.0 },
        up: Enu { e: 0.0, n: 0.0, u: 1.0 },
    }
}

#[test]
fn race_course_roundtrips_with_any_gate_count() {
    for n in [0, 1, 16, 200] {
        let c = RaceCourse {
            seed: u64::MAX,
            opening_m: 1.016,
            bar_m: 0.0762,
            top_bar_m: 1.1684,
            gates: (0..n).map(gate).collect(),
        };
        let bytes = encode_race_course(&c);
        assert_eq!(decode_race_course(&bytes).unwrap(), c);
        assert!(decode_race_course(&bytes[..bytes.len() - 8]).is_err());
    }
}

#[test]
fn race_state_roundtrips_every_outcome() {
    for outcome in [
        RaceOutcome::Running,
        RaceOutcome::Complete,
        RaceOutcome::CrashedGround,
        RaceOutcome::CrashedGate,
        RaceOutcome::TimedOut,
        RaceOutcome::SpawnFailed,
    ] {
        let s = RaceState {
            aircraft_id: 3,
            pending_gate: 4,
            passed: 4,
            missed: 1,
            gate_count: 9,
            last_pass_s: 6.5,
            sim_time_s: 7.0,
            finished: outcome != RaceOutcome::Running,
            outcome,
        };
        assert_eq!(decode_race_state(&encode_race_state(&s)).unwrap(), s);
    }
}

#[test]
fn race_events_and_end_reasons_roundtrip() {
    let kinds = [
        EventKind::GatePassed { aircraft_id: 1, gate: 2, offset_x_m: -0.25, offset_y_m: 0.5 },
        EventKind::GateMissed { aircraft_id: 1, gate: 3 },
        EventKind::Crashed { aircraft_id: 1, outcome: RaceOutcome::CrashedGround, gate: 3 },
        EventKind::RunEnded { reason: EndReason::RaceComplete, message: String::new() },
        EventKind::RunEnded { reason: EndReason::Crashed, message: String::new() },
    ];
    for kind in kinds {
        let e = Event { tick: 7, sim_time_s: 0.5, kind };
        assert_eq!(decode_event(&encode_event(&e)).unwrap(), e);
    }
}

#[test]
fn an_event_kind_this_build_does_not_know_is_a_decode_error_to_skip() {
    // Event = tick, simTimeS, then the 16-bit union discriminant in the third data word,
    // after the 8-byte segment table and the 8-byte root pointer.
    const DISCRIMINANT: usize = 8 + 8 + 16;
    let e =
        Event { tick: 1, sim_time_s: 0.0, kind: EventKind::GateMissed { aircraft_id: 1, gate: 0 } };
    let mut bytes = encode_event(&e);
    // A known discriminant (controllerUp, member 5) proves the offset: it decodes as that kind.
    bytes[DISCRIMINANT..DISCRIMINANT + 2].copy_from_slice(&5u16.to_le_bytes());
    assert!(matches!(decode_event(&bytes).unwrap().kind, EventKind::ControllerUp { .. }));
    // An ordinal past the schema's last kind is an error (a subscriber skips the event).
    bytes[DISCRIMINANT..DISCRIMINANT + 2].copy_from_slice(&99u16.to_le_bytes());
    assert!(decode_event(&bytes).is_err());
}

#[test]
fn an_outcome_this_build_does_not_know_decodes_as_unknown() {
    let s = RaceState { outcome: RaceOutcome::Complete, ..Default::default() };
    let mut bytes = encode_race_state(&s);
    // Locate the outcome's byte as the one that differs between two outcomes.
    let probe = encode_race_state(&RaceState { outcome: RaceOutcome::SpawnFailed, ..s });
    let at = bytes.iter().zip(&probe).position(|(a, b)| a != b).unwrap();
    bytes[at] = 99;
    assert_eq!(decode_race_state(&bytes).unwrap().outcome, RaceOutcome::Unknown(99));
}
