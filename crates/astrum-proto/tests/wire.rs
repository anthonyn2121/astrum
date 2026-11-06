//! Wire-format behaviour tests: round-trips for every payload and every command
//! arm, plus the decoder's rejection rules.

use astrum_proto::{
    decode_command, decode_telemetry, decode_tick_end, encode_command, encode_telemetry,
    encode_tick_end, AutopilotTargets, Command, CommandStatus, Control, ControlMode, MultirotorRaw,
    RawActuators, Refusal, Telemetry, TickEnd, VehicleKind,
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
