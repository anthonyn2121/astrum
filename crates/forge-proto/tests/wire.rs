//! Wire-format integration tests: encode/decode round-trips for telemetry, both
//! command union arms, and the bag frame stream, all through the public helpers.

use forge_proto::{
    decode_command, decode_telemetry, encode_command, encode_telemetry, write_bag_frame, BagReader,
    CommandMsg, Control, TelemetrySample,
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
    }
}

#[test]
fn telemetry_roundtrip() {
    let s = sample(7, 12.5);
    assert_eq!(decode_telemetry(&encode_telemetry(&s)).unwrap(), s);
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
