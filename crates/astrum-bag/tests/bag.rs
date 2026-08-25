//! Bag record/replay integration test: append frames with [`BagWriter`] and read
//! them back with [`BagReader`], confirming a bag round-trips losslessly.

use astrum_bag::{BagReader, BagWriter};
use astrum_proto::TelemetrySample;

fn sample(id: u32, t: f64) -> TelemetrySample {
    TelemetrySample {
        aircraft_id: id,
        sim_time_s: t,
        lat_deg: 37.6188,
        lon_deg: -122.3756,
        alt_m: 1524.0,
        enu_e: t,
        enu_n: 2.0 * t,
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
fn write_then_read_roundtrips() {
    let mut buf = Vec::new();
    {
        let mut bag = BagWriter::new(&mut buf);
        for i in 0..4u32 {
            bag.write_frame(10_000 + i as u64, &sample(i, i as f64)).unwrap();
        }
        assert_eq!(bag.frames(), 4);
    }

    let mut reader = BagReader::new(std::io::Cursor::new(buf));
    let mut got = Vec::new();
    while let Some(frame) = reader.next().unwrap() {
        got.push(frame);
    }
    assert_eq!(got.len(), 4);
    assert_eq!(got[2].0, 10_002);
    assert_eq!(got[2].1.aircraft_id, 2);
    assert_eq!(got[3].1.sim_time_s, 3.0);
}
