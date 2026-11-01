//! `export_bag` turns an MCAP bag's telemetry into the renderer's document.

use astrum_bag::BagWriter;
use astrum_net::keys;
use astrum_proto::{
    encode_command, encode_telemetry, Command, Control, ControlMode, RawActuators, Telemetry,
    VehicleKind,
};
use astrum_web::export::export_bag;
use astrum_web::ExportDoc;

const DT: f64 = 1.0 / 120.0;

fn telemetry(id: u32, tick: u64, kind: VehicleKind) -> Telemetry {
    Telemetry {
        aircraft_id: id,
        tick,
        sim_time_s: tick as f64 * DT,
        vehicle_kind: kind,
        control_mode: ControlMode::Autopilot,
        ..Telemetry::default()
    }
}

#[test]
fn export_keeps_every_telemetry_frame_and_summarises_the_bag() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("run.mcap");
    let scenario = "[world]\norigin_lat_deg = 37.6\n";

    let mut w = BagWriter::create(&path, Some(scenario)).unwrap();
    let mut log_ns = 1_700_000_000_000_000_000u64;
    let mut write = |w: &mut BagWriter, topic: &str, publish_ns: u64, payload: &[u8]| {
        log_ns += 1_000;
        w.write(topic, log_ns, publish_ns, payload).unwrap();
    };
    // Ticks 10..=59 for aircraft 1 (fixed-wing) and 7 (multirotor).
    for tick in 10..60u64 {
        for (id, kind) in [(1, VehicleKind::FixedWing), (7, VehicleKind::Multirotor)] {
            let t = telemetry(id, tick, kind);
            write(&mut w, &keys::telemetry(id), (t.sim_time_s * 1e9) as u64, &encode_telemetry(&t));
        }
        let cmd = Command {
            target_id: 1,
            seq: tick,
            telemetry_tick: tick,
            trigger: false,
            control: Control::Raw(RawActuators::default()),
        };
        write(&mut w, &keys::command(1), tick, &encode_command(&cmd));
    }
    // Untrusted junk: an undecodable payload and a payload filed under the wrong id.
    write(&mut w, &keys::telemetry(1), 0, b"not capnp");
    let misfiled = telemetry(3, 60, VehicleKind::FixedWing);
    write(&mut w, &keys::telemetry(4), 0, &encode_telemetry(&misfiled));
    w.finish().unwrap();

    let doc = export_bag(&path).unwrap();
    assert_eq!((doc.kind, doc.version), (ExportDoc::KIND, ExportDoc::VERSION));
    assert_eq!(doc.frames.len(), 100);
    assert_eq!(doc.meta.frames, 100);
    assert_eq!(doc.meta.aircraft, vec![1, 7]);
    assert_eq!(doc.meta.rejected, 2);
    assert!((doc.meta.sim_time_start - 10.0 * DT).abs() < 1e-12);
    assert!((doc.meta.sim_time_end - 59.0 * DT).abs() < 1e-12);
    assert_eq!(doc.meta.scenario.as_deref(), Some(scenario));

    let first = &doc.frames[0].frame;
    assert_eq!((first.id, first.tick, first.kind), (1, 10, "fixed-wing"));
    assert_eq!(first.control_mode, "autopilot");
    assert_eq!(doc.frames[1].frame.kind, "multirotor");
    assert!(doc.frames.windows(2).all(|p| p[0].log_time_ns <= p[1].log_time_ns));

    let json: serde_json::Value = serde_json::to_value(&doc).unwrap();
    assert_eq!(json["kind"], "astrum-replay");
    assert_eq!(json["version"], 2);
    assert_eq!(json["frames"][0]["id"], 1);
    assert!(json["frames"][0]["log_time_ns"].is_u64());
}

#[test]
fn export_of_a_bag_without_telemetry_or_scenario_is_empty() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("empty.mcap");
    BagWriter::create(&path, None).unwrap().finish().unwrap();

    let doc = export_bag(&path).unwrap();
    assert!(doc.frames.is_empty());
    assert!(doc.meta.aircraft.is_empty());
    assert_eq!((doc.meta.sim_time_start, doc.meta.sim_time_end), (0.0, 0.0));
    assert_eq!(doc.meta.scenario, None);
}
