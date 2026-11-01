//! `export_bag` turns an MCAP bag's telemetry and events into the renderer's
//! document.

use astrum_bag::BagWriter;
use astrum_net::keys;
use astrum_proto::{
    encode_command, encode_event, encode_telemetry, Command, Control, ControlMode, EndReason, Enu,
    Event, EventKind, ModeReason, RawActuators, Telemetry, VehicleKind,
};
use astrum_web::export::export_bag;
use astrum_web::{AircraftModel, EventFields, ExportDoc};

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

fn event(tick: u64, kind: EventKind) -> Event {
    Event { tick, sim_time_s: tick as f64 * DT, kind }
}

#[test]
fn export_keeps_every_telemetry_frame_and_event_and_summarises_the_bag() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("run.mcap");
    let scenario = "[world]\norigin_lat_deg = 37.6\n\n\
                    [[aircraft]]\nid = 7\nmodel = \"f450\"\nkind = \"multirotor\"\n\n\
                    [[aircraft]]\nid = 1\nmodel = \"f16\"\nkind = \"fixed-wing\"\n";

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
    // Events in arrival order, except a hit at tick 25 that the recorder logged
    // late; a mode change shares tick 30 with a hit.
    let at = Enu { e: 100.0, n: -50.0, u: 1500.0 };
    for e in [
        event(20, EventKind::TriggerPressed { aircraft_id: 7, ammo: 500 }),
        event(
            30,
            EventKind::Hit {
                shooter_id: 7,
                target_id: 1,
                damage: 2.0,
                hit_points: 98.0,
                pos_enu: at,
            },
        ),
        event(
            30,
            EventKind::ModeChange {
                aircraft_id: 1,
                from: ControlMode::Autopilot,
                to: ControlMode::Failsafe,
                reason: ModeReason::LeaseExpired,
            },
        ),
        event(40, EventKind::Kill { victim_id: 1, killer_id: Some(7), pos_enu: at }),
        event(45, EventKind::Kill { victim_id: 3, killer_id: None, pos_enu: at }),
        event(59, EventKind::RunEnded { reason: EndReason::EngagementDecided, message: "".into() }),
        event(
            25,
            EventKind::Hit {
                shooter_id: 7,
                target_id: 1,
                damage: 2.0,
                hit_points: 100.0,
                pos_enu: at,
            },
        ),
    ] {
        write(&mut w, keys::EVENTS, (e.sim_time_s * 1e9) as u64, &encode_event(&e));
    }
    // Untrusted junk: an undecodable telemetry payload, a payload filed under the
    // wrong id, and an undecodable event.
    write(&mut w, &keys::telemetry(1), 0, b"not capnp");
    let misfiled = telemetry(3, 60, VehicleKind::FixedWing);
    write(&mut w, &keys::telemetry(4), 0, &encode_telemetry(&misfiled));
    write(&mut w, keys::EVENTS, 0, b"not an event");
    w.finish().unwrap();

    let doc = export_bag(&path).unwrap();
    assert_eq!((doc.kind, doc.version), (ExportDoc::KIND, ExportDoc::VERSION));
    assert_eq!(doc.frames.len(), 100);
    assert_eq!(doc.meta.frames, 100);
    assert_eq!(doc.meta.aircraft, vec![1, 7]);
    assert_eq!(doc.meta.rejected, 3);
    assert!((doc.meta.sim_time_start - 10.0 * DT).abs() < 1e-12);
    assert!((doc.meta.sim_time_end - 59.0 * DT).abs() < 1e-12);
    assert_eq!(doc.meta.scenario.as_deref(), Some(scenario));
    assert_eq!(
        doc.meta.models,
        [
            AircraftModel { id: 1, model: "f16".to_owned() },
            AircraftModel { id: 7, model: "f450".to_owned() }
        ]
    );

    let first = &doc.frames[0].frame;
    assert_eq!((first.id, first.tick, first.kind), (1, 10, "fixed-wing"));
    assert_eq!(first.control_mode, "autopilot");
    assert_eq!(doc.frames[1].frame.kind, "multirotor");
    assert!(doc.frames.windows(2).all(|p| p[0].log_time_ns <= p[1].log_time_ns));

    // Sorted by tick; the tick-30 tie keeps bag order (hit, then mode change).
    assert_eq!(doc.meta.events, 7);
    let order: Vec<(u64, &str)> = doc.events.iter().map(|e| (e.tick, e.kind)).collect();
    assert_eq!(
        order,
        [
            (20, "trigger-pressed"),
            (25, "hit"),
            (30, "hit"),
            (30, "mode-change"),
            (40, "kill"),
            (45, "kill"),
            (59, "run-ended"),
        ]
    );
    assert_eq!(
        doc.events[2].fields,
        EventFields::Hit {
            shooter_id: 7,
            target_id: 1,
            damage: 2.0,
            hit_points: 98.0,
            pos_enu: [100.0, -50.0, 1500.0]
        }
    );
    assert!((doc.events[2].sim_time_s - 30.0 * DT).abs() < 1e-12);

    let json: serde_json::Value = serde_json::to_value(&doc).unwrap();
    assert_eq!(json["kind"], "astrum-replay");
    assert_eq!(json["version"], 3);
    assert_eq!(json["meta"]["events"], 7);
    assert_eq!(
        json["meta"]["models"],
        serde_json::json!([{"id": 1, "model": "f16"}, {"id": 7, "model": "f450"}])
    );
    assert_eq!(json["frames"][0]["id"], 1);
    assert!(json["frames"][0]["log_time_ns"].is_u64());
    let events = &json["events"];
    assert_eq!(
        events[0],
        serde_json::json!({"tick": 20, "sim_time_s": 20.0 * DT, "kind": "trigger-pressed",
                           "aircraft_id": 7, "ammo": 500})
    );
    assert_eq!(events[2]["pos_enu"], serde_json::json!([100.0, -50.0, 1500.0]));
    assert_eq!(events[3]["from"], "autopilot");
    assert_eq!(events[3]["to"], "failsafe");
    assert_eq!(events[3]["reason"], "lease-expired");
    assert_eq!(
        (events[4]["victim_id"].as_u64(), events[4]["killer_id"].as_u64()),
        (Some(1), Some(7))
    );
    assert!(events[5].as_object().unwrap()["killer_id"].is_null());
    assert_eq!(events[6]["reason"], "engagement-decided");
    assert_eq!(events[6]["message"], "");
}

#[test]
fn export_of_a_bag_without_telemetry_or_scenario_is_empty() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("empty.mcap");
    BagWriter::create(&path, None).unwrap().finish().unwrap();

    let doc = export_bag(&path).unwrap();
    assert!(doc.frames.is_empty());
    assert!(doc.events.is_empty());
    assert!(doc.meta.aircraft.is_empty());
    assert_eq!((doc.meta.sim_time_start, doc.meta.sim_time_end), (0.0, 0.0));
    assert_eq!(doc.meta.scenario, None);
    assert!(doc.meta.models.is_empty());
}
