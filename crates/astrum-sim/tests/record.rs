//! In-process recording (`astrum-sim --record`): the bag holds every tick of
//! every aircraft — even one whose published telemetry is decimated — plus the
//! commands the sim received, readable by the `astrum-bag` tools.

mod common;

use std::time::Duration;

use astrum_bag::{BagReader, TopicKind};
use astrum_net::{keys, CommandReceiver, CommandSender, Session, TelemetryPublisher};
use astrum_proto::{decode_command, decode_telemetry, Control, RawActuators};
use astrum_sim::net::LastPublished;
use astrum_sim::{build_app, record, RunMode};
use common::{two_ship, DT};

#[test]
fn recording_keeps_every_tick_and_every_received_command() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("run.mcap");
    let mut scenario = two_ship();
    // Aircraft 2 publishes at 30 Hz; its recording must still be every tick.
    scenario.aircraft[1].control_rate_hz = Some(30.0);
    let toml = scenario.to_toml();

    let session = Session::in_process().unwrap();
    let mut app = build_app(scenario, RunMode::Embedded);
    app.insert_non_send(TelemetryPublisher::new(&session));
    app.insert_non_send(CommandReceiver::bind(&session).unwrap());
    let (recorder, recording) = record::start(&path, &toml).unwrap();
    app.insert_resource(recorder);

    let raw =
        Control::Raw(RawActuators { aileron: 0.2, elevator: 0.0, rudder: 0.0, throttle: 0.7 });
    let mut tx = CommandSender::new(&session, 1).unwrap();
    app.update(); // startup (tick 0) + tick 1
    for _ in 0..10 {
        let tick = app.world().resource::<LastPublished>().0[0].tick;
        tx.send(raw, false, tick).unwrap();
        app.update();
    }
    std::thread::sleep(Duration::from_millis(50));
    for _ in 0..109 {
        app.update();
    }
    let final_tick = app.world().resource::<LastPublished>().0[0].tick;
    assert_eq!(final_tick, 120);
    drop(app); // closes the recording queue
    let stats = recording.finish().unwrap();
    assert_eq!((stats.telemetry, stats.commands), (2 * 121, 10));

    let bag = BagReader::open(&path).unwrap();
    assert_eq!(bag.scenario(), Some(toml.as_str()));
    let messages = bag.messages().unwrap();
    for id in [1, 2] {
        let ticks: Vec<u64> = messages
            .iter()
            .filter(|m| TopicKind::of(&m.topic) == Some(TopicKind::Telemetry(id)))
            .map(|m| {
                let t = decode_telemetry(&m.payload).unwrap();
                assert_eq!(m.publish_time_ns, (t.sim_time_s * 1e9).round() as u64);
                assert!((t.sim_time_s - t.tick as f64 * DT).abs() < 1e-9);
                t.tick
            })
            .collect();
        assert_eq!(ticks, (0..=120).collect::<Vec<_>>(), "aircraft {id}: every tick, in order");
    }
    let commands: Vec<_> = messages.iter().filter(|m| m.topic == keys::command(1)).collect();
    assert_eq!(commands.len(), 10);
    for m in commands {
        assert_eq!(decode_command(&m.payload).unwrap().control, raw);
    }
}
