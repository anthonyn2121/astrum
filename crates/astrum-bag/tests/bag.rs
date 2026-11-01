//! Behaviour of the MCAP bag: file round trip, recording from a live session,
//! replay pacing, and damaged-file handling.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use astrum_bag::{
    replay, replay_schedule, BagError, BagMessage, BagReader, BagWriter, RecordStats, Recorder,
    COMMAND_SCHEMA, MAX_REPLAY_GAP, SCHEMA_TEXT, TELEMETRY_SCHEMA,
};
use astrum_net::zenoh::Wait;
use astrum_net::{keys, CommandSender, Session, TelemetryPublisher};
use astrum_proto::{
    decode_command, encode_telemetry, AutopilotTargets, Control, RawActuators, Telemetry,
};

const SCENARIO: &str =
    "[world]\norigin_lat_deg = 37.6188\n\n[[aircraft]]\nid = 1\nmodel = \"f16\"\n";

fn temp_dir() -> tempfile::TempDir {
    tempfile::tempdir().expect("temp dir")
}

fn bag_path(dir: &tempfile::TempDir, name: &str) -> PathBuf {
    dir.path().join(name)
}

fn telemetry(id: u32, tick: u64) -> Telemetry {
    Telemetry {
        aircraft_id: id,
        tick,
        sim_time_s: tick as f64 / 120.0,
        lat_deg: 37.6188,
        lon_deg: -122.3756,
        alt_hae_m: 1524.0 + tick as f64,
        tas_mps: 55.0,
        ..Telemetry::default()
    }
}

fn msg(topic: &str, log: u64, publish: u64, payload: &[u8]) -> BagMessage {
    BagMessage {
        topic: topic.to_owned(),
        log_time_ns: log,
        publish_time_ns: publish,
        payload: payload.to_vec(),
    }
}

fn write_bag(path: &Path, scenario: Option<&str>, messages: &[BagMessage]) {
    let mut bag = BagWriter::create(path, scenario).expect("create bag");
    for m in messages {
        bag.write(&m.topic, m.log_time_ns, m.publish_time_ns, &m.payload).expect("write");
    }
    bag.finish().expect("finish");
}

#[test]
fn roundtrip_preserves_topics_times_bytes_order_and_scenario() {
    let dir = temp_dir();
    let path = bag_path(&dir, "roundtrip.mcap");
    let t1 = encode_telemetry(&telemetry(1, 10));
    let t2 = encode_telemetry(&telemetry(2, 10));
    // Written slightly out of log-time order, with a tie; payloads are opaque
    // bytes (the writer never re-encodes), including empty and non-capnp ones.
    let written = vec![
        msg("astrum/telemetry/1", 1_000, 83_333_333, &t1),
        msg("astrum/telemetry/2", 1_000, 83_333_333, &t2),
        msg("astrum/command/1", 3_000, 3_000, &[0xde, 0xad, 0xbe, 0xef]),
        msg("astrum/command/2", 2_000, 2_000, &[]),
        msg("astrum/telemetry/1", 4_000, u64::MAX, &[7; 300]),
    ];
    write_bag(&path, Some(SCENARIO), &written);

    let reader = BagReader::open(&path).expect("open");
    assert_eq!(reader.scenario(), Some(SCENARIO));
    let read = reader.messages().expect("messages");
    let expected = vec![
        written[0].clone(),
        written[1].clone(),
        written[3].clone(),
        written[2].clone(),
        written[4].clone(),
    ];
    assert_eq!(read, expected);
}

#[test]
fn channels_carry_the_capnp_schema_text() {
    let dir = temp_dir();
    let path = bag_path(&dir, "schema.mcap");
    write_bag(
        &path,
        None,
        &[msg("astrum/telemetry/3", 1, 1, &[1]), msg("astrum/command/3", 2, 2, &[2])],
    );

    let bytes = std::fs::read(&path).expect("read");
    let mut seen = Vec::new();
    for m in mcap::MessageStream::new(&bytes).expect("stream") {
        let m = m.expect("message");
        let schema = m.channel.schema.as_ref().expect("schema");
        assert_eq!(m.channel.message_encoding, "capnp");
        assert_eq!(schema.encoding, "capnp");
        assert_eq!(&schema.data[..], SCHEMA_TEXT.as_bytes());
        seen.push((m.channel.topic.clone(), schema.name.clone()));
    }
    assert_eq!(
        seen,
        vec![
            ("astrum/telemetry/3".to_owned(), TELEMETRY_SCHEMA.to_owned()),
            ("astrum/command/3".to_owned(), COMMAND_SCHEMA.to_owned()),
        ]
    );
    assert_eq!(BagReader::open(&path).expect("open").scenario(), None);
}

#[test]
fn writer_rejects_non_astrum_topics() {
    let dir = temp_dir();
    let mut bag = BagWriter::create(&bag_path(&dir, "topics.mcap"), None).expect("create");
    for topic in ["astrum/scenario", "astrum/telemetry/x", "astrum/telemetry", "other/1"] {
        assert!(
            matches!(bag.write(topic, 1, 1, &[1]), Err(BagError::UnknownTopic(t)) if t == topic),
            "{topic}"
        );
    }
    bag.finish().expect("finish");
}

#[test]
fn truncated_or_garbage_files_are_errors() {
    let dir = temp_dir();
    let good = bag_path(&dir, "good.mcap");
    let payload = encode_telemetry(&telemetry(1, 1));
    let messages: Vec<_> = (0..200).map(|i| msg("astrum/telemetry/1", i, i, &payload)).collect();
    write_bag(&good, Some(SCENARIO), &messages);
    let bytes = std::fs::read(&good).expect("read");

    let damaged = bag_path(&dir, "damaged.mcap");
    for len in [0, 4, 8, bytes.len() / 3, bytes.len() / 2, bytes.len() - 9, bytes.len() - 1] {
        std::fs::write(&damaged, &bytes[..len]).expect("write");
        let result = BagReader::open(&damaged).and_then(|r| r.messages());
        assert!(result.is_err(), "truncated to {len} of {} bytes was accepted", bytes.len());
    }
    std::fs::write(&damaged, b"definitely not an mcap file").expect("write");
    assert!(BagReader::open(&damaged).is_err());
    assert!(matches!(BagReader::open(&bag_path(&dir, "missing.mcap")), Err(BagError::Io(_))));
}

/// Run `recorder` until `done(stats)` or a generous deadline.
fn record_until(
    recorder: Recorder,
    bag: &mut BagWriter,
    done: impl Fn(&RecordStats) -> bool,
) -> RecordStats {
    let deadline = Instant::now() + Duration::from_secs(10);
    recorder.run(bag, |s| done(s) || Instant::now() >= deadline).expect("record")
}

#[test]
fn recorder_captures_every_telemetry_and_command_sample() {
    const TICKS: u64 = 300;
    let session = Session::in_process().expect("session");
    let recorder = Recorder::subscribe(&session).expect("subscribe");

    let publishing = {
        let session = session.clone();
        std::thread::spawn(move || {
            let mut telemetry_pub = TelemetryPublisher::new(&session);
            let mut cmd1 = CommandSender::new(&session, 1).expect("sender 1");
            let mut cmd2 = CommandSender::new(&session, 2).expect("sender 2");
            for tick in 1..=TICKS {
                telemetry_pub.publish(&telemetry(1, tick)).expect("publish 1");
                telemetry_pub.publish(&telemetry(2, tick)).expect("publish 2");
                let raw = Control::Raw(RawActuators { throttle: 0.5, ..RawActuators::default() });
                cmd1.send(raw, false, tick).expect("cmd 1");
                let ap = Control::Autopilot(AutopilotTargets {
                    heading_rad: 1.0,
                    alt_hae_m: 1500.0,
                    tas_mps: 60.0,
                });
                cmd2.send(ap, tick % 2 == 0, tick).expect("cmd 2");
            }
            // An undecodable telemetry payload is counted and skipped.
            session.raw().put(keys::telemetry(9), vec![1u8, 2, 3]).wait().expect("garbage");
        })
    };

    let dir = temp_dir();
    let path = bag_path(&dir, "record.mcap");
    let mut bag = BagWriter::create(&path, Some(SCENARIO)).expect("create");
    let stats = record_until(recorder, &mut bag, |s| {
        s.telemetry == 2 * TICKS && s.commands == 2 * TICKS && s.rejected == 1
    });
    publishing.join().expect("publisher thread");
    bag.finish().expect("finish");
    assert_eq!(stats, RecordStats { telemetry: 2 * TICKS, commands: 2 * TICKS, rejected: 1 });

    let reader = BagReader::open(&path).expect("open");
    assert_eq!(reader.scenario(), Some(SCENARIO));
    let messages = reader.messages().expect("messages");
    assert!(messages.windows(2).all(|w| w[0].log_time_ns <= w[1].log_time_ns));
    for id in [1u32, 2] {
        let topic = keys::telemetry(id);
        let recorded: Vec<_> = messages.iter().filter(|m| m.topic == topic).collect();
        assert_eq!(recorded.len() as u64, TICKS);
        for (m, tick) in recorded.iter().zip(1..) {
            let sent = telemetry(id, tick);
            assert_eq!(m.payload, encode_telemetry(&sent), "{topic} tick {tick}");
            assert_eq!(m.publish_time_ns, (sent.sim_time_s * 1e9).round() as u64);
        }

        let topic = keys::command(id);
        let commands: Vec<_> = messages.iter().filter(|m| m.topic == topic).collect();
        let seqs: Vec<u64> =
            commands.iter().map(|m| decode_command(&m.payload).expect("command").seq).collect();
        assert_eq!(seqs, (1..=TICKS).collect::<Vec<_>>());
        assert!(commands.iter().all(|m| m.publish_time_ns == m.log_time_ns));
    }
    assert!(messages.iter().all(|m| m.topic != keys::telemetry(9)));
}

#[test]
fn replay_republishes_recorded_telemetry_bytes_but_not_commands() {
    let dir = temp_dir();
    let source = bag_path(&dir, "source.mcap");
    let mut written = Vec::new();
    for tick in 1..=50u64 {
        for id in [4u32, 5] {
            let t = telemetry(id, tick);
            written.push(msg(
                &keys::telemetry(id),
                tick * 10,
                (t.sim_time_s * 1e9).round() as u64,
                &encode_telemetry(&t),
            ));
        }
        written.push(msg(&keys::command(4), tick * 10 + 1, tick * 10 + 1, &[9, 9]));
    }
    write_bag(&source, Some(SCENARIO), &written);
    let bag = BagReader::open(&source).expect("open");

    let session = Session::in_process().expect("session");
    let recorder = Recorder::subscribe(&session).expect("subscribe");
    let stats = replay(&session, &bag, 0.0).expect("replay");
    assert_eq!((stats.telemetry, stats.commands, stats.skipped), (100, 50, 0));

    let copy = bag_path(&dir, "copy.mcap");
    let mut out = BagWriter::create(&copy, None).expect("create");
    let recorded = record_until(recorder, &mut out, |s| s.telemetry == 100);
    out.finish().expect("finish");
    assert_eq!(recorded, RecordStats { telemetry: 100, commands: 0, rejected: 0 });

    let replayed = BagReader::open(&copy).expect("open copy").messages().expect("messages");
    let sent: Vec<_> =
        written.iter().filter(|m| m.topic.starts_with(keys::TELEMETRY_PREFIX)).collect();
    assert_eq!(replayed.len(), sent.len());
    for (got, want) in replayed.iter().zip(sent) {
        assert_eq!(
            (&got.topic, &got.payload, got.publish_time_ns),
            (&want.topic, &want.payload, want.publish_time_ns)
        );
    }
}

const S: u64 = 1_000_000_000;

fn secs(v: &[f64]) -> Vec<Duration> {
    v.iter().map(|&s| Duration::from_secs_f64(s)).collect()
}

#[test]
fn schedule_follows_sim_time_scaled_by_rate() {
    let times = [10 * S, 10 * S, 11 * S, 13 * S, 13 * S + S / 2];
    assert_eq!(replay_schedule(&times, 1.0), secs(&[0.0, 0.0, 1.0, 3.0, 3.5]));
    assert_eq!(replay_schedule(&times, 2.0), secs(&[0.0, 0.0, 0.5, 1.5, 1.75]));
    assert_eq!(replay_schedule(&times, 0.5), secs(&[0.0, 0.0, 2.0, 6.0, 7.0]));
    assert!(replay_schedule(&[], 1.0).is_empty());
}

#[test]
fn schedule_caps_each_gap_at_five_seconds() {
    let times = [0, 60 * S, 61 * S, 1_000 * S];
    assert_eq!(
        replay_schedule(&times, 1.0),
        vec![
            Duration::ZERO,
            MAX_REPLAY_GAP,
            MAX_REPLAY_GAP + Duration::from_secs(1),
            2 * MAX_REPLAY_GAP + Duration::from_secs(1)
        ]
    );
    // The cap applies after scaling: 8 s of sim time at 2x is a 4 s pause.
    assert_eq!(replay_schedule(&[0, 8 * S, 28 * S], 2.0), secs(&[0.0, 4.0, 9.0]));
    // A vanishingly slow rate saturates at the cap instead of overflowing.
    assert_eq!(replay_schedule(&[0, S], 1e-300), vec![Duration::ZERO, MAX_REPLAY_GAP]);
}

#[test]
fn schedule_treats_backwards_time_as_no_gap() {
    // A sim restart mid-recording: time jumps back, then pacing resumes.
    assert_eq!(replay_schedule(&[5 * S, 6 * S, S, 2 * S], 1.0), secs(&[0.0, 1.0, 1.0, 2.0]));
}

#[test]
fn non_positive_rate_is_unpaced() {
    let times = [0, S, 100 * S];
    for rate in [0.0, -1.0, f64::NAN] {
        assert_eq!(replay_schedule(&times, rate), vec![Duration::ZERO; 3], "rate {rate}");
    }
}
