//! The event stream end to end: a gun pass between two F-16s reports
//! attributed hits and the kill, lined up with the telemetry and identical on
//! every run; mode changes carry their reason; and every event the sim reports
//! reaches `astrum/events` and the `--record` bag unchanged.

mod common;

use std::time::Duration;

use astrum_bag::{BagReader, TopicKind};
use astrum_net::{CommandSender, EventReceiver, Session};
use astrum_proto::{
    decode_event, Control, ControlMode, EndReason, Event, EventKind, ModeReason, Telemetry,
};
use astrum_sim::components::SimClock;
use astrum_sim::env::{Action, SimEnv};
use astrum_sim::events::EventLog;
use astrum_sim::net::LastPublished;
use astrum_sim::scenario::Scenario;
use astrum_sim::{build_app, net, record, RunMode};
use common::{autopilot, of, raw, scenario, single_ship, spec, DT, ORIGIN};

/// Two F-16s on opposing teams in trail, 200 m apart at 250 m/s: close enough
/// that the chaser's boresight (a few degrees of alpha above its flight path)
/// still passes within the hit radius of the lead. The lead dies after 5 hits.
fn duel() -> Scenario {
    let mut lead = spec(1, "f16", ORIGIN.0 + 0.0018, 0.0, 4572.0, 250.0);
    lead.team = 1;
    lead.health = 10.0;
    let chaser = spec(2, "f16", ORIGIN.0, 0.0, 4572.0, 250.0);
    scenario(vec![lead, chaser])
}

fn hold() -> Control {
    autopilot(0.0, 4572.0, 250.0).control
}

/// Fly the duel with the chaser firing until the lead dies; every step's
/// events with that step's telemetry.
fn gun_pass() -> Vec<(Vec<Event>, Vec<Telemetry>)> {
    let mut env = SimEnv::new(duel()).unwrap();
    let mut steps = Vec::new();
    for _ in 0..600 {
        let fire = Action { control: hold(), trigger: true };
        let (obs, r) =
            env.step(&[(2, fire), (1, Action { control: hold(), trigger: false })]).unwrap();
        steps.push((env.events().to_vec(), obs));
        if r.terminated {
            return steps;
        }
    }
    panic!("the chaser never killed the lead");
}

#[test]
fn a_gun_pass_reports_attributed_hits_then_the_kill_identically_every_run() {
    let run = gun_pass();
    let events: Vec<&Event> = run.iter().flat_map(|(e, _)| e).collect();
    let kinds: Vec<&EventKind> = events.iter().map(|e| &e.kind).collect();

    // The first step: both pilots take their commands, the chaser presses.
    let autopilot = |id| EventKind::ModeChange {
        aircraft_id: id,
        from: ControlMode::Uncommanded,
        to: ControlMode::Autopilot,
        reason: ModeReason::Command,
    };
    assert_eq!(
        kinds[..3],
        [&autopilot(1), &autopilot(2), &EventKind::TriggerPressed { aircraft_id: 2, ammo: 511 }]
    );

    let hits: Vec<(u32, u32, f32)> = kinds
        .iter()
        .filter_map(|k| match k {
            EventKind::Hit { shooter_id, target_id, hit_points, .. } => {
                Some((*shooter_id, *target_id, *hit_points))
            }
            _ => None,
        })
        .collect();
    assert!(hits.len() >= 5, "{hits:?}");
    assert!(
        hits.iter().all(|&(s, t, _)| (s, t) == (2, 1)),
        "every hit is the chaser's on the lead"
    );
    for (i, &(_, _, hp)) in hits.iter().enumerate() {
        assert_eq!(hp, 10.0 - 2.0 * (i + 1) as f32, "hit {i} leaves the lead its remaining HP");
    }

    let kills: Vec<&Event> =
        events.iter().copied().filter(|e| matches!(e.kind, EventKind::Kill { .. })).collect();
    assert_eq!(kills.len(), 1);
    let EventKind::Kill { victim_id, killer_id, pos_enu } = kills[0].kind else { unreachable!() };
    assert_eq!((victim_id, killer_id), (1, Some(2)), "the lead, killed by the chaser");
    assert_eq!(kinds.last(), Some(&&kills[0].kind), "nothing follows the kill on its tick");

    // Events line up with telemetry: the kill's tick is the first frame that
    // shows the lead dead, at the position the event reports.
    let (_, obs) = run.iter().find(|(e, _)| e.iter().any(|e| e.tick == kills[0].tick)).unwrap();
    let lead = of(obs, 1);
    assert_eq!((lead.tick, lead.combat.alive), (kills[0].tick, false));
    assert_eq!(lead.pos_enu, pos_enu);
    let before = run.iter().rev().find(|(_, o)| of(o, 1).tick < kills[0].tick).unwrap();
    assert!(of(&before.1, 1).combat.alive);

    // Hits are where the rounds struck: within the hit radius of the lead at
    // the moment of contact, i.e. of its end-of-tick position plus one tick of
    // its own motion.
    for e in &events {
        if let EventKind::Hit { pos_enu: p, .. } = e.kind {
            let (_, o) = run.iter().find(|(_, o)| of(o, 1).tick == e.tick).unwrap();
            let (t, v) = (of(o, 1).pos_enu, of(o, 1).vel_enu);
            let d = ((p.e - t.e).powi(2) + (p.n - t.n).powi(2) + (p.u - t.u).powi(2)).sqrt();
            let step = (v.e.powi(2) + v.n.powi(2) + v.u.powi(2)).sqrt() * DT;
            assert!(d <= 8.0 + step + 1e-6, "a hit {d:.2} m from the lead (step {step:.2} m)");
        }
    }

    assert_eq!(gun_pass(), run, "the same scenario and actions report the same events");
}

#[test]
fn an_expired_lease_reports_the_failsafe_with_its_reason() {
    let mut env = SimEnv::new(single_ship()).unwrap(); // lease 0.5 s = 60 ticks
    env.step(&[(1, raw(0.0, -0.05, 0.6))]).unwrap();
    let mut seen = env.events().to_vec();
    let mut failsafe_tick = None;
    for _ in 0..80 {
        let (obs, _) = env.step(&[]).unwrap();
        seen.extend_from_slice(env.events());
        if failsafe_tick.is_none() && of(&obs, 1).control_mode == ControlMode::Failsafe {
            failsafe_tick = Some(obs[0].tick);
        }
    }
    let kinds: Vec<&EventKind> = seen.iter().map(|e| &e.kind).collect();
    assert_eq!(
        kinds,
        [
            &EventKind::ModeChange {
                aircraft_id: 1,
                from: ControlMode::Uncommanded,
                to: ControlMode::Raw,
                reason: ModeReason::Command,
            },
            &EventKind::ModeChange {
                aircraft_id: 1,
                from: ControlMode::Raw,
                to: ControlMode::Failsafe,
                reason: ModeReason::LeaseExpired,
            },
        ]
    );
    assert_eq!(Some(seen[1].tick), failsafe_tick, "reported on the first failsafe frame");
}

#[test]
fn every_event_reaches_the_bus_and_the_recording_unchanged() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("duel.mcap");
    let scenario = duel();
    let session = Session::in_process().unwrap();
    let bus = EventReceiver::subscribe(&session).unwrap();
    let mut app = build_app(scenario.clone(), RunMode::Batch); // ends itself on the kill
    net::connect(&mut app, &session).unwrap();
    let (recorder, recording) = record::start(&path, &scenario.to_toml()).unwrap();
    app.insert_resource(recorder);
    let mut chaser = CommandSender::new(&session, 2).unwrap();

    app.finish();
    app.cleanup();
    let mut emitted = Vec::new();
    for _ in 0..600 {
        let tick = app.world().resource::<LastPublished>().0.first().map_or(0, |t| t.tick);
        chaser.send(hold(), true, tick).unwrap();
        std::thread::sleep(Duration::from_millis(1)); // let the command arrive
        app.update();
        emitted.extend_from_slice(app.world().resource::<EventLog>().last());
        if app.should_exit().is_some() {
            break;
        }
    }
    let end_tick = app.world().resource::<SimClock>().tick;
    drop(app);
    recording.finish().unwrap();

    let ended = emitted.last().expect("events were emitted");
    assert_eq!(ended.tick, end_tick);
    assert!(matches!(
        ended.kind,
        EventKind::RunEnded { reason: EndReason::EngagementDecided, ref message } if message.is_empty()
    ));
    assert!(emitted.iter().any(|e| matches!(e.kind, EventKind::Kill { killer_id: Some(2), .. })));

    let mut received = Vec::new();
    while received.len() < emitted.len() {
        received.push(bus.recv_timeout(Duration::from_secs(2)).expect("every event arrives"));
    }
    assert_eq!(received, emitted, "the bus carries every event, in order");

    let recorded: Vec<Event> = BagReader::open(&path)
        .unwrap()
        .messages()
        .unwrap()
        .iter()
        .filter(|m| TopicKind::of(&m.topic) == Some(TopicKind::Events))
        .map(|m| decode_event(&m.payload).unwrap())
        .collect();
    assert_eq!(recorded, emitted, "the bag holds every event");
}
