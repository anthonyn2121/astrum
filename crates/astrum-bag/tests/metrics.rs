//! Run metrics computed from a bag: separation, extremes, event counts, end
//! reason, refusals, final state and models.

use std::path::{Path, PathBuf};

use astrum_bag::metrics::{compute, RunMetrics};
use astrum_bag::{BagReader, BagWriter};
use astrum_proto::{
    encode_event, encode_telemetry, CombatStatus, CommandStatus, ControlMode, EndReason, Enu,
    Euler, Event, EventKind, ModeReason, Telemetry,
};

const RATE_HZ: f64 = 120.0;

fn sample(id: u32, tick: u64, enu: [f64; 3]) -> Telemetry {
    Telemetry {
        aircraft_id: id,
        tick,
        sim_time_s: tick as f64 / RATE_HZ,
        lat_deg: 37.6188,
        lon_deg: -122.3756,
        alt_hae_m: enu[2],
        pos_enu: Enu { e: enu[0], n: enu[1], u: enu[2] },
        tas_mps: 200.0,
        combat: CombatStatus {
            hit_points: 100.0,
            hit_points_max: 100.0,
            alive: true,
            ..Default::default()
        },
        ..Telemetry::default()
    }
}

fn event(tick: u64, kind: EventKind) -> Event {
    Event { tick, sim_time_s: tick as f64 / RATE_HZ, kind }
}

fn mode_change(id: u32, from: ControlMode, to: ControlMode) -> EventKind {
    EventKind::ModeChange { aircraft_id: id, from, to, reason: ModeReason::LeaseExpired }
}

/// A recorded message: topic and encoded payload, in bag (log-time) order.
struct Item {
    topic: String,
    payload: Vec<u8>,
}

fn tel(t: Telemetry) -> Item {
    Item { topic: format!("astrum/telemetry/{}", t.aircraft_id), payload: encode_telemetry(&t) }
}

fn ev(e: Event) -> Item {
    Item { topic: "astrum/events".to_owned(), payload: encode_event(&e) }
}

fn write_bag(path: &Path, scenario: Option<&str>, items: &[Item]) {
    let mut bag = BagWriter::create(path, scenario).expect("create bag");
    for (i, item) in items.iter().enumerate() {
        bag.write(&item.topic, i as u64 + 1, 0, &item.payload).expect("write");
    }
    bag.finish().expect("finish");
}

fn metrics_of(scenario: Option<&str>, items: &[Item]) -> RunMetrics {
    let dir = tempfile::tempdir().expect("temp dir");
    let path: PathBuf = dir.path().join("run.mcap");
    write_bag(&path, scenario, items);
    compute(&BagReader::open(&path).expect("open")).expect("compute")
}

#[test]
fn separation_is_the_minimum_same_tick_pair_over_living_aircraft() {
    let mut dead = sample(3, 2, [1.0, 0.0, 1000.0]);
    dead.combat.alive = false;
    let items = vec![
        // Tick 1: 1–2 are 100 m apart, 2–3 are 50 m apart.
        tel(sample(1, 1, [0.0, 0.0, 1000.0])),
        tel(sample(2, 1, [100.0, 0.0, 1000.0])),
        tel(sample(3, 1, [100.0, 30.0, 1040.0])),
        // Tick 2: 3 is dead right next to 1 (ignored); 1–2 close to 3-4-12 = 13 m.
        tel(sample(1, 2, [0.0, 0.0, 1000.0])),
        tel(sample(2, 2, [3.0, 4.0, 1012.0])),
        tel(dead),
        // Tick 3: only aircraft 1 at tick 3, aircraft 2 at tick 4 — no common
        // tick, so their 1 m gap never counts.
        tel(sample(1, 3, [0.0, 0.0, 1000.0])),
        tel(sample(2, 4, [1.0, 0.0, 1000.0])),
    ];
    let m = metrics_of(None, &items);
    assert_eq!(m.min_separation_m, Some(13.0));
    assert_eq!(m.min_separation_pair, Some([1, 2]));
    assert_eq!(m.min_separation_t_s, Some(2.0 / RATE_HZ));
}

#[test]
fn separation_matches_samples_by_tick_not_by_bag_order() {
    // Telemetry of one tick interleaved with another's (a live recording).
    let items = vec![
        tel(sample(2, 5, [10.0, 0.0, 0.0])),
        tel(sample(1, 6, [500.0, 0.0, 0.0])),
        tel(sample(1, 5, [0.0, 0.0, 0.0])),
        tel(sample(2, 6, [0.0, 0.0, 0.0])),
    ];
    let m = metrics_of(None, &items);
    assert_eq!(m.min_separation_m, Some(10.0));
    assert_eq!(m.min_separation_pair, Some([1, 2]));
    assert_eq!(m.min_separation_t_s, Some(5.0 / RATE_HZ));
}

#[test]
fn a_single_aircraft_has_no_separation() {
    let items: Vec<Item> = (0..10).map(|k| tel(sample(1, k, [k as f64, 0.0, 900.0]))).collect();
    let m = metrics_of(None, &items);
    assert_eq!(m.min_separation_m, None);
    assert_eq!(m.min_separation_pair, None);
    assert_eq!(m.min_separation_t_s, None);
    assert_eq!(m.aircraft.len(), 1);
}

#[test]
fn failsafe_counts_only_transitions_into_failsafe() {
    use ControlMode::*;
    let items = vec![
        tel(sample(1, 1, [0.0; 3])),
        tel(sample(2, 1, [0.0; 3])),
        ev(event(1, mode_change(1, Uncommanded, Raw))),
        ev(event(2, mode_change(1, Raw, Failsafe))),
        ev(event(3, mode_change(1, Failsafe, Autopilot))),
        ev(event(4, mode_change(1, Autopilot, Failsafe))),
        ev(event(5, mode_change(2, Autopilot, Failsafe))),
        ev(event(6, mode_change(2, Failsafe, Raw))),
        ev(event(7, EventKind::ControllerDown { aircraft_id: 2 })),
        ev(event(7, EventKind::ControllerUp { aircraft_id: 2 })),
    ];
    let m = metrics_of(None, &items);
    assert_eq!(m.failsafe_entries, 3);
    assert_eq!(m.controller_losses, 1);
    let per: Vec<(u32, u32)> = m.aircraft.iter().map(|a| (a.id, a.failsafe_entries)).collect();
    assert_eq!(per, vec![(1, 2), (2, 1)]);
}

#[test]
fn an_event_only_aircraft_counts_in_the_fleet_but_has_no_entry() {
    // Aircraft 9 never appears in telemetry: no state to report, but its
    // failsafe entry is still a fleet event.
    let items = vec![
        tel(sample(1, 1, [0.0; 3])),
        ev(event(2, mode_change(9, ControlMode::Autopilot, ControlMode::Failsafe))),
    ];
    let m = metrics_of(None, &items);
    assert_eq!(m.failsafe_entries, 1);
    let ids: Vec<u32> = m.aircraft.iter().map(|a| a.id).collect();
    assert_eq!(ids, vec![1]);
    assert_eq!(m.aircraft[0].failsafe_entries, 0);
}

#[test]
fn hits_kills_and_end_come_from_events() {
    let pos = Enu { e: 0.0, n: 0.0, u: 1000.0 };
    let items = vec![
        ev(event(
            1,
            EventKind::Hit {
                shooter_id: 1,
                target_id: 2,
                damage: 50.0,
                hit_points: 50.0,
                pos_enu: pos,
            },
        )),
        ev(event(
            2,
            EventKind::Hit {
                shooter_id: 1,
                target_id: 2,
                damage: 50.0,
                hit_points: 0.0,
                pos_enu: pos,
            },
        )),
        ev(event(2, EventKind::Kill { victim_id: 2, killer_id: Some(1), pos_enu: pos })),
        ev(event(
            3,
            EventKind::RunEnded { reason: EndReason::Fatal, message: "fdm diverged".into() },
        )),
    ];
    let m = metrics_of(None, &items);
    assert_eq!((m.hits, m.kills), (2, 1));
    assert_eq!(m.end_reason.as_deref(), Some("fatal"));
    assert_eq!(m.end_message.as_deref(), Some("fdm diverged"));

    let ended = metrics_of(
        None,
        &[ev(event(
            9,
            EventKind::RunEnded { reason: EndReason::EngagementDecided, message: String::new() },
        ))],
    );
    assert_eq!(ended.end_reason.as_deref(), Some("engagement-decided"));
    assert_eq!(ended.end_message.as_deref(), Some(""));
}

#[test]
fn an_empty_bag_has_no_extremes_and_zero_counts() {
    let m = metrics_of(None, &[]);
    assert_eq!(m.end_reason, None);
    assert_eq!(m.end_message, None);
    assert_eq!(m.sim_time_s, 0.0);
    assert_eq!(m.min_separation_m, None);
    assert_eq!(m.min_alt_hae_m, None);
    assert_eq!(m.max_bank_deg, None);
    assert_eq!(m.max_nz_g, None);
    assert_eq!(
        (m.failsafe_entries, m.controller_losses, m.command_refusals, m.hits, m.kills, m.rejected),
        (0, 0, 0, 0, 0, 0)
    );
    assert!(m.aircraft.is_empty());
}

#[test]
fn refusals_are_summed_from_each_aircrafts_last_telemetry() {
    let with_refusals = |id, tick, refusals| {
        let mut t = sample(id, tick, [0.0; 3]);
        t.command = CommandStatus { refusals, ..CommandStatus::default() };
        t
    };
    let items = vec![
        tel(with_refusals(1, 1, 5)),
        tel(with_refusals(1, 2, 7)),
        tel(with_refusals(2, 1, 4_000_000_000)),
        tel(with_refusals(2, 2, u32::MAX)),
    ];
    let m = metrics_of(None, &items);
    // Summed without wrapping: 7 + u32::MAX.
    assert_eq!(m.command_refusals, 7 + u64::from(u32::MAX));
}

#[test]
fn final_state_is_the_last_sample_and_extremes_skip_dead_samples() {
    let mut early = sample(1, 1, [1.0, 2.0, 800.0]);
    early.attitude = Euler { phi_rad: (-60f64).to_radians(), theta_rad: 0.0, psi_rad: 0.0 };
    early.aero.nz_g = 2.0;
    let mut last = sample(1, 3, [10.0, 20.0, 1500.0]);
    last.lat_deg = 37.7;
    last.lon_deg = -122.4;
    last.tas_mps = 180.0;
    last.attitude = Euler { phi_rad: 0.1, theta_rad: 0.0, psi_rad: (-90f64).to_radians() };
    last.combat.alive = false;
    last.combat.hit_points = 0.0;
    // Dead: these extremes never count.
    last.alt_hae_m = 10.0;
    last.aero.nz_g = 9.0;
    let mid = sample(1, 2, [5.0, 5.0, 900.0]);
    // The last tick arrives before an earlier one in the bag.
    let m = metrics_of(None, &[tel(early), tel(last), tel(mid)]);

    let a = &m.aircraft[0];
    assert!(!a.alive);
    assert_eq!(a.hit_points, 0.0);
    let f = &a.final_state;
    assert_eq!(f.sim_time_s, 3.0 / RATE_HZ);
    assert_eq!((f.lat_deg, f.lon_deg, f.alt_hae_m, f.tas_mps), (37.7, -122.4, 10.0, 180.0));
    assert!((f.heading_deg - 270.0).abs() < 1e-9, "{}", f.heading_deg);
    assert_eq!(f.enu_m, [10.0, 20.0, 1500.0]);
    assert_eq!(m.sim_time_s, 3.0 / RATE_HZ);

    assert_eq!(a.min_alt_hae_m, Some(800.0));
    assert!((a.max_bank_deg.unwrap() - 60.0).abs() < 1e-9);
    assert_eq!(a.max_nz_g, Some(2.0));
    assert_eq!((m.min_alt_hae_m, m.max_nz_g), (Some(800.0), Some(2.0)));
}

#[test]
fn fleet_extremes_span_aircraft_which_are_sorted_by_id() {
    let mut low = sample(7, 1, [0.0, 0.0, 300.0]);
    low.aero.nz_g = 1.5;
    let mut banked = sample(3, 1, [0.0, 0.0, 2000.0]);
    banked.attitude.phi_rad = 45f64.to_radians();
    banked.aero.nz_g = 4.0;
    let m = metrics_of(None, &[tel(low), tel(banked)]);
    let ids: Vec<u32> = m.aircraft.iter().map(|a| a.id).collect();
    assert_eq!(ids, vec![3, 7]);
    assert_eq!(m.min_alt_hae_m, Some(300.0));
    assert_eq!(m.max_nz_g, Some(4.0));
    assert!((m.max_bank_deg.unwrap() - 45.0).abs() < 1e-9);
}

#[test]
fn models_come_from_the_recorded_scenario() {
    let scenario = "[world]\norigin_lat_deg = 37.6\n\n[[aircraft]]\nid = 1\nmodel = \"f16\"\n\n\
                    [[aircraft]]\nid = 2\nmodel = \"c172x\"\n";
    let items =
        vec![tel(sample(1, 1, [0.0; 3])), tel(sample(2, 1, [0.0; 3])), tel(sample(5, 1, [0.0; 3]))];
    let m = metrics_of(Some(scenario), &items);
    let models: Vec<Option<&str>> = m.aircraft.iter().map(|a| a.model.as_deref()).collect();
    assert_eq!(models, vec![Some("f16"), Some("c172x"), None]);

    let none = metrics_of(None, &items[..1]);
    assert_eq!(none.aircraft[0].model, None);
}

#[test]
fn an_unparseable_recorded_scenario_only_drops_the_models() {
    let items = vec![
        tel(sample(1, 1, [0.0, 0.0, 500.0])),
        tel(sample(2, 1, [30.0, 40.0, 500.0])),
        ev(event(2, EventKind::RunEnded { reason: EndReason::MaxTime, message: String::new() })),
    ];
    let m = metrics_of(Some("[[aircraft]\nid = 1"), &items);
    let models: Vec<Option<&str>> = m.aircraft.iter().map(|a| a.model.as_deref()).collect();
    assert_eq!(models, vec![None, None]);
    assert_eq!(m.min_separation_m, Some(50.0));
    assert_eq!(m.min_alt_hae_m, Some(500.0));
    assert_eq!(m.end_reason.as_deref(), Some("max-time"));
    assert_eq!(m.sim_time_s, 1.0 / RATE_HZ);
}

#[test]
fn final_heading_is_in_zero_to_360() {
    let mut t = sample(1, 1, [0.0; 3]);
    // -1e-17 rad in degrees wraps to 360 - 5.7e-16, which rounds to 360.0.
    t.attitude.psi_rad = -1e-17;
    let m = metrics_of(None, &[tel(t)]);
    assert_eq!(m.aircraft[0].final_state.heading_deg, 0.0);

    t.attitude.psi_rad = 3.0 * std::f64::consts::FRAC_PI_2;
    let m = metrics_of(None, &[tel(t)]);
    assert!((m.aircraft[0].final_state.heading_deg - 270.0).abs() < 1e-9);
}

#[test]
fn undecodable_or_mislabelled_payloads_are_rejected_not_counted() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("junk.mcap");
    let mut bag = BagWriter::create(&path, None).expect("create");
    bag.write("astrum/telemetry/1", 1, 0, &[1, 2, 3]).expect("write");
    // Aircraft 2's sample on aircraft 1's topic.
    bag.write("astrum/telemetry/1", 2, 0, &encode_telemetry(&sample(2, 1, [0.0; 3])))
        .expect("write");
    bag.write("astrum/events", 3, 0, &[0xff]).expect("write");
    bag.finish().expect("finish");
    let m = compute(&BagReader::open(&path).expect("open")).expect("compute");
    assert_eq!(m.rejected, 3);
    assert!(m.aircraft.is_empty());
}

fn keys(v: &serde_json::Value) -> Vec<&str> {
    let mut keys: Vec<&str> = v.as_object().expect("object").keys().map(String::as_str).collect();
    keys.sort_unstable();
    keys
}

#[test]
fn json_uses_the_metric_ids() {
    let m = metrics_of(None, &[tel(sample(1, 1, [0.0; 3]))]);
    let json = serde_json::to_value(&m).expect("json");
    let mut fleet = vec![
        "end_reason",
        "end_message",
        "sim_time_s",
        "min_separation_m",
        "min_separation_pair",
        "min_separation_t_s",
        "min_alt_hae_m",
        "max_bank_deg",
        "max_nz_g",
        "failsafe_entries",
        "controller_losses",
        "command_refusals",
        "hits",
        "kills",
        "rejected",
        "aircraft",
    ];
    fleet.sort_unstable();
    assert_eq!(keys(&json), fleet);

    let aircraft = &json["aircraft"][0];
    let mut per = vec![
        "id",
        "model",
        "alive",
        "hit_points",
        "min_alt_hae_m",
        "max_bank_deg",
        "max_nz_g",
        "failsafe_entries",
        "final_state",
    ];
    per.sort_unstable();
    assert_eq!(keys(aircraft), per);

    let mut fin =
        vec!["sim_time_s", "lat_deg", "lon_deg", "alt_hae_m", "heading_deg", "tas_mps", "enu_m"];
    fin.sort_unstable();
    assert_eq!(keys(&aircraft["final_state"]), fin);
    assert_eq!(aircraft["final_state"]["enu_m"], serde_json::json!([0.0, 0.0, 0.0]));
}
