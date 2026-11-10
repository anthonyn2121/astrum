//! Flight tracks computed from a bag: sim-time thinning, ordering, death times,
//! models, rounding and the samples that are skipped.

use std::path::{Path, PathBuf};

use astrum_bag::tracks::{tracks, TrackSet};
use astrum_bag::{BagReader, BagWriter};
use astrum_proto::{encode_telemetry, CombatStatus, Enu, Telemetry};

const RATE_HZ: f64 = 120.0;

fn sample(id: u32, tick: u64, enu: [f64; 3]) -> Telemetry {
    Telemetry {
        aircraft_id: id,
        tick,
        sim_time_s: tick as f64 / RATE_HZ,
        pos_enu: Enu { e: enu[0], n: enu[1], u: enu[2] },
        combat: CombatStatus {
            hit_points: 100.0,
            hit_points_max: 100.0,
            alive: true,
            ..Default::default()
        },
        ..Telemetry::default()
    }
}

fn dead(id: u32, tick: u64) -> Telemetry {
    let mut t = sample(id, tick, [0.0; 3]);
    t.combat.alive = false;
    t
}

/// A recorded message: topic and encoded payload, in bag (log-time) order.
struct Item {
    topic: String,
    payload: Vec<u8>,
}

fn tel(t: &Telemetry) -> Item {
    Item { topic: format!("astrum/telemetry/{}", t.aircraft_id), payload: encode_telemetry(t) }
}

fn write_bag(path: &Path, scenario: Option<&str>, items: &[Item]) {
    let mut bag = BagWriter::create(path, scenario).expect("create bag");
    for (i, item) in items.iter().enumerate() {
        bag.write(&item.topic, i as u64 + 1, 0, &item.payload).expect("write");
    }
    bag.finish().expect("finish");
}

fn tracks_of(scenario: Option<&str>, items: &[Item], max_points: usize) -> TrackSet {
    let dir = tempfile::tempdir().expect("temp dir");
    let path: PathBuf = dir.path().join("run.mcap");
    write_bag(&path, scenario, items);
    tracks(&BagReader::open(&path).expect("open"), max_points).expect("tracks")
}

#[test]
fn fewer_samples_than_max_points_are_all_kept_in_time_order() {
    // Written out of sim-time order: the bag is log-time ordered, the track is not.
    let items = vec![
        tel(&sample(1, 3, [3.0; 3])),
        tel(&sample(1, 1, [1.0; 3])),
        tel(&sample(1, 2, [2.0; 3])),
    ];
    let set = tracks_of(None, &items, 10);
    assert_eq!(set.sim_time_s, 0.025);
    assert_eq!(set.aircraft.len(), 1);
    let track = &set.aircraft[0];
    assert_eq!(track.t, vec![0.008, 0.017, 0.025]);
    assert_eq!(track.e, vec![1.0, 2.0, 3.0]);
    assert_eq!(track.n, vec![1.0, 2.0, 3.0]);
    assert_eq!(track.u, vec![1.0, 2.0, 3.0]);
}

#[test]
fn a_long_track_is_thinned_to_max_points_nearest_evenly_spaced_samples() {
    let items: Vec<Item> = (0..1000).map(|tick| tel(&sample(1, tick, [tick as f64; 3]))).collect();
    let track = &tracks_of(None, &items, 50).aircraft[0];
    assert_eq!(track.t.len(), 50);
    assert_eq!(track.e.len(), 50);
    assert_eq!(track.n.len(), 50);
    assert_eq!(track.u.len(), 50);

    // Samples are exactly tick / 120 s and their e is the tick, so the nearest
    // sample to a grid time is the tick nearest grid_time * 120.
    let last_tick = 999.0;
    for i in [0usize, 1, 25, 49] {
        let grid_tick = (i as f64 * last_tick / 49.0).round();
        assert_eq!(track.e[i], grid_tick, "grid point {i}");
        assert_eq!(track.t[i], round3(grid_tick / RATE_HZ), "grid point {i}");
    }
    assert_eq!(track.e[0], 0.0);
    assert_eq!(track.e[49], 999.0);

    assert!(track.t.windows(2).all(|w| w[0] < w[1]));
    let ideal = (999.0 / RATE_HZ) / 49.0;
    for gap in track.t.windows(2).map(|w| w[1] - w[0]) {
        assert!(gap <= 2.0 * ideal + 1e-9 && gap >= ideal / 2.0 - 1e-9, "gap {gap}");
    }
}

#[test]
fn max_points_below_two_is_two() {
    let items: Vec<Item> = (0..100).map(|tick| tel(&sample(1, tick, [0.0; 3]))).collect();
    let zero = tracks_of(None, &items, 0);
    let one = tracks_of(None, &items, 1);
    let two = tracks_of(None, &items, 2);
    assert_eq!(zero, two);
    assert_eq!(one, two);
    let track = &two.aircraft[0];
    assert_eq!(track.t, vec![0.0, 0.825]);
}

#[test]
fn aircraft_are_ascending_by_id_with_models_from_the_scenario() {
    let scenario =
        "[[aircraft]]\nid = 1\nmodel = \"f16\"\n\n[[aircraft]]\nid = 2\nmodel = \"c172x\"\n";
    let items = vec![tel(&sample(2, 1, [0.0; 3])), tel(&sample(1, 1, [0.0; 3]))];
    let set = tracks_of(Some(scenario), &items, 10);
    let ids: Vec<u32> = set.aircraft.iter().map(|a| a.id).collect();
    assert_eq!(ids, vec![1, 2]);
    let models: Vec<Option<&str>> = set.aircraft.iter().map(|a| a.model.as_deref()).collect();
    assert_eq!(models, vec![Some("f16"), Some("c172x")]);

    let none = tracks_of(None, &items, 10);
    assert!(none.aircraft.iter().all(|a| a.model.is_none()));
}

#[test]
fn a_dead_aircraft_reports_its_first_dead_sample_time() {
    let items = vec![
        tel(&sample(1, 1, [0.0; 3])),
        tel(&sample(1, 2, [0.0; 3])),
        tel(&dead(1, 3)),
        tel(&dead(1, 4)),
        tel(&sample(2, 1, [0.0; 3])),
        tel(&sample(2, 2, [0.0; 3])),
    ];
    let set = tracks_of(None, &items, 10);
    assert_eq!(set.aircraft[0].died_at_s, Some(0.025));
    assert_eq!(set.aircraft[0].t.len(), 4, "dead samples are kept");
    assert_eq!(set.aircraft[1].died_at_s, None);
}

#[test]
fn an_empty_bag_has_no_time_and_no_aircraft() {
    let set = tracks_of(None, &[], 10);
    assert_eq!(set.sim_time_s, 0.0);
    assert!(set.aircraft.is_empty());
}

#[test]
fn payloads_whose_id_disagrees_with_the_topic_are_skipped() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("junk.mcap");
    let mut bag = BagWriter::create(&path, None).expect("create");
    bag.write("astrum/telemetry/1", 1, 0, &encode_telemetry(&sample(2, 1, [0.0; 3])))
        .expect("write");
    bag.write("astrum/telemetry/1", 2, 0, &[1, 2, 3]).expect("write");
    bag.write("astrum/telemetry/3", 3, 0, &encode_telemetry(&sample(3, 5, [0.0; 3])))
        .expect("write");
    bag.finish().expect("finish");
    let set = tracks(&BagReader::open(&path).expect("open"), 10).expect("tracks");
    let ids: Vec<u32> = set.aircraft.iter().map(|a| a.id).collect();
    assert_eq!(ids, vec![3]);
    assert_eq!(set.sim_time_s, round3(5.0 / RATE_HZ));
}

#[test]
fn positions_are_rounded_to_a_decimetre() {
    let items = vec![tel(&sample(1, 1, [12.3456, -0.04, 1000.04]))];
    let track = &tracks_of(None, &items, 10).aircraft[0];
    assert_eq!(track.e, vec![12.3]);
    assert_eq!(track.n, vec![0.0]);
    assert_eq!(track.u, vec![1000.0]);
}

fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}
