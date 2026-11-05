//! Sweeps: the grid and repeats expand in a fixed order with the right seeds,
//! every invalid sweep fails before anything flies, controller commands get
//! their placeholders, and an in-process sweep records every run and
//! summarises it.

use std::path::Path;

use astrum_bag::{BagReader, TopicKind};
use astrum_sim::sweep::{
    execute, run_in_process, write_summary, Purpose, Status, Sweep, SweepFile,
};
use astrum_sim::systems::StopRequested;

const BASE: &str = r#"
[world]
origin_lat_deg = 37.6188
origin_lon_deg = -122.3756
origin_alt_hae_m = 0.0

[sim]
rate_hz = 120.0
max_time_s = 1.0
seed = 40

[[aircraft]]
id = 7
model = "c172x"
lat_deg = 37.6188
lon_deg = -122.3756
alt_hae_m = 1524.0
heading_deg = 90.0
tas_mps = 55.0
"#;

fn base_file(dir: &Path, text: &str) -> String {
    let path = dir.join("base.toml");
    std::fs::write(&path, text).unwrap();
    path.display().to_string().replace('\\', "/")
}

fn plan(dir: &Path, body: &str) -> Result<Sweep, String> {
    let text = format!("base = {:?}\n{body}", base_file(dir, BASE));
    Sweep::from_toml_str(&text, "t").map_err(|e| e.0)
}

#[test]
fn grid_and_repeats_expand_in_order_with_seeds() {
    let dir = tempfile::tempdir().unwrap();
    let sweep = plan(
        dir.path(),
        r#"
repeats = 2
speed = 0.0
[set]
"aircraft.7.tas_mps" = 60.0
[vary]
"sim.rate_hz" = [60.0, 120.0]
"aircraft.7.heading_deg" = [0.0, 90.0]
"#,
    )
    .unwrap();
    // Keys in sorted order; the last key fastest, repeats innermost.
    assert_eq!(sweep.vary_keys, ["aircraft.7.heading_deg", "sim.rate_hz"]);
    let got: Vec<(f64, f64, u64)> = sweep
        .runs
        .iter()
        .map(|r| (r.scenario.aircraft[0].heading_deg, r.scenario.sim.rate_hz, r.scenario.sim.seed))
        .collect();
    assert_eq!(
        got,
        [
            (0.0, 60.0, 40),
            (0.0, 60.0, 41),
            (0.0, 120.0, 40),
            (0.0, 120.0, 41),
            (90.0, 60.0, 40),
            (90.0, 60.0, 41),
            (90.0, 120.0, 40),
            (90.0, 120.0, 41),
        ]
    );
    for (i, r) in sweep.runs.iter().enumerate() {
        assert_eq!(r.index, i);
        assert_eq!(r.scenario.aircraft[0].tas_mps, 60.0, "`set` applies to every run");
        assert_eq!(r.scenario.sim.speed, 0.0);
    }
    assert!(sweep.in_process());
    assert_eq!(sweep.out, Path::new("runs/t"));
}

#[test]
fn invalid_sweeps_fail_before_any_run() {
    let dir = tempfile::tempdir().unwrap();
    let lockstep = "[set]\n\"sim.lockstep\" = true\n\"aircraft.7.lockstep\" = true\n";
    for (body, expected) in [
        ("[vary]\n\"physics.rate\" = [1.0]", "unknown key"),
        ("[vary]\n\"aircraft.8.tas_mps\" = [1.0]", "no aircraft 8"),
        ("[vary]\n\"aircraft.7.tas_kts\" = [1.0]", "tas_kts"),
        ("[vary]\n\"aircraft.7.tas_mps\" = [\"fast\"]", "aircraft.7.tas_mps = \"fast\""),
        ("[vary]\n\"aircraft.7.tas_mps\" = [-5.0]", "tas_mps must be >= 0"),
        ("[vary]\n\"sim.speed\" = [1.0]", "sim.speed"),
        ("[vary]\n\"aircraft.7.tas_mps\" = []", "no values"),
        ("[set]\n\"sim.max_time_s\" = 0.0", "max_time_s must be > 0"),
        ("repeats = 0", "repeats"),
        ("controllers = [\"ctl --connect {endpoint}\"]", "need sim.lockstep"),
        (lockstep, "needs controllers"),
        ("[set]\n\"param.gain\" = 2.0", "only used by controllers"),
        (&format!("controllers = [\"ctl --gain {{param.gain}}\"]\n{lockstep}"), "{param.gain}"),
    ] {
        let e = plan(dir.path(), body).err().unwrap_or_else(|| panic!("accepted: {body}"));
        assert!(e.contains(expected), "{body:?}: expected {expected:?} in {e:?}");
    }
}

#[test]
fn a_run_may_run_until_stopped_at_the_scenarios_pace() {
    let dir = tempfile::tempdir().unwrap();
    let unbounded = BASE
        .replace("max_time_s = 1.0", "max_time_s = 0.0\nspeed = 2.0\nlockstep = true")
        .replace("tas_mps = 55.0", "tas_mps = 55.0\nlockstep = true");
    let mut file = SweepFile::for_scenario(base_file(dir.path(), &unbounded).into());
    let plan = |file: &SweepFile, purpose| Sweep::plan(file.clone(), "t", purpose).map_err(|e| e.0);

    // `astrum run`: unbounded and lockstep without listed controllers are fine,
    // and the pace is the scenario's own unless `speed` says otherwise.
    let run = plan(&file, Purpose::Run).unwrap();
    assert_eq!(run.runs.len(), 1);
    assert_eq!(run.runs[0].scenario.sim.speed, 2.0);
    // A sweep of the same file must end by itself and needs controllers for lockstep.
    assert!(plan(&file, Purpose::Sweep).unwrap_err().contains("max_time_s must be > 0"));
    file.set.insert("sim.max_time_s".into(), toml::Value::Float(1.0));
    assert!(plan(&file, Purpose::Sweep).unwrap_err().contains("needs controllers"));

    file.speed = Some(0.5);
    assert_eq!(plan(&file, Purpose::Run).unwrap().runs[0].scenario.sim.speed, 0.5);
}

#[test]
fn controller_commands_get_their_placeholders() {
    let dir = tempfile::tempdir().unwrap();
    let sweep = plan(
        dir.path(),
        r#"
controllers = ["ctl --connect {endpoint} --gain {param.gain} --tag run{run}-s{seed}"]
[set]
"sim.lockstep" = true
"aircraft.7.lockstep" = true
[vary]
"param.gain" = [0.5, 2.0]
"#,
    )
    .unwrap();
    let cmd = &sweep.file.controllers[0];
    assert_eq!(
        sweep.runs[1].render(cmd, "tcp/127.0.0.1:9"),
        "ctl --connect tcp/127.0.0.1:9 --gain 2.0 --tag run2-s40"
    );
}

#[test]
fn an_in_process_sweep_records_and_summarises_every_run() {
    let dir = tempfile::tempdir().unwrap();
    let sweep = plan(dir.path(), "[vary]\n\"aircraft.7.heading_deg\" = [0.0, 180.0]").unwrap();
    let stop = StopRequested::default();
    let outcomes = execute(&sweep, 2, &stop, false, |run| {
        run_in_process(run, &dir.path().join(format!("{}.mcap", run.stem())), &stop)
    });
    for (run, outcome) in sweep.runs.iter().zip(&outcomes) {
        let outcome = outcome.as_ref().expect("every run ran");
        assert_eq!(outcome.status, Status::Ok);
        assert_eq!(outcome.sim_time_s, Some(1.0));
        let bag = BagReader::open(&dir.path().join(format!("{}.mcap", run.stem()))).unwrap();
        let samples = bag
            .messages()
            .unwrap()
            .iter()
            .filter(|m| TopicKind::of(&m.topic) == Some(TopicKind::Telemetry(7)))
            .count();
        assert_eq!(samples, 121, "{}: ticks 0..=120", run.stem());
        assert!(bag
            .scenario()
            .unwrap()
            .contains(&format!("heading_deg = {:?}", run.values[0].as_float().unwrap())));
    }
    let summary = dir.path().join("summary.csv");
    write_summary(&summary, &sweep, &outcomes).unwrap();
    let csv = std::fs::read_to_string(summary).unwrap();
    let rows: Vec<Vec<&str>> = csv.lines().map(|l| l.split(',').collect()).collect();
    assert_eq!(
        rows[0],
        [
            "run",
            "status",
            "seed",
            "aircraft.7.heading_deg",
            "sim_time_s",
            "wall_s",
            "speed_x",
            "bag"
        ]
    );
    assert_eq!(rows.len(), 3);
    for (i, row) in rows[1..].iter().enumerate() {
        assert_eq!(row[..5], [&(i + 1).to_string(), "ok", "40", ["0.0", "180.0"][i], "1.000"]);
        assert_eq!(row[7], format!("run-{:04}.mcap", i + 1));
    }
}
