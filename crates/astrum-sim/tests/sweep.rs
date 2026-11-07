//! Sweeps: the grid, samples and repeats expand in a fixed order with the right
//! seeds; offsets land relative to each run's initial conditions; samples
//! reproduce per `(sample_seed, k)` and are shared by every grid point, redrawn
//! within the separation / altitude limits; every invalid sweep (including one
//! that cannot trim) fails before anything flies; controller commands get
//! their placeholders; and a sweep reports every run as it starts and
//! finishes, keeping each run's metrics and the summary up to date.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Mutex;

use astrum_bag::{BagReader, TopicKind};
use astrum_geo::{Geodetic, LocalFrame};
use astrum_sim::catalog::METRICS;
use astrum_sim::perturb::{self, Axis, Group, Offset};
use astrum_sim::scenario::Scenario;
use astrum_sim::sweep::{
    execute, run_in_process, Outcome, Purpose, RunEvent, Status, Sweep, SweepFile,
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

/// [`BASE`] plus aircraft 8, about 133 m north of 7.
fn two_ship() -> String {
    format!(
        "{BASE}\n[[aircraft]]\nid = 8\nmodel = \"c172x\"\nlat_deg = 37.6200\nlon_deg = -122.3756\n\
         alt_hae_m = 1524.0\nheading_deg = 0.0\ntas_mps = 55.0\n"
    )
}

fn base_file(dir: &Path, text: &str) -> String {
    let path = dir.join("base.toml");
    std::fs::write(&path, text).unwrap();
    path.display().to_string().replace('\\', "/")
}

fn plan_on(dir: &Path, base: &str, body: &str) -> Result<Sweep, String> {
    let text = format!("base = {:?}\n{body}", base_file(dir, base));
    Sweep::from_toml_str(&text, "t").map_err(|e| e.0)
}

fn plan(dir: &Path, body: &str) -> Result<Sweep, String> {
    plan_on(dir, BASE, body)
}

fn aircraft(s: &Scenario, id: u32) -> &astrum_sim::scenario::AircraftSpec {
    s.aircraft.iter().find(|a| a.id == id).unwrap()
}

/// ENU of aircraft `id` of `after` in a frame at its place in `before`.
fn moved(before: &Scenario, after: &Scenario, id: u32) -> [f64; 3] {
    let at = |s: &Scenario| {
        let a = aircraft(s, id);
        Geodetic::from_degrees(a.lat_deg, a.lon_deg, a.alt_hae_m)
    };
    let p = LocalFrame::new(at(before)).geodetic_to_enu(&at(after));
    [p.e, p.n, p.u]
}

fn floats(values: &[toml::Value]) -> Vec<f64> {
    values.iter().map(|v| v.as_float().unwrap()).collect()
}

fn ok() -> Outcome {
    Outcome { status: Status::Ok, sim_time_s: Some(1.0), wall_s: 0.5 }
}

/// Every event `execute` reported, in order.
#[derive(Default)]
struct Events(Mutex<Vec<(bool, usize)>>);

impl Events {
    fn on(&self, e: RunEvent) {
        let entry = match e {
            RunEvent::Started { index } => (false, index),
            RunEvent::Finished { index, .. } => (true, index),
        };
        self.0.lock().unwrap().push(entry);
    }

    /// Each run started once and finished once, after it started.
    fn assert_once_each(&self, runs: usize) {
        let events = self.0.lock().unwrap();
        assert_eq!(events.len(), 2 * runs, "{events:?}");
        for i in 0..runs {
            let at = |finished| events.iter().position(|&e| e == (finished, i));
            let (start, end) = (at(false).expect("started"), at(true).expect("finished"));
            assert!(start < end, "run {i}: {events:?}");
            assert_eq!(events.iter().filter(|e| e.1 == i).count(), 2, "run {i}: {events:?}");
        }
    }
}

fn csv(path: &Path) -> Vec<Vec<String>> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|l| l.split(',').map(String::from).collect())
        .collect()
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
    assert_eq!(sweep.value_keys, ["aircraft.7.heading_deg", "sim.rate_hz"]);
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
        assert_eq!(r.sample, None);
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
    let perturb = "[[perturb]]\naircraft = [7]\nup_m = { uniform = 10.0 }\n";
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
        // Offset keys.
        ("[set]\n\"aircraft.7.offset.sideways_m\" = 1.0", "unknown offset axis `sideways_m`"),
        ("[set]\n\"aircraft.8.offset.up_m\" = 1.0", "aircraft.8.offset.up_m: the base scenario has no aircraft 8"),
        ("[set]\n\"aircraft.7.offset.up_m\" = \"high\"", "aircraft.7.offset.up_m must be a finite number"),
        (
            "[vary]\n\"aircraft.7.offset.climb_deg\" = [0.0, 95.0]",
            "run 2 (aircraft.7.offset.climb_deg = 95.0): aircraft 7: aircraft.7.offset.climb_deg = 95 \
             makes climb_deg 95, outside (-90, 90)",
        ),
        // Sampling.
        ("samples = 3", "samples needs [[perturb]]"),
        (perturb, "[[perturb]] needs samples > 0"),
        ("min_separation_m = 10.0", "min_separation_m limits [[perturb]] draws"),
        (&format!("samples = 2\nmin_separation_m = -1.0\n{perturb}"), "min_separation_m must be > 0"),
        ("samples = 2\n[[perturb]]\naircraft = []\nup_m = { uniform = 1.0 }", "aircraft lists no aircraft"),
        ("samples = 2\n[[perturb]]\naircraft = [9]\nup_m = { uniform = 1.0 }", "[[perturb]] 1: the base scenario has no aircraft 9"),
        ("samples = 2\n[[perturb]]\naircraft = [7, 7]\nup_m = { uniform = 1.0 }", "aircraft 7 is listed twice"),
        ("samples = 2\n[[perturb]]\naircraft = [7]", "draws no axis"),
        ("samples = 2\n[[perturb]]\naircraft = [7]\nup_m = { uniform = 0.0 }", "up_m must be > 0"),
        ("samples = 2\n[[perturb]]\naircraft = [7]\nup_m = { sideways = 1.0 }", "sideways"),
        ("samples = 2\n[[perturb]]\naircraft = [7]\nup = { uniform = 1.0 }", "unknown field `up`"),
        (
            &format!("samples = 2\n{perturb}{perturb}"),
            "aircraft.7.offset.up_m is drawn by two [[perturb]] tables",
        ),
        (
            &format!("samples = 2\n[vary]\n\"aircraft.7.offset.up_m\" = [1.0]\n{perturb}"),
            "aircraft.7.offset.up_m is both varied and drawn",
        ),
        (
            "samples = 20\n[[perturb]]\naircraft = [7]\nbank_deg = { uniform = 1000.0 }",
            "aircraft.7.offset.bank_deg",
        ),
    ] {
        let e = plan(dir.path(), body).err().unwrap_or_else(|| panic!("accepted: {body}"));
        assert!(e.contains(expected), "{body:?}: expected {expected:?} in {e:?}");
    }
    // A drawn bank beyond ±90 names its run and sample.
    let e = plan(
        dir.path(),
        "samples = 20\n[[perturb]]\naircraft = [7]\nbank_deg = { uniform = 1000.0 }",
    )
    .unwrap_err();
    assert!(e.starts_with("run ") && e.contains("(sample ") && e.contains("aircraft 7"), "{e}");
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
fn offsets_apply_after_every_absolute_key_in_the_aircrafts_own_frame() {
    let dir = tempfile::tempdir().unwrap();
    let base = Scenario::from_toml_str(BASE).unwrap();
    // The base heads east; `set` turns it north first, so forward is north.
    let sweep = plan(
        dir.path(),
        r#"
[set]
"aircraft.7.offset.forward_m" = 500.0
"aircraft.7.heading_deg" = 0.0
[vary]
"aircraft.7.offset.right_m" = [0.0, 100.0]
"aircraft.7.offset.heading_deg" = [-30.0]
"#,
    )
    .unwrap();
    assert_eq!(sweep.runs.len(), 2);
    for (run, east) in sweep.runs.iter().zip([0.0, 100.0]) {
        let [e, n, u] = moved(&base, &run.scenario, 7);
        assert!((e - east).abs() < 1e-3 && (n - 500.0).abs() < 1e-3 && u.abs() < 1e-3);
        assert_eq!(aircraft(&run.scenario, 7).heading_deg, 330.0, "0 - 30 wraps");
    }
}

const SAMPLED: &str = r#"
samples = 3
repeats = 2
sample_seed = 5
[vary]
"aircraft.7.tas_mps" = [50.0, 60.0]
[[perturb]]
aircraft = [7]
forward_m = { uniform = 100.0 }
heading_deg = { normal = 5.0 }
"#;

#[test]
fn samples_multiply_the_grid_and_share_their_draws_across_it() {
    let dir = tempfile::tempdir().unwrap();
    let sweep = plan(dir.path(), SAMPLED).unwrap();
    assert_eq!(
        sweep.value_keys,
        ["aircraft.7.tas_mps", "aircraft.7.offset.forward_m", "aircraft.7.offset.heading_deg"]
    );
    // Grid point, then sample, then repeat.
    let got: Vec<(f64, Option<u64>, u64)> = sweep
        .runs
        .iter()
        .map(|r| (r.scenario.aircraft[0].tas_mps, r.sample, r.scenario.sim.seed))
        .collect();
    let mut want = Vec::new();
    for tas in [50.0, 60.0] {
        for k in 0..3 {
            for seed in [40, 41] {
                want.push((tas, Some(k), seed));
            }
        }
    }
    assert_eq!(got, want);

    let draws = |s: &Sweep, k: u64| -> Vec<Vec<f64>> {
        s.runs.iter().filter(|r| r.sample == Some(k)).map(|r| floats(&r.values[1..])).collect()
    };
    for k in 0..3 {
        let d = draws(&sweep, k);
        assert!(d.iter().all(|v| *v == d[0]), "sample {k} is the same at every grid point: {d:?}");
        assert!(d[0][0].abs() <= 100.0 && d[0][1].abs() <= 15.0, "{d:?}");
    }
    assert_ne!(draws(&sweep, 0)[0], draws(&sweep, 1)[0]);
    assert_ne!(draws(&sweep, 1)[0], draws(&sweep, 2)[0]);

    // The recorded values reproduce each run's initial conditions.
    let base = Scenario::from_toml_str(BASE).unwrap();
    for run in &sweep.runs {
        let mut want = base.clone();
        want.aircraft[0].tas_mps = run.values[0].as_float().unwrap();
        let v = floats(&run.values[1..]);
        let mut offset = Offset::default();
        offset.set(Axis::ForwardM, v[0]);
        offset.set(Axis::HeadingDeg, v[1]);
        Group { lead: 7, members: vec![7], offset }.apply(&mut want).unwrap();
        assert_eq!(run.scenario.aircraft, want.aircraft, "{}", run.stem());
    }

    // Sample k depends on (sample_seed, k) only: replanning — even with a
    // smaller grid — draws the same, another seed draws otherwise.
    let again = plan(dir.path(), SAMPLED).unwrap();
    let fewer = plan(dir.path(), &SAMPLED.replace("[50.0, 60.0]", "[70.0]")).unwrap();
    let reseeded =
        plan(dir.path(), &SAMPLED.replace("sample_seed = 5", "sample_seed = 6")).unwrap();
    for k in 0..3 {
        assert_eq!(draws(&again, k), draws(&sweep, k));
        assert_eq!(draws(&fewer, k)[0], draws(&sweep, k)[0]);
        assert_ne!(draws(&reseeded, k)[0], draws(&sweep, k)[0]);
    }
}

#[test]
fn a_rigid_group_moves_as_one_and_is_recorded_under_its_lead() {
    let dir = tempfile::tempdir().unwrap();
    let base = Scenario::from_toml_str(&two_ship()).unwrap();
    let sweep = plan_on(
        dir.path(),
        &two_ship(),
        r#"
samples = 5
[[perturb]]
aircraft = [7, 8]
rigid = true
forward_m = { uniform = 1000.0 }
heading_deg = { uniform = 90.0 }
"#,
    )
    .unwrap();
    assert_eq!(sweep.value_keys, ["aircraft.7.offset.forward_m", "aircraft.7.offset.heading_deg"]);
    let gap = perturb::min_separation_m(&base).unwrap();
    for run in &sweep.runs {
        let d = perturb::min_separation_m(&run.scenario).unwrap();
        assert!((d - gap).abs() < 1e-3, "{}: {gap} → {d}", run.stem());
        let turn = run.values[1].as_float().unwrap();
        for id in [7, 8] {
            let want = perturb::wrap_heading(aircraft(&base, id).heading_deg + turn);
            assert!((aircraft(&run.scenario, id).heading_deg - want).abs() < 1e-9);
        }
    }
}

#[test]
fn draws_too_close_or_too_low_are_redrawn_within_bounds() {
    let dir = tempfile::tempdir().unwrap();
    let body = |limits: &str| {
        format!(
            "samples = 40\n{limits}\n[vary]\n\"aircraft.7.offset.up_m\" = [0.0, 20.0]\n\
             [[perturb]]\naircraft = [8]\nforward_m = {{ uniform = 200.0 }}\n\
             right_m = {{ uniform = 200.0 }}\nup_m = {{ uniform = 50.0 }}\n"
        )
    };
    let free = plan_on(dir.path(), &two_ship(), &body("")).unwrap();
    let limited =
        plan_on(dir.path(), &two_ship(), &body("min_separation_m = 150.0\nmin_alt_hae_m = 1490.0"))
            .unwrap();
    let violates = |s: &Scenario| {
        perturb::min_separation_m(s).unwrap() < 150.0
            || s.aircraft.iter().any(|a| a.alt_hae_m < 1490.0)
    };
    assert!(free.runs.iter().any(|r| violates(&r.scenario)), "the limits matter here");
    assert!(limited.runs.iter().all(|r| !violates(&r.scenario)));
    // A sample the limits accept at every grid point is kept as drawn.
    for k in 0..40 {
        let runs = |s: &Sweep| -> Vec<Vec<toml::Value>> {
            s.runs.iter().filter(|r| r.sample == Some(k)).map(|r| r.values.clone()).collect()
        };
        let free_runs: Vec<_> = free.runs.iter().filter(|r| r.sample == Some(k)).collect();
        if free_runs.iter().all(|r| !violates(&r.scenario)) {
            assert_eq!(runs(&limited), runs(&free), "sample {k}");
        }
    }

    for limits in ["min_separation_m = 100000.0", "min_alt_hae_m = 1600.0"] {
        let e = plan_on(dir.path(), &two_ship(), &body(limits)).unwrap_err();
        assert!(
            e.contains("sample 1: no draw in 1000 attempts meets") && e.contains(limits),
            "{e}"
        );
    }
}

#[test]
fn a_run_that_cannot_trim_fails_the_check_naming_run_and_aircraft() {
    let dir = tempfile::tempdir().unwrap();
    let sweep = plan(dir.path(), "[vary]\n\"aircraft.7.tas_mps\" = [55.0, 10.0]").unwrap();
    let e = sweep.check_trim().unwrap_err().0;
    assert!(e.starts_with("run 2 (aircraft.7.tas_mps = 10.0): aircraft 7 (c172x): "), "{e}");
    assert!(e.to_lowercase().contains("trim"), "{e}");

    let fine =
        plan(dir.path(), "repeats = 2\n[vary]\n\"aircraft.7.tas_mps\" = [55.0, 60.0]").unwrap();
    assert_eq!(fine.check_trim().unwrap().trims, 4, "every run's own build");
    // An untrimmed aircraft is not built.
    let untrimmed =
        plan(dir.path(), "[set]\n\"aircraft.7.trim\" = false\n\"aircraft.7.tas_mps\" = 10.0")
            .unwrap();
    assert_eq!(untrimmed.check_trim().unwrap().trims, 0);
}

#[test]
fn a_sweep_file_round_trips_through_toml() {
    let text = r#"
base = "config/scenario.example.toml"
repeats = 2
jobs = 3
speed = 4.0
run_timeout_s = 9.5
out = "runs/x"
samples = 7
sample_seed = 11
min_separation_m = 120.0
min_alt_hae_m = 300.0
controllers = ["ctl --id 1 --gain {param.gain}"]
[set]
"sim.max_time_s" = 30.0
"param.gain" = 2
[vary]
"aircraft.1.offset.heading_deg" = [-10.0, 10.0]
"aircraft.2.tas_mps" = [200, 250.5]
[[perturb]]
aircraft = [1, 2]
rigid = true
forward_m = { uniform = 100.0 }
bank_deg = { normal = 2.5 }
[[perturb]]
aircraft = [3]
up_m = { normal = 10.0 }
"#;
    let file: SweepFile = toml::from_str(text).unwrap();
    let written = file.to_toml().unwrap();
    let back: SweepFile = toml::from_str(&written).unwrap();
    assert_eq!(back, file, "{written}");
    let bare = SweepFile::for_scenario("b.toml".into());
    assert_eq!(toml::from_str::<SweepFile>(&bare.to_toml().unwrap()).unwrap(), bare);
}

#[test]
fn planning_from_an_in_memory_base_equals_planning_from_its_file() {
    let dir = tempfile::tempdir().unwrap();
    let from_file = plan(dir.path(), SAMPLED).unwrap();
    let file: SweepFile = toml::from_str(&format!("base = \"nowhere.toml\"\n{SAMPLED}")).unwrap();
    let base = Scenario::from_toml_str(BASE).unwrap();
    let in_memory = Sweep::plan_scenario(file, base, "t", Purpose::Sweep).unwrap();
    assert_eq!(in_memory.value_keys, from_file.value_keys);
    assert_eq!(in_memory.out, from_file.out);
    assert_eq!(in_memory.runs.len(), from_file.runs.len());
    for (a, b) in in_memory.runs.iter().zip(&from_file.runs) {
        assert_eq!(
            (a.index, &a.scenario, &a.values, a.sample, &a.params),
            (b.index, &b.scenario, &b.values, b.sample, &b.params)
        );
    }
}

#[test]
fn an_in_process_sweep_records_reports_and_summarises_every_run() {
    let dir = tempfile::tempdir().unwrap();
    let mut sweep = plan(dir.path(), "[vary]\n\"aircraft.7.heading_deg\" = [0.0, 180.0]").unwrap();
    sweep.out = dir.path().join("out");
    let stop = StopRequested::default();
    let events = Events::default();
    let outcomes = execute(
        &sweep,
        2,
        &stop,
        |e| events.on(e),
        |run| run_in_process(run, &sweep.out.join(format!("{}.mcap", run.stem())), &stop),
    )
    .unwrap();
    events.assert_once_each(2);
    for (run, outcome) in sweep.runs.iter().zip(&outcomes) {
        let outcome = outcome.as_ref().expect("every run ran");
        assert_eq!(outcome.status, Status::Ok);
        assert_eq!(outcome.sim_time_s, Some(1.0));
        let bag = BagReader::open(&sweep.out.join(format!("{}.mcap", run.stem()))).unwrap();
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
        let metrics: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(sweep.out.join(format!("{}.metrics.json", run.stem())))
                .unwrap(),
        )
        .unwrap();
        assert_eq!(metrics["end_reason"], "max-time");
    }
    let rows = csv(&sweep.out.join("summary.csv"));
    let mut header: Vec<&str> = vec![
        "run",
        "status",
        "seed",
        "aircraft.7.heading_deg",
        "sim_time_s",
        "wall_s",
        "speed_x",
        "bag",
    ];
    // Then every metric that is not a column already.
    header.extend(METRICS.iter().filter(|id| **id != "sim_time_s"));
    assert_eq!(rows[0], header);
    assert_eq!(rows.len(), 3);
    let col = |name: &str| rows[0].iter().position(|h| h == name).unwrap();
    for (i, row) in rows[1..].iter().enumerate() {
        assert_eq!(row[..5], [&(i + 1).to_string(), "ok", "40", ["0.0", "180.0"][i], "1.000"]);
        assert_eq!(row[7], format!("run-{:04}.mcap", i + 1));
        assert_eq!(row[col("end_reason")], "max-time");
        assert_eq!(row[col("min_separation_m")], "", "one aircraft: no separation");
        let min_alt: f64 = row[col("min_alt_hae_m")].parse().unwrap();
        assert!((min_alt - 1524.0).abs() < 50.0, "{min_alt}");
        assert_eq!(row[col("kills")], "0");
    }
}

#[test]
fn the_summary_follows_the_runs_as_they_start_and_finish() {
    let dir = tempfile::tempdir().unwrap();
    let mut sweep = plan(dir.path(), &SAMPLED.replace("repeats = 2", "repeats = 1")).unwrap();
    sweep.out = dir.path().join("out");
    let summary = sweep.out.join("summary.csv");
    let seen: Mutex<BTreeMap<usize, Vec<String>>> = Mutex::default();
    let events = Events::default();
    let stop = StopRequested::default();
    let outcomes = execute(
        &sweep,
        1,
        &stop,
        |e| events.on(e),
        |run| {
            let status: Vec<String> = csv(&summary)[1..].iter().map(|r| r[1].clone()).collect();
            seen.lock().unwrap().insert(run.index, status);
            ok()
        },
    )
    .unwrap();
    assert!(outcomes.iter().all(Option::is_some));
    events.assert_once_each(6);
    assert_eq!(
        *events.0.lock().unwrap(),
        [
            (false, 0),
            (true, 0),
            (false, 1),
            (true, 1),
            (false, 2),
            (true, 2),
            (false, 3),
            (true, 3),
            (false, 4),
            (true, 4),
            (false, 5),
            (true, 5)
        ]
    );
    let seen = seen.into_inner().unwrap();
    assert_eq!(seen[&0], ["running", "not run", "not run", "not run", "not run", "not run"]);
    assert_eq!(seen[&2], ["ok", "ok", "running", "not run", "not run", "not run"]);

    let rows = csv(&summary);
    let mut header = vec![
        "run",
        "status",
        "seed",
        "sample",
        "aircraft.7.tas_mps",
        "aircraft.7.offset.forward_m",
        "aircraft.7.offset.heading_deg",
        "sim_time_s",
        "wall_s",
        "speed_x",
        "bag",
    ];
    header.extend(METRICS.iter().filter(|id| **id != "sim_time_s"));
    assert_eq!(rows[0], header);
    for (row, run) in rows[1..].iter().zip(&sweep.runs) {
        assert_eq!(row[1], "ok");
        assert_eq!(row[3], (run.sample.unwrap() + 1).to_string());
        let drawn: Vec<f64> = row[5..7].iter().map(|c| c.parse().unwrap()).collect();
        assert_eq!(drawn, floats(&run.values[1..]));
        assert!(row[11..].iter().all(String::is_empty), "no bag, no metrics: {row:?}");
    }
}

#[test]
fn a_stopped_sweep_marks_the_runs_it_never_started() {
    let dir = tempfile::tempdir().unwrap();
    let mut sweep = plan(dir.path(), "repeats = 3").unwrap();
    sweep.out = dir.path().join("out");
    let stop = StopRequested::default();
    let events = Events::default();
    let outcomes = execute(
        &sweep,
        1,
        &stop,
        |e| events.on(e),
        |_| {
            stop.0.store(true, std::sync::atomic::Ordering::Relaxed);
            ok()
        },
    )
    .unwrap();
    assert_eq!(outcomes.iter().filter(|o| o.is_some()).count(), 1);
    events.assert_once_each(1);
    let status: Vec<String> =
        csv(&sweep.out.join("summary.csv"))[1..].iter().map(|r| r[1].clone()).collect();
    assert_eq!(status, ["ok", "not run", "not run"]);
}
