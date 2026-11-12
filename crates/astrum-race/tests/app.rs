//! The assembled race: scenario resolution (what a seed, a pinned course seed or
//! explicit gates give), the resolved scenario that a recording stores, and a real
//! app on JSBSim. The tests that reach JSBSim take the process-wide FDM guard.

mod common;

use astrum_bag::BagReader;
use astrum_geo::Enu;
use astrum_proto::{decode_event, EndReason, EventKind, RaceOutcome};
use astrum_race::components::RaceProgress;
use astrum_race::course::{generate, CourseParams};
use astrum_race::gate::{BAR_M, OPENING_M, TOP_BAR_M};
use astrum_race::scenario::RaceScenario;
use astrum_race::{build_race_app, course_view, is_race_scenario, RaceScenarioError};
use astrum_sim::components::AircraftState;
use astrum_sim::{record, run_to_exit, world_frame, RunMode};
use bevy::math::DVec3;
use bevy::prelude::*;
use common::{fdm_guard, flat_course, scenario_toml};

fn resolve(toml: &str, seed: Option<u64>) -> astrum_race::scenario::ResolvedRace {
    RaceScenario::from_toml_str(toml).unwrap().resolve(seed).unwrap()
}

/// The shipped scenario parses, validates, resolves and is a race.
#[test]
fn shipped_scenario_is_valid() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../config/scenario.race.toml");
    let text = std::fs::read_to_string(path).unwrap();
    assert!(is_race_scenario(&text));
    let race = RaceScenario::from_toml_str(&text).expect("config/scenario.race.toml should load");
    let resolved = race.resolve(None).expect("and its envelope should be feasible");
    assert_eq!(resolved.course.len(), 8);
    assert_eq!(resolved.course.seed, race.sim.sim.seed, "no [course].seed: the sim's seed");
}

/// A scenario with no `[course]` table is not a race (to the web catalogue); one with
/// a `[course]` table is.
#[test]
fn a_course_table_marks_a_race() {
    assert!(is_race_scenario("[course]\ngates = 4"));
    assert!(!is_race_scenario("[sim]\nrate_hz = 120"));
    assert!(!is_race_scenario("not toml ["));
}

/// The course seed is the override, else `[course].seed`, else `sim.seed`; pinning
/// `[course].seed` makes the course independent of `sim.seed`.
#[test]
fn the_course_seed_can_be_pinned_apart_from_the_sim_seed() {
    // `scenario_toml` writes `[sim] seed = 1`.
    let unpinned = |sim_seed: u64| {
        scenario_toml("", "", "gates = 5").replace("seed = 1\n", &format!("seed = {sim_seed}\n"))
    };
    let pinned = |sim_seed: u64| unpinned(sim_seed).replace("gates = 5", "gates = 5\nseed = 9");

    let a = resolve(&unpinned(1), None);
    let b = resolve(&unpinned(2), None);
    assert_eq!((a.course.seed, b.course.seed), (1, 2), "the sim seed is the default");
    assert_ne!(a.course.gates, b.course.gates, "so a different sim seed is a different course");

    let (p1, p2) = (resolve(&pinned(1), None), resolve(&pinned(2), None));
    assert_eq!((p1.course.seed, p2.course.seed), (9, 9));
    assert_eq!(p1.course.gates, p2.course.gates, "a pinned course ignores the sim seed");
    let expect = generate(9, &CourseParams { gates: 5, ..CourseParams::default() }).unwrap();
    assert_eq!(p1.course.gates, expect.gates, "and is exactly what the sampler draws for 9");

    let o = resolve(&pinned(1), Some(4));
    assert_eq!(o.course.seed, 4, "an override beats the pin");
    assert_eq!(o.scenario.sim.seed, 1, "and leaves the sim's own randomness alone");
    assert_eq!(o.course.gates, generate(4, &expect.params).unwrap().gates);
}

/// Explicit `[[course.gate]]` are used verbatim and nothing is sampled; the drone is
/// put 10 m upstream of gate 0, on its centreline, facing along it.
#[test]
fn explicit_gates_are_used_verbatim_and_the_drone_is_aimed_at_them() {
    let gates = r#"
gates = 2
[[course.gate]]
index = 0
east_m = 12.5
north_m = -40.25
up_m = 2.0
yaw_rad = 1.0
[[course.gate]]
index = 1
east_m = 30.0
north_m = -30.0
up_m = 2.5
yaw_rad = 0.5
"#;
    let toml = scenario_toml("", "", gates);
    let r = resolve(&toml, None);
    let got: Vec<_> = r.course.gates.iter().map(|g| (g.index, g.center, g.yaw_rad)).collect();
    assert_eq!(
        got,
        [(0, DVec3::new(12.5, -40.25, 2.0), 1.0), (1, DVec3::new(30.0, -30.0, 2.5), 0.5)]
    );

    let frame = world_frame(&r.scenario);
    let drone = &r.scenario.aircraft[0];
    let at = frame.0.geodetic_to_enu(&astrum_geo::Geodetic::from_degrees(
        drone.lat_deg,
        drone.lon_deg,
        drone.alt_hae_m,
    ));
    let first = r.course.gates[0];
    let want = first.center - 10.0 * first.normal();
    assert!(
        (DVec3::new(at.e, at.n, at.u) - want).length() < 1e-3,
        "10 m up gate 0's centreline, at its height: {at:?} vs {want:?}"
    );
    assert!((drone.heading_deg - 1.0f64.to_degrees()).abs() < 1e-9);

    let err = RaceScenario::from_toml_str(&toml).unwrap().resolve(Some(3)).unwrap_err();
    assert!(matches!(err, RaceScenarioError::Invalid(_)), "a seed would do nothing: {err}");
}

/// The resolved scenario is what a recording stores: explicit gates, the effective
/// seed — and reading it back resolves to the same course, bit for bit, whatever the
/// sampler does in the future.
#[test]
fn the_resolved_scenario_replays_to_the_same_course() {
    let toml = scenario_toml("", "", "gates = 6\nseed = 21");
    let r = resolve(&toml, Some(7));
    let text = r.to_toml();
    assert!(text.contains("[[course.gate]]"), "gates are explicit:\n{text}");
    assert_eq!(text.matches("[[course.gate]]").count(), 6);

    let again = resolve(&text, None);
    assert_eq!(again.course.gates, r.course.gates, "the same gates, bit for bit");
    assert_eq!(again.course.seed, 7, "with the seed that drew them");
    assert_eq!(again.scenario, r.scenario, "and the same start pose");
    assert_eq!(again.to_toml(), text, "resolving a resolved scenario changes nothing");

    let view = course_view(&text, None).unwrap();
    assert_eq!(view, r.view());
    assert_eq!(
        (view.seed, view.opening_m, view.bar_m, view.top_bar_m),
        (7, OPENING_M, BAR_M, TOP_BAR_M)
    );
    assert_eq!(view.gates.len(), 6);
}

/// The web preview of a plain envelope samples; with a seed override it is that seed's
/// course.
#[test]
fn the_course_view_of_an_envelope_samples_it() {
    let toml = scenario_toml("", "", "gates = 4");
    let (a, b) = (course_view(&toml, None).unwrap(), course_view(&toml, Some(5)).unwrap());
    assert_eq!((a.seed, b.seed), (1, 5));
    assert_ne!(a.gates, b.gates);
    let sampled = generate(5, &CourseParams { gates: 4, ..CourseParams::default() }).unwrap();
    assert_eq!(b.gates[1].east_m, sampled.gates[1].center.x);
}

/// A scenario laid out for different gates cannot be flown on these ones.
#[test]
fn a_scenario_for_other_gates_is_refused() {
    let toml = scenario_toml("", "", "gates = 3\nopening_m = 0.5");
    let err = RaceScenario::from_toml_str(&toml).unwrap_err();
    assert!(err.to_string().contains("opening_m"), "{err}");
}

/// A race flies one F450 on the ground-level origin, and nothing else.
#[test]
fn only_the_race_airframe_is_accepted() {
    let toml = scenario_toml("", "", "gates = 3");
    let c172 = toml.replace("\"F450\"", "\"c172x\"");
    assert!(RaceScenario::from_toml_str(&c172).is_err());
    let raised = toml.replace("origin_alt_hae_m = 0.0", "origin_alt_hae_m = 5.0");
    assert!(RaceScenario::from_toml_str(&raised).is_err());
    let two = format!("{toml}\n[[aircraft]]\nid = 2\nmodel = \"F450\"\nlat_deg = 44.7\nlon_deg = -93.0\nalt_hae_m = 5.0\nheading_deg = 0.0\ntas_mps = 0.0\n");
    assert!(RaceScenario::from_toml_str(&two).is_err());
}

/// Startup spawns the drone through the sim's own fleet spawn: upstream of gate 0, on
/// its centreline, at its height, facing along it — and scores nothing yet.
#[test]
fn the_app_spawns_the_drone_on_the_approach() {
    let _fdm = fdm_guard();
    let race = resolve(&scenario_toml("", "", &flat_course(3.0, 4)), None);
    let first = race.course.gates[0];
    let mut app = build_race_app(race, RunMode::Embedded);
    app.update(); // Startup, then tick 1

    let mut q = app.world_mut().query::<&AircraftState>();
    let st = *q.single(app.world()).unwrap();
    let Enu { e, n, u } = st.pos_enu;
    let offset = DVec3::new(e, n, u) - first.center;
    let along = offset.dot(first.normal());
    assert!((along + 10.0).abs() < 0.5, "10 m upstream, got {along:.2}");
    assert!((offset - along * first.normal()).length() < 0.5, "on the centreline");
    let p = *app.world().resource::<RaceProgress>();
    assert_eq!((p.passed, p.missed, p.next_gate), (0, 0, 0));
    assert_eq!(p.outcome, RaceOutcome::Running);
}

/// `sim.max_time_s` ends the race `TimedOut` and the run `MaxTime`; the bag of the run
/// holds the resolved scenario, and replaying that text gives the course that was flown.
#[test]
fn a_recorded_run_carries_the_resolved_course_and_ends_on_the_time_limit() {
    let _fdm = fdm_guard();
    // 30 m up, an unflown drone takes ~2.5 s to fall: a 1 s limit ends it first.
    let toml = scenario_toml("max_time_s = 1.0", "", &flat_course(30.0, 4));
    let race = resolve(&toml, Some(11));
    let resolved_toml = race.to_toml();
    let flown_gates = race.course.gates.clone();

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("race.mcap");
    let (recorder, recording) = record::start(&path, &resolved_toml).unwrap();
    let mut app = build_race_app(race, RunMode::Batch);
    app.insert_resource(recorder);
    assert!(matches!(run_to_exit(&mut app), AppExit::Success));
    let p = *app.world().resource::<RaceProgress>();
    drop(app);
    let stats = recording.finish().unwrap();

    assert_eq!(p.outcome, RaceOutcome::TimedOut);
    assert_eq!(stats.telemetry, 121, "every tick from 0 to the limit");

    let bag = BagReader::open(&path).unwrap();
    let scenario = bag.scenario().expect("the bag carries its scenario");
    assert_eq!(scenario, resolved_toml);
    assert_eq!(scenario.matches("[[course.gate]]").count(), 4);
    assert_eq!(
        resolve(scenario, None).course.gates,
        flown_gates,
        "replaying reproduces the course"
    );

    let ends: Vec<EndReason> = bag
        .messages()
        .unwrap()
        .iter()
        .filter(|m| m.topic == astrum_net::keys::EVENTS)
        .filter_map(|m| match decode_event(&m.payload).unwrap().kind {
            EventKind::RunEnded { reason, .. } => Some(reason),
            _ => None,
        })
        .collect();
    assert_eq!(ends, [EndReason::MaxTime]);
}
