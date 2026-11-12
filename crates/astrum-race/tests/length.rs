//! `[course].length_m`: a target total length from which the leg limits and the course
//! radius are derived, per the gate count — so the user picks the number of gates and the
//! course stays about that long. Pure arithmetic and the sampler: no JSBSim.

mod common;

use astrum_race::course::CourseParams;
use astrum_race::scenario::MAX_GATES;
use astrum_race::{course_view, RaceScenario, RaceScenarioError};
use common::scenario_toml;

fn race(course_body: &str) -> Result<RaceScenario, RaceScenarioError> {
    RaceScenario::from_toml_str(&scenario_toml("", "", course_body))
}

fn effective(course_body: &str) -> CourseParams {
    race(course_body).unwrap().course.effective_params().unwrap()
}

fn err(course_body: &str) -> String {
    let e = race(course_body).and_then(|r| r.resolve(None)).unwrap_err();
    assert!(matches!(e, RaceScenarioError::Invalid(_)), "{e:?}");
    e.to_string()
}

/// Without `length_m` the envelope is exactly what the file says.
#[test]
fn without_a_length_the_envelope_is_as_written() {
    let defaults = CourseParams::default();
    assert_eq!(effective("gates = 8"), CourseParams { gates: 8, ..defaults });
    let p = effective("gates = 20\nleg_max_m = 40.0");
    assert_eq!((p.gates, p.leg_min_m, p.leg_max_m), (20, defaults.leg_min_m, 40.0));
    assert_eq!(p.course_radius_m, defaults.course_radius_m);
}

/// With `length_m`, the legs spread over `gates - 1` legs at 0.7x..1.3x the mean and the
/// radius is half the length (at least the default).
#[test]
fn a_length_derives_the_legs_and_the_radius() {
    let p = effective("gates = 8\nlength_m = 500.0");
    let mean = 500.0 / 7.0;
    assert!((p.leg_min_m - 0.7 * mean).abs() < 1e-9, "{}", p.leg_min_m);
    assert!((p.leg_max_m - 1.3 * mean).abs() < 1e-9, "{}", p.leg_max_m);
    assert_eq!(p.course_radius_m, 250.0);

    // A short course keeps at least the default radius.
    let short = effective("gates = 4\nlength_m = 120.0");
    assert_eq!(short.course_radius_m, CourseParams::default().course_radius_m);
}

/// A field the file states wins over its derived value; the rest still derive.
#[test]
fn stated_fields_win() {
    let p = effective("gates = 8\nlength_m = 500.0\nleg_min_m = 20.0\ncourse_radius_m = 400.0");
    assert_eq!((p.leg_min_m, p.course_radius_m), (20.0, 400.0));
    assert!((p.leg_max_m - 1.3 * 500.0 / 7.0).abs() < 1e-9, "leg_max_m still derives");
}

/// Changing the gate count re-derives the legs and keeps the length: that is what
/// `--gates` and the web's `gates` do (they set `course.gates` and resolve again).
#[test]
fn the_gate_count_rederives_the_legs() {
    let mut r = race("gates = 8\nlength_m = 500.0").unwrap();
    let eight = r.course.effective_params().unwrap();
    r.course.params.gates = 12;
    let twelve = r.course.effective_params().unwrap();
    assert!(twelve.leg_max_m < eight.leg_max_m, "more gates, shorter legs");
    assert!((twelve.leg_max_m - 1.3 * 500.0 / 11.0).abs() < 1e-9);
    let resolved = r.resolve(Some(5)).unwrap();
    assert_eq!(resolved.course.len(), 12);
    let length = resolved.course.total_length_m();
    assert!((length - 500.0).abs() < 75.0, "{length}");
    assert_eq!(resolved.view().length_m, length, "the view reports the realised length");
}

/// The realised length of the sampled legs stays near `length_m`, for any seed, at a few
/// gate counts. The legs are drawn uniformly over 0.7x..1.3x the mean, so their sum has a
/// standard deviation of ~17%/sqrt(gates-1) of the target: ~6% at 8 gates, the worst of
/// 200 seeds ~15%. The bound is 20%, and the average error must be small.
#[test]
fn the_realised_length_is_near_the_target() {
    for gates in [8u32, 12, 24] {
        let toml = scenario_toml("", "", &format!("gates = {gates}\nlength_m = 500.0"));
        let race = RaceScenario::from_toml_str(&toml).unwrap();
        let (mut worst, mut total): (f64, f64) = (0.0, 0.0);
        for seed in 0..200 {
            let resolved = race
                .resolve(Some(seed))
                .unwrap_or_else(|e| panic!("{gates} gates, seed {seed}: {e}"));
            assert_eq!(resolved.course.len(), gates as usize);
            let off = (resolved.course.total_length_m() - 500.0).abs() / 500.0;
            worst = worst.max(off);
            total += off;
            assert!(
                off < 0.20,
                "{gates} gates, seed {seed}: {:.1} m",
                resolved.course.total_length_m()
            );
        }
        assert!(total / 200.0 < 0.06, "{gates} gates: mean error {:.1}%", total / 2.0);
        eprintln!("{gates} gates: worst length error {:.1}%", worst * 100.0);
    }
}

/// The recorded scenario keeps its target, and the view of a plain envelope reports the
/// length it realised.
#[test]
fn the_length_survives_the_resolved_scenario() {
    let toml = scenario_toml("", "", "gates = 6\nlength_m = 300.0");
    let resolved = RaceScenario::from_toml_str(&toml).unwrap().resolve(Some(2)).unwrap();
    let text = resolved.to_toml();
    assert!(text.contains("length_m = 300.0"), "{text}");
    let again = RaceScenario::from_toml_str(&text).unwrap().resolve(None).unwrap();
    assert_eq!(again.course.gates, resolved.course.gates, "explicit gates, no re-derivation");
    let view = course_view(&toml, Some(2)).unwrap();
    assert_eq!(view.gates.len(), 6);
    assert_eq!(view.length_m, resolved.course.total_length_m());
}

/// Impossible combinations are errors that name the gate count and `length_m`, not an
/// exhausted sampler.
#[test]
fn infeasible_combinations_are_clear_errors() {
    let too_many = err("gates = 64\nlength_m = 500.0");
    assert!(
        too_many.contains("course.gates = 64") && too_many.contains("length_m = 500"),
        "{too_many}"
    );
    assert!(too_many.contains("too many gates"), "{too_many}");

    let too_few = err("gates = 3\nlength_m = 500.0");
    assert!(too_few.contains("course.gates = 3") && too_few.contains("too few gates"), "{too_few}");

    assert!(err("gates = 1").contains("between 2 and"));
    assert!(err("gates = 1\nlength_m = 500.0").contains("between 2 and"));
    let over = format!("gates = {}\nlength_m = 500.0", MAX_GATES + 1);
    assert!(err(&over).contains("between 2 and 64"));
    assert!(err("gates = 8\nlength_m = -5.0").contains("length_m"));
}
