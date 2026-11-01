//! The embedded env seam (`astrum_sim::env::SimEnv`): what an RL trainer relies
//! on — observation layout, actions reaching the FDM, the trigger, bit-exact
//! determinism, and a fast reset that is indistinguishable from a fresh build.

mod common;

use astrum_proto::{Control, RawActuators, Telemetry};
use astrum_sim::env::{Action, EnvError, SimEnv};
use common::{of, raw, single_ship, two_f16_opposing, two_ship, DT};

fn varying(k: usize, fire: bool) -> Action {
    let ail = (k as f64 * 0.05).sin() * 0.4;
    Action {
        control: Control::Raw(RawActuators {
            aileron: ail,
            elevator: -0.05,
            rudder: 0.0,
            throttle: 0.7,
        }),
        trigger: fire,
    }
}

fn rollout(env: &mut SimEnv, steps: usize) -> Vec<Vec<Telemetry>> {
    (0..steps)
        .map(|k| env.step(&[(1, varying(k, true)), (2, varying(k + 7, false))]).unwrap().0)
        .collect()
}

#[test]
fn first_observation_is_the_trimmed_fleet_at_tick_1() {
    let env = SimEnv::new(two_f16_opposing()).unwrap();
    let obs = env.observe();
    assert_eq!(obs.iter().map(|t| t.aircraft_id).collect::<Vec<_>>(), vec![1, 2], "sorted by id");
    for t in &obs {
        assert_eq!(t.tick, 1);
        assert!(t.combat.alive && t.combat.hit_points == 100.0);
        assert_eq!(t.combat.ammo, 511, "an F-16 carries the M61A1's 511 rounds");
        assert!(
            (t.aero.nz_g - 1.0).abs() < 0.05,
            "trimmed level flight is ~1 g, got {}",
            t.aero.nz_g
        );
    }
    assert_eq!(of(&obs, 1).combat.team, 1);
}

#[test]
fn raw_action_reaches_the_fdm() {
    let mut env = SimEnv::new(two_ship()).unwrap();
    let mut obs = Vec::new();
    for _ in 0..40 {
        obs = env.step(&[(1, raw(1.0, 0.0, 0.6)), (2, raw(0.0, 0.0, 0.6))]).unwrap().0;
    }
    assert!(of(&obs, 1).rates.p_rad_s > 0.3, "full right aileron rolls right");
    assert!(of(&obs, 2).rates.p_rad_s.abs() < 0.1, "neutral stays wings-level");
}

#[test]
fn held_trigger_fires_at_the_gun_rate() {
    let mut env = SimEnv::new(two_f16_opposing()).unwrap();
    let mut obs = Vec::new();
    for _ in 0..120 {
        let fire = Action { trigger: true, ..raw(0.0, 0.0, 0.7) };
        obs = env.step(&[(2, fire), (1, raw(0.0, 0.0, 0.7))]).unwrap().0;
    }
    // 1 s at 100 rounds/s.
    assert_eq!(of(&obs, 2).combat.ammo, 511 - 100);
    assert_eq!(of(&obs, 1).combat.ammo, 511);
}

#[test]
fn identical_runs_are_bit_identical() {
    let a = rollout(&mut SimEnv::new(two_f16_opposing()).unwrap(), 240);
    let b = rollout(&mut SimEnv::new(two_f16_opposing()).unwrap(), 240);
    assert_eq!(a, b);
}

#[test]
fn reset_is_indistinguishable_from_a_fresh_build() {
    let mut env = SimEnv::new(two_f16_opposing()).unwrap();
    let first = rollout(&mut env, 240);
    // Reuse the loaded FDMs (and the bullets/damage of the first episode).
    let obs0 = env.reset(None, None).unwrap();
    assert_eq!(obs0, SimEnv::new(two_f16_opposing()).unwrap().observe());
    assert_eq!(rollout(&mut env, 240), first);
}

#[test]
fn reset_with_new_initial_conditions_matches_a_fresh_env_of_that_scenario() {
    let mut moved = two_f16_opposing();
    moved.aircraft[1].alt_hae_m += 300.0;
    moved.aircraft[1].heading_deg = 20.0;
    moved.sim.seed = 9;
    let mut env = SimEnv::new(two_f16_opposing()).unwrap();
    rollout(&mut env, 60);
    let via_reset = env.reset(None, Some(moved.clone())).unwrap();
    let mut fresh = SimEnv::new(moved).unwrap();
    assert_eq!(via_reset, fresh.observe());
    assert_eq!(rollout(&mut env, 120), rollout(&mut fresh, 120));
}

#[test]
fn episode_truncates_at_the_time_cap() {
    let mut s = single_ship();
    s.sim.max_time_s = 0.5;
    let mut env = SimEnv::new(s).unwrap();
    let mut steps = 0;
    loop {
        let (_, r) = env.step(&[]).unwrap();
        steps += 1;
        assert!(!r.terminated, "a single-team scenario never terminates");
        if r.truncated {
            assert!(r.sim_time_s >= 0.5 && r.sim_time_s < 0.5 + DT);
            break;
        }
        assert!(steps < 100);
    }
}

#[test]
fn actions_for_unknown_aircraft_are_errors() {
    let mut env = SimEnv::new(single_ship()).unwrap();
    assert_eq!(env.step(&[(99, raw(0.0, 0.0, 0.5))]).unwrap_err(), EnvError::UnknownAircraft(99));
}

#[test]
fn invalid_scenarios_are_rejected_before_spawning() {
    let mut s = single_ship();
    s.aircraft[0].model = "no-such-aircraft".into();
    assert!(
        matches!(SimEnv::new(s), Err(EnvError::Scenario(_))),
        "an unknown model needs an explicit kind"
    );
    let mut s = single_ship();
    s.aircraft[0].model = "no-such-aircraft".into();
    s.aircraft[0].kind = Some(astrum_sim::components::VehicleKind::FixedWing);
    s.aircraft[0].autopilot_profile = Some(astrum_sim::components::AutopilotProfile::Trainer);
    assert!(
        matches!(SimEnv::new(s), Err(EnvError::Sim(_))),
        "a model JSBSim cannot load is a sim error"
    );
}
