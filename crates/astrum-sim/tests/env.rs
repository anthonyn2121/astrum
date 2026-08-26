//! Integration tests for the embedded RL env seam ([`astrum_sim::env`]).
//!
//! These drive the *real* app (JSBSim FDMs via `build_app`) through the
//! [`SimEnv`] `reset`/`step`/`observe` API — the same surface the pyo3 binding
//! (`crates/astrum-env`) marshals — proving the seam works headlessly from Rust
//! before any Python is involved. Because they build FDMs they take the
//! process-wide [`fdm_guard`] first, exactly like the other JSBSim-backed tests.
//!
//! Run:  `cargo test -p astrum-sim --test env -- --nocapture`

mod common;

use astrum_sim::env::{Action, AgentObs, SimEnv};
use astrum_sim::scenario::{AircraftSpec, Scenario, Sim, World};
use common::fdm_guard;

const DT: f64 = 1.0 / 120.0;

/// A c172x spec at the SF-Bay datum, heading `hdg`, on `team`.
fn c172x(id: u32, team: u8, lat_deg: f64, heading_deg: f64) -> AircraftSpec {
    AircraftSpec {
        id,
        model: "c172x".into(),
        kind: None,
        lat_deg,
        lon_deg: -122.3756,
        alt_ft: 5000.0,
        heading_deg,
        airspeed_kts: 100.0,
        team,
        health: 100.0,
        ammo: None,
    }
}

/// Two c172x on opposing teams (a real, self-terminating engagement), fastest
/// mode, uncapped time.
fn two_team() -> Scenario {
    Scenario {
        world: World { origin_lat_deg: 37.6188, origin_lon_deg: -122.3756, origin_alt_m: 0.0 },
        sim: Sim { rate_hz: 120.0, max_time_s: 0.0, realtime: false, seed: 7 },
        aircraft: vec![c172x(1, 0, 37.6188, 90.0), c172x(2, 1, 37.6190, 270.0)],
    }
}

/// One c172x (single team → never self-terminates) with a hard time cap, for the
/// truncation test.
fn single_capped(max_time_s: f64) -> Scenario {
    Scenario {
        world: World { origin_lat_deg: 37.6188, origin_lon_deg: -122.3756, origin_alt_m: 0.0 },
        sim: Sim { rate_hz: 120.0, max_time_s, realtime: false, seed: 1 },
        aircraft: vec![c172x(1, 0, 37.6188, 90.0)],
    }
}

fn obs_of(obs: &[AgentObs], id: u32) -> AgentObs {
    *obs.iter().find(|o| o.id == id).expect("aircraft present in observation")
}

/// Every field of an observation as raw bits — for byte-exact determinism checks.
fn obs_bits(o: &AgentObs) -> Vec<u64> {
    vec![
        o.id as u64,
        o.team as u64,
        o.alive as u64,
        o.pos_e.to_bits(),
        o.pos_n.to_bits(),
        o.pos_u.to_bits(),
        o.vel_n.to_bits(),
        o.vel_e.to_bits(),
        o.vel_d.to_bits(),
        o.phi.to_bits(),
        o.theta.to_bits(),
        o.psi.to_bits(),
        o.p.to_bits(),
        o.q.to_bits(),
        o.r.to_bits(),
        o.alpha_rad.to_bits(),
        o.beta_rad.to_bits(),
        o.nz_g.to_bits(),
        o.airspeed_mps.to_bits(),
        o.alt_m.to_bits(),
        o.health.to_bits() as u64,
        o.ammo as u64,
        o.sim_time_s.to_bits(),
    ]
}

fn fleet_bits(obs: &[AgentObs]) -> Vec<Vec<u64>> {
    obs.iter().map(obs_bits).collect()
}

#[test]
fn reset_observes_full_fleet() {
    let _fdm = fdm_guard();
    let mut env = SimEnv::new(two_team());

    let obs = env.reset();
    assert_eq!(obs.len(), 2, "both aircraft should be observed");
    assert_eq!(obs.iter().map(|o| o.id).collect::<Vec<_>>(), vec![1, 2], "sorted by id");

    for o in &obs {
        assert!(o.alive, "aircraft {} starts alive", o.id);
        assert!((o.health - 100.0).abs() < f32::EPSILON, "starts at full HP");
        // c172x is fixed-wing → armed with the M61A1's 511-round magazine.
        assert_eq!(o.ammo, 511, "fixed-wing aircraft {} starts armed", o.id);
        assert!(o.p.is_finite() && o.q.is_finite() && o.r.is_finite(), "body rates finite");
        assert!(
            o.alpha_rad.is_finite() && o.beta_rad.is_finite() && o.nz_g.is_finite(),
            "aero angles + load factor finite"
        );
        assert!(o.nz_g.abs() < 10.0, "trimmed flight is near 1 g, got Nz={} g", o.nz_g);
        assert!(o.airspeed_mps > 0.0, "trimmed to ~100 kts, got {} m/s", o.airspeed_mps);
        assert!(o.sim_time_s > 0.0, "one settling tick has run at reset");
    }

    // A neutral step: the engagement isn't decided (both alive, two teams) and
    // there's no time cap to hit.
    let neutral = [(1, Action::default()), (2, Action::default())];
    let (_, result) = env.step(&neutral);
    assert!(!result.terminated, "both teams alive ⇒ not terminated");
    assert!(!result.truncated, "uncapped scenario ⇒ not truncated");
}

#[test]
fn continuous_action_drives_the_fdm() {
    // Full aileron on agent 1 must produce a real roll rate, while a neutral
    // agent 2 stays roughly wings-level — proving the continuous channels reach
    // the FDM through the env.
    let _fdm = fdm_guard();
    let mut env = SimEnv::new(two_team());
    env.reset();

    let roll = Action { aileron: 1.0, throttle: 0.6, ..Default::default() };
    let neutral = Action { throttle: 0.6, ..Default::default() };
    let mut last = Vec::new();
    for _ in 0..40 {
        let (obs, _) = env.step(&[(1, roll), (2, neutral)]);
        last = obs;
    }

    let a1 = obs_of(&last, 1);
    let a2 = obs_of(&last, 2);
    assert!(
        a1.p.abs() > 0.3,
        "full aileron should command a clear roll rate, got p={} rad/s",
        a1.p
    );
    assert!(
        a2.p.abs() < 0.3,
        "a neutral aircraft should stay near wings-level, got p={} rad/s",
        a2.p
    );
}

#[test]
fn trigger_action_consumes_ammo() {
    // The trigger is a learned action (design doc §8): holding it must fire and
    // deplete the magazine; not holding it must not.
    let _fdm = fdm_guard();
    let mut env = SimEnv::new(two_team());
    env.reset();

    let firing = Action { throttle: 0.6, fire: true, ..Default::default() };
    let holding = Action { throttle: 0.6, fire: false, ..Default::default() };
    let mut last = Vec::new();
    for _ in 0..40 {
        let (obs, _) = env.step(&[(1, firing), (2, holding)]);
        last = obs;
    }

    let shooter = obs_of(&last, 1);
    let holder = obs_of(&last, 2);
    assert!(shooter.ammo < 511, "a held trigger fires rounds, ammo={}", shooter.ammo);
    assert!(shooter.ammo > 470, "~33 rounds in 40 ticks at 100 Hz, ammo={}", shooter.ammo);
    assert_eq!(holder.ammo, 511, "a released trigger fires nothing");
}

#[test]
fn rollout_is_deterministic() {
    // Two envs from the same scenario, driven by an identical action sequence,
    // must produce byte-identical observations at every step — the reproducibility
    // an RL trainer relies on. (Purely FDM determinism here: no firing.)
    let _fdm = fdm_guard();

    let action_at = |k: usize| {
        let ail = ((k as f64) * 0.05).sin(); // a varying but deterministic input
        Action { aileron: ail, elevator: -0.05, throttle: 0.7, ..Default::default() }
    };

    let run = || {
        let mut env = SimEnv::new(two_team());
        env.reset();
        let mut frames = Vec::new();
        for k in 0..120 {
            let a = action_at(k);
            let (obs, _) = env.step(&[(1, a), (2, a)]);
            frames.push(fleet_bits(&obs));
        }
        frames
    };

    let a = run();
    let b = run();
    assert_eq!(a, b, "identical scenario + actions must yield byte-identical observations");
}

#[test]
fn engagement_terminates_on_reset_boundary() {
    // reset() must clear a prior episode's termination: after stepping a two-team
    // engagement, a fresh reset shows both teams alive again and not terminated.
    let _fdm = fdm_guard();
    let mut env = SimEnv::new(two_team());
    env.reset();

    // Sanity: fresh reset is a live, undecided two-team state.
    let (_, before) = env.step(&[(1, Action::default()), (2, Action::default())]);
    assert!(!before.terminated);

    let obs = env.reset();
    assert_eq!(obs.len(), 2);
    assert!(obs.iter().all(|o| o.alive), "reset restores both aircraft alive");
}

#[test]
fn episode_truncates_at_max_time() {
    // A single-team, time-capped scenario never *terminates* (no fight) but must
    // *truncate* once sim time reaches the cap.
    let _fdm = fdm_guard();
    let cap = 0.5;
    let mut env = SimEnv::new(single_capped(cap));
    env.reset();

    let mut truncated = false;
    let mut ticks = 0;
    for _ in 0..500 {
        let (_, result) = env.step(&[(1, Action { throttle: 0.6, ..Default::default() })]);
        ticks += 1;
        assert!(!result.terminated, "a single-team scenario never self-terminates");
        if result.truncated {
            truncated = true;
            assert!(result.sim_time_s >= cap, "truncation implies the cap was reached");
            break;
        }
    }
    assert!(truncated, "the episode should truncate at the time cap");
    // ~cap/dt steps, plus the settling tick already consumed at reset.
    assert!(ticks <= (cap / DT) as i32 + 2, "truncation should fire right at the cap");
}
