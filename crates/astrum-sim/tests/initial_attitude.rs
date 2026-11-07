//! `climb_deg` / `bank_deg`: trimmed aircraft start in the steady climbing turn,
//! untrimmed ones start with the attitude as given — on build and on reset.

mod common;

use astrum_sim::env::SimEnv;
use common::{of, scenario, single_quad, spec, ORIGIN};

#[test]
fn untrimmed_multirotor_starts_banked() {
    let mut s = single_quad();
    s.aircraft[0].bank_deg = 10.0;
    let mut env = SimEnv::new(s).unwrap();
    // The first telemetry follows one tick from rest at zero body rates, so φ is
    // still the IC to rounding; the next tick of SCAS correction moves it little.
    let phi0 = env.observe()[0].attitude.phi_rad.to_degrees();
    assert!((phi0 - 10.0).abs() < 1e-6, "spawn φ {phi0}");
    let (obs, _) = env.step(&[]).unwrap();
    let phi1 = obs[0].attitude.phi_rad.to_degrees();
    assert!((phi1 - 10.0).abs() < 0.5, "first step φ {phi1}");
    let phi_reset = env.reset(None, None).unwrap()[0].attitude.phi_rad.to_degrees();
    assert!((phi_reset - 10.0).abs() < 1e-6, "reset φ {phi_reset}");
}

#[test]
fn untrimmed_fixed_wing_starts_with_the_ics_as_given() {
    let mut c172 = spec(1, "c172x", ORIGIN.0, 123.0, 1000.0, 45.0);
    (c172.trim, c172.climb_deg, c172.bank_deg) = (Some(false), 4.0, -15.0);
    let mut env = SimEnv::new(scenario(vec![c172])).unwrap();
    // The first telemetry is one 1/120 s tick after spawn. φ (zero body rates)
    // is still the IC to rounding; the untrimmed airframe, at α = 0 short of
    // lift, has already sunk 0.049 m/s (measured) — 0.1 m/s bounds that tick.
    for (t, what) in [(env.observe()[0], "spawn"), (env.reset(None, None).unwrap()[0], "reset")] {
        let want_d = -45.0 * 4f64.to_radians().sin();
        assert!((t.vel_ned.d - want_d).abs() < 0.1, "{what}: vel_ned.d {}", t.vel_ned.d);
        assert!((t.attitude.phi_rad.to_degrees() + 15.0).abs() < 1e-6, "{what}: φ {t:?}");
    }
}

#[test]
fn trimmed_fixed_wing_starts_in_a_climbing_turn() {
    let mut c172 = spec(1, "c172x", ORIGIN.0, 0.0, 1000.0, 45.0);
    c172.climb_deg = 3.0;
    c172.bank_deg = -30.0;
    let mut env = SimEnv::new(scenario(vec![c172])).unwrap();
    let check = |t: astrum_proto::Telemetry, what: &str| {
        let phi = t.attitude.phi_rad.to_degrees();
        let want = 45.0 * 3f64.to_radians().sin();
        assert!((phi + 30.0).abs() < 0.5, "{what}: φ {phi}");
        assert!((-t.vel_ned.d - want).abs() < 0.05 * want, "{what}: climb {}", -t.vel_ned.d);
    };
    check(env.observe()[0], "spawn");
    // Hold the trimmed controls: no command means the trim stays in.
    let mut t = env.observe()[0];
    for _ in 0..600 {
        t = of(&env.step(&[]).unwrap().0, 1);
    }
    check(t, "after 5 s");
    check(env.reset(None, None).unwrap()[0], "reset");
}
