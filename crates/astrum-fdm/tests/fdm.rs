//! JSBSim FFI integration tests against the real libJSBSim and its data root
//! (`$JSBSIM_ROOT`, defaulting to the dev image's `/opt/jsbsim/share/jsbsim`).
//! No test-side locking: the crate serialises JSBSim's lifecycle itself.

use std::path::PathBuf;

use astrum_fdm::{Controls, Fdm, FdmError, InitialConditions, TrimMode};

fn root() -> PathBuf {
    std::env::var_os("JSBSIM_ROOT").map(PathBuf::from).unwrap_or("/opt/jsbsim/share/jsbsim".into())
}

const DT: f64 = 1.0 / 120.0;

fn c172x_trimmed(psi_deg: f64) -> Fdm {
    let mut f = Fdm::load(&root(), "c172x").expect("load c172x");
    f.set_initial_conditions(&InitialConditions {
        lat_geod_deg: 37.6188,
        lon_deg: -122.3756,
        alt_hae_m: 1524.0,
        psi_true_deg: psi_deg,
        tas_mps: 55.0,
        gamma_deg: 0.0,
        phi_deg: 0.0,
    })
    .unwrap();
    f.run_ic().unwrap();
    f.start_engines().unwrap();
    f.trim(TrimMode::Full).expect("c172x trims in level cruise");
    f.set_dt(DT).unwrap();
    f
}

#[test]
fn trimmed_aircraft_holds_level_flight() {
    let mut f = c172x_trimmed(90.0);
    let s0 = f.state().unwrap();
    for _ in 0..1200 {
        f.step().unwrap();
    }
    let s1 = f.state().unwrap();
    assert!((s1.sim_time_s - s0.sim_time_s - 10.0).abs() < 1e-6, "10 s of 1/120 s steps");
    // A trimmed start has no initial transient: speed, altitude and heading hold.
    assert!((s1.tas_mps - 55.0).abs() < 1.0, "TAS held near 55 m/s, got {}", s1.tas_mps);
    assert!((s1.alt_hae_m - 1524.0).abs() < 15.0, "altitude held, got {}", s1.alt_hae_m);
    assert!((s1.psi_rad.to_degrees() - 90.0).abs() < 2.0, "heading held, got {}", s1.psi_rad);
    assert!(s1.nz_g > 0.9 && s1.nz_g < 1.1, "≈1 g in level flight, got {}", s1.nz_g);
}

#[test]
fn unknown_properties_are_errors_not_new_nodes() {
    let mut f = Fdm::load(&root(), "c172x").unwrap();
    assert!(matches!(f.get("velocities/p-rad-sec-typo"), Err(FdmError::MissingProperty(_))));
    assert!(matches!(f.set("fcs/aileron-cmd-nrom", 0.5), Err(FdmError::MissingProperty(_))));
    // Still missing afterwards: the failed set did not create it.
    assert!(matches!(f.get("fcs/aileron-cmd-nrom"), Err(FdmError::MissingProperty(_))));
}

#[test]
fn unknown_model_is_a_load_error() {
    match Fdm::load(&root(), "no-such-aircraft") {
        Err(FdmError::Load(msg)) => assert!(msg.contains("no-such-aircraft"), "{msg}"),
        Err(e) => panic!("expected a load error, got {e}"),
        Ok(_) => panic!("loading a missing model must fail"),
    }
}

#[test]
fn failed_trim_is_reported_not_thrown() {
    // A C++ TrimFailureException must surface as an Err, never unwind into Rust.
    let mut f = Fdm::load(&root(), "c172x").unwrap();
    f.set_initial_conditions(&InitialConditions {
        lat_geod_deg: 37.0,
        lon_deg: -122.0,
        alt_hae_m: 1500.0,
        psi_true_deg: 0.0,
        tas_mps: 300.0, // far beyond what a C172 can sustain
        gamma_deg: 0.0,
        phi_deg: 0.0,
    })
    .unwrap();
    f.run_ic().unwrap();
    assert!(matches!(f.trim(TrimMode::Full), Err(FdmError::Jsbsim(_))));
}

#[test]
fn reset_reproduces_the_trajectory_bit_for_bit() {
    let fly = |f: &mut Fdm| {
        for k in 0..600 {
            let ail = (k as f64 * 0.02).sin() * 0.3;
            f.set_controls(Controls { aileron: ail, elevator: 0.0, rudder: 0.0, throttle: 0.7 })
                .unwrap();
            f.step().unwrap();
        }
        f.state().unwrap()
    };
    let mut f = c172x_trimmed(0.0);
    let first = fly(&mut f);
    f.reset().unwrap();
    f.start_engines().unwrap();
    f.trim(TrimMode::Full).unwrap();
    let second = fly(&mut f);
    assert_eq!(first, second, "reset + re-trim must replay the same trajectory exactly");
}

#[test]
fn instances_are_independent() {
    let mut a = c172x_trimmed(0.0);
    let mut b = c172x_trimmed(0.0);
    let up = Controls { elevator: -0.4, throttle: 0.7, ..Default::default() };
    let down = Controls { elevator: 0.4, throttle: 0.7, ..Default::default() };
    for _ in 0..600 {
        a.set_controls(up).unwrap();
        b.set_controls(down).unwrap();
        a.step().unwrap();
        b.step().unwrap();
    }
    let (alt_a, alt_b) = (a.state().unwrap().alt_hae_m, b.state().unwrap().alt_hae_m);
    assert!((alt_a - alt_b).abs() > 1.0, "instances must diverge: a={alt_a}, b={alt_b}");
}

#[test]
fn controls_are_clamped_and_nan_safe() {
    let c = Controls { aileron: 3.0, elevator: f64::NAN, rudder: -2.0, throttle: 1.5 }.clamped();
    assert_eq!(c, Controls { aileron: 1.0, elevator: 0.0, rudder: -1.0, throttle: 1.0 });
}

const G: f64 = 9.80665;

fn trimmed(model: &str, alt_hae_m: f64, tas_mps: f64, gamma_deg: f64, phi_deg: f64) -> Fdm {
    let mut f = Fdm::load(&root(), model).expect("load");
    f.set_initial_conditions(&InitialConditions {
        lat_geod_deg: 37.6188,
        lon_deg: -122.3756,
        alt_hae_m,
        psi_true_deg: 0.0,
        tas_mps,
        gamma_deg,
        phi_deg,
    })
    .unwrap();
    f.run_ic().unwrap();
    f.start_engines().unwrap();
    let mode = if phi_deg == 0.0 { TrimMode::Full } else { TrimMode::Turn };
    f.trim(mode).unwrap_or_else(|e| panic!("{model} trims (γ {gamma_deg}°, φ {phi_deg}°): {e}"));
    f.set_dt(DT).unwrap();
    f
}

fn climb_rate(s: &astrum_fdm::FdmState) -> f64 {
    -s.vel_down_mps
}

/// Heading change over one second of free flight (rad/s), wrapped to ±π.
fn heading_rate_over_1s(f: &mut Fdm) -> (f64, astrum_fdm::FdmState) {
    let psi0 = f.state().unwrap().psi_rad;
    for _ in 0..120 {
        f.step().unwrap();
    }
    let s = f.state().unwrap();
    let d = (s.psi_rad - psi0 + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU)
        - std::f64::consts::PI;
    (d, s)
}

#[test]
fn trimmed_climb_starts_and_holds_the_flight_path_angle() {
    // Measured: both start within 0.1 % of TAS·sin γ and lose < 0.5 % over 5 s
    // (speed bleeds slowly); 1 % / 3 % leave margin without admitting a level start.
    for (model, alt, tas) in [("c172x", 1000.0, 45.0), ("f16", 3000.0, 200.0)] {
        let mut f = trimmed(model, alt, tas, 5.0, 0.0);
        let want = tas * 5f64.to_radians().sin();
        let s0 = f.state().unwrap();
        assert!(
            (climb_rate(&s0) - want).abs() < 0.01 * want,
            "{model}: initial climb {} m/s, want ≈ {want}",
            climb_rate(&s0)
        );
        for _ in 0..5 {
            let (_, s) = heading_rate_over_1s(&mut f);
            assert!(
                (climb_rate(&s) - want).abs() < 0.03 * want,
                "{model}: climb {} m/s at {} s, want ≈ {want}",
                climb_rate(&s),
                s.sim_time_s
            );
            assert!(s.phi_rad.to_degrees().abs() < 1.0, "{model}: wings stay level");
        }
    }
}

#[test]
fn turn_trim_holds_a_coordinated_bank() {
    // Measured on the c172x: heading rate within 0.1 % of g·tanφ/V, φ drifts
    // < 0.01°, climb rate < 0.01 m/s over 5 s — the trim is a true steady turn.
    for phi in [30.0, -30.0] {
        let tas = 45.0;
        let mut f = trimmed("c172x", 1000.0, tas, 0.0, phi);
        let want = G * f64::tan(phi.to_radians()) / tas; // + = turning right
        assert!((f.state().unwrap().phi_rad.to_degrees() - phi).abs() < 0.1, "initial φ");
        for sec in 1..=5 {
            let (rate, s) = heading_rate_over_1s(&mut f);
            assert!(
                (rate - want).abs() < 0.02 * want.abs(),
                "φ {phi}: heading rate {rate} rad/s in second {sec}, want ≈ {want}"
            );
            assert!((s.phi_rad.to_degrees() - phi).abs() < 0.5, "φ {phi}: held, got {s:?}");
            assert!(climb_rate(&s).abs() < 0.5, "φ {phi}: level turn, got {s:?}");
        }
    }
}

#[test]
fn f16_turn_trim_starts_in_the_turn() {
    // The trim itself is exact (φ = 30°, body rates of the steady turn), but the
    // F-16's fly-by-wire yaw damper and filters start from zero state and pull
    // the turn rate to ≈ 0.6–0.8 of g·tanφ/V within the first second while φ
    // relaxes ≈ 1.3°/s (measured). So pin the trimmed start, and only that the
    // turn continues the right way for a few seconds.
    let (tas, phi) = (200.0, 30.0);
    let mut f = trimmed("f16", 5000.0, tas, 0.0, phi);
    let omega = G * f64::tan(phi.to_radians()) / tas;
    let s0 = f.state().unwrap();
    let (sp, cp, ct) = (s0.phi_rad.sin(), s0.phi_rad.cos(), s0.theta_rad.cos());
    assert!((s0.phi_rad.to_degrees() - phi).abs() < 0.1, "initial φ {s0:?}");
    assert!((s0.q_rad_s - omega * sp * ct).abs() < 0.05 * omega, "trimmed pitch rate {s0:?}");
    assert!((s0.r_rad_s - omega * cp * ct).abs() < 0.05 * omega, "trimmed yaw rate {s0:?}");
    for sec in 1..=3 {
        let (rate, s) = heading_rate_over_1s(&mut f);
        assert!(rate > 0.5 * omega && rate < 1.05 * omega, "second {sec}: heading rate {rate}");
        assert!((s.phi_rad.to_degrees() - phi).abs() < 6.0, "second {sec}: φ {s:?}");
    }
}
