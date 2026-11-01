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
