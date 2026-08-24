//! JSBSim FFI integration tests: load the bundled `c172x`, step it, and confirm
//! each `Fdm` owns a fully independent flight-dynamics instance.
//!
//! These drive a real `libJSBSim` and need its data root; `JSBSIM_ROOT` points at
//! it in the dev container (defaulting to `/opt/jsbsim/share/jsbsim`).

use std::sync::Mutex;

use forge_fdm::{prop, Fdm};

/// JSBSim has process-global initialisation state, so constructing `FGFDMExec`
/// from two threads at once races and segfaults. Cargo runs a binary's tests in
/// parallel by default, so we serialise the JSBSim-touching tests here — this
/// keeps a plain `cargo test` reliable without needing `--test-threads=1`.
static JSBSIM_LOCK: Mutex<()> = Mutex::new(());

fn jsbsim_root() -> String {
    std::env::var("JSBSIM_ROOT").unwrap_or_else(|_| "/opt/jsbsim/share/jsbsim".into())
}

fn spawn_c172x() -> Fdm {
    let mut f = Fdm::new();
    f.set_root_dir(&jsbsim_root());
    assert!(f.load_model("c172x"), "failed to load c172x aircraft");
    f.set("ic/h-sl-ft", 5000.0);
    f.set("ic/vc-kts", 100.0);
    f.set("ic/gamma-deg", 0.0);
    assert!(f.run_ic(), "RunIC failed");
    f.set_dt(1.0 / 120.0);
    f
}

#[test]
fn loads_and_steps() {
    let _guard = JSBSIM_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut fdm = spawn_c172x();
    fdm.set(prop::THROTTLE_CMD, 0.6);
    let t0 = fdm.get(prop::SIM_TIME);
    for _ in 0..240 {
        assert!(fdm.step(), "step returned false");
    }
    let elapsed = fdm.get(prop::SIM_TIME) - t0;
    assert!((elapsed - 2.0).abs() < 1e-3, "sim time should advance ~2s, got {elapsed}");
    assert!(fdm.get(prop::VT_FPS) > 0.0, "aircraft should be moving");
    let alt = fdm.get(prop::ALT_SL_M);
    assert!(alt.is_finite() && alt > 0.0, "altitude should be finite/positive, got {alt}");
}

#[test]
fn instances_are_independent() {
    let _guard = JSBSIM_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    // Two FDMs, same start, opposite elevator -> their altitudes must diverge,
    // proving each entity owns a fully independent flight-dynamics instance.
    let mut a = spawn_c172x();
    let mut b = spawn_c172x();
    a.set(prop::ELEVATOR_CMD, -0.4);
    b.set(prop::ELEVATOR_CMD, 0.4);
    for _ in 0..600 {
        a.step();
        b.step();
    }
    let (alt_a, alt_b) = (a.get(prop::ALT_SL_M), b.get(prop::ALT_SL_M));
    assert!(
        (alt_a - alt_b).abs() > 1.0,
        "independent instances should diverge: a={alt_a}, b={alt_b}"
    );
}
