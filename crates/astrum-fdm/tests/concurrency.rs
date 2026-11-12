//! Regression test for JSBSim's construction race: building instances from many
//! threads at once used to segfault (c172x) or abort (F450) on the first
//! `Element` parse in a process. Kept in its own test binary so it is the first
//! JSBSim activity in the process — the only time the race window is open.

use std::path::PathBuf;
use std::sync::{Arc, Barrier};

use astrum_fdm::{Fdm, InitialConditions};

#[test]
fn concurrent_construction_and_stepping_from_safe_threads() {
    let root: PathBuf = std::env::var_os("JSBSIM_ROOT")
        .map(PathBuf::from)
        .unwrap_or("/opt/jsbsim/share/jsbsim".into());
    let models = ["c172x", "F450", "f16", "c172x", "F450", "f16", "c172x", "F450"];
    let start = Arc::new(Barrier::new(models.len()));
    let handles: Vec<_> = models
        .iter()
        .enumerate()
        .map(|(i, model)| {
            let (root, start, model) = (root.clone(), Arc::clone(&start), model.to_string());
            std::thread::spawn(move || {
                start.wait(); // maximise overlap of the first loads
                let mut f = Fdm::load(&root, &model).expect("load");
                f.set_initial_conditions(&InitialConditions {
                    lat_geod_deg: 37.0 + i as f64 * 0.01,
                    lon_deg: -122.0,
                    alt_hae_m: 1500.0,
                    psi_true_deg: 0.0,
                    tas_mps: if model == "F450" { 0.0 } else { 60.0 },
                    gamma_deg: 0.0,
                    phi_deg: 0.0,
                })
                .unwrap();
                f.run_ic().unwrap();
                f.set_dt(1.0 / 120.0).unwrap();
                for _ in 0..240 {
                    f.step().unwrap();
                }
                f.state().unwrap().sim_time_s
            })
        })
        .collect();
    for h in handles {
        let t = h.join().expect("a JSBSim thread panicked");
        assert!((t - 2.0).abs() < 1e-6);
    }
}
