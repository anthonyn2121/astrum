//! Scale / mixed-fleet integration test. Builds the real headless app with a
//! 40-entity fleet — 20 fixed-wing (`c172x`) and 20 quadrotors (`F450`) — engages
//! every aircraft's (vehicle-class-specific) autopilot, and steps the whole fleet
//! together. Proves the engine runs many independent FDMs of *both* classes
//! concurrently: all spawn, none diverge, quads hold a level hover-climb while
//! fixed-wing craft cruise, and every entity carries its own state.

mod common;

use astrum_sim::build_app;
use astrum_sim::components::VehicleKind;
use common::{
    alt_of, attitude_of, engage_ap, fdm_guard, heading_of, kind_of, mixed_fleet, snapshot,
};

const N: u32 = 40;

#[test]
fn mixed_fleet_of_40_flies_without_divergence() {
    let _fdm = fdm_guard();
    let mut app = build_app(mixed_fleet(N));
    app.update(); // Startup: build one FDM per aircraft + first step.

    // Whole fleet spawned as an even mix of the two vehicle classes.
    let start = snapshot(&mut app);
    assert_eq!(start.len() as u32, N, "all {N} aircraft should spawn");
    let (mut fixed, mut multi) = (0u32, 0u32);
    for id in 1..=N {
        match kind_of(&mut app, id) {
            VehicleKind::FixedWing => fixed += 1,
            VehicleKind::Multirotor => multi += 1,
        }
    }
    assert_eq!((fixed, multi), (20, 20), "20/20 class mix, got {fixed} fixed / {multi} multi");

    // Engage every autopilot: hold the start heading, climb 15 m, and (fixed-wing)
    // hold ~100 kt cruise. Quads ignore the airspeed target. Each craft is flown by
    // the inner loop for its own class.
    let mut targets = Vec::with_capacity(N as usize);
    for id in 1..=N {
        let (hdg, target_alt) = (heading_of(&mut app, id), alt_of(&mut app, id) + 15.0);
        targets.push((id, hdg, target_alt));
        engage_ap(&mut app, id, hdg, target_alt, 51.0);
    }

    // Fly the whole fleet ~10 s of deterministic 120 Hz steps.
    for _ in 0..1200 {
        app.update();
    }

    let end = snapshot(&mut app);
    assert_eq!(end.len() as u32, N, "no aircraft lost");
    assert!((end[0].2 - 10.0).abs() < 0.1, "sim advanced ~10 s, got {}", end[0].2);
    for (id, pos, _) in &end {
        assert!(pos.is_finite(), "aircraft {id} position must stay finite (no divergence)");
    }

    // Per-class behaviour under load.
    for (id, _target_hdg, target_alt) in &targets {
        let (id, target_alt) = (*id, *target_alt);
        let alt = alt_of(&mut app, id);
        let (phi, theta) = attitude_of(&mut app, id);
        assert!(alt.is_finite() && phi.is_finite() && theta.is_finite(), "aircraft {id} finite");
        match kind_of(&mut app, id) {
            VehicleKind::Multirotor => {
                // Hover-hold: climbed to the commanded altitude and stayed level.
                assert!(
                    (alt - target_alt).abs() < 5.0,
                    "quad {id} should hold ~{target_alt:.0} m, got {alt:.1} m"
                );
                assert!(
                    phi.abs() < 0.15 && theta.abs() < 0.15,
                    "quad {id} should stay near level, φ={phi:.2} θ={theta:.2}"
                );
            }
            VehicleKind::FixedWing => {
                // Cruised a meaningful distance from its start.
                let start_pos = start.iter().find(|r| r.0 == id).unwrap().1;
                let end_pos = end.iter().find(|r| r.0 == id).unwrap().1;
                let moved = (end_pos - start_pos).length();
                assert!(moved > 50.0, "plane {id} should fly, moved only {moved:.0} m");
            }
        }
    }

    // Per-entity isolation at scale: the 20 hovering quads stay near their (spread)
    // start points, so their positions must remain pairwise distinct — 20 identical
    // states would betray a single shared FDM.
    let quads: Vec<_> = end.iter().filter(|(id, ..)| id % 2 == 0).collect();
    for i in 0..quads.len() {
        for j in (i + 1)..quads.len() {
            let sep = (quads[i].1 - quads[j].1).length();
            assert!(
                sep > 50.0,
                "quads {} and {} must be distinct, sep {sep:.1} m",
                quads[i].0,
                quads[j].0
            );
        }
    }
}
