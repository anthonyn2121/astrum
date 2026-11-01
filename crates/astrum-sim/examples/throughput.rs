//! Headless throughput of the sim core: `make bench`.
//!
//! Steps fixed fleets through the embedded app (the same schedule as the
//! `astrum-sim` binary minus the Zenoh bridge and pacing) and reports wall time
//! per tick and the speed relative to real time. The aircraft hold their spawn
//! trim uncommanded, so the cost measured is the sim's own: FDM stepping, the
//! ECS schedule and telemetry assembly.
//!
//! `cargo run --release -p astrum-sim --example throughput [-- --ticks N]`

use std::time::Instant;

use astrum_sim::scenario::Scenario;
use astrum_sim::{build_app, RunMode};

const RATE_HZ: f64 = 120.0;
/// Timed windows per case; the fastest is reported.
const WINDOWS: u32 = 3;

/// (model, aircraft, threads)
const CASES: &[(&str, u32, usize)] = &[
    ("c172x", 1, 1),
    ("f16", 1, 1),
    ("f16", 4, 1),
    ("f16", 4, 4),
    ("f16", 16, 1),
    ("f16", 16, 8),
    ("f16", 64, 1),
    ("f16", 64, 8),
    ("f16", 64, 16),
];

fn scenario(model: &str, n: u32, threads: usize) -> Scenario {
    let (alt, tas) = if model == "f16" { (4572.0, 150.0) } else { (1524.0, 55.0) };
    let mut toml = format!(
        "[world]\norigin_lat_deg = 37.6188\norigin_lon_deg = -122.3756\norigin_alt_hae_m = 0.0\n\
         [sim]\nrate_hz = {RATE_HZ}\nthreads = {threads}\n"
    );
    for i in 0..n {
        // A 10-wide grid, ~1 km apart: independent aircraft, no interaction.
        let (row, col) = (f64::from(i / 10), f64::from(i % 10));
        toml += &format!(
            "[[aircraft]]\nid = {}\nmodel = \"{model}\"\nlat_deg = {}\nlon_deg = {}\n\
             alt_hae_m = {alt}\nheading_deg = 90.0\ntas_mps = {tas}\n",
            i + 1,
            37.6188 + row * 0.01,
            -122.3756 + col * 0.01,
        );
    }
    Scenario::from_toml_str(&toml).expect("bench scenario is valid")
}

fn main() {
    let ticks: u32 = std::env::args()
        .skip_while(|a| a != "--ticks")
        .nth(1)
        .map_or(1200, |v| v.parse().expect("--ticks takes a positive integer"));
    println!("| fleet | threads | µs/tick | × real time @{RATE_HZ} Hz |");
    println!("|---|---|---|---|");
    for &(model, n, threads) in CASES {
        let mut app = build_app(scenario(model, n, threads), RunMode::Embedded);
        // Startup (load + trim) and a warm-up second are not timed.
        for _ in 0..=RATE_HZ as u32 {
            app.update();
        }
        // Best of a few windows: the floor is the sim's cost, the rest is other
        // load on the host (and, on hybrid CPUs, landing on an efficiency core).
        let us = (0..WINDOWS)
            .map(|_| {
                let t0 = Instant::now();
                for _ in 0..ticks {
                    app.update();
                }
                t0.elapsed().as_secs_f64() / f64::from(ticks) * 1e6
            })
            .fold(f64::INFINITY, f64::min);
        println!("| {n} × {model} | {threads} | {us:.1} | {:.0}× |", 1e6 / us / RATE_HZ);
    }
}
