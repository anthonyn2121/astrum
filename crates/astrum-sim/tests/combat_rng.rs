//! Tests for the deterministic combat PRNG (`astrum_sim::combat::CombatRng`).
//! Gun dispersion must be reproducible from the scenario seed (design doc §7),
//! so this locks the two properties the systems rely on: same seed → identical
//! stream, and the derived `f64`/Gaussian draws stay in-distribution. App-free,
//! mirroring `tests/ballistics.rs`.
//!
//! Run:  `cargo test -p astrum-sim --test combat_rng -- --nocapture`

use astrum_sim::combat::CombatRng;

#[test]
fn same_seed_reproduces_the_stream() {
    let mut a = CombatRng::seeded(0xDEAD_BEEF);
    let mut b = CombatRng::seeded(0xDEAD_BEEF);
    for i in 0..10_000 {
        assert_eq!(a.next_u64(), b.next_u64(), "streams diverged at draw {i}");
    }
}

#[test]
fn different_seeds_diverge() {
    let mut a = CombatRng::seeded(1);
    let mut b = CombatRng::seeded(2);
    // Not a strong statistical claim — just that the seed actually matters.
    let differs = (0..64).any(|_| a.next_u64() != b.next_u64());
    assert!(differs, "distinct seeds produced identical streams");
}

#[test]
fn uniform_draws_stay_in_unit_interval() {
    let mut rng = CombatRng::seeded(12345);
    let n = 100_000;
    let mut sum = 0.0;
    for _ in 0..n {
        let x = rng.next_f64();
        assert!((0.0..1.0).contains(&x), "next_f64 out of [0,1): {x}");
        sum += x;
    }
    // Mean of a uniform on [0,1) is 0.5; std error over 1e5 draws ≈ 0.0009.
    let mean = sum / f64::from(n);
    assert!((mean - 0.5).abs() < 0.01, "uniform mean should be ~0.5, got {mean}");
}

#[test]
fn gaussian_is_roughly_standard_normal() {
    let mut rng = CombatRng::seeded(0xC0FFEE);
    let n = 100_000;
    let (mut sum, mut sum_sq) = (0.0_f64, 0.0_f64);
    for _ in 0..n {
        let z = rng.next_gaussian();
        sum += z;
        sum_sq += z * z;
    }
    let mean = sum / f64::from(n);
    let var = sum_sq / f64::from(n) - mean * mean;
    // Standard normal: mean 0, variance 1. Tolerances comfortably exceed the
    // sampling error at n = 1e5 (SE(mean) ≈ 0.003, SE(var) ≈ 0.004).
    assert!(mean.abs() < 0.03, "gaussian mean should be ~0, got {mean}");
    assert!((var - 1.0).abs() < 0.05, "gaussian variance should be ~1, got {var}");
}
