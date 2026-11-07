//! The one deterministic PRNG the crate uses (design doc §7 — never
//! `thread_rng`): SplitMix64. Gun dispersion draws from it
//! ([`crate::combat::CombatRng`]), and so do a sweep's random initial-condition
//! offsets ([`crate::perturb`]). Cheap and good enough for both — not for
//! cryptography.

const GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// SplitMix64's output function: a bijective 64-bit mix.
fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A SplitMix64 generator.
#[derive(Clone, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Seed the generator (any 64-bit value).
    pub fn seeded(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Stream `k` of `seed`: independent of every other `k` (the seed is
    /// hashed with `k`, so streams never overlap as shifted copies of one
    /// another), and the same for the same `(seed, k)`.
    pub fn stream(seed: u64, k: u64) -> Self {
        Self::seeded(mix(seed ^ mix(k.wrapping_add(1).wrapping_mul(GAMMA))))
    }

    /// Next raw 64-bit value.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(GAMMA);
        mix(self.state)
    }

    /// Uniform `f64` in `[0, 1)` (top 53 bits → full double mantissa).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// A standard-normal sample (mean 0, variance 1) via Box–Muller.
    pub fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_f64().max(f64::MIN_POSITIVE);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}
