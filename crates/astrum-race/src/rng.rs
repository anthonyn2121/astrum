//! The racing environment's deterministic pseudo-random source.
//!
//! Course randomization has to be reproducible: a seed plus a [`CourseParams`]
//! (see [`crate::course`]) must always yield byte-identical gate poses, so a
//! training run, a replay, and a bug report all describe the same track. Every
//! draw therefore comes from a [`RaceRng`] seeded off the scenario — never from a
//! thread-local RNG.
//!
//! The generator is SplitMix64 (Steele/Lea/Flood, 2014): two multiplies and a few
//! shifts per draw, no state beyond a `u64`, statistically fine for sampling a
//! course — and not for cryptography.
//!
//! [`CourseParams`]: crate::course::CourseParams

/// A seeded SplitMix64 stream.
#[derive(Clone, Copy, Debug)]
pub struct RaceRng {
    state: u64,
}

impl RaceRng {
    /// Seed the generator (any 64-bit value).
    pub fn seeded(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Next raw 64-bit value.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform `f64` in `[0, 1)` (top 53 bits → a full double mantissa).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform `f64` in `[lo, hi)`.
    ///
    /// A degenerate or inverted range returns `lo` **without consuming a draw**, so
    /// a zero-width parameter band (a fixed gate height, a turn limit of 0°) is a
    /// legal input to the sampler instead of a NaN — and costs no randomness.
    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        if hi <= lo {
            lo
        } else {
            lo + (hi - lo) * self.next_f64()
        }
    }
}
