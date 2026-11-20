//! The racing environment's deterministic pseudo-random source.
//!
//! Course randomization has to be reproducible: a seed plus a [`CourseParams`]
//! (see [`crate::course`]) must always yield byte-identical gate poses, so a
//! training run, a replay, and a bug report all describe the same track. Every
//! draw therefore comes from an [`astrum_sim::rng::SplitMix64`] seeded off the
//! scenario — never from a thread-local RNG. It is the sim's own generator
//! (Steele/Lea/Flood, 2014: statistically fine for sampling a course, not for
//! cryptography); this module only adds the range draw the sampler needs.
//!
//! [`CourseParams`]: crate::course::CourseParams

pub use astrum_sim::rng::SplitMix64;

/// A uniform draw over a range.
pub trait Uniform {
    /// Uniform `f64` in `[lo, hi)`.
    ///
    /// A degenerate or inverted range returns `lo` **without consuming a draw**, so
    /// a zero-width parameter band (a fixed gate height, a turn limit of 0°) is a
    /// legal input to the sampler instead of a NaN — and costs no randomness.
    fn uniform(&mut self, lo: f64, hi: f64) -> f64;
}

impl Uniform for SplitMix64 {
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        if hi <= lo {
            lo
        } else {
            lo + (hi - lo) * self.next_f64()
        }
    }
}
