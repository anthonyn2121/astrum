//! Global combat resources: tunable parameters and the seeded PRNG that keeps
//! the whole simulation reproducible (design doc §7 — never `thread_rng`).

use bevy::prelude::*;

use super::ballistics::Ballistics;

/// Tunable combat constants, shared by every shooter and round.
#[derive(Resource, Clone, Copy, Debug)]
pub struct CombatParams {
    /// Drag properties of a gun round (the constant-`Cd` model; §12.1).
    pub round: Ballistics,
    /// HP removed per connecting round (tuned so a ~0.5 s burst kills).
    pub damage_per_hit: f32,
    /// Target bounding-sphere radius for the swept hit test (m; §4).
    pub hit_radius_m: f64,
    /// Round lifetime before it is culled (s).
    pub bullet_ttl_s: f64,
}

impl Default for CombatParams {
    fn default() -> Self {
        Self {
            // 20 mm round: Cd 0.3, 20 mm diameter, 0.10 kg (§3.3).
            round: Ballistics::new(0.3, std::f64::consts::PI * 0.010 * 0.010, 0.10),
            // ~50 rounds (0.5 s at 100 rps) on a 100-HP target.
            damage_per_hit: 2.0,
            hit_radius_m: 8.0,
            bullet_ttl_s: 3.0,
        }
    }
}

/// Episode-level combat state, distinct from the per-round tuning in
/// [`CombatParams`]. `initial_living_teams` is captured at spawn (step 4 wiring)
/// so `end_on_elimination` only ever terminates a genuine ≥ 2-team fight — a
/// single-team or non-combat scenario leaves this at its default and never
/// self-terminates.
#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct CombatState {
    pub initial_living_teams: u8,
}

/// Deterministic PRNG (SplitMix64) for gun dispersion. A resource so the whole
/// sim is reproducible from the scenario seed; MUST be used in place of any
/// thread-local RNG (§7). Cheap and good enough for a dispersion cone — not for
/// cryptography.
#[derive(Resource, Clone, Debug)]
pub struct CombatRng {
    state: u64,
}

impl CombatRng {
    /// Seed the generator (any 64-bit value; the scenario exposes the seed).
    pub fn seeded(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Next raw 64-bit value (SplitMix64).
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform `f64` in `[0, 1)` (top 53 bits → full double mantissa).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// A standard-normal sample (mean 0, variance 1) via Box–Muller — used to
    /// scatter rounds within the muzzle-dispersion cone.
    pub fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_f64().max(f64::MIN_POSITIVE);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}
