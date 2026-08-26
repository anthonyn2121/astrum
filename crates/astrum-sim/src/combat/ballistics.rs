//! Free-projectile ballistics in the ENU world frame — the physics under every
//! gun round. Deliberately pure functions with no ECS/Bevy-App dependency (only
//! the `DVec3` math type), so they can be unit-tested in isolation
//! (`tests/ballistics.rs`) before the `fire_guns`/`step_bullets` systems bolt
//! them onto the sim.
//!
//! A round is a point mass under three accelerations (design doc §3):
//!
//!   * **gravity** → bullet drop;
//!   * **aerodynamic drag** → the dominant term at gun range. Constant-`Cd` for
//!     now, with a Mach-dependent G1/G7 curve left as a seam (decision §12.1) —
//!     it will swap the constant `Cd` inside [`Ballistics`] for a lookup without
//!     changing any call site;
//!   * **Earth Coriolis** → `−2·Ω×v`. Correct but only ~cm over a WVR
//!     time-of-flight (§12.2); included and commented so nobody tunes against
//!     the noise. We do **not** add any "turn-Coriolis": integrating in the
//!     world frame already makes a round fired from a turning jet appear to
//!     curve when viewed from the cockpit, so a turn force would double-count
//!     (§3.1b).
//!
//! ENU matches `astrum-geo`: East = +X, North = +Y, Up = +Z, right-handed.

// Re-exported so downstream code (and `tests/ballistics.rs`) can name the vector
// type through this module without depending on `bevy` directly.
pub use bevy::math::DVec3;

/// Standard gravity (m/s²).
pub const G0: f64 = 9.806_65;

/// Earth's rotation rate (rad/s).
pub const OMEGA_EARTH: f64 = 7.292_115e-5;

/// Earth's rotation vector expressed in an ENU frame at geodetic latitude
/// `lat_rad`: no East component, `Ω·cos(lat)` North, `Ω·sin(lat)` Up.
pub fn earth_rate_enu(lat_rad: f64) -> DVec3 {
    let (s, c) = lat_rad.sin_cos();
    OMEGA_EARTH * DVec3::new(0.0, c, s)
}

/// ISA air density (kg/m³) at geometric altitude `alt_m`. Models the troposphere
/// (≤ 11 km) and the isothermal lower stratosphere above it (good to ~20 km) —
/// ample for any WVR engagement. Sub-sea-level altitudes clamp to the sea-level
/// value rather than extrapolate.
pub fn air_density(alt_m: f64) -> f64 {
    const RHO0: f64 = 1.225; // kg/m³ at sea level
    const T0: f64 = 288.15; // K, sea-level standard temperature
    const LAPSE: f64 = 0.006_5; // K/m, tropospheric temperature lapse
    const R: f64 = 287.052_87; // J/(kg·K), specific gas constant of dry air
    const TROPOPAUSE: f64 = 11_000.0; // m

    // Density exponent from the barometric formula: ρ = ρ0·(T/T0)^(g/(R·L) − 1).
    let exp = G0 / (R * LAPSE) - 1.0;
    let h = alt_m.max(0.0);
    if h <= TROPOPAUSE {
        let t = T0 - LAPSE * h;
        RHO0 * (t / T0).powf(exp)
    } else {
        // Isothermal above the tropopause: density decays exponentially.
        let t_trop = T0 - LAPSE * TROPOPAUSE;
        let rho_trop = RHO0 * (t_trop / T0).powf(exp);
        rho_trop * (-G0 * (h - TROPOPAUSE) / (R * t_trop)).exp()
    }
}

/// Ballistic properties of a projectile under the constant-`Cd` model. The drag
/// acceleration is `−½·ρ·(Cd·A/m)·|v|·v`, so the only quantity that matters is
/// the lumped `drag_factor = Cd·A/m` (m²/kg). The Mach-dependent seam (§12.1)
/// will replace the constant `Cd` folded in here with a lookup on `|v|`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ballistics {
    /// `Cd · A / m` (m²/kg): drag coefficient × reference area per unit mass.
    pub drag_factor: f64,
}

impl Ballistics {
    /// Build from physical parameters: drag coefficient, reference cross-section
    /// area (m²), and mass (kg).
    pub fn new(cd: f64, area_m2: f64, mass_kg: f64) -> Self {
        Self { drag_factor: cd * area_m2 / mass_kg }
    }
}

/// Total acceleration (gravity + drag + Coriolis, m/s²) on a round moving at
/// `vel` (ENU) at geometric altitude `alt_m`, with the local Earth rate
/// `omega_enu` (see [`earth_rate_enu`]). Factored out of the integrator so tests
/// can probe the individual terms.
pub fn acceleration(vel: DVec3, alt_m: f64, b: &Ballistics, omega_enu: DVec3) -> DVec3 {
    let gravity = DVec3::new(0.0, 0.0, -G0);
    let speed = vel.length();
    let drag = if speed > 0.0 {
        -0.5 * air_density(alt_m) * b.drag_factor * speed * vel
    } else {
        DVec3::ZERO
    };
    let coriolis = -2.0 * omega_enu.cross(vel);
    gravity + drag + coriolis
}

/// Advance a round one `dt`-second step with semi-implicit (symplectic) Euler:
/// velocity is updated first, then position with the *new* velocity (design doc
/// §3). `alt_m` is the round's current geometric altitude (ENU origin altitude +
/// its Up coordinate), used only for the air-density lookup. Returns the new
/// `(position, velocity)` in the same ENU frame.
pub fn step_projectile(
    pos: DVec3,
    vel: DVec3,
    alt_m: f64,
    b: &Ballistics,
    omega_enu: DVec3,
    dt: f64,
) -> (DVec3, DVec3) {
    let a = acceleration(vel, alt_m, b, omega_enu);
    let new_vel = vel + a * dt;
    let new_pos = pos + new_vel * dt;
    (new_pos, new_vel)
}
