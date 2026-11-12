//! Ballistic fire control: where a round fired now goes, and where to point the
//! gun so one reaches the target. Uses exactly the projectile model the sim
//! integrates (gravity, drag, Coriolis, velocity inheritance, muzzle offset), so
//! a policy's or script's firing decisions agree with what the sim will score.

use astrum_geo::LocalFrame;

use crate::components::AircraftState;

use super::ballistics::{earth_rate_enu, grid_altitude, step_projectile, DVec3};
use super::components::Gun;
use super::geometry::closest_approach;
use super::resources::CombatParams;
use super::systems::{dvec, muzzle_and_boresight};

/// A fire-control solution against one target, assuming it holds its velocity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GunSolution {
    /// Closest approach (m) of a round fired now along the boresight.
    pub miss_m: f64,
    /// Time of flight (s) to that closest approach.
    pub tof_s: f64,
    /// A round fired now would hit (`miss_m` ≤ the hit radius).
    pub hit: bool,
    /// Unit world-grid direction the boresight must point along for a hit.
    pub aim_dir: DVec3,
    /// Angle (rad) between the current boresight and `aim_dir`.
    pub aim_error_rad: f64,
    /// The aim solution reaches the target within the round's lifetime.
    pub in_range: bool,
}

struct Flight {
    miss: f64,
    tof: f64,
    /// Round position at closest approach, relative to the muzzle.
    round_rel: DVec3,
    /// Target minus round at closest approach.
    miss_vec: DVec3,
    passed: bool,
}

#[allow(clippy::too_many_arguments)]
fn fly(
    muzzle: DVec3,
    vel0: DVec3,
    target: DVec3,
    target_vel: DVec3,
    combat: &CombatParams,
    origin_alt: f64,
    omega: DVec3,
    dt: f64,
) -> Flight {
    let (mut p, mut v) = (muzzle, vel0);
    let mut t_pos = target;
    let mut best = Flight {
        miss: (target - muzzle).length(),
        tof: 0.0,
        round_rel: DVec3::ZERO,
        miss_vec: target - muzzle,
        passed: false,
    };
    let steps = (combat.bullet_ttl_s / dt).ceil() as usize;
    for k in 0..steps {
        let (np, nv) =
            step_projectile(p, v, grid_altitude(origin_alt, p), &combat.round, omega, dt);
        let nt = t_pos + target_vel * dt;
        let (s, d) = closest_approach(p, np, t_pos, nt);
        if d < best.miss {
            let round = p + (np - p) * s;
            let tgt = t_pos + (nt - t_pos) * s;
            best = Flight {
                miss: d,
                tof: (k as f64 + s) * dt,
                round_rel: round - muzzle,
                miss_vec: tgt - round,
                passed: false,
            };
        } else if k > 0 && s == 0.0 {
            // Receding from the target since the start of this step: the single
            // closest approach of a ballistic round is behind us.
            best.passed = true;
            break;
        }
        (p, v, t_pos) = (np, nv, nt);
    }
    best
}

/// Solve fire control for `shooter` (with `gun`) against `target` at `dt` steps.
pub fn gun_solution(
    frame: &LocalFrame,
    shooter: &AircraftState,
    gun: &Gun,
    combat: &CombatParams,
    target: &AircraftState,
    dt: f64,
) -> GunSolution {
    let origin = frame.origin();
    let omega = earth_rate_enu(origin.lat_rad);
    let (muzzle, bore) = muzzle_and_boresight(frame, shooter, gun);
    let own_vel = dvec(shooter.vel_enu);
    let (tgt, tgt_vel) = (dvec(target.pos_enu), dvec(target.vel_enu));
    let shoot = |dir: DVec3| {
        fly(
            muzzle,
            own_vel + dir * gun.muzzle_speed_mps,
            tgt,
            tgt_vel,
            combat,
            origin.alt_hae_m,
            omega,
            dt,
        )
    };

    let now = shoot(bore);
    // Aim iteration (a secant step on the direction): the part of the round's
    // displacement the gun controls is its travel minus what it inherited from
    // the shooter's velocity; re-point that displacement at the miss. Converges
    // in a few steps at gun ranges.
    let mut dir = (tgt - muzzle).normalize_or_zero();
    let mut aim = shoot(dir);
    for _ in 0..8 {
        if aim.miss < 0.05 {
            break;
        }
        let corrected = (aim.round_rel - own_vel * aim.tof + aim.miss_vec).normalize_or_zero();
        if corrected == DVec3::ZERO {
            break;
        }
        dir = corrected;
        aim = shoot(dir);
    }
    GunSolution {
        miss_m: now.miss,
        tof_s: now.tof,
        hit: now.miss <= combat.hit_radius_m,
        aim_dir: dir,
        aim_error_rad: bore.dot(dir).clamp(-1.0, 1.0).acos(),
        in_range: aim.miss <= combat.hit_radius_m && (aim.passed || aim.tof < combat.bullet_ttl_s),
    }
}
