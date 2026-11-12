//! Monte-Carlo course generation: sample a randomized sequence of gates that a
//! quadrotor can actually fly.
//!
//! # Method
//!
//! Gates are placed one at a time by **rejection sampling with backtracking**.
//! For each new gate the sampler draws a turn, a leg length and a target altitude,
//! then tests the resulting placement against every clearance rule; a rejected
//! candidate is simply redrawn. If a gate cannot be placed within
//! [`CourseParams::tries_per_gate`] attempts the sampler *backtracks* — it discards
//! the previous gate and re-draws it — because the usual cause of a dead end is not
//! the gate being placed but the one before it having boxed the track in.
//!
//! # Why the limits exist
//!
//! Unconstrained sampling produces tracks that are topologically impossible to
//! fly: gates on top of each other, legs that thread other gates' frames, a
//! doubling-back course that intersects itself. The parameters fall into three
//! groups:
//!
//! * **Flyability** — [`leg_min_m`]/[`leg_max_m`] keep gates 15–30 m apart so the
//!   drone has room to line up; [`max_turn_deg`] caps the heading change at a gate
//!   (under 120°, so with the gate's normal bisecting the turn no crossing is more
//!   than 60° off the normal); [`max_climb_deg`] caps the flight-path angle.
//! * **Non-overlap** — [`min_gate_gap_m`] separates gate centres,
//!   [`min_leg_gate_gap_m`] keeps a leg clear of frames it is not flying through
//!   (measured against [`frame_radius_m`], the frame's bounding sphere), and
//!   [`min_leg_leg_gap_m`] stops non-adjacent legs from crossing.
//! * **Containment** — [`alt_min_m`]/[`alt_max_m`] and [`course_radius_m`] keep the
//!   whole track inside a flight volume, with the altitude floor high enough that
//!   the lowest hardware (the gate sill) clears the ground.
//!
//! Two of these hold **by construction** rather than by rejection: the altitude is
//! drawn from the intersection of the altitude band and what the climb limit allows
//! from the current height, and the turn is drawn inside its own limit. Rejecting
//! on them instead would waste samples, and would make a zero-width altitude band
//! (a deliberately flat course) look infeasible.
//!
//! [`leg_min_m`]: CourseParams::leg_min_m
//! [`leg_max_m`]: CourseParams::leg_max_m
//! [`max_turn_deg`]: CourseParams::max_turn_deg
//! [`max_climb_deg`]: CourseParams::max_climb_deg
//! [`min_gate_gap_m`]: CourseParams::min_gate_gap_m
//! [`min_leg_gate_gap_m`]: CourseParams::min_leg_gate_gap_m
//! [`min_leg_leg_gap_m`]: CourseParams::min_leg_leg_gap_m
//! [`alt_min_m`]: CourseParams::alt_min_m
//! [`alt_max_m`]: CourseParams::alt_max_m
//! [`course_radius_m`]: CourseParams::course_radius_m

use std::fmt;

use bevy::math::DVec3;
use serde::{Deserialize, Serialize};

use crate::gate::{frame_radius_m, Gate, OPENING_M};
use crate::geom::{segment_point_dist_sq, segment_segment_dist_sq};
use crate::rng::{SplitMix64, Uniform};

/// The randomization envelope: what the sampler is allowed to draw.
///
/// Deserialized from the scenario's `[course]` table; every field has a default, so
/// an absent table is a valid (and reasonable) course.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct CourseParams {
    /// How many gates to place.
    pub gates: u32,
    /// Shortest gate-to-gate leg (m, 3-D).
    pub leg_min_m: f64,
    /// Longest gate-to-gate leg (m, 3-D).
    pub leg_max_m: f64,
    /// Largest horizontal heading change from one leg to the next (deg).
    pub max_turn_deg: f64,
    /// Largest flight-path angle of a leg, climbing or descending (deg).
    pub max_climb_deg: f64,
    /// Lowest allowed gate-centre altitude above the ENU origin plane (m).
    pub alt_min_m: f64,
    /// Highest allowed gate-centre altitude (m).
    pub alt_max_m: f64,
    /// Minimum distance between any two gate centres (m).
    pub min_gate_gap_m: f64,
    /// Minimum clearance from a leg to the bounding sphere of a gate frame it is
    /// not flying through (m).
    pub min_leg_gate_gap_m: f64,
    /// Minimum clearance between two non-adjacent legs (m).
    pub min_leg_leg_gap_m: f64,
    /// Horizontal radius of the flight volume about the first gate (m).
    pub course_radius_m: f64,
    /// Candidate draws per gate before the sampler backtracks.
    pub tries_per_gate: u32,
    /// Total backtracks allowed before the course is declared infeasible.
    pub max_backtracks: u32,
    /// Heading of the first leg (deg; `0` = North, `+` = East). Fixed, not drawn, so
    /// a course's entry direction is repeatable across seeds.
    pub start_heading_deg: f64,
}

impl Default for CourseParams {
    fn default() -> Self {
        Self {
            gates: 8,
            leg_min_m: 15.0,
            leg_max_m: 30.0,
            max_turn_deg: 120.0,
            max_climb_deg: 20.0,
            // 1.5 m centre leaves the 1.016 m opening's sill ~1.0 m off the deck:
            // clear of ground effect, low enough to stay a racing gate.
            alt_min_m: 1.5,
            alt_max_m: 4.0,
            min_gate_gap_m: 10.0,
            min_leg_gate_gap_m: 1.5,
            min_leg_leg_gap_m: 2.0,
            course_radius_m: 100.0,
            tries_per_gate: 200,
            max_backtracks: 200,
            start_heading_deg: 0.0,
        }
    }
}

impl CourseParams {
    /// Reject a parameter set that cannot describe a course, so the failure is a
    /// clear message about the input rather than an exhausted sampler.
    pub fn validate(&self) -> Result<(), CourseError> {
        let bad = |m: &str| Err(CourseError::Params(m.to_string()));
        if self.gates < 2 {
            return bad("course.gates must be at least 2");
        }
        if !self.leg_min_m.is_finite()
            || !self.leg_max_m.is_finite()
            || self.leg_min_m <= 0.0
            || self.leg_max_m < self.leg_min_m
        {
            return bad("course legs must satisfy 0 < leg_min_m <= leg_max_m");
        }
        if !(0.0..180.0).contains(&self.max_turn_deg) {
            return bad("course.max_turn_deg must be in [0, 180)");
        }
        if !(0.0..90.0).contains(&self.max_climb_deg) {
            return bad("course.max_climb_deg must be in [0, 90)");
        }
        if self.alt_max_m < self.alt_min_m {
            return bad("course.alt_min_m must not exceed course.alt_max_m");
        }
        // The gate's lowest hardware is the sill, half an opening below the centre.
        if self.alt_min_m < OPENING_M / 2.0 {
            return bad("course.alt_min_m is too low: the gate sill would be underground");
        }
        if self.min_gate_gap_m < 0.0
            || self.min_leg_gate_gap_m < 0.0
            || self.min_leg_leg_gap_m < 0.0
        {
            return bad("course clearances must not be negative");
        }
        // Consecutive gates are exactly one leg apart, so a gap wider than the
        // shortest leg would reject every candidate.
        if self.min_gate_gap_m > self.leg_min_m {
            return bad("course.min_gate_gap_m must not exceed course.leg_min_m");
        }
        if self.course_radius_m < self.leg_max_m {
            return bad("course.course_radius_m must be at least course.leg_max_m");
        }
        if self.tries_per_gate < 1 {
            return bad("course.tries_per_gate must be at least 1");
        }
        Ok(())
    }
}

/// Why a course could not be generated.
#[derive(Clone, Debug, PartialEq)]
pub enum CourseError {
    /// The parameters themselves are contradictory; see [`CourseParams::validate`].
    Params(String),
    /// The parameters are legal but the sampler could not satisfy them — typically
    /// too many gates for the volume, or clearances too wide for the leg lengths.
    Infeasible { placed: usize, wanted: u32, samples: u32, backtracks: u32 },
    /// An explicit gate list was rejected; see [`Course::from_gates`].
    Gates(String),
}

impl fmt::Display for CourseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CourseError::Params(m) => write!(f, "invalid course parameters: {m}"),
            CourseError::Gates(m) => write!(f, "invalid course gates: {m}"),
            CourseError::Infeasible { placed, wanted, samples, backtracks } => write!(
                f,
                "course infeasible: placed {placed}/{wanted} gates after {samples} samples \
                 and {backtracks} backtracks (loosen clearances, widen the volume, \
                 or ask for fewer gates)"
            ),
        }
    }
}

impl std::error::Error for CourseError {}

/// A generated course: the gates in flight order, plus what it cost to find them.
#[derive(Clone, Debug)]
pub struct Course {
    /// Gates in the order they must be flown.
    pub gates: Vec<Gate>,
    /// The seed that produced this course.
    pub seed: u64,
    /// The envelope it was drawn from.
    pub params: CourseParams,
    /// Candidate placements drawn, accepted or not — a cheap difficulty signal.
    pub samples: u32,
    /// How many times the sampler had to undo a gate.
    pub backtracks: u32,
}

impl Course {
    /// A course from gates given explicitly: used verbatim, nothing is sampled.
    ///
    /// This is what makes a recorded race replayable independent of the generator
    /// version: the bag stores the gates, and this rebuilds the course from them.
    /// `seed` and `params` are carried as provenance only (`samples` and `backtracks`
    /// are `0`: nothing was drawn).
    ///
    /// Rejected: an empty list, a gate whose `index` is not its position (so
    /// duplicates and out-of-order lists both fail), a non-finite centre or yaw, and
    /// a sill at or below the ground. The sampler's *clearance* rules are deliberately
    /// not imposed — a hand-built course may stack gates however it likes; call
    /// [`Course::validate_clearance`] to ask.
    pub fn from_gates(
        gates: Vec<Gate>,
        seed: u64,
        params: CourseParams,
    ) -> Result<Course, CourseError> {
        let bad = |m: String| Err(CourseError::Gates(m));
        if gates.is_empty() {
            return bad("a course needs at least one gate".to_string());
        }
        for (i, g) in gates.iter().enumerate() {
            if g.index as usize != i {
                return bad(format!(
                    "gate at position {i} has index {} (indices must be 0, 1, 2, ... in flight order)",
                    g.index
                ));
            }
            if !(g.center.is_finite() && g.yaw_rad.is_finite()) {
                return bad(format!("gate {i} has a non-finite centre or yaw"));
            }
            if g.sill_m() <= 0.0 {
                return bad(format!(
                    "gate {i} sill is at {:.3} m: the frame would be underground",
                    g.sill_m()
                ));
            }
        }
        Ok(Course { gates, seed, params, samples: 0, backtracks: 0 })
    }

    /// Apply the sampler's clearance rules (gate gap, leg-to-frame, leg-to-leg) to
    /// this course against `params`, reporting the first gate that breaks one.
    ///
    /// The flight-volume radius is *not* checked: the sampler measures it from gate 0
    /// at the origin, which an explicit course need not respect.
    pub fn validate_clearance(&self, params: &CourseParams) -> Result<(), CourseError> {
        let centers: Vec<DVec3> = self.gates.iter().map(|g| g.center).collect();
        for i in 1..centers.len() {
            if !clearance_ok(centers[i], &centers[..i], params) {
                return Err(CourseError::Gates(format!(
                    "gate {i} (or the leg into it) breaks a clearance rule"
                )));
            }
        }
        Ok(())
    }

    /// Number of gates.
    pub fn len(&self) -> usize {
        self.gates.len()
    }

    /// Is the course empty? (Only a degenerate one is.)
    pub fn is_empty(&self) -> bool {
        self.gates.is_empty()
    }

    /// Straight-line 3-D length of each leg (m).
    pub fn leg_lengths_m(&self) -> Vec<f64> {
        self.gates.windows(2).map(|w| (w[1].center - w[0].center).length()).collect()
    }

    /// Horizontal heading change at each interior gate (deg, unsigned).
    pub fn turn_angles_deg(&self) -> Vec<f64> {
        let headings: Vec<f64> = self
            .gates
            .windows(2)
            .map(|w| {
                let d = w[1].center - w[0].center;
                d.x.atan2(d.y) // ENU heading: 0 = North, + = East
            })
            .collect();
        headings.windows(2).map(|h| wrap_pi(h[1] - h[0]).abs().to_degrees()).collect()
    }

    /// Flight-path angle of each leg (deg; positive climbing).
    pub fn climb_angles_deg(&self) -> Vec<f64> {
        self.gates
            .windows(2)
            .map(|w| {
                let d = w[1].center - w[0].center;
                d.z.atan2(d.x.hypot(d.y)).to_degrees()
            })
            .collect()
    }

    /// Total distance flown gate to gate (m).
    pub fn total_length_m(&self) -> f64 {
        self.leg_lengths_m().iter().sum()
    }

    /// One-line description for the startup log.
    pub fn summary(&self) -> String {
        let legs = self.leg_lengths_m();
        let turns = self.turn_angles_deg();
        let alt_lo = self.gates.iter().map(|g| g.center.z).fold(f64::MAX, f64::min);
        let alt_hi = self.gates.iter().map(|g| g.center.z).fold(f64::MIN, f64::max);
        format!(
            "{} gates, {:.1} m total, legs {:.1}-{:.1} m, turns <= {:.0} deg, \
             alt {:.1}-{:.1} m (seed {}, {} samples, {} backtracks)",
            self.len(),
            self.total_length_m(),
            legs.iter().copied().fold(f64::MAX, f64::min),
            legs.iter().copied().fold(f64::MIN, f64::max),
            turns.iter().copied().fold(0.0, f64::max),
            alt_lo,
            alt_hi,
            self.seed,
            self.samples,
            self.backtracks,
        )
    }
}

/// Wrap an angle to `(-pi, pi]`.
fn wrap_pi(a: f64) -> f64 {
    let mut x = a % std::f64::consts::TAU;
    if x > std::f64::consts::PI {
        x -= std::f64::consts::TAU;
    } else if x <= -std::f64::consts::PI {
        x += std::f64::consts::TAU;
    }
    x
}

/// Consecutive backtracks at one frontier before the sampler unwinds a gate deeper.
///
/// Undoing only the last gate assumes it is the one that boxed the track in. Usually
/// it is — but when a *pair* of gates has closed off the volume, re-drawing just the
/// last one samples the same trap over and over until the budget is gone. Unwinding
/// progressively deeper the longer the sampler stays stuck escapes that.
///
/// "Stuck" is measured against the high-water mark of gates placed, not against the
/// last successful placement: re-filling a gate that was just popped is not progress,
/// and counting it as such is exactly what lets the sampler thrash at depth 1. A
/// course that never gets stuck never escalates, so it pays nothing for this.
const BACKTRACK_ESCALATION: u32 = 8;

/// A horizontal unit vector on `heading` (ENU; `0` = North, `+` = East).
fn heading_vec(heading: f64) -> DVec3 {
    let (s, c) = heading.sin_cos();
    DVec3::new(s, c, 0.0)
}

/// Generate a course from `seed`.
///
/// Deterministic: the same `(seed, params)` always yields the same gates. See the
/// module docs for the sampling scheme; `Err` distinguishes a bad envelope
/// ([`CourseError::Params`]) from one the sampler could not satisfy
/// ([`CourseError::Infeasible`]).
pub fn generate(seed: u64, params: &CourseParams) -> Result<Course, CourseError> {
    params.validate()?;

    let mut rng = SplitMix64::seeded(seed);
    let max_turn = params.max_turn_deg.to_radians();
    let sin_climb_max = params.max_climb_deg.to_radians().sin();
    let start_heading = params.start_heading_deg.to_radians();

    // Gate 0 anchors the volume at the horizontal origin; only its height is drawn.
    let mut centers = vec![DVec3::new(0.0, 0.0, rng.uniform(params.alt_min_m, params.alt_max_m))];
    // headings[k] is the horizontal heading of the leg from gate k to gate k+1.
    let mut headings: Vec<f64> = Vec::new();

    let mut samples = 0u32;
    let mut backtracks = 0u32;
    // Backtracks since the course last got *longer* than it has ever been; drives how
    // deep the sampler unwinds. See [`BACKTRACK_ESCALATION`].
    let mut stuck = 0u32;
    let mut deepest = centers.len();

    while centers.len() < params.gates as usize {
        let last = *centers.last().expect("gate 0 is always present");
        let prev_heading = headings.last().copied();

        let mut placed = None;
        for _ in 0..params.tries_per_gate {
            samples = samples.saturating_add(1);

            // Turn: fixed for the entry leg, otherwise inside the turn limit.
            let heading = match prev_heading {
                None => start_heading,
                Some(h) => h + rng.uniform(-max_turn, max_turn),
            };

            // Leg length, then a target altitude inside both the band and what the
            // climb limit permits over that length — so neither can be violated.
            let len = rng.uniform(params.leg_min_m, params.leg_max_m);
            let d_alt_max = len * sin_climb_max;
            let z = rng.uniform(
                params.alt_min_m.max(last.z - d_alt_max),
                params.alt_max_m.min(last.z + d_alt_max),
            );
            let du = z - last.z;
            // Spend the rest of the leg horizontally (|du| <= len by the draw above).
            let horiz = (len * len - du * du).max(0.0).sqrt();
            let cand = last + horiz * heading_vec(heading) + DVec3::new(0.0, 0.0, du);

            if accepts(cand, &centers, params) {
                placed = Some((cand, heading));
                break;
            }
        }

        match placed {
            Some((cand, heading)) => {
                centers.push(cand);
                headings.push(heading);
                // Only a new record counts as escaping the dead end.
                if centers.len() > deepest {
                    deepest = centers.len();
                    stuck = 0;
                }
            }
            // Dead end: undo placed gates and let them be re-drawn. One is normally
            // enough — the gate that boxed us in is the previous one — but the longer
            // re-drawing keeps failing, the further back the trap was set, so unwind
            // deeper. Never past gate 0, which anchors the volume.
            None => {
                if centers.len() <= 1 || backtracks >= params.max_backtracks {
                    return Err(CourseError::Infeasible {
                        placed: centers.len(),
                        wanted: params.gates,
                        samples,
                        backtracks,
                    });
                }
                backtracks += 1;
                stuck += 1;
                let depth = (1 + stuck / BACKTRACK_ESCALATION) as usize;
                let keep = centers.len().saturating_sub(depth).max(1);
                centers.truncate(keep);
                headings.truncate(keep - 1);
            }
        }
    }

    Ok(Course { gates: orient(&centers, &headings), seed, params: *params, samples, backtracks })
}

/// Every placement rule, applied to one candidate centre. `centers` holds the gates
/// already placed; the new leg runs from the last of them to `cand`.
fn accepts(cand: DVec3, centers: &[DVec3], params: &CourseParams) -> bool {
    // Inside the flight volume. Gate 0 is the axis; a leg between two points inside
    // a disc stays inside it, so testing centres is enough.
    cand.x.hypot(cand.y) <= params.course_radius_m && clearance_ok(cand, centers, params)
}

/// The clearance rules alone (everything but the flight volume): gate centres stay
/// apart, and the new leg neither grazes a frame nor crosses a non-adjacent leg.
fn clearance_ok(cand: DVec3, centers: &[DVec3], params: &CourseParams) -> bool {
    let n = centers.len();
    let last = centers[n - 1];
    let frame_clear_sq = (frame_radius_m() + params.min_leg_gate_gap_m).powi(2);

    // Gate centres stay apart.
    let gap_sq = params.min_gate_gap_m * params.min_gate_gap_m;
    if centers.iter().any(|c| (cand - *c).length_squared() < gap_sq) {
        return false;
    }

    // The new leg must not thread or graze any frame other than the one it starts
    // at (the candidate's own frame is where it ends).
    if centers[..n - 1].iter().any(|c| segment_point_dist_sq(last, cand, *c) < frame_clear_sq) {
        return false;
    }

    // ...and no existing leg may graze the candidate's frame.
    if centers.windows(2).any(|w| segment_point_dist_sq(w[0], w[1], cand) < frame_clear_sq) {
        return false;
    }

    // Non-adjacent legs must not cross. Leg n-2 shares the candidate leg's start
    // point, so it is adjacent by construction and excluded.
    let leg_gap_sq = params.min_leg_leg_gap_m * params.min_leg_leg_gap_m;
    for i in 0..n.saturating_sub(2) {
        if segment_segment_dist_sq(centers[i], centers[i + 1], last, cand) < leg_gap_sq {
            return false;
        }
    }

    true
}

/// Turn placed centres into gates by choosing each yaw.
///
/// A gate's normal **bisects** its incoming and outgoing legs: that is the
/// orientation a pilot would set, since it splits the turn evenly and leaves the
/// crossing at most half the turn angle off the normal (so under 60° for a 120°
/// turn). The end gates have only one leg, and face along it.
fn orient(centers: &[DVec3], headings: &[f64]) -> Vec<Gate> {
    centers
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let incoming = i.checked_sub(1).and_then(|k| headings.get(k)).copied();
            let outgoing = headings.get(i).copied();
            let yaw = match (incoming, outgoing) {
                // Sum the two unit headings and take the direction of the sum: the
                // bisector, wrap-safe (unlike averaging the angles).
                (Some(a), Some(b)) => {
                    let v = heading_vec(a) + heading_vec(b);
                    v.x.atan2(v.y)
                }
                (Some(a), None) => a,
                (None, Some(b)) => b,
                (None, None) => 0.0, // a single-gate course; nothing to align to
            };
            Gate::new(i as u32, *c, wrap_pi(yaw))
        })
        .collect()
}
