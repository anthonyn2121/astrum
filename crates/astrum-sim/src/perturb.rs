//! Initial-condition offsets for sweeps: move and turn an aircraft relative to
//! the initial conditions it already has, and draw such offsets at random.
//!
//! **Offsets** (`aircraft.<id>.offset.<axis>` in a sweep's `set`/`vary`, and
//! every `[[perturb]]` draw) are taken in the aircraft's *level frame* at its
//! own position, with its heading before any heading offset: `forward_m` is
//! horizontal along the true heading, `right_m` 90° clockwise of it, `up_m` the
//! local vertical (neither is tilted by climb or bank). With heading ψ, the
//! shift in a local ENU frame at the aircraft is `e = f·sinψ + r·cosψ`,
//! `n = f·cosψ − r·sinψ`, `u = up`. `heading_deg` adds and wraps to [0, 360);
//! `climb_deg` and `bank_deg` add and must stay strictly within ±90.
//!
//! **Groups.** A [`Group`] applies one offset to several aircraft as a rigid
//! body in its lead's frame: the shift moves every member alike, a heading
//! offset turns every member's position about the lead (and adds to every
//! member's heading), so every pairwise distance is kept. A single aircraft is
//! a group of one.
//!
//! **Draws.** A [`Perturb`] table draws offsets per axis — `{ uniform = a }`
//! (±a) or `{ normal = s }` (σ = s, truncated at 3σ) — either per aircraft
//! (each listed aircraft its own draw) or, `rigid`, once for the whole group in
//! the first listed aircraft's frame. Drawn values are rounded to 1e-3 (mm,
//! millidegrees), so the values a sweep records reproduce its runs exactly.

use astrum_geo::{Enu, Geodetic, LocalFrame};
use serde::{Deserialize, Serialize};

use crate::rng::SplitMix64;
use crate::scenario::Scenario;

/// One offset axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Axis {
    ForwardM,
    RightM,
    UpM,
    HeadingDeg,
    ClimbDeg,
    BankDeg,
}

impl Axis {
    /// Every axis, in key order.
    pub const ALL: [Axis; 6] =
        [Self::ForwardM, Self::RightM, Self::UpM, Self::HeadingDeg, Self::ClimbDeg, Self::BankDeg];

    /// The axis's key (`forward_m`, …).
    pub fn name(self) -> &'static str {
        match self {
            Self::ForwardM => "forward_m",
            Self::RightM => "right_m",
            Self::UpM => "up_m",
            Self::HeadingDeg => "heading_deg",
            Self::ClimbDeg => "climb_deg",
            Self::BankDeg => "bank_deg",
        }
    }

    /// The axis named `name`.
    pub fn parse(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|a| a.name() == name)
    }
}

/// The sweep key of `axis` for aircraft `id`: `aircraft.<id>.offset.<axis>`.
pub fn offset_key(id: u32, axis: Axis) -> String {
    format!("aircraft.{id}.offset.{}", axis.name())
}

/// An offset: a value per axis, `None` where the axis is not offset.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Offset([Option<f64>; 6]);

impl Offset {
    pub fn get(&self, axis: Axis) -> Option<f64> {
        self.0[axis as usize]
    }

    pub fn set(&mut self, axis: Axis, value: f64) {
        self.0[axis as usize] = Some(value);
    }

    fn or_zero(&self, axis: Axis) -> f64 {
        self.get(axis).unwrap_or(0.0)
    }
}

/// `(e, n)` of a level-frame `(forward, right)` displacement at true heading
/// `heading_deg`.
pub fn level_to_en(heading_deg: f64, forward_m: f64, right_m: f64) -> (f64, f64) {
    let (sin, cos) = heading_deg.to_radians().sin_cos();
    (forward_m * sin + right_m * cos, forward_m * cos - right_m * sin)
}

/// `heading_deg` wrapped to [0, 360).
pub fn wrap_heading(heading_deg: f64) -> f64 {
    let h = heading_deg.rem_euclid(360.0);
    // A tiny negative input rounds to exactly 360.
    if h >= 360.0 {
        0.0
    } else {
        h
    }
}

/// One offset applied to `members` (non-empty, `lead` among them) as a rigid
/// body in `lead`'s level frame; see the [module docs](self). Keys in errors
/// name the lead's axis (`aircraft.<lead>.offset.<axis>`).
#[derive(Debug, Clone, PartialEq)]
pub struct Group {
    pub lead: u32,
    pub members: Vec<u32>,
    pub offset: Offset,
}

impl Group {
    /// Apply the offset to `scenario`. Errors name the aircraft and key when a
    /// climb or bank leaves (-90, 90), or when an aircraft is missing.
    pub fn apply(&self, scenario: &mut Scenario) -> Result<(), String> {
        let find = |s: &Scenario, id: u32| {
            s.aircraft
                .iter()
                .position(|a| a.id == id)
                .ok_or_else(|| format!("the scenario has no aircraft {id}"))
        };
        let lead = &scenario.aircraft[find(scenario, self.lead)?];
        let at = Geodetic::from_degrees(lead.lat_deg, lead.lon_deg, lead.alt_hae_m);
        let off = &self.offset;
        let (de, dn) =
            level_to_en(lead.heading_deg, off.or_zero(Axis::ForwardM), off.or_zero(Axis::RightM));
        let shift = Enu { e: de, n: dn, u: off.or_zero(Axis::UpM) };
        let turn = off.or_zero(Axis::HeadingDeg).to_radians();
        let moves = shift.e != 0.0
            || shift.n != 0.0
            || shift.u != 0.0
            || (turn != 0.0 && self.members.len() > 1);
        let frame = moves.then(|| LocalFrame::new(at));
        let (sin, cos) = turn.sin_cos();
        for &id in &self.members {
            let i = find(scenario, id)?;
            let a = &mut scenario.aircraft[i];
            if let Some(frame) = &frame {
                // The lead sits at the frame origin exactly.
                let p = if id == self.lead {
                    Enu { e: 0.0, n: 0.0, u: 0.0 }
                } else {
                    frame.geodetic_to_enu(&Geodetic::from_degrees(
                        a.lat_deg,
                        a.lon_deg,
                        a.alt_hae_m,
                    ))
                };
                // Clockwise (from north toward east) by the heading offset.
                let moved = Enu {
                    e: p.e * cos + p.n * sin + shift.e,
                    n: p.n * cos - p.e * sin + shift.n,
                    u: p.u + shift.u,
                };
                let g = frame.enu_to_geodetic(&moved);
                (a.lat_deg, a.lon_deg, a.alt_hae_m) = (g.lat_deg(), g.lon_deg(), g.alt_hae_m);
            }
            if let Some(dh) = off.get(Axis::HeadingDeg) {
                a.heading_deg = wrap_heading(a.heading_deg + dh);
            }
            for (axis, value) in
                [(Axis::ClimbDeg, &mut a.climb_deg), (Axis::BankDeg, &mut a.bank_deg)]
            {
                if let Some(d) = off.get(axis) {
                    let v = *value + d;
                    if v.abs() >= 90.0 {
                        return Err(format!(
                            "aircraft {id}: {} = {d} makes {} {v}, outside (-90, 90)",
                            offset_key(self.lead, axis),
                            axis.name()
                        ));
                    }
                    *value = v;
                }
            }
        }
        Ok(())
    }
}

/// A random spread: `{ uniform = a }` draws from [-a, a], `{ normal = s }`
/// from N(0, s²) truncated at ±3s.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Spread {
    Uniform(f64),
    Normal(f64),
}

impl Spread {
    /// The largest |draw|.
    pub fn bound(self) -> f64 {
        match self {
            Self::Uniform(a) => a,
            Self::Normal(s) => 3.0 * s,
        }
    }

    /// One draw, rounded to 1e-3 and within [`Self::bound`].
    pub fn draw(self, rng: &mut SplitMix64) -> f64 {
        let x = match self {
            Self::Uniform(a) => a * (2.0 * rng.next_f64() - 1.0),
            Self::Normal(s) => loop {
                let z = rng.next_gaussian();
                if z.abs() <= 3.0 {
                    break s * z;
                }
            },
        };
        let bound = self.bound();
        ((x * 1000.0).round() / 1000.0).clamp(-bound, bound)
    }

    fn width(self) -> f64 {
        match self {
            Self::Uniform(v) | Self::Normal(v) => v,
        }
    }
}

/// A `[[perturb]]` table: random offsets for `aircraft`.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Perturb {
    pub aircraft: Vec<u32>,
    /// One draw for the whole group, in the first listed aircraft's frame
    /// (else: one per listed aircraft).
    #[serde(default)]
    pub rigid: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub forward_m: Option<Spread>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub right_m: Option<Spread>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub up_m: Option<Spread>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub heading_deg: Option<Spread>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub climb_deg: Option<Spread>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bank_deg: Option<Spread>,
}

impl Perturb {
    /// The spread of `axis`, if drawn.
    pub fn spread(&self, axis: Axis) -> Option<Spread> {
        match axis {
            Axis::ForwardM => self.forward_m,
            Axis::RightM => self.right_m,
            Axis::UpM => self.up_m,
            Axis::HeadingDeg => self.heading_deg,
            Axis::ClimbDeg => self.climb_deg,
            Axis::BankDeg => self.bank_deg,
        }
    }

    /// The drawn axes, in key order.
    pub fn axes(&self) -> impl Iterator<Item = (Axis, Spread)> + '_ {
        Axis::ALL.into_iter().filter_map(|a| self.spread(a).map(|s| (a, s)))
    }

    /// The aircraft each draw is recorded under: every listed aircraft, or
    /// the lead alone when rigid.
    fn leads(&self) -> &[u32] {
        if self.rigid {
            &self.aircraft[..self.aircraft.len().min(1)]
        } else {
            &self.aircraft
        }
    }
}

/// The keys of every draw `tables` make, in draw order: per table, per lead
/// (every listed aircraft, or a rigid group's first), per drawn axis.
pub fn keys(tables: &[Perturb]) -> Vec<String> {
    tables
        .iter()
        .flat_map(|t| t.leads().iter().flat_map(move |&id| t.axes().map(move |(a, _)| (id, a))))
        .map(|(id, a)| offset_key(id, a))
        .collect()
}

/// Check `tables` against the scenario they perturb; errors name the table
/// (1-based) and the offending aircraft, axis or key.
pub fn validate(tables: &[Perturb], base: &Scenario) -> Result<(), String> {
    for (i, t) in tables.iter().enumerate() {
        let at = format!("[[perturb]] {}", i + 1);
        if t.aircraft.is_empty() {
            return Err(format!("{at}: aircraft lists no aircraft"));
        }
        for (k, id) in t.aircraft.iter().enumerate() {
            if !base.aircraft.iter().any(|a| a.id == *id) {
                return Err(format!("{at}: the base scenario has no aircraft {id}"));
            }
            if t.aircraft[..k].contains(id) {
                return Err(format!("{at}: aircraft {id} is listed twice"));
            }
        }
        if t.axes().next().is_none() {
            let names: Vec<&str> = Axis::ALL.iter().map(|a| a.name()).collect();
            return Err(format!("{at}: draws no axis (set any of {})", names.join(", ")));
        }
        for (axis, spread) in t.axes() {
            let w = spread.width();
            if !(w.is_finite() && w > 0.0) {
                return Err(format!("{at}: {} must be > 0, got {w}", axis.name()));
            }
        }
    }
    let keys = keys(tables);
    for (k, key) in keys.iter().enumerate() {
        if keys[..k].contains(key) {
            return Err(format!("{key} is drawn by two [[perturb]] tables"));
        }
    }
    Ok(())
}

/// One draw of every table, from `rng`: the groups to apply and the drawn
/// values in [`keys`] order.
pub fn draw(tables: &[Perturb], rng: &mut SplitMix64) -> (Vec<Group>, Vec<f64>) {
    let mut groups = Vec::new();
    let mut values = Vec::new();
    for t in tables {
        for &lead in t.leads() {
            let mut offset = Offset::default();
            for (axis, spread) in t.axes() {
                let v = spread.draw(rng);
                offset.set(axis, v);
                values.push(v);
            }
            let members = if t.rigid { t.aircraft.clone() } else { vec![lead] };
            groups.push(Group { lead, members, offset });
        }
    }
    (groups, values)
}

/// The smallest straight-line (3-D) distance between two of `scenario`'s
/// aircraft at their initial positions; `None` with fewer than two.
pub fn min_separation_m(scenario: &Scenario) -> Option<f64> {
    let w = &scenario.world;
    let frame = LocalFrame::new(Geodetic::from_degrees(
        w.origin_lat_deg,
        w.origin_lon_deg,
        w.origin_alt_hae_m,
    ));
    let p: Vec<Enu> = scenario
        .aircraft
        .iter()
        .map(|a| frame.geodetic_to_enu(&Geodetic::from_degrees(a.lat_deg, a.lon_deg, a.alt_hae_m)))
        .collect();
    let mut min: Option<f64> = None;
    for (i, a) in p.iter().enumerate() {
        for b in &p[i + 1..] {
            let d = ((a.e - b.e).powi(2) + (a.n - b.n).powi(2) + (a.u - b.u).powi(2)).sqrt();
            min = Some(min.map_or(d, |m| m.min(d)));
        }
    }
    min
}
