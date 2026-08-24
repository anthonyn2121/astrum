//! `forge-geo` — WGS84 geodetic ⇄ ECEF ⇄ local ENU conversions.
//!
//! FORGE maps JSBSim's geodetic aircraft state (lat/lon/alt) onto a local
//! Cartesian **ENU** grid anchored at a configurable origin. That ENU frame *is*
//! the world grid: East = +X, North = +Y, Up = +Z (right-handed). The future
//! replay renderer is responsible for adapting ENU to its own axis convention.
//!
//! ```
//! use forge_geo::{Geodetic, LocalFrame};
//! let origin = Geodetic::from_degrees(40.0, -105.0, 1600.0);
//! let frame = LocalFrame::new(origin);
//! let enu = frame.geodetic_to_enu(&Geodetic::from_degrees(40.001, -105.0, 1600.0));
//! assert!(enu.n > 0.0); // 0.001° north -> positive North
//! ```

/// WGS84 ellipsoid parameters.
pub mod wgs84 {
    /// Semi-major axis (m).
    pub const A: f64 = 6_378_137.0;
    /// Flattening.
    pub const F: f64 = 1.0 / 298.257_223_563;
    /// First eccentricity squared, `f * (2 - f)`.
    pub const E2: f64 = F * (2.0 - F);
    /// Semi-minor axis (m).
    pub const B: f64 = A * (1.0 - F);
}

/// Geodetic position on the WGS84 ellipsoid. Angles stored in radians.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Geodetic {
    pub lat_rad: f64,
    pub lon_rad: f64,
    pub alt_m: f64,
}

impl Geodetic {
    /// Construct from degrees + metres.
    pub fn from_degrees(lat_deg: f64, lon_deg: f64, alt_m: f64) -> Self {
        Self { lat_rad: lat_deg.to_radians(), lon_rad: lon_deg.to_radians(), alt_m }
    }
    pub fn lat_deg(&self) -> f64 {
        self.lat_rad.to_degrees()
    }
    pub fn lon_deg(&self) -> f64 {
        self.lon_rad.to_degrees()
    }
}

/// Earth-Centred, Earth-Fixed position (m).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ecef {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// Local East-North-Up position (m) relative to a [`LocalFrame`] origin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Enu {
    pub e: f64,
    pub n: f64,
    pub u: f64,
}

/// Geodetic → ECEF.
pub fn geodetic_to_ecef(g: &Geodetic) -> Ecef {
    let (s_lat, c_lat) = g.lat_rad.sin_cos();
    let (s_lon, c_lon) = g.lon_rad.sin_cos();
    let n = wgs84::A / (1.0 - wgs84::E2 * s_lat * s_lat).sqrt();
    Ecef {
        x: (n + g.alt_m) * c_lat * c_lon,
        y: (n + g.alt_m) * c_lat * s_lon,
        z: (n * (1.0 - wgs84::E2) + g.alt_m) * s_lat,
    }
}

/// ECEF → geodetic (iterative; converges in a few iterations away from the poles).
pub fn ecef_to_geodetic(e: &Ecef) -> Geodetic {
    let lon = e.y.atan2(e.x);
    let p = (e.x * e.x + e.y * e.y).sqrt();
    let mut lat = e.z.atan2(p * (1.0 - wgs84::E2));
    let mut alt = 0.0;
    for _ in 0..8 {
        let s_lat = lat.sin();
        let n = wgs84::A / (1.0 - wgs84::E2 * s_lat * s_lat).sqrt();
        alt = p / lat.cos() - n;
        lat = e.z.atan2(p * (1.0 - wgs84::E2 * n / (n + alt)));
    }
    Geodetic { lat_rad: lat, lon_rad: lon, alt_m: alt }
}

/// A local tangent-plane (ENU) frame anchored at a geodetic origin.
///
/// Precomputes the origin's ECEF position and the ECEF→ENU rotation so repeated
/// per-aircraft conversions are just a subtraction and a 3×3 multiply.
#[derive(Clone, Copy, Debug)]
pub struct LocalFrame {
    origin_geo: Geodetic,
    origin_ecef: Ecef,
    /// Rows of the ECEF→ENU rotation matrix.
    r: [[f64; 3]; 3],
}

impl LocalFrame {
    /// Build a frame anchored at `origin`.
    pub fn new(origin: Geodetic) -> Self {
        let (s_lat, c_lat) = origin.lat_rad.sin_cos();
        let (s_lon, c_lon) = origin.lon_rad.sin_cos();
        let r = [
            [-s_lon, c_lon, 0.0],
            [-s_lat * c_lon, -s_lat * s_lon, c_lat],
            [c_lat * c_lon, c_lat * s_lon, s_lat],
        ];
        Self { origin_geo: origin, origin_ecef: geodetic_to_ecef(&origin), r }
    }

    pub fn origin(&self) -> Geodetic {
        self.origin_geo
    }

    /// ECEF → ENU: `R · (ecef − origin)`.
    pub fn ecef_to_enu(&self, e: &Ecef) -> Enu {
        let d = [e.x - self.origin_ecef.x, e.y - self.origin_ecef.y, e.z - self.origin_ecef.z];
        Enu {
            e: self.r[0][0] * d[0] + self.r[0][1] * d[1] + self.r[0][2] * d[2],
            n: self.r[1][0] * d[0] + self.r[1][1] * d[1] + self.r[1][2] * d[2],
            u: self.r[2][0] * d[0] + self.r[2][1] * d[1] + self.r[2][2] * d[2],
        }
    }

    /// ENU → ECEF: `Rᵀ · enu + origin`.
    pub fn enu_to_ecef(&self, p: &Enu) -> Ecef {
        let v = [p.e, p.n, p.u];
        Ecef {
            x: self.r[0][0] * v[0] + self.r[1][0] * v[1] + self.r[2][0] * v[2] + self.origin_ecef.x,
            y: self.r[0][1] * v[0] + self.r[1][1] * v[1] + self.r[2][1] * v[2] + self.origin_ecef.y,
            z: self.r[0][2] * v[0] + self.r[1][2] * v[1] + self.r[2][2] * v[2] + self.origin_ecef.z,
        }
    }

    /// Geodetic → ENU relative to this frame's origin.
    pub fn geodetic_to_enu(&self, g: &Geodetic) -> Enu {
        self.ecef_to_enu(&geodetic_to_ecef(g))
    }

    /// ENU → geodetic.
    pub fn enu_to_geodetic(&self, p: &Enu) -> Geodetic {
        ecef_to_geodetic(&self.enu_to_ecef(p))
    }
}
