//! `astrum-geo` — WGS84 geodetic ⇄ ECEF ⇄ local ENU conversions.
//!
//! ASTRUM maps JSBSim's geodetic aircraft state onto a local Cartesian **ENU**
//! grid anchored at a configurable origin. That ENU frame *is* the world grid:
//! East = +X, North = +Y, Up = +Z (right-handed). Vectors measured in an
//! aircraft's *local-level* NED frame (velocity, body axes) are rotated into the
//! grid with [`LocalFrame::ned_to_enu`]; away from the origin the two frames are
//! tilted relative to each other (≈ 0.009° per km), so swapping axes is not enough.
//!
//! ```
//! use astrum_geo::{Geodetic, LocalFrame};
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

/// Geodetic position on the WGS84 ellipsoid. Angles in radians; latitude is
/// geodetic, altitude is height above the ellipsoid (HAE).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Geodetic {
    pub lat_rad: f64,
    pub lon_rad: f64,
    pub alt_hae_m: f64,
}

impl Geodetic {
    /// Construct from degrees + metres (HAE).
    pub fn from_degrees(lat_deg: f64, lon_deg: f64, alt_hae_m: f64) -> Self {
        Self { lat_rad: lat_deg.to_radians(), lon_rad: lon_deg.to_radians(), alt_hae_m }
    }
    pub fn lat_deg(&self) -> f64 {
        self.lat_rad.to_degrees()
    }
    pub fn lon_deg(&self) -> f64 {
        self.lon_rad.to_degrees()
    }
}

/// Earth-Centred, Earth-Fixed position or vector (m).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Ecef {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// East-North-Up position or vector in a [`LocalFrame`] grid.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Enu {
    pub e: f64,
    pub n: f64,
    pub u: f64,
}

/// North-East-Down vector in the local-level frame at some geodetic point.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Ned {
    pub n: f64,
    pub e: f64,
    pub d: f64,
}

/// Geodetic → ECEF.
pub fn geodetic_to_ecef(g: &Geodetic) -> Ecef {
    let (s_lat, c_lat) = g.lat_rad.sin_cos();
    let (s_lon, c_lon) = g.lon_rad.sin_cos();
    let n = wgs84::A / (1.0 - wgs84::E2 * s_lat * s_lat).sqrt();
    Ecef {
        x: (n + g.alt_hae_m) * c_lat * c_lon,
        y: (n + g.alt_hae_m) * c_lat * s_lon,
        z: (n * (1.0 - wgs84::E2) + g.alt_hae_m) * s_lat,
    }
}

/// ECEF → geodetic. Iterates the latitude with a height formula that stays
/// well-conditioned at the poles (no division by `cos(lat)`); converges to
/// sub-millimetre in a few iterations anywhere outside the Earth's core.
pub fn ecef_to_geodetic(e: &Ecef) -> Geodetic {
    let lon = e.y.atan2(e.x);
    let p = e.x.hypot(e.y);
    let mut lat = e.z.atan2(p * (1.0 - wgs84::E2));
    let mut alt = 0.0;
    for _ in 0..10 {
        let (s, c) = lat.sin_cos();
        let n = wgs84::A / (1.0 - wgs84::E2 * s * s).sqrt();
        alt = p * c + e.z * s - wgs84::A * wgs84::A / n;
        let next = e.z.atan2(p * (1.0 - wgs84::E2 * n / (n + alt)));
        let done = (next - lat).abs() < 1e-14;
        lat = next;
        if done {
            break;
        }
    }
    Geodetic { lat_rad: lat, lon_rad: lon, alt_hae_m: alt }
}

/// Rows of the ECEF→ENU rotation at geodetic `lat`/`lon` (rows: E, N, U).
fn ecef_to_enu_rotation(lat: f64, lon: f64) -> [[f64; 3]; 3] {
    let (s_lat, c_lat) = lat.sin_cos();
    let (s_lon, c_lon) = lon.sin_cos();
    [
        [-s_lon, c_lon, 0.0],
        [-s_lat * c_lon, -s_lat * s_lon, c_lat],
        [c_lat * c_lon, c_lat * s_lon, s_lat],
    ]
}

/// A local tangent-plane (ENU) frame anchored at a geodetic origin.
///
/// Precomputes the origin's ECEF position and the ECEF→ENU rotation so repeated
/// per-aircraft conversions are just a subtraction and a 3×3 multiply.
#[derive(Clone, Copy, Debug)]
pub struct LocalFrame {
    origin_geo: Geodetic,
    origin_ecef: Ecef,
    /// Rows of the ECEF→ENU rotation matrix at the origin.
    r: [[f64; 3]; 3],
}

impl LocalFrame {
    /// Build a frame anchored at `origin`.
    pub fn new(origin: Geodetic) -> Self {
        Self {
            origin_geo: origin,
            origin_ecef: geodetic_to_ecef(&origin),
            r: ecef_to_enu_rotation(origin.lat_rad, origin.lon_rad),
        }
    }

    pub fn origin(&self) -> Geodetic {
        self.origin_geo
    }

    /// Rotate an ECEF vector into the grid (no translation).
    pub fn ecef_vec_to_enu(&self, d: &Ecef) -> Enu {
        let r = &self.r;
        Enu {
            e: r[0][0] * d.x + r[0][1] * d.y + r[0][2] * d.z,
            n: r[1][0] * d.x + r[1][1] * d.y + r[1][2] * d.z,
            u: r[2][0] * d.x + r[2][1] * d.y + r[2][2] * d.z,
        }
    }

    /// Rotate a grid vector into ECEF (no translation).
    pub fn enu_vec_to_ecef(&self, v: &Enu) -> Ecef {
        let r = &self.r;
        Ecef {
            x: r[0][0] * v.e + r[1][0] * v.n + r[2][0] * v.u,
            y: r[0][1] * v.e + r[1][1] * v.n + r[2][1] * v.u,
            z: r[0][2] * v.e + r[1][2] * v.n + r[2][2] * v.u,
        }
    }

    /// ECEF position → ENU: `R · (ecef − origin)`.
    pub fn ecef_to_enu(&self, e: &Ecef) -> Enu {
        self.ecef_vec_to_enu(&Ecef {
            x: e.x - self.origin_ecef.x,
            y: e.y - self.origin_ecef.y,
            z: e.z - self.origin_ecef.z,
        })
    }

    /// ENU position → ECEF: `Rᵀ · enu + origin`.
    pub fn enu_to_ecef(&self, p: &Enu) -> Ecef {
        let d = self.enu_vec_to_ecef(p);
        Ecef {
            x: d.x + self.origin_ecef.x,
            y: d.y + self.origin_ecef.y,
            z: d.z + self.origin_ecef.z,
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

    /// Rotate a vector given in the local-level NED frame at `at` into the grid.
    pub fn ned_to_enu(&self, at: &Geodetic, v: &Ned) -> Enu {
        // NED at `at` → ECEF via the transpose of that point's ECEF→ENU rotation
        // (ENU = (e, n, −d)), then ECEF → grid ENU.
        let l = ecef_to_enu_rotation(at.lat_rad, at.lon_rad);
        let (e, n, u) = (v.e, v.n, -v.d);
        let ecef = Ecef {
            x: l[0][0] * e + l[1][0] * n + l[2][0] * u,
            y: l[0][1] * e + l[1][1] * n + l[2][1] * u,
            z: l[0][2] * e + l[1][2] * n + l[2][2] * u,
        };
        self.ecef_vec_to_enu(&ecef)
    }

    /// Rotate a grid vector into the local-level NED frame at `at`.
    pub fn enu_to_ned(&self, at: &Geodetic, v: &Enu) -> Ned {
        let d = self.enu_vec_to_ecef(v);
        let l = ecef_to_enu_rotation(at.lat_rad, at.lon_rad);
        let e = l[0][0] * d.x + l[0][1] * d.y + l[0][2] * d.z;
        let n = l[1][0] * d.x + l[1][1] * d.y + l[1][2] * d.z;
        let u = l[2][0] * d.x + l[2][1] * d.y + l[2][2] * d.z;
        Ned { n, e, d: -u }
    }
}
