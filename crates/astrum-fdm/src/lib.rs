//! `astrum-fdm` — a safe Rust wrapper over one JSBSim `FGFDMExec`, via the C ABI
//! shim in `shim/`.
//!
//! Each [`Fdm`] owns exactly one flight-dynamics instance: the building block that
//! lets every ASTRUM entity carry its own independent aircraft physics.
//!
//! # Thread safety
//! JSBSim is not thread-safe across instances during construction: the first XML
//! element parsed in a process lazily fills a process-global unit-conversion table
//! (`Element::convert`), and every `FGFDMExec` constructor writes the static
//! `FGJSBBase::debug_lvl`. Concurrent construction segfaults. This crate owns that
//! invariant with one process-wide reader/writer lock:
//!
//! * lifecycle calls — [`Fdm::load`], [`Fdm::run_ic`], [`Fdm::trim`],
//!   [`Fdm::reset`] and `Drop` — take it exclusively;
//! * every other call takes it shared, so distinct instances step in parallel.
//!
//! With that, [`Fdm`] is [`Send`]: an instance shares no other mutable state, so it
//! may move between threads. It is deliberately **not** `Sync` — every method takes
//! `&mut self` (JSBSim mutates even on reads: errors are recorded per instance).
//!
//! ```no_run
//! use astrum_fdm::{Controls, Fdm, InitialConditions, TrimMode};
//! let mut fdm = Fdm::load("/opt/jsbsim/share/jsbsim".as_ref(), "c172x")?;
//! fdm.set_initial_conditions(&InitialConditions {
//!     lat_geod_deg: 37.6, lon_deg: -122.4, alt_hae_m: 1500.0, psi_true_deg: 90.0, tas_mps: 55.0,
//!     gamma_deg: 0.0, phi_deg: 0.0,
//! })?;
//! fdm.run_ic()?;
//! fdm.start_engines()?;
//! fdm.trim(TrimMode::Full)?;
//! fdm.set_dt(1.0 / 120.0)?;
//! fdm.set_controls(Controls { throttle: 0.6, ..Default::default() })?;
//! fdm.step()?;
//! let alt = fdm.state()?.alt_hae_m;
//! # Ok::<(), astrum_fdm::FdmError>(())
//! ```

use std::ffi::{c_char, c_double, c_int, CStr, CString};
use std::fmt;
use std::marker::{PhantomData, PhantomPinned};
use std::path::Path;
use std::ptr::NonNull;
use std::sync::{PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Feet → metres.
pub const FT_TO_M: f64 = 0.3048;

/// Property names callers address directly through [`Fdm::get`] / [`Fdm::set`].
/// The per-step state and the four FCS inputs go through [`Fdm::state`] and
/// [`Fdm::set_controls`], which use nodes resolved once at load.
pub mod prop {
    /// Multirotor stability-augmentation gate (F450 `FlightControl`). Defaults to
    /// `0`, which disables the inner rate loop entirely — the quad then has no
    /// attitude control. Multirotor entities set it to `1`.
    pub const SCAS_ENGAGE: &str = "fcs/ScasEngage";
    /// Seed of the instance's JSBSim random generator (sensor noise, dispersions).
    pub const RANDOM_SEED: &str = "simulation/randomseed";
}

/// Why a JSBSim call failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FdmError {
    /// The model could not be loaded (missing files, bad XML, or a standard
    /// property ASTRUM relies on is absent).
    Load(String),
    /// [`Fdm::get`] / [`Fdm::set`] named a property the model does not have.
    /// Never silently created: a typo is an error, not a new node.
    MissingProperty(String),
    /// JSBSim reported a failure or threw (trim not converging, RunIC failing, …).
    Jsbsim(String),
    /// JSBSim reported the run complete (`Run()` returned false).
    Terminated,
    /// A name or path contained an interior NUL byte.
    InvalidName(String),
}

impl fmt::Display for FdmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FdmError::Load(m) => write!(f, "JSBSim load failed: {m}"),
            FdmError::MissingProperty(m) => write!(f, "JSBSim: {m}"),
            FdmError::Jsbsim(m) => write!(f, "JSBSim: {m}"),
            FdmError::Terminated => write!(f, "JSBSim run complete"),
            FdmError::InvalidName(n) => write!(f, "name contains a NUL byte: {n:?}"),
        }
    }
}

impl std::error::Error for FdmError {}

/// Result alias for this crate.
pub type Result<T, E = FdmError> = std::result::Result<T, E>;

/// Initial conditions, in SI units. Applied by [`Fdm::set_initial_conditions`]
/// and taking effect on the next [`Fdm::run_ic`] / [`Fdm::reset`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InitialConditions {
    /// Geodetic (WGS84) latitude, degrees.
    pub lat_geod_deg: f64,
    /// Longitude, degrees.
    pub lon_deg: f64,
    /// Height above the WGS84 ellipsoid, metres.
    pub alt_hae_m: f64,
    /// True heading, degrees.
    pub psi_true_deg: f64,
    /// True airspeed, m/s.
    pub tas_mps: f64,
    /// Flight-path angle (climb), degrees; + = climbing. A trim holds it.
    pub gamma_deg: f64,
    /// Bank angle, degrees; + = right wing down. [`TrimMode::Full`] re-trims it
    /// to wings level; [`TrimMode::Turn`] holds it as a coordinated turn.
    pub phi_deg: f64,
}

/// One instance's state after a step, in SI units (angles in radians).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FdmState {
    /// Geodetic latitude, degrees.
    pub lat_deg: f64,
    /// Longitude, degrees.
    pub lon_deg: f64,
    /// Height above the WGS84 ellipsoid, metres.
    pub alt_hae_m: f64,
    /// Euler attitude (3-2-1): roll, pitch, true heading.
    pub phi_rad: f64,
    pub theta_rad: f64,
    pub psi_rad: f64,
    /// Velocity relative to the Earth in the local-level NED frame at the aircraft.
    pub vel_north_mps: f64,
    pub vel_east_mps: f64,
    pub vel_down_mps: f64,
    /// True airspeed.
    pub tas_mps: f64,
    /// Body angular rates relative to the Earth (roll p, pitch q, yaw r).
    pub p_rad_s: f64,
    pub q_rad_s: f64,
    pub r_rad_s: f64,
    /// Angle of attack and sideslip.
    pub alpha_rad: f64,
    pub beta_rad: f64,
    /// Normal load factor (g, body −z); ≈ +1 in level flight.
    pub nz_g: f64,
    pub mach: f64,
    /// JSBSim's own simulation clock.
    pub sim_time_s: f64,
}

/// The four normalised FCS inputs (`fcs/*-cmd-norm`). Their meaning is
/// vehicle-class specific: surfaces + throttle on a fixed-wing, body-rate demands
/// + collective on a multirotor.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Controls {
    /// [-1, 1]
    pub aileron: f64,
    /// [-1, 1]
    pub elevator: f64,
    /// [-1, 1]
    pub rudder: f64,
    /// [0, 1] (engine 0; the collective on a multirotor)
    pub throttle: f64,
}

impl Controls {
    /// Clamp each channel to its valid range. A NaN channel becomes neutral (0)
    /// rather than poisoning the FCS.
    pub fn clamped(self) -> Self {
        fn c(v: f64, lo: f64, hi: f64) -> f64 {
            if v.is_nan() {
                0.0
            } else {
                v.clamp(lo, hi)
            }
        }
        Self {
            aileron: c(self.aileron, -1.0, 1.0),
            elevator: c(self.elevator, -1.0, 1.0),
            rudder: c(self.rudder, -1.0, 1.0),
            throttle: c(self.throttle, 0.0, 1.0),
        }
    }
}

/// JSBSim trim modes ASTRUM uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrimMode {
    /// Longitudinal only (pitch/throttle).
    Longitudinal = 0,
    /// Full six-axis trim for steady flight.
    Full = 1,
    /// On-ground trim.
    Ground = 2,
    /// Steady coordinated turn at the IC bank (`ic/phi-deg`): load factor
    /// 1/cos φ, turn rate g·tan φ / V (JSBSim `tTurn`).
    Turn = 5,
}

// --- FFI -------------------------------------------------------------------

#[repr(C)]
struct FdmHandle {
    _data: [u8; 0],
    _marker: PhantomData<(*mut u8, PhantomPinned)>,
}

#[repr(C)]
#[derive(Default)]
struct RawState {
    lat_geod_deg: c_double,
    lon_deg: c_double,
    alt_hae_ft: c_double,
    phi_rad: c_double,
    theta_rad: c_double,
    psi_rad: c_double,
    v_north_fps: c_double,
    v_east_fps: c_double,
    v_down_fps: c_double,
    vt_fps: c_double,
    p_rad_s: c_double,
    q_rad_s: c_double,
    r_rad_s: c_double,
    alpha_rad: c_double,
    beta_rad: c_double,
    nz_g: c_double,
    mach: c_double,
    sim_time_s: c_double,
}

#[repr(C)]
struct RawControls {
    aileron: c_double,
    elevator: c_double,
    rudder: c_double,
    throttle: c_double,
}

const FDM_OK: c_int = 0;
const FDM_MISSING: c_int = 2;

extern "C" {
    fn fdm_create(
        root: *const c_char,
        model: *const c_char,
        debug_level: c_int,
        err: *mut c_char,
        err_len: usize,
    ) -> *mut FdmHandle;
    fn fdm_destroy(h: *mut FdmHandle);
    fn fdm_last_error(h: *const FdmHandle) -> *const c_char;
    fn fdm_get(h: *mut FdmHandle, name: *const c_char, out: *mut c_double) -> c_int;
    fn fdm_set(h: *mut FdmHandle, name: *const c_char, value: c_double) -> c_int;
    fn fdm_set_dt(h: *mut FdmHandle, dt: c_double) -> c_int;
    fn fdm_run_ic(h: *mut FdmHandle) -> c_int;
    fn fdm_trim(h: *mut FdmHandle, mode: c_int) -> c_int;
    fn fdm_reset(h: *mut FdmHandle) -> c_int;
    fn fdm_run(h: *mut FdmHandle, running: *mut c_int) -> c_int;
    fn fdm_read_state(h: *mut FdmHandle, out: *mut RawState) -> c_int;
    fn fdm_write_controls(h: *mut FdmHandle, c: *const RawControls) -> c_int;
    fn fdm_set_ic_position(
        h: *mut FdmHandle,
        lat_geod_deg: c_double,
        lon_deg: c_double,
        alt_hae_ft: c_double,
    ) -> c_int;
}

/// Serialises JSBSim lifecycle calls against everything else (see module docs).
static JSBSIM: RwLock<()> = RwLock::new(());

fn exclusive() -> RwLockWriteGuard<'static, ()> {
    JSBSIM.write().unwrap_or_else(PoisonError::into_inner)
}

fn shared() -> RwLockReadGuard<'static, ()> {
    JSBSIM.read().unwrap_or_else(PoisonError::into_inner)
}

fn cstring(s: &str) -> Result<CString> {
    CString::new(s).map_err(|_| FdmError::InvalidName(s.to_owned()))
}

/// A single JSBSim flight-dynamics instance.
pub struct Fdm {
    handle: NonNull<FdmHandle>,
}

// SAFETY: an `FGFDMExec` shares no mutable state with other instances except the
// construction-time globals described in the module docs, and every call that can
// touch those holds `JSBSIM` exclusively while all other calls hold it shared. The
// instance itself is only reachable through `&mut self`, so moving it between
// threads is sound. `Fdm` is intentionally not `Sync`.
unsafe impl Send for Fdm {}

impl Fdm {
    /// Create an instance rooted at `root` (containing `aircraft/`, `engine/`,
    /// `systems/`) and load `model`. JSBSim's console chatter is silenced unless
    /// `$JSBSIM_DEBUG` is set.
    pub fn load(root: &Path, model: &str) -> Result<Self> {
        let root_c = cstring(&root.to_string_lossy())?;
        let model_c = cstring(model)?;
        let debug_level: c_int = if std::env::var_os("JSBSIM_DEBUG").is_some() { -1 } else { 0 };
        let mut err = [0 as c_char; 512];
        let _g = exclusive();
        // SAFETY: valid NUL-terminated strings and an `err` buffer of the stated length.
        let h = unsafe {
            fdm_create(root_c.as_ptr(), model_c.as_ptr(), debug_level, err.as_mut_ptr(), err.len())
        };
        match NonNull::new(h) {
            Some(handle) => Ok(Self { handle }),
            None => {
                // SAFETY: the shim always NUL-terminates `err` on failure.
                let msg = unsafe { CStr::from_ptr(err.as_ptr()) }.to_string_lossy().into_owned();
                Err(FdmError::Load(msg))
            }
        }
    }

    fn last_error(&self) -> String {
        // SAFETY: the handle is live; the shim returns a NUL-terminated string it owns.
        unsafe { CStr::from_ptr(fdm_last_error(self.handle.as_ptr())) }
            .to_string_lossy()
            .into_owned()
    }

    fn check(&self, status: c_int) -> Result<()> {
        match status {
            FDM_OK => Ok(()),
            FDM_MISSING => Err(FdmError::MissingProperty(self.last_error())),
            _ => Err(FdmError::Jsbsim(self.last_error())),
        }
    }

    /// Read a property-tree value. Fails if the model has no such property.
    pub fn get(&mut self, name: &str) -> Result<f64> {
        let c = cstring(name)?;
        let mut out = 0.0;
        let _g = shared();
        // SAFETY: live handle, valid string, valid out pointer.
        let st = unsafe { fdm_get(self.handle.as_ptr(), c.as_ptr(), &mut out) };
        self.check(st).map(|()| out)
    }

    /// Write a property-tree value. Fails if the model has no such property or it
    /// is read-only — a typo can never silently create a dead node.
    pub fn set(&mut self, name: &str, value: f64) -> Result<()> {
        let c = cstring(name)?;
        let _g = shared();
        // SAFETY: live handle and valid string.
        let st = unsafe { fdm_set(self.handle.as_ptr(), c.as_ptr(), value) };
        self.check(st)
    }

    /// Stage initial conditions (position, heading, airspeed, flight-path angle, bank).
    /// They take effect on the next [`run_ic`](Self::run_ic) / [`reset`](Self::reset).
    pub fn set_initial_conditions(&mut self, ic: &InitialConditions) -> Result<()> {
        {
            let _g = shared();
            // SAFETY: live handle.
            let st = unsafe {
                fdm_set_ic_position(
                    self.handle.as_ptr(),
                    ic.lat_geod_deg,
                    ic.lon_deg,
                    ic.alt_hae_m / FT_TO_M,
                )
            };
            self.check(st)?;
        }
        self.set("ic/psi-true-deg", ic.psi_true_deg)?;
        // Speed first: JSBSim scales the existing velocity vector, so the angles
        // set afterwards orient a non-zero velocity rather than being lost.
        self.set("ic/vt-fps", ic.tas_mps / FT_TO_M)?;
        self.set("ic/phi-deg", ic.phi_deg)?;
        self.set("ic/gamma-deg", ic.gamma_deg)
    }

    /// Apply the staged initial conditions.
    pub fn run_ic(&mut self) -> Result<()> {
        let _g = exclusive();
        // SAFETY: live handle.
        let st = unsafe { fdm_run_ic(self.handle.as_ptr()) };
        self.check(st)
    }

    /// Trim the aircraft for steady flight at the current conditions. A trim that
    /// does not converge is an error, never a silently untrimmed start.
    pub fn trim(&mut self, mode: TrimMode) -> Result<()> {
        let _g = exclusive();
        // SAFETY: live handle; `mode` is a valid JSBSim trim mode.
        let st = unsafe { fdm_trim(self.handle.as_ptr(), mode as c_int) };
        self.check(st)
    }

    /// Re-initialise every JSBSim model and re-apply the staged initial
    /// conditions (`ResetToInitialConditions`), without reloading the model.
    pub fn reset(&mut self) -> Result<()> {
        let _g = exclusive();
        // SAFETY: live handle.
        let st = unsafe { fdm_reset(self.handle.as_ptr()) };
        self.check(st)
    }

    /// Start every engine running (an airborne start). Without this a piston
    /// engine stays off and throttle is inert.
    pub fn start_engines(&mut self) -> Result<()> {
        self.set("propulsion/set-running", -1.0)
    }

    /// Set the integration timestep, in seconds.
    pub fn set_dt(&mut self, dt: f64) -> Result<()> {
        let _g = shared();
        // SAFETY: live handle.
        let st = unsafe { fdm_set_dt(self.handle.as_ptr(), dt) };
        self.check(st)
    }

    /// Advance one timestep.
    pub fn step(&mut self) -> Result<()> {
        let mut running: c_int = 0;
        let _g = shared();
        // SAFETY: live handle, valid out pointer.
        let st = unsafe { fdm_run(self.handle.as_ptr(), &mut running) };
        self.check(st)?;
        if running == 0 {
            return Err(FdmError::Terminated);
        }
        Ok(())
    }

    /// Read the standard state in one call (no property-name lookups).
    pub fn state(&mut self) -> Result<FdmState> {
        let mut r = RawState::default();
        let _g = shared();
        // SAFETY: live handle, valid out pointer to a `#[repr(C)]` mirror of `FdmState`.
        let st = unsafe { fdm_read_state(self.handle.as_ptr(), &mut r) };
        self.check(st)?;
        Ok(FdmState {
            lat_deg: r.lat_geod_deg,
            lon_deg: r.lon_deg,
            alt_hae_m: r.alt_hae_ft * FT_TO_M,
            phi_rad: r.phi_rad,
            theta_rad: r.theta_rad,
            psi_rad: r.psi_rad,
            vel_north_mps: r.v_north_fps * FT_TO_M,
            vel_east_mps: r.v_east_fps * FT_TO_M,
            vel_down_mps: r.v_down_fps * FT_TO_M,
            tas_mps: r.vt_fps * FT_TO_M,
            p_rad_s: r.p_rad_s,
            q_rad_s: r.q_rad_s,
            r_rad_s: r.r_rad_s,
            alpha_rad: r.alpha_rad,
            beta_rad: r.beta_rad,
            nz_g: r.nz_g,
            mach: r.mach,
            sim_time_s: r.sim_time_s,
        })
    }

    /// Write the four FCS inputs (clamped to their valid ranges).
    pub fn set_controls(&mut self, c: Controls) -> Result<()> {
        let c = c.clamped();
        let raw = RawControls {
            aileron: c.aileron,
            elevator: c.elevator,
            rudder: c.rudder,
            throttle: c.throttle,
        };
        let _g = shared();
        // SAFETY: live handle, valid pointer to a `#[repr(C)]` controls struct.
        let st = unsafe { fdm_write_controls(self.handle.as_ptr(), &raw) };
        self.check(st)
    }
}

impl Drop for Fdm {
    fn drop(&mut self) {
        let _g = exclusive();
        // SAFETY: the handle is live and uniquely owned; it is never used again.
        unsafe { fdm_destroy(self.handle.as_ptr()) };
    }
}
