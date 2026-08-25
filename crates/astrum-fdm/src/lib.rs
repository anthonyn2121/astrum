//! `astrum-fdm` — a thin, safe Rust wrapper over a single JSBSim `FGFDMExec`
//! instance, via a small C++ shim (see `shim/`).
//!
//! Each [`Fdm`] owns exactly one flight-dynamics instance: the building block
//! that lets every ASTRUM ECS entity carry its own independent aircraft physics.
//!
//! ```no_run
//! use astrum_fdm::{Fdm, prop};
//! let mut fdm = Fdm::new();
//! fdm.set_root_dir("/opt/jsbsim/share/jsbsim");
//! assert!(fdm.load_model("c172x"));
//! fdm.set("ic/h-sl-ft", 5000.0);
//! fdm.set("ic/vc-kts", 100.0);
//! assert!(fdm.run_ic());
//! fdm.set_dt(1.0 / 120.0);
//! fdm.set(prop::THROTTLE_CMD, 0.6);
//! fdm.step();
//! let alt_m = fdm.get(prop::ALT_SL_M);
//! ```

use std::ffi::{c_char, c_double, c_int, CString};

/// Opaque handle to the C++ `FGFDMExec` (matches `shim/jsbsim_shim.h`).
#[repr(C)]
struct FdmHandle {
    _private: [u8; 0],
}

extern "C" {
    fn fdm_new() -> *mut FdmHandle;
    fn fdm_free(h: *mut FdmHandle);
    fn fdm_set_root_dir(h: *mut FdmHandle, path: *const c_char);
    fn fdm_load_model(h: *mut FdmHandle, model: *const c_char) -> c_int;
    fn fdm_set_dt(h: *mut FdmHandle, dt: c_double);
    fn fdm_run_ic(h: *mut FdmHandle) -> c_int;
    fn fdm_run(h: *mut FdmHandle) -> c_int;
    fn fdm_get_prop(h: *mut FdmHandle, name: *const c_char) -> c_double;
    fn fdm_set_prop(h: *mut FdmHandle, name: *const c_char, value: c_double);
}

/// A single JSBSim flight-dynamics instance.
pub struct Fdm {
    handle: *mut FdmHandle,
}

// Safety: `Fdm` uniquely owns its `FGFDMExec`; instances share no state. ASTRUM only
// ever touches a given `Fdm` through exclusive (`&mut`) access — the Bevy scheduler
// guarantees no two systems alias the same component concurrently — so it is safe to
// move between threads and to hold as a `Send + Sync` ECS component.
unsafe impl Send for Fdm {}
unsafe impl Sync for Fdm {}

impl Fdm {
    /// Create a new, empty instance. Call [`Fdm::set_root_dir`] and
    /// [`Fdm::load_model`] before running.
    pub fn new() -> Self {
        let handle = unsafe { fdm_new() };
        assert!(!handle.is_null(), "fdm_new returned null");
        Self { handle }
    }

    /// Set the JSBSim root directory (contains `aircraft/`, `engine/`, `systems/`).
    pub fn set_root_dir(&mut self, path: &str) {
        let c = to_cstring(path);
        unsafe { fdm_set_root_dir(self.handle, c.as_ptr()) }
    }

    /// Load an aircraft model by name (e.g. `"c172x"`). Returns `true` on success.
    pub fn load_model(&mut self, model: &str) -> bool {
        let c = to_cstring(model);
        unsafe { fdm_load_model(self.handle, c.as_ptr()) != 0 }
    }

    /// Set the integration timestep, in seconds.
    pub fn set_dt(&mut self, dt: f64) {
        unsafe { fdm_set_dt(self.handle, dt) }
    }

    /// Apply the initial conditions set via the `ic/*` properties. Returns `true` on success.
    pub fn run_ic(&mut self) -> bool {
        unsafe { fdm_run_ic(self.handle) != 0 }
    }

    /// Advance the simulation by one timestep. Returns `true` while the model is running.
    pub fn step(&mut self) -> bool {
        unsafe { fdm_run(self.handle) != 0 }
    }

    /// Read a property-tree value (string-keyed double).
    pub fn get(&self, name: &str) -> f64 {
        let c = to_cstring(name);
        unsafe { fdm_get_prop(self.handle, c.as_ptr()) }
    }

    /// Write a property-tree value (string-keyed double).
    pub fn set(&mut self, name: &str, value: f64) {
        let c = to_cstring(name);
        unsafe { fdm_set_prop(self.handle, c.as_ptr(), value) }
    }
}

impl Drop for Fdm {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { fdm_free(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

impl Default for Fdm {
    fn default() -> Self {
        Self::new()
    }
}

fn to_cstring(s: &str) -> CString {
    CString::new(s).expect("JSBSim property/model names must not contain interior NUL bytes")
}

/// JSBSim property names ASTRUM relies on (verified against the JSBSim 1.3.1
/// `c172x --catalog` output in the container).
pub mod prop {
    // --- control inputs ---
    pub const AILERON_CMD: &str = "fcs/aileron-cmd-norm";
    pub const ELEVATOR_CMD: &str = "fcs/elevator-cmd-norm";
    pub const RUDDER_CMD: &str = "fcs/rudder-cmd-norm";
    /// Throttle is registered per-engine and indexed; engine 0. For a multirotor
    /// (e.g. the F450) index 0 is the FCS **collective** input feeding the motor
    /// mixer, so the same constant drives both vehicle classes.
    pub const THROTTLE_CMD: &str = "fcs/throttle-cmd-norm[0]";
    /// Write `-1` to start all engines running (airborne start); an index `n`
    /// starts engine `n`. Without this a piston engine stays off.
    pub const PROPULSION_SET_RUNNING: &str = "propulsion/set-running";
    /// Multirotor stability-augmentation gate (F450 `FlightControl`). Defaults to
    /// `0`, which **disables the inner rate-PID loop entirely** — with it off the
    /// quad has no attitude control at all (roll/pitch/yaw commands do nothing).
    /// ASTRUM sets this to `1.0` for multirotor entities. Absent on fixed-wing
    /// models (setting it there is a harmless no-op).
    pub const SCAS_ENGAGE: &str = "fcs/ScasEngage";

    // --- state outputs ---
    /// **Geodetic** latitude (deg). Pairs with WGS84 lat/lon/alt; do NOT use the
    /// geocentric `position/lat-gc-deg`, which differs by up to ~0.19°.
    pub const LAT_GEOD_DEG: &str = "position/lat-geod-deg";
    /// Longitude (deg). Longitude is datum-independent, so `long-gc` is correct.
    pub const LON_GC_DEG: &str = "position/long-gc-deg";
    /// Altitude above sea level (m). JSBSim's sea level is the WGS84 ellipsoid
    /// (no geoid model), so this is the ellipsoidal height to pair with LAT_GEOD_DEG.
    pub const ALT_SL_M: &str = "position/h-sl-meters";
    pub const PHI_RAD: &str = "attitude/phi-rad";
    pub const THETA_RAD: &str = "attitude/theta-rad";
    pub const PSI_RAD: &str = "attitude/psi-rad";
    pub const VT_FPS: &str = "velocities/vt-fps";
    pub const V_NORTH_FPS: &str = "velocities/v-north-fps";
    pub const V_EAST_FPS: &str = "velocities/v-east-fps";
    pub const V_DOWN_FPS: &str = "velocities/v-down-fps";
    pub const P_RAD_S: &str = "velocities/p-rad_sec";
    pub const Q_RAD_S: &str = "velocities/q-rad_sec";
    pub const R_RAD_S: &str = "velocities/r-rad_sec";
    pub const SIM_TIME: &str = "simulation/sim-time-sec";

    // --- initial conditions (RW; set before RunIC) ---
    pub const IC_LAT_GEOD_DEG: &str = "ic/lat-geod-deg";
    pub const IC_LON_GC_DEG: &str = "ic/long-gc-deg";
    pub const IC_H_SL_FT: &str = "ic/h-sl-ft";
    pub const IC_PSI_TRUE_DEG: &str = "ic/psi-true-deg";
    /// Calibrated airspeed (kts).
    pub const IC_VC_KTS: &str = "ic/vc-kts";
}

/// Feet → metres.
pub const FT_TO_M: f64 = 0.3048;
