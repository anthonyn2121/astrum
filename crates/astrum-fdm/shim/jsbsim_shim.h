/* C ABI over one JSBSim FGFDMExec, consumed by astrum-fdm (Rust).
 *
 * Contract:
 *  - Every function catches every C++ exception; none can unwind into the caller.
 *    Failure is reported by a non-FDM_OK status, and the message is available via
 *    fdm_last_error() until the next call on the same handle.
 *  - Property access never creates nodes: an unknown name returns FDM_MISSING.
 *  - The standard state/control properties are resolved once in fdm_create, so
 *    fdm_read_state / fdm_write_controls do no string lookups.
 *  - Thread safety is the caller's job (see astrum-fdm's lifecycle lock): JSBSim
 *    lazily initialises a process-global unit-conversion table
 *    (Element::convert) and writes the static FGJSBBase::debug_lvl during
 *    construction, so fdm_create / fdm_run_ic / fdm_trim / fdm_reset /
 *    fdm_destroy must not run concurrently with any other JSBSim call.
 */
#ifndef ASTRUM_JSBSIM_SHIM_H
#define ASTRUM_JSBSIM_SHIM_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct FdmHandle FdmHandle;

enum { FDM_OK = 0, FDM_ERR = 1, FDM_MISSING = 2 };

/* Standard per-step outputs, in JSBSim's native units. */
typedef struct FdmState {
  double lat_geod_deg;
  double lon_deg;
  double alt_hae_ft; /* position/geod-alt-ft: height above the WGS84 ellipsoid */
  double phi_rad;
  double theta_rad;
  double psi_rad;
  double v_north_fps;
  double v_east_fps;
  double v_down_fps;
  double vt_fps; /* true airspeed */
  double p_rad_s;
  double q_rad_s;
  double r_rad_s;
  double alpha_rad;
  double beta_rad;
  double nz_g;
  double mach;
  double sim_time_s;
} FdmState;

/* Normalised FCS inputs: surfaces in [-1, 1], throttle (engine 0 / collective) in [0, 1]. */
typedef struct FdmControls {
  double aileron;
  double elevator;
  double rudder;
  double throttle;
} FdmControls;

/* Create an instance rooted at `root` and load `model`. `debug_level` < 0 keeps
 * JSBSim's own choice ($JSBSIM_DEBUG or its default); >= 0 forces that level.
 * Returns NULL on failure and writes a NUL-terminated message into `err`. */
FdmHandle* fdm_create(const char* root, const char* model, int debug_level, char* err,
                      size_t err_len);
void fdm_destroy(FdmHandle* h);
const char* fdm_last_error(const FdmHandle* h);

int fdm_get(FdmHandle* h, const char* name, double* out);
int fdm_set(FdmHandle* h, const char* name, double value);
int fdm_set_dt(FdmHandle* h, double dt);
int fdm_run_ic(FdmHandle* h);
/* JSBSim trim mode (0 longitudinal, 1 full, 2 ground, ...). FDM_ERR if it fails. */
int fdm_trim(FdmHandle* h, int mode);
/* ResetToInitialConditions: re-initialise every model and re-run the staged ICs. */
int fdm_reset(FdmHandle* h);
/* One integration step. `*running` is 0 once JSBSim reports the run complete. */
int fdm_run(FdmHandle* h, int* running);
int fdm_read_state(FdmHandle* h, FdmState* out);
int fdm_write_controls(FdmHandle* h, const FdmControls* in);

/* Discard every staged initial condition (back to JSBSim's defaults) and stage
 * the position as geodetic latitude, longitude and height above the WGS84
 * ellipsoid. Call this first, then stage the remaining ICs. JSBSim's
 * `ic/geod-alt-ft` is read-only and `ic/h-sl-ft` is radius-based, so this
 * converts HAE to JSBSim's sea-level-radius altitude with JSBSim's own ellipsoid. */
int fdm_set_ic_position(FdmHandle* h, double lat_geod_deg, double lon_deg, double alt_hae_ft);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // ASTRUM_JSBSIM_SHIM_H
