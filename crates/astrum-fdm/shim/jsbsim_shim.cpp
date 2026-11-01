// C++ implementation of the JSBSim C ABI shim (see jsbsim_shim.h for the contract).
#include "jsbsim_shim.h"

#include <FGFDMExec.h>
#include <FGJSBBase.h>
#include <initialization/FGInitialCondition.h>
#include <input_output/FGPropertyManager.h>
#include <math/FGLocation.h>
#include <models/FGInertial.h>

#include <cstdio>
#include <cstring>
#include <exception>
#include <simgear/misc/sg_path.hxx>
#include <string>

using JSBSim::FGFDMExec;

struct FdmHandle {
  FGFDMExec exec;
  std::string last_error;

  // Standard properties, resolved once after LoadModel (see resolve_nodes).
  SGPropertyNode_ptr lat, lon, alt, phi, theta, psi, vn, ve, vd, vt, p, q, r, alpha, beta, nz, mach,
      time;
  SGPropertyNode_ptr ail, elev, rud, thr;
};

namespace {

void set_error(FdmHandle* h, const char* msg) noexcept {
  try {
    h->last_error = msg;
  } catch (...) {
  }
}

void write_err(char* err, size_t err_len, const char* msg) noexcept {
  if (err != nullptr && err_len > 0) std::snprintf(err, err_len, "%s", msg);
}

// Some models declare <output> files (c172x writes JSBout172B.csv into the
// working directory). JSBSim opens them at every model initialisation even when
// output is disabled, and every instance would race on the same file, so each
// output is pointed at the null device. Renaming also resets JSBSim's run-id
// suffix, so a START_NEW_OUTPUT reset reopens exactly /dev/null.
void silence_outputs(FdmHandle* h) {
  for (int n = 0; h->exec.SetOutputFileName(n, "/dev/null"); ++n) {
  }
  h->exec.DisableOutput();
}

SGPropertyNode* node(FdmHandle* h, const char* name) {
  return h->exec.GetPropertyManager()->GetNode(std::string(name), false);
}

// Resolve every standard node or throw naming the first one the model lacks.
void resolve_nodes(FdmHandle* h) {
  auto need = [h](const char* name) -> SGPropertyNode_ptr {
    SGPropertyNode* n = node(h, name);
    if (n == nullptr) throw std::runtime_error(std::string("model lacks property ") + name);
    return SGPropertyNode_ptr(n);
  };
  h->lat = need("position/lat-geod-deg");
  h->lon = need("position/long-gc-deg");
  h->alt = need("position/geod-alt-ft");
  h->phi = need("attitude/phi-rad");
  h->theta = need("attitude/theta-rad");
  h->psi = need("attitude/psi-rad");
  h->vn = need("velocities/v-north-fps");
  h->ve = need("velocities/v-east-fps");
  h->vd = need("velocities/v-down-fps");
  h->vt = need("velocities/vt-fps");
  h->p = need("velocities/p-rad_sec");
  h->q = need("velocities/q-rad_sec");
  h->r = need("velocities/r-rad_sec");
  h->alpha = need("aero/alpha-rad");
  h->beta = need("aero/beta-rad");
  h->nz = need("accelerations/Nz");
  h->mach = need("velocities/mach");
  h->time = need("simulation/sim-time-sec");
  h->ail = need("fcs/aileron-cmd-norm");
  h->elev = need("fcs/elevator-cmd-norm");
  h->rud = need("fcs/rudder-cmd-norm");
  h->thr = need("fcs/throttle-cmd-norm[0]");
}

}  // namespace

#define FDM_GUARD(h, ...)                    \
  try {                                      \
    __VA_ARGS__                              \
  } catch (const std::exception& e) {        \
    set_error((h), e.what());                \
    return FDM_ERR;                          \
  } catch (...) {                            \
    set_error((h), "unknown C++ exception"); \
    return FDM_ERR;                          \
  }

extern "C" {

FdmHandle* fdm_create(const char* root, const char* model, int debug_level, char* err,
                      size_t err_len) {
  FdmHandle* h = nullptr;
  try {
    // debug_lvl is static and the constructor prints at the current level, so
    // set it first; the constructor then lets $JSBSIM_DEBUG override it.
    if (debug_level >= 0) JSBSim::FGJSBBase::debug_lvl = static_cast<short>(debug_level);
    h = new FdmHandle();
    if (debug_level >= 0) h->exec.SetDebugLevel(debug_level);

    h->exec.SetRootDir(SGPath(std::string(root)));
    // SetRootDir only stores the root; the model paths are re-based onto it only
    // when (re)set. Mirror the JSBSim CLI's --root handling.
    h->exec.SetAircraftPath(SGPath("aircraft"));
    h->exec.SetEnginePath(SGPath("engine"));
    h->exec.SetSystemsPath(SGPath("systems"));
    if (!h->exec.LoadModel(std::string(model))) {
      throw std::runtime_error(std::string("LoadModel failed for '") + model + "' under " + root);
    }
    silence_outputs(h);
    resolve_nodes(h);
    return h;
  } catch (const std::exception& e) {
    write_err(err, err_len, e.what());
  } catch (...) {
    write_err(err, err_len, "unknown C++ exception");
  }
  try {
    delete h;
  } catch (...) {
  }
  return nullptr;
}

void fdm_destroy(FdmHandle* h) {
  try {
    delete h;
  } catch (...) {
    // A throwing destructor must not unwind into Rust; the instance is gone either way.
  }
}

const char* fdm_last_error(const FdmHandle* h) { return h->last_error.c_str(); }

int fdm_get(FdmHandle* h, const char* name, double* out) {
  FDM_GUARD(h, {
    SGPropertyNode* n = node(h, name);
    if (n == nullptr) {
      set_error(h, (std::string("no such property: ") + name).c_str());
      return FDM_MISSING;
    }
    *out = n->getDoubleValue();
    return FDM_OK;
  })
}

int fdm_set(FdmHandle* h, const char* name, double value) {
  FDM_GUARD(h, {
    SGPropertyNode* n = node(h, name);
    if (n == nullptr) {
      set_error(h, (std::string("no such property: ") + name).c_str());
      return FDM_MISSING;
    }
    if (!n->setDoubleValue(value)) {
      set_error(h, (std::string("property is not writable: ") + name).c_str());
      return FDM_ERR;
    }
    return FDM_OK;
  })
}

int fdm_set_dt(FdmHandle* h, double dt) {
  FDM_GUARD(h, {
    h->exec.Setdt(dt);
    return FDM_OK;
  })
}

int fdm_set_ic_position(FdmHandle* h, double lat_geod_deg, double lon_deg, double alt_hae_ft) {
  FDM_GUARD(h, {
    constexpr double kDegToRad = 3.14159265358979323846 / 180.0;
    const double lat = lat_geod_deg * kDegToRad;
    const double lon = lon_deg * kDegToRad;
    const auto inertial = h->exec.GetInertial();
    JSBSim::FGLocation loc;
    loc.SetEllipse(inertial->GetSemimajor(), inertial->GetSemiminor());
    loc.SetPositionGeodetic(lon, lat, alt_hae_ft);
    const double asl_ft = loc.GetRadius() - loc.GetSeaLevelRadius();
    const auto ic = h->exec.GetIC();
    // Start from JSBSim's default IC so the result never depends on what was
    // staged before: JSBSim's IC setters solve relative to the current IC
    // state, so restaging the same values over an earlier IC (a reset) would
    // otherwise differ from a fresh load in the last bits.
    ic->InitializeIC();
    // Latitude first: with a geodetic latitude set, SetAltitudeASLFtIC solves for
    // the geodetic altitude while holding that latitude.
    ic->SetGeodLatitudeRadIC(lat);
    ic->SetLongitudeRadIC(lon);
    ic->SetAltitudeASLFtIC(asl_ft);
    return FDM_OK;
  })
}

int fdm_run_ic(FdmHandle* h) {
  FDM_GUARD(h, {
    if (!h->exec.RunIC()) {
      set_error(h, "RunIC failed");
      return FDM_ERR;
    }
    return FDM_OK;
  })
}

int fdm_trim(FdmHandle* h, int mode) {
  // DoTrim throws TrimFailureException when the trim does not converge.
  FDM_GUARD(h, {
    h->exec.DoTrim(mode);
    return FDM_OK;
  })
}

int fdm_reset(FdmHandle* h) {
  FDM_GUARD(h, {
    // START_NEW_OUTPUT closes each output before the models re-initialise and
    // reopen it (re-opening a still-open stream fails); silence_outputs first
    // re-points them at /dev/null with no run-id suffix.
    silence_outputs(h);
    h->exec.ResetToInitialConditions(FGFDMExec::START_NEW_OUTPUT);
    return FDM_OK;
  })
}

int fdm_run(FdmHandle* h, int* running) {
  FDM_GUARD(h, {
    *running = h->exec.Run() ? 1 : 0;
    return FDM_OK;
  })
}

int fdm_read_state(FdmHandle* h, FdmState* s) {
  FDM_GUARD(h, {
    s->lat_geod_deg = h->lat->getDoubleValue();
    s->lon_deg = h->lon->getDoubleValue();
    s->alt_hae_ft = h->alt->getDoubleValue();
    s->phi_rad = h->phi->getDoubleValue();
    s->theta_rad = h->theta->getDoubleValue();
    s->psi_rad = h->psi->getDoubleValue();
    s->v_north_fps = h->vn->getDoubleValue();
    s->v_east_fps = h->ve->getDoubleValue();
    s->v_down_fps = h->vd->getDoubleValue();
    s->vt_fps = h->vt->getDoubleValue();
    s->p_rad_s = h->p->getDoubleValue();
    s->q_rad_s = h->q->getDoubleValue();
    s->r_rad_s = h->r->getDoubleValue();
    s->alpha_rad = h->alpha->getDoubleValue();
    s->beta_rad = h->beta->getDoubleValue();
    s->nz_g = h->nz->getDoubleValue();
    s->mach = h->mach->getDoubleValue();
    s->sim_time_s = h->time->getDoubleValue();
    return FDM_OK;
  })
}

int fdm_write_controls(FdmHandle* h, const FdmControls* c) {
  FDM_GUARD(h, {
    h->ail->setDoubleValue(c->aileron);
    h->elev->setDoubleValue(c->elevator);
    h->rud->setDoubleValue(c->rudder);
    h->thr->setDoubleValue(c->throttle);
    return FDM_OK;
  })
}

}  // extern "C"
