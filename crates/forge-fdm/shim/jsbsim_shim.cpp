// C++ implementation of the JSBSim C ABI shim (see jsbsim_shim.h).
//
// The property-tree API (Get/SetPropertyValue) plus the LoadModel/RunIC/Run
// lifecycle is all we need to drive JSBSim programmatically, and it maps cleanly
// to plain C strings + doubles across the FFI boundary.
#include "jsbsim_shim.h"

#include <FGFDMExec.h>
#include <simgear/misc/sg_path.hxx>

#include <string>

using JSBSim::FGFDMExec;

namespace {
inline FGFDMExec *as_fdm(FdmHandle *h) {
    return reinterpret_cast<FGFDMExec *>(h);
}
}  // namespace

extern "C" {

FdmHandle *fdm_new(void) {
    return reinterpret_cast<FdmHandle *>(new FGFDMExec());
}

void fdm_free(FdmHandle *h) {
    delete as_fdm(h);
}

void fdm_set_root_dir(FdmHandle *h, const char *path) {
    FGFDMExec *fdm = as_fdm(h);
    fdm->SetRootDir(SGPath(std::string(path)));
    // SetRootDir only stores the root; the aircraft/engine/systems paths are only
    // re-based onto it when explicitly (re)set. Mirror the JSBSim CLI's --root
    // handling so LoadModel resolves <root>/aircraft/<model>/... rather than CWD.
    fdm->SetAircraftPath(SGPath("aircraft"));
    fdm->SetEnginePath(SGPath("engine"));
    fdm->SetSystemsPath(SGPath("systems"));
}

int fdm_load_model(FdmHandle *h, const char *model) {
    return as_fdm(h)->LoadModel(std::string(model)) ? 1 : 0;
}

void fdm_set_dt(FdmHandle *h, double dt) {
    as_fdm(h)->Setdt(dt);
}

int fdm_run_ic(FdmHandle *h) {
    return as_fdm(h)->RunIC() ? 1 : 0;
}

int fdm_run(FdmHandle *h) {
    return as_fdm(h)->Run() ? 1 : 0;
}

double fdm_get_prop(FdmHandle *h, const char *name) {
    return as_fdm(h)->GetPropertyValue(std::string(name));
}

void fdm_set_prop(FdmHandle *h, const char *name, double value) {
    as_fdm(h)->SetPropertyValue(std::string(name), value);
}

}  // extern "C"
