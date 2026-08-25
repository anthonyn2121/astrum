/* C ABI over JSBSim's FGFDMExec. Each FdmHandle wraps one FGFDMExec instance.
 * Consumed by astrum-fdm's Rust FFI (crates/astrum-fdm/src/lib.rs). */
#ifndef ASTRUM_JSBSIM_SHIM_H
#define ASTRUM_JSBSIM_SHIM_H

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle to a JSBSim FGFDMExec. */
typedef struct FdmHandle FdmHandle;

FdmHandle *fdm_new(void);
void       fdm_free(FdmHandle *h);

/* Root dir containing aircraft/ engine/ systems/. */
void   fdm_set_root_dir(FdmHandle *h, const char *path);
/* Load an aircraft model by name; returns 1 on success, 0 on failure. */
int    fdm_load_model(FdmHandle *h, const char *model);
void   fdm_set_dt(FdmHandle *h, double dt);
int    fdm_run_ic(FdmHandle *h);  /* apply initial-condition properties; 1 = ok */
int    fdm_run(FdmHandle *h);     /* one step; 1 = still running */
double fdm_get_prop(FdmHandle *h, const char *name);
void   fdm_set_prop(FdmHandle *h, const char *name, double value);

#ifdef __cplusplus
}
#endif

#endif /* ASTRUM_JSBSIM_SHIM_H */
