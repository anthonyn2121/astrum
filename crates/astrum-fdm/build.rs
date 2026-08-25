// Build the JSBSim C++ shim and link it against libJSBSim.
//
// JSBSim is located via pkg-config (the `JSBSim.pc` synthesized in the astrum-dev
// image). We deliberately suppress pkg-config's own link metadata and emit it
// AFTER compiling the shim, so the linker sees our shim's static archive before
// libJSBSim (correct left-to-right resolution order for GNU ld).

fn main() {
    let jsbsim = pkg_config::Config::new().cargo_metadata(false).probe("JSBSim").expect(
        "JSBSim not found via pkg-config. Build inside the astrum-dev image \
             (see docker/Dockerfile), which provides /opt/jsbsim + JSBSim.pc.",
    );

    // Compile the shim (emits the link directive for libjsbsim_shim.a).
    let mut build = cc::Build::new();
    build.cpp(true).std("c++17").file("shim/jsbsim_shim.cpp");
    for inc in &jsbsim.include_paths {
        build.include(inc);
    }
    build.compile("jsbsim_shim");

    // Now emit libJSBSim's link flags (after the shim) + rpath so the test/binary
    // runs without needing LD_LIBRARY_PATH set.
    for path in &jsbsim.link_paths {
        println!("cargo:rustc-link-search=native={}", path.display());
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", path.display());
    }
    for lib in &jsbsim.libs {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }

    println!("cargo:rerun-if-changed=shim/jsbsim_shim.cpp");
    println!("cargo:rerun-if-changed=shim/jsbsim_shim.h");
}
