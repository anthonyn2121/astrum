// Build the JSBSim C++ shim and link it against libJSBSim.
//
// JSBSim is located via pkg-config (the `JSBSim.pc` synthesized in the astrum-dev
// image). pkg-config's own link metadata is suppressed and emitted AFTER compiling
// the shim, so the linker sees the shim's static archive before libJSBSim (GNU ld
// resolves left to right).
//
// `rustc-link-arg` only reaches this package's own targets, so the library
// directory is also exported as `DEP_JSBSIM_LIBDIR` (this crate declares
// `links = "jsbsim"`); every final artifact that links JSBSim (the astrum-sim
// binaries and the astrum-env cdylib) reads it in its own build script and embeds
// the rpath itself.

fn main() {
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");
    println!("cargo:rerun-if-changed=shim/jsbsim_shim.cpp");
    println!("cargo:rerun-if-changed=shim/jsbsim_shim.h");

    let jsbsim = pkg_config::Config::new().cargo_metadata(false).probe("JSBSim").expect(
        "JSBSim not found via pkg-config. Build inside the astrum-dev image \
         (see docker/Dockerfile), which provides /opt/jsbsim + JSBSim.pc.",
    );

    let mut build = cc::Build::new();
    build.cpp(true).std("c++17").file("shim/jsbsim_shim.cpp");
    for inc in &jsbsim.include_paths {
        // -isystem: JSBSim's headers are third-party; their warnings are not ours.
        build.flag("-isystem").flag(inc.to_str().expect("JSBSim include path is not UTF-8"));
        // Rebuild the shim when JSBSim is upgraded under the same prefix.
        println!("cargo:rerun-if-changed={}", inc.join("FGFDMExec.h").display());
    }
    build.flag("-Werror");
    build.compile("jsbsim_shim");

    for path in &jsbsim.link_paths {
        println!("cargo:rustc-link-search=native={}", path.display());
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", path.display());
        println!("cargo:rerun-if-changed={}", path.join("libJSBSim.so").display());
    }
    for lib in &jsbsim.libs {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
    let libdir = jsbsim.link_paths.first().expect("JSBSim.pc has no -L path");
    println!("cargo:libdir={}", libdir.display());
}
