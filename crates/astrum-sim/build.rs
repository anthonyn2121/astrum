// Embed libJSBSim's directory as an rpath in this package's binaries, tests
// and examples (cargo only applies astrum-fdm's own link args to astrum-fdm's
// targets; it exports the directory to direct dependents as DEP_JSBSIM_LIBDIR).
fn main() {
    let libdir = std::env::var("DEP_JSBSIM_LIBDIR").expect("astrum-fdm exports DEP_JSBSIM_LIBDIR");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{libdir}");
}
