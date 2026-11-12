// Embed libJSBSim's directory as an rpath in the extension module (cargo only
// applies astrum-fdm's own link args to astrum-fdm's targets).
fn main() {
    let libdir = std::env::var("DEP_JSBSIM_LIBDIR").expect("astrum-fdm exports DEP_JSBSIM_LIBDIR");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{libdir}");
}
