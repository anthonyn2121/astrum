//! Compile `schema/astrum.capnp` to Rust via the system `capnp` compiler
//! (installed in the dev image). Output lands in `$OUT_DIR/astrum_capnp.rs`.

fn main() {
    capnpc::CompilerCommand::new()
        .src_prefix("schema")
        .file("schema/astrum.capnp")
        .run()
        .expect("compiling schema/astrum.capnp failed");

    println!("cargo:rerun-if-changed=schema/astrum.capnp");
}
