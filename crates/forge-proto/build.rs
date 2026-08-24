//! Compile `schema/forge.capnp` to Rust via the system `capnp` compiler
//! (installed in the dev image). Output lands in `$OUT_DIR/forge_capnp.rs`.

fn main() {
    capnpc::CompilerCommand::new()
        .src_prefix("schema")
        .file("schema/forge.capnp")
        .run()
        .expect("compiling schema/forge.capnp failed");

    println!("cargo:rerun-if-changed=schema/forge.capnp");
}
