//! Write the Rust encoding of the canonical wire fixtures
//! (`tests/fixtures/<kind>.rust.bin`). Run via `scripts/wire-fixtures.sh`.

// Shared with the decode test, which also uses the language list.
#[allow(dead_code)]
#[path = "../tests/fixture/mod.rs"]
mod fixture;

use astrum_proto::{encode_command, encode_telemetry};

fn main() {
    let values = fixture::values();
    for kind in fixture::KINDS {
        let bytes = if kind == "telemetry" {
            encode_telemetry(&fixture::telemetry(&values[kind]))
        } else {
            encode_command(&fixture::command(&values[kind]))
        };
        let path = fixture::fixtures_dir().join(format!("{kind}.rust.bin"));
        std::fs::write(&path, bytes).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        println!("wrote {}", path.display());
    }
}
