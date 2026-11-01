//! Write the Rust encoding of the canonical wire fixtures
//! (`tests/fixtures/<kind>.rust.bin`). Run via `scripts/wire-fixtures.sh`.

// Shared with the decode test, which also uses the language list.
#[allow(dead_code)]
#[path = "../tests/fixture/mod.rs"]
mod fixture;

use astrum_proto::{encode_command, encode_event, encode_telemetry};

fn main() {
    let values = fixture::values();
    let encoded = fixture::KINDS.iter().map(|&kind| {
        let bytes = if kind == "telemetry" {
            encode_telemetry(&fixture::telemetry(&values[kind]))
        } else {
            encode_command(&fixture::command(&values[kind]))
        };
        (kind, bytes)
    });
    let events = fixture::EVENT_KINDS
        .iter()
        .map(|&kind| (kind, encode_event(&fixture::event(&values[kind]))));
    for (kind, bytes) in encoded.chain(events) {
        let path = fixture::fixtures_dir().join(format!("{kind}.rust.bin"));
        std::fs::write(&path, bytes).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        println!("wrote {}", path.display());
    }
}
