//! Write the Rust encoding of the canonical wire fixtures
//! (`tests/fixtures/<kind>.rust.bin`). Run via `scripts/wire-fixtures.sh`.

// Shared with the decode test, which also uses the decode half.
#[allow(dead_code)]
#[path = "../tests/fixture/mod.rs"]
mod fixture;

fn main() {
    let values = fixture::values();
    for kind in fixture::KINDS {
        let path = fixture::fixtures_dir().join(format!("{kind}.rust.bin"));
        std::fs::write(&path, fixture::encode(&values, kind))
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        println!("wrote {}", path.display());
    }
}
