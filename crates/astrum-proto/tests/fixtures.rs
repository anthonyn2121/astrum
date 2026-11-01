//! Cross-language wire check: every language's encoding of the canonical fixture
//! values (`tests/fixtures/<kind>.<lang>.bin`, written by `scripts/wire-fixtures.sh`)
//! must decode in Rust to exactly those values.

#[path = "fixture/mod.rs"]
mod fixture;

use astrum_proto::{decode_command, decode_telemetry};

#[test]
fn every_language_fixture_decodes_to_the_canonical_values() {
    let values = fixture::values();
    for lang in fixture::LANGS {
        for kind in fixture::KINDS {
            let path = fixture::fixtures_dir().join(format!("{kind}.{lang}.bin"));
            let bytes = std::fs::read(&path).unwrap_or_else(|e| {
                panic!("{}: {e} (run scripts/wire-fixtures.sh)", path.display())
            });
            if kind == "telemetry" {
                let got = decode_telemetry(&bytes).unwrap_or_else(|e| panic!("{lang} {kind}: {e}"));
                assert_eq!(got, fixture::telemetry(&values[kind]), "{lang} {kind}");
            } else {
                let got = decode_command(&bytes).unwrap_or_else(|e| panic!("{lang} {kind}: {e}"));
                assert_eq!(got, fixture::command(&values[kind]), "{lang} {kind}");
            }
        }
    }
}
