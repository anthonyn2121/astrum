//! Cross-language wire check: every language's encoding of the canonical fixture
//! values (`tests/fixtures/<kind>.<lang>.bin`, written by `scripts/wire-fixtures.sh`)
//! must decode in Rust to exactly those values.

// Shared with the `wire_fixture` example, which uses the encode half.
#[allow(dead_code)]
#[path = "fixture/mod.rs"]
mod fixture;

#[test]
fn every_language_fixture_decodes_to_the_canonical_values() {
    let values = fixture::values();
    let mut failures = Vec::new();
    for lang in fixture::LANGS {
        for kind in fixture::KINDS {
            let path = fixture::fixtures_dir().join(format!("{kind}.{lang}.bin"));
            let result = std::fs::read(&path)
                .map_err(|e| format!("{e} (run scripts/wire-fixtures.sh)"))
                .and_then(|bytes| fixture::check(&values, kind, &bytes));
            if let Err(e) = result {
                failures.push(format!("{}: {e}", path.display()));
            }
        }
    }
    assert!(failures.is_empty(), "{} fixture(s) failed:\n{}", failures.len(), failures.join("\n"));
}
