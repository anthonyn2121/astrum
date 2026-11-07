//! The catalog: `astrum list` classifies files by their keys and shows each
//! one's first comment sentence, JSBSim models are found by their directory
//! layout, and the environments new scenarios are authored in.

mod common;

use std::path::{Path, PathBuf};

use astrum_sim::catalog::{
    environment, environments, jsbsim_models, list, summary, CatalogEntry, FileKind,
};
use astrum_sim::scenario::{Scenario, World, KNOWN_MODELS};
use astrum_sim::sweep::{Purpose, Sweep, SweepFile};

fn write(root: &Path, rel: &str, text: &str) -> PathBuf {
    let path = root.join(rel);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, text).unwrap();
    path
}

#[test]
fn files_are_classified_by_their_keys_and_sorted_by_kind_then_path() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let b = write(root, "b.toml", "# Scenario B. More.\n[[aircraft]]\nid = 1\n");
    let a = write(root, "nested/deep/a.toml", "[[aircraft]]\nid = 2\n");
    let run = write(root, "run.toml", "# Run.\nbase = \"b.toml\"\n");
    let single = write(root, "single.toml", "base = \"b.toml\"\nrepeats = 1\n[vary]\n");
    let grid = write(root, "grid.toml", "base = \"b.toml\"\n[vary]\n\"sim.seed\" = [1, 2]\n");
    let repeats = write(root, "repeats.toml", "base = \"b.toml\"\nrepeats = 3\n");
    let broken = write(root, "broken.toml", "# Broken. Really.\nbase = \n");
    write(root, "other.toml", "[package]\nname = \"x\"\n");
    write(root, "notes.txt", "[[aircraft]]\n");

    let entries = list(root).unwrap();
    let got: Vec<(FileKind, &Path)> = entries.iter().map(|e| (e.kind, e.path.as_path())).collect();
    assert_eq!(
        got,
        [
            (FileKind::Run, run.as_path()),
            (FileKind::Run, single.as_path()),
            (FileKind::Scenario, b.as_path()),
            (FileKind::Scenario, a.as_path()),
            (FileKind::Sweep, grid.as_path()),
            (FileKind::Sweep, repeats.as_path()),
            (FileKind::Invalid, broken.as_path()),
        ]
    );
    let entry = |p: &Path| entries.iter().find(|e| e.path == p).unwrap();
    assert_eq!(
        *entry(&b),
        CatalogEntry {
            path: b.clone(),
            kind: FileKind::Scenario,
            summary: "Scenario B.".into(),
            error: None
        }
    );
    assert_eq!(entry(&a).summary, "");
    let broken = entry(&broken);
    assert!(broken.error.as_deref().is_some_and(|e| !e.is_empty()), "{broken:?}");
    assert_eq!(broken.summary, "Broken.");
}

#[test]
fn listing_a_missing_directory_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    assert!(list(&dir.path().join("missing")).is_err());
}

#[test]
fn the_summary_is_the_first_sentence_of_the_leading_comment_block() {
    assert_eq!(summary("# One. Two.\nx = 1\n"), "One.");
    assert_eq!(
        summary("\n\n#\n#   Spread over\n# two lines. Then more.\n"),
        "Spread over two lines."
    );
    assert_eq!(summary("# Is it? Yes.\n"), "Is it?");
    assert_eq!(summary("# Stop! Now.\n"), "Stop!");
    assert_eq!(summary("# Version 1.2 is out\n"), "Version 1.2 is out");
    assert_eq!(summary("# Ends at the line.\n"), "Ends at the line.");
    assert_eq!(summary("# First block\n#\n# Second block.\n"), "First block");
    assert_eq!(summary("# First block\n\n# Second block.\n"), "First block");
    assert_eq!(summary("x = 1\n# Late comment.\n"), "");
    assert_eq!(summary(""), "");
}

#[test]
fn jsbsim_models_are_aircraft_directories_holding_their_own_xml() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    write(root, "aircraft/f16/f16.xml", "<fdm_config/>");
    write(root, "aircraft/c172x/c172x.xml", "<fdm_config/>");
    write(root, "aircraft/c172x/reset00.xml", "<initialize/>");
    write(root, "aircraft/empty/readme.txt", "");
    write(root, "aircraft/odd/other.xml", "<fdm_config/>");
    write(root, "aircraft/loose.xml", "<fdm_config/>");
    write(root, "engine/f16/f16.xml", "<fdm_config/>");
    assert_eq!(jsbsim_models(root).unwrap(), ["c172x", "f16"]);
    assert!(jsbsim_models(&root.join("aircraft")).is_err(), "no aircraft/ under it");
}

#[test]
fn there_is_exactly_one_environment_the_empty_world() {
    let all = environments();
    assert_eq!(all.iter().map(|e| e.id).collect::<Vec<_>>(), ["empty"]);
    assert_eq!(environment("empty"), Some(all[0].clone()));
    assert_eq!(environment("dogfight"), None);
    let empty = &all[0];
    assert_eq!(empty.name, "Empty world");
    let known: Vec<&str> = KNOWN_MODELS.iter().map(|(m, ..)| *m).collect();
    assert_eq!(empty.models, known);
    assert_eq!(
        empty.metrics,
        [
            "end_reason",
            "sim_time_s",
            "min_separation_m",
            "min_alt_hae_m",
            "max_bank_deg",
            "max_nz_g",
            "failsafe_entries",
            "controller_losses",
            "command_refusals",
            "hits",
            "kills",
        ]
    );
}

#[test]
fn every_environment_metric_is_a_field_of_the_run_metrics() {
    // The ids an environment advertises name `astrum_bag::metrics::RunMetrics`
    // fields; the web reads them out of that JSON by these names.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty.mcap");
    astrum_bag::BagWriter::create(&path, None).unwrap().finish().unwrap();
    let bag = astrum_bag::BagReader::open(&path).unwrap();
    let json = serde_json::to_value(astrum_bag::metrics::compute(&bag).unwrap()).unwrap();
    let fields = json.as_object().unwrap();
    for env in environments() {
        for id in &env.metrics {
            assert!(fields.contains_key(*id), "environment {}: metric {id} is not a field", env.id);
        }
    }
}

#[test]
fn the_empty_world_offers_every_bundled_scenario_origin_once() {
    let config = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../config");
    let empty = environment("empty").unwrap();
    let mut origins = 0;
    for entry in std::fs::read_dir(&config).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        if !(name.starts_with("scenario.") && name.ends_with(".toml")) {
            continue;
        }
        origins += 1;
        let w = Scenario::load(&path).unwrap().world;
        let at = |l: &&astrum_sim::catalog::Location| {
            (l.origin_lat_deg, l.origin_lon_deg, l.origin_alt_hae_m)
                == (w.origin_lat_deg, w.origin_lon_deg, w.origin_alt_hae_m)
        };
        assert!(empty.locations.iter().any(|l| at(&l)), "{name}'s origin is not offered");
    }
    assert!(origins > 0, "no bundled scenarios under {}", config.display());
    for (i, a) in empty.locations.iter().enumerate() {
        assert!(!a.name.is_empty());
        for b in &empty.locations[i + 1..] {
            assert_ne!(
                (a.origin_lat_deg, a.origin_lon_deg, a.origin_alt_hae_m),
                (b.origin_lat_deg, b.origin_lon_deg, b.origin_alt_hae_m),
                "{} and {} are the same place",
                a.name,
                b.name
            );
        }
    }
}

#[test]
fn a_new_scenario_with_the_environment_defaults_is_valid_and_can_be_swept() {
    let empty = environment("empty").unwrap();
    let place = &empty.locations[0];
    let mut scenario = common::single_ship();
    scenario.world = World {
        origin_lat_deg: place.origin_lat_deg,
        origin_lon_deg: place.origin_lon_deg,
        origin_alt_hae_m: place.origin_alt_hae_m,
    };
    scenario.sim = empty.sim;
    scenario.validate().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let base = write(dir.path(), "scenario.new.toml", &scenario.to_toml());
    let mut file = SweepFile::for_scenario(base);
    file.repeats = 2;
    Sweep::plan(file, "new", Purpose::Sweep).unwrap();
}
