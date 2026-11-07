//! The controller catalog: load-time validation of `config/controllers.toml` and
//! rendering entries into `controllers` command lines.

use std::collections::BTreeMap;

use astrum_sim::components::VehicleKind;
use astrum_sim::controllers::{ControllerCatalog, ControllerError, ParamTarget};

fn shipped() -> ControllerCatalog {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../config/controllers.toml");
    ControllerCatalog::load(path).expect("shipped catalog loads")
}

fn split(line: &str) -> Vec<String> {
    shlex::split(line).expect("rendered line splits")
}

fn none() -> BTreeMap<String, toml::Value> {
    BTreeMap::new()
}

const MINI: &str = r#"
[[controller]]
id = "mini"
name = "Mini"
description = "test entry"
command = ["prog", "--law", "x"]
id_flag = "--id"
kinds = ["multirotor"]

[[controller.param]]
name = "alt"
flag = "--alt"
label = "Altitude"
type = "number"
default = 100.0
min = 0.0
max = 1000.0

[[controller.param]]
name = "steps"
flag = "--steps"
label = "Steps"
type = "integer"
default = 3
min = 1

[[controller.param]]
name = "slot"
flag = "--slot"
label = "Slot"
type = "choice"
choices = ["left", "far right", "it's"]
default = "left"

[[controller.param]]
name = "leader"
flag = "--leader"
label = "Leader"
type = "aircraft"
"#;

fn mini() -> ControllerCatalog {
    ControllerCatalog::from_toml_str(MINI).unwrap()
}

fn with(pairs: &[(&str, toml::Value)]) -> BTreeMap<String, toml::Value> {
    pairs.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
}

#[test]
fn shipped_catalog_has_every_program_law() {
    let cat = shipped();
    let ids: Vec<&str> = cat.entries().iter().map(|e| e.id.as_str()).collect();
    assert_eq!(
        ids,
        [
            "cpp-hold",
            "cpp-autopilot",
            "py-hold",
            "py-autopilot",
            "cpp-lqr",
            "cpp-mpc",
            "py-lqr",
            "py-mpc",
            "py-figure8",
            "py-formation"
        ]
    );
    assert!(cat.get("py-formation").unwrap().needs_fleet);
    assert!(cat.get("nope").is_none());
}

#[test]
fn every_shipped_entry_renders_its_defaults() {
    let cat = shipped();
    for e in cat.entries() {
        let line = e.render(2, &none(), &[1, 2]).unwrap();
        assert!(!line.contains(['{', '}']), "{}: {line}", e.id);
        let mut want = e.command.clone();
        want.extend(["--id".to_string(), "2".to_string()]);
        for p in &e.params {
            want.push(p.flag.clone());
            want.push(match p.default.as_ref().unwrap() {
                toml::Value::Float(x) => x.to_string(),
                toml::Value::Integer(n) => n.to_string(),
                toml::Value::String(s) => s.clone(),
                v => panic!("{}: unexpected default {v}", e.id),
            });
        }
        assert_eq!(split(&line), want, "{}", e.id);
    }
}

#[test]
fn shipped_entries_render_to_the_programs_flags() {
    let cat = shipped();
    let hold = cat.get("cpp-hold").unwrap().render(1, &none(), &[1]).unwrap();
    assert_eq!(
        split(&hold),
        [
            "build/controllers/cpp/astrum-controller",
            "--law",
            "hold",
            "--id",
            "1",
            "--heading",
            "0",
            "--alt",
            "1500",
            "--speed",
            "50",
            "--max-bank-deg",
            "25",
            "--max-pitch-deg",
            "10"
        ]
    );
    let params = with(&[
        ("leader", toml::Value::Integer(1)),
        ("slot", toml::Value::String("left".into())),
        ("alt", toml::Value::Float(4572.0)),
    ]);
    let line = cat.get("py-formation").unwrap().render(3, &params, &[1, 2, 3]).unwrap();
    let argv = split(&line);
    assert_eq!(
        argv[..11],
        [
            "python3",
            "controllers/optimal/python/run.py",
            "--law",
            "formation",
            "--id",
            "3",
            "--leader",
            "1",
            "--slot",
            "left",
            "--alt"
        ]
    );
    assert_eq!(argv[11], "4572");
}

#[test]
fn shipped_kinds_and_targets() {
    let cat = shipped();
    let multirotor: Vec<&str> = cat
        .entries()
        .iter()
        .filter(|e| e.flies(VehicleKind::Multirotor))
        .map(|e| e.id.as_str())
        .collect();
    assert_eq!(multirotor, ["cpp-autopilot", "py-autopilot"]);
    assert!(cat.entries().iter().all(|e| e.flies(VehicleKind::FixedWing)));
    let hold = cat.get("py-hold").unwrap();
    let target = |n: &str| hold.params.iter().find(|p| p.name == n).unwrap().target;
    assert_eq!(target("heading"), Some(ParamTarget::HeadingDeg));
    assert_eq!(target("alt"), Some(ParamTarget::AltHaeM));
    assert_eq!(target("speed"), Some(ParamTarget::TasMps));
    assert_eq!(target("max-bank"), None);
}

#[test]
fn render_quotes_so_split_reproduces_argv() {
    let e = mini().entries()[0].clone();
    let params = with(&[
        ("slot", toml::Value::String("it's".into())),
        ("leader", toml::Value::Integer(7)),
        ("alt", toml::Value::Integer(250)),
    ]);
    let argv = split(&e.render(9, &params, &[7, 9]).unwrap());
    assert_eq!(
        argv,
        [
            "prog", "--law", "x", "--id", "9", "--alt", "250", "--steps", "3", "--slot", "it's",
            "--leader", "7"
        ]
    );
    let params = with(&[
        ("slot", toml::Value::String("far right".into())),
        ("leader", toml::Value::Integer(7)),
    ]);
    let argv = split(&e.render(9, &params, &[7, 9]).unwrap());
    assert_eq!(argv[10], "far right");
}

fn bad(params: &[(&str, toml::Value)], fleet: &[u32]) -> ControllerError {
    mini().entries()[0].render(9, &with(params), fleet).unwrap_err()
}

fn bad_value(err: ControllerError) -> String {
    match err {
        ControllerError::BadValue { param, .. } => param,
        e => panic!("expected BadValue, got {e:?}"),
    }
}

#[test]
fn render_errors_name_the_param() {
    let lead = ("leader", toml::Value::Integer(7));
    let fleet = [7, 9];
    assert_eq!(
        bad(&[lead.clone(), ("bogus", toml::Value::Integer(1))], &fleet),
        ControllerError::UnknownParam("bogus".into())
    );
    assert_eq!(bad(&[], &fleet), ControllerError::MissingParam("leader".into()));
    // Wrong type.
    let s = toml::Value::String("1".into());
    assert_eq!(bad_value(bad(&[lead.clone(), ("alt", s)], &fleet)), "alt");
    assert_eq!(
        bad_value(bad(&[lead.clone(), ("steps", toml::Value::Float(2.0))], &fleet)),
        "steps"
    );
    // Bounds are inclusive.
    let alt = |x: f64| bad(&[lead.clone(), ("alt", toml::Value::Float(x))], &fleet);
    assert_eq!(bad_value(alt(1000.5)), "alt");
    assert_eq!(bad_value(alt(-0.1)), "alt");
    assert_eq!(bad_value(alt(f64::NAN)), "alt");
    let e = mini().entries()[0].clone();
    for edge in [0.0, 1000.0] {
        let p = with(&[lead.clone(), ("alt", toml::Value::Float(edge))]);
        assert!(e.render(9, &p, &fleet).is_ok(), "{edge}");
    }
    assert_eq!(
        bad_value(bad(&[lead.clone(), ("steps", toml::Value::Integer(0))], &fleet)),
        "steps"
    );
    // Choice outside the list, including placeholder braces.
    for v in ["middle", "{id}"] {
        let p = [lead.clone(), ("slot", toml::Value::String(v.into()))];
        assert_eq!(bad_value(bad(&p, &fleet)), "slot");
    }
    // Aircraft: not in the fleet, the controlled aircraft itself, not an id.
    assert_eq!(bad_value(bad(&[("leader", toml::Value::Integer(5))], &fleet)), "leader");
    assert_eq!(bad_value(bad(&[("leader", toml::Value::Integer(9))], &fleet)), "leader");
    assert_eq!(bad_value(bad(&[("leader", toml::Value::Integer(0))], &[0, 9])), "leader");
    assert_eq!(bad_value(bad(&[("leader", toml::Value::Float(7.0))], &fleet)), "leader");
}

fn load_err(toml: &str) -> String {
    match ControllerCatalog::from_toml_str(toml).unwrap_err() {
        ControllerError::Catalog(msg) => msg,
        e => panic!("expected Catalog, got {e:?}"),
    }
}

#[test]
fn load_rejects_invalid_catalogs() {
    let dup = format!("{MINI}{}", &MINI[..MINI.find("[[controller.param]]").unwrap()]);
    assert!(load_err(&dup).contains("duplicate controller id `mini`"));
    let head = &MINI[..MINI.find("[[controller.param]]").unwrap()];
    let same_name = format!("{MINI}{}", head.replace("id = \"mini\"", "id = \"mini-2\""));
    assert!(load_err(&same_name).contains("duplicate controller name `Mini`"));

    let oob = MINI.replace("default = 100.0", "default = 1000.5");
    let msg = load_err(&oob);
    assert!(msg.contains("`mini`") && msg.contains("`alt`"), "{msg}");

    let not_choice = MINI.replace("default = \"left\"", "default = \"up\"");
    assert!(load_err(&not_choice).contains("`slot`"));

    let wrong_type = MINI.replace("default = 3", "default = \"3\"");
    assert!(load_err(&wrong_type).contains("`steps`"));

    assert!(load_err(&MINI.replace("\"multirotor\"", "\"rocket\"")).contains("rocket"));
    assert!(load_err(&MINI.replace("kinds = [\"multirotor\"]", "kinds = []")).contains("kinds"));
    assert!(load_err(&MINI.replace("[\"prog\", \"--law\", \"x\"]", "[]")).contains("command"));
    assert!(load_err(&MINI.replace("\"x\"]", "\"{x}\"]")).contains("command"));
    assert!(load_err(&MINI.replace("id = \"mini\"", "id = \"Mini\"")).contains("[a-z0-9-]+"));
    assert!(load_err(&MINI.replace("name = \"steps\"", "name = \"alt\"")).contains("`alt`"));
    assert!(load_err(&MINI.replace("flag = \"--steps\"", "flag = \"--alt\"")).contains("--alt"));
    assert!(load_err(&MINI.replace("flag = \"--steps\"", "flag = \"--id\"")).contains("--id"));
    // A param may not re-declare a fixed `-`-prefixed argument of the command.
    assert!(load_err(&MINI.replace("flag = \"--steps\"", "flag = \"--law\"")).contains("--law"));
    let fixed_eq = MINI.replace("\"--law\", \"x\"", "\"--law=x\"");
    assert!(load_err(&fixed_eq.replace("flag = \"--steps\"", "flag = \"--law\"")).contains("--law"));
    assert!(ControllerCatalog::from_toml_str(&MINI.replace("\"x\"]", "\"--steps\"]")).is_err());
    assert!(load_err(&MINI.replace("label = \"Slot\"", "label = \"Slot\"\nextra = 1"))
        .contains("extra"));
}

#[test]
fn flies_filters_by_kind() {
    let e = mini().entries()[0].clone();
    assert!(e.flies(VehicleKind::Multirotor));
    assert!(!e.flies(VehicleKind::FixedWing));
}

#[test]
fn render_unbounded_numbers_exactly_and_rejects_unprintable_ones() {
    let cat = shipped();
    let e = cat.get("cpp-autopilot").unwrap();
    let alt = |x: f64| e.render(1, &with(&[("alt", toml::Value::Float(x))]), &[1]);
    let argv = |x: f64| split(&alt(x).unwrap());
    let rest = |v: &str| {
        [
            "build/controllers/cpp/astrum-controller",
            "--law",
            "autopilot",
            "--id",
            "1",
            "--heading",
            "0",
            "--alt",
            v,
            "--speed",
            "50",
        ]
        .map(String::from)
        .to_vec()
    };
    assert_eq!(argv(-50.5), rest("-50.5"));
    assert_eq!(argv(1e300), rest(&format!("1{}", "0".repeat(300))));
    for x in [f64::INFINITY, f64::NEG_INFINITY, 1e-310, -1e-310] {
        assert_eq!(bad_value(alt(x).unwrap_err()), "alt", "{x}");
    }
    assert_eq!(argv(-0.0), rest("-0"));
}
