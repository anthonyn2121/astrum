//! Scenario validation: configuration mistakes fail loudly before anything spawns.

use astrum_sim::components::VehicleKind;
use astrum_sim::scenario::Scenario;

const BASE: &str = r#"
[world]
origin_lat_deg = 37.0
origin_lon_deg = -122.0
origin_alt_hae_m = 0.0
[sim]
rate_hz = 120.0
[[aircraft]]
id = 1
model = "c172x"
lat_deg = 37.0
lon_deg = -122.0
alt_hae_m = 1500.0
heading_deg = 0.0
tas_mps = 55.0
"#;

fn err(toml: &str) -> String {
    Scenario::from_toml_str(toml).unwrap_err().to_string()
}

#[test]
fn defaults_fill_in_and_known_models_resolve() {
    let s = Scenario::from_toml_str(BASE).unwrap();
    assert!(
        s.sim.speed == 1.0 && s.sim.command_lease_s == 0.5 && s.sim.threads == 1 && !s.sim.lockstep
    );
    let c = s.aircraft[0].class().unwrap();
    assert_eq!(c.kind, VehicleKind::FixedWing);
    assert!(c.trim, "fixed-wing aircraft trim by default");
    // The served TOML round-trips.
    assert_eq!(Scenario::from_toml_str(&s.to_toml()).unwrap(), s);
}

#[test]
fn unknown_and_legacy_keys_are_rejected() {
    assert!(err(&BASE.replace("tas_mps = 55.0", "airspeed_kts = 100.0")).contains("airspeed_kts"));
    assert!(err(&BASE.replace("rate_hz = 120.0", "rate_hz = 120.0\nmax_time = 60.0"))
        .contains("max_time"));
    // `realtime = true|false` became `speed`.
    assert!(err(&BASE.replace("rate_hz = 120.0", "rate_hz = 120.0\nrealtime = false"))
        .contains("realtime"));
}

#[test]
fn unknown_models_must_state_their_class() {
    let other = BASE.replace("\"c172x\"", "\"c310\"");
    assert!(err(&other).contains("kind"));
    let fixed = other.replace("tas_mps = 55.0", "tas_mps = 55.0\nkind = \"fixed-wing\"");
    assert!(err(&fixed).contains("autopilot_profile"));
    let ok = fixed
        .replace("kind = \"fixed-wing\"", "kind = \"fixed-wing\"\nautopilot_profile = \"trainer\"");
    assert!(Scenario::from_toml_str(&ok).is_ok());
}

#[test]
fn invalid_values_are_rejected() {
    assert!(err(&BASE.replace("rate_hz = 120.0", "rate_hz = nan")).contains("rate_hz"));
    assert!(err(&BASE.replace("tas_mps = 55.0", "tas_mps = -1.0")).contains("tas_mps"));
    let dup = format!("{BASE}{}", &BASE[BASE.find("[[aircraft]]").unwrap()..]);
    assert!(err(&dup).contains("duplicate aircraft id 1"));
    let no_fleet = &BASE[..BASE.find("[[aircraft]]").unwrap()];
    assert!(Scenario::from_toml_str(no_fleet).is_err());
    let lockstep_alone = BASE.replace("tas_mps = 55.0", "tas_mps = 55.0\nlockstep = true");
    assert!(err(&lockstep_alone).contains("sim.lockstep"));
    for bad in ["-1.0", "nan", "inf", "0.0001"] {
        let slow = BASE.replace("rate_hz = 120.0", &format!("rate_hz = 120.0\nspeed = {bad}"));
        assert!(err(&slow).contains("sim.speed"), "speed = {bad} accepted");
    }
    for ok in ["0.0", "0.001", "1000.0"] {
        let s = BASE.replace("rate_hz = 120.0", &format!("rate_hz = 120.0\nspeed = {ok}"));
        assert!(Scenario::from_toml_str(&s).is_ok(), "speed = {ok} rejected");
    }
    let control = |rate: &str| {
        BASE.replace("tas_mps = 55.0", &format!("tas_mps = 55.0\ncontrol_rate_hz = {rate}"))
    };
    for bad in ["50.0", "0.0", "-30.0", "240.0", "nan"] {
        assert!(err(&control(bad)).contains("control_rate_hz"), "control_rate_hz = {bad} accepted");
    }
    for ok in ["120.0", "60.0", "30.0", "4.0"] {
        assert!(Scenario::from_toml_str(&control(ok)).is_ok(), "control_rate_hz = {ok} rejected");
    }
    // 1 Hz control under the default 0.5 s lease would trip the failsafe between commands.
    assert!(err(&control("1.0")).contains("command_lease_s"));
    let no_lease =
        control("1.0").replace("rate_hz = 120.0", "rate_hz = 120.0\ncommand_lease_s = 0.0");
    assert!(Scenario::from_toml_str(&no_lease).is_ok());
}
