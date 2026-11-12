//! The studio's trim check of one aircraft: a class that trims must trim at its
//! initial conditions (the error names the aircraft); a class that does not
//! trim passes without being built.

use astrum_sim::scenario::Scenario;
use astrum_sim::trim::check_aircraft;

const WORLD: &str = "[world]\norigin_lat_deg = 37.0\norigin_lon_deg = -122.0\n\
                     origin_alt_hae_m = 0.0\n[sim]\nrate_hz = 120.0\n";

const C172X_CRUISE: &str = "[[aircraft]]\nid = 1\nmodel = \"c172x\"\nlat_deg = 37.0\n\
                            lon_deg = -122.0\nalt_hae_m = 1500.0\nheading_deg = 0.0\n\
                            tas_mps = 55.0\n";

const F16_CRUISE: &str = "[[aircraft]]\nid = 2\nmodel = \"f16\"\nlat_deg = 37.0\n\
                          lon_deg = -122.0\nalt_hae_m = 3000.0\nheading_deg = 90.0\n\
                          tas_mps = 200.0\n";

const F16_BANK_60: &str = "[[aircraft]]\nid = 2\nmodel = \"f16\"\nlat_deg = 37.0\n\
                           lon_deg = -122.0\nalt_hae_m = 3000.0\nheading_deg = 90.0\n\
                           tas_mps = 200.0\nbank_deg = 60.0\n";

const C172X_UNTRIMMED_BANK_80: &str = "[[aircraft]]\nid = 1\nmodel = \"c172x\"\nlat_deg = 37.0\n\
                                       lon_deg = -122.0\nalt_hae_m = 1500.0\nheading_deg = 0.0\n\
                                       tas_mps = 55.0\ntrim = false\nbank_deg = 80.0\n";

const QUAD: &str = "[[aircraft]]\nid = 7\nmodel = \"F450\"\nlat_deg = 37.0\nlon_deg = -122.0\n\
                    alt_hae_m = 50.0\nheading_deg = 0.0\ntas_mps = 0.0\n";

/// The scenario made of `WORLD` and the given aircraft tables.
fn scenario(aircraft: &str) -> Scenario {
    Scenario::from_toml_str(&format!("{WORLD}{aircraft}")).unwrap()
}

#[test]
fn trimming_aircraft_at_level_cruise_pass() {
    for table in [C172X_CRUISE, F16_CRUISE] {
        let s = scenario(table);
        check_aircraft(&s, &s.aircraft[0]).unwrap();
    }
}

#[test]
fn a_fixed_wing_that_cannot_trim_fails_naming_the_aircraft() {
    let s = scenario(F16_BANK_60);
    let e = check_aircraft(&s, &s.aircraft[0]).unwrap_err();
    assert!(e.starts_with("aircraft 2 (f16): "), "{e}");
    assert!(e.to_lowercase().contains("trim"), "{e}");
}

#[test]
fn an_aircraft_that_does_not_trim_is_not_built() {
    let s = scenario(C172X_UNTRIMMED_BANK_80);
    check_aircraft(&s, &s.aircraft[0]).unwrap();

    let s = scenario(QUAD);
    check_aircraft(&s, &s.aircraft[0]).unwrap();
}
