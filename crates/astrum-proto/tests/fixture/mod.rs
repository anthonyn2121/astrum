//! JSON → plain-data conversion for the cross-language wire fixtures
//! (`tests/fixtures/values.json`). Shared by the fixture test and the
//! `wire_fixture` example that writes the Rust-encoded `.bin` files.

use std::path::PathBuf;

use astrum_proto::{
    Aero, AutopilotTargets, BodyRates, CombatStatus, Command, Control, ControlMode, Enu, Euler,
    Ned, RawActuators, Telemetry, VehicleKind,
};
use serde_json::Value;

pub const KINDS: [&str; 3] = ["telemetry", "command_raw", "command_autopilot"];
pub const LANGS: [&str; 3] = ["rust", "cpp", "python"];

pub fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

pub fn values() -> Value {
    let text = std::fs::read_to_string(fixtures_dir().join("values.json")).expect("values.json");
    serde_json::from_str(&text).expect("values.json parses")
}

fn f(v: &Value, k: &str) -> f64 {
    v[k].as_f64().unwrap_or_else(|| panic!("{k} must be a number"))
}

fn u(v: &Value, k: &str) -> u64 {
    v[k].as_u64().unwrap_or_else(|| panic!("{k} must be an unsigned integer"))
}

fn enu(v: &Value) -> Enu {
    Enu { e: f(v, "e"), n: f(v, "n"), u: f(v, "u") }
}

pub fn telemetry(v: &Value) -> Telemetry {
    Telemetry {
        aircraft_id: u(v, "aircraft_id") as u32,
        tick: u(v, "tick"),
        sim_time_s: f(v, "sim_time_s"),
        lat_deg: f(v, "lat_deg"),
        lon_deg: f(v, "lon_deg"),
        alt_hae_m: f(v, "alt_hae_m"),
        pos_enu: enu(&v["pos_enu"]),
        vel_enu: enu(&v["vel_enu"]),
        vel_ned: Ned {
            n: f(&v["vel_ned"], "n"),
            e: f(&v["vel_ned"], "e"),
            d: f(&v["vel_ned"], "d"),
        },
        attitude: Euler {
            phi_rad: f(&v["attitude"], "phi_rad"),
            theta_rad: f(&v["attitude"], "theta_rad"),
            psi_rad: f(&v["attitude"], "psi_rad"),
        },
        rates: BodyRates {
            p_rad_s: f(&v["rates"], "p_rad_s"),
            q_rad_s: f(&v["rates"], "q_rad_s"),
            r_rad_s: f(&v["rates"], "r_rad_s"),
        },
        aero: Aero {
            alpha_rad: f(&v["aero"], "alpha_rad"),
            beta_rad: f(&v["aero"], "beta_rad"),
            nz_g: f(&v["aero"], "nz_g"),
            mach: f(&v["aero"], "mach"),
        },
        tas_mps: f(v, "tas_mps"),
        fault_flags: u(v, "fault_flags") as u8,
        vehicle_kind: match v["vehicle_kind"].as_str() {
            Some("fixed-wing") => VehicleKind::FixedWing,
            Some("multirotor") => VehicleKind::Multirotor,
            other => panic!("bad vehicle_kind {other:?}"),
        },
        control_mode: match v["control_mode"].as_str() {
            Some("uncommanded") => ControlMode::Uncommanded,
            Some("raw") => ControlMode::Raw,
            Some("autopilot") => ControlMode::Autopilot,
            Some("failsafe") => ControlMode::Failsafe,
            other => panic!("bad control_mode {other:?}"),
        },
        combat: {
            let c = &v["combat"];
            CombatStatus {
                team: u(c, "team") as u8,
                hit_points: f(c, "hit_points") as f32,
                hit_points_max: f(c, "hit_points_max") as f32,
                alive: c["alive"].as_bool().expect("alive"),
                ammo: u(c, "ammo") as u16,
            }
        },
    }
}

pub fn command(v: &Value) -> Command {
    let ctrl = &v["control"];
    let control = if let Some(r) = ctrl.get("raw") {
        Control::Raw(RawActuators {
            aileron: f(r, "aileron"),
            elevator: f(r, "elevator"),
            rudder: f(r, "rudder"),
            throttle: f(r, "throttle"),
        })
    } else {
        let a = &ctrl["autopilot"];
        Control::Autopilot(AutopilotTargets {
            heading_rad: f(a, "heading_rad"),
            alt_hae_m: f(a, "alt_hae_m"),
            tas_mps: f(a, "tas_mps"),
        })
    };
    Command {
        target_id: u(v, "target_id") as u32,
        seq: u(v, "seq"),
        telemetry_tick: u(v, "telemetry_tick"),
        trigger: v["trigger"].as_bool().expect("trigger"),
        control,
    }
}
