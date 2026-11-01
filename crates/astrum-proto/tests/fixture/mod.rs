//! JSON → plain-data conversion for the cross-language wire fixtures
//! (`tests/fixtures/values.json`). Shared by the fixture test and the
//! `wire_fixture` example that writes the Rust-encoded `.bin` files.

use std::path::PathBuf;

use astrum_proto::{
    Aero, AutopilotTargets, BodyRates, CombatStatus, Command, Control, ControlMode, EndReason, Enu,
    Euler, Event, EventKind, ModeReason, Ned, RawActuators, Telemetry, VehicleKind,
};
use serde_json::Value;

pub const KINDS: [&str; 3] = ["telemetry", "command_raw", "command_autopilot"];
pub const LANGS: [&str; 3] = ["rust", "cpp", "python"];
/// Event fixtures (the shared event JSON shape); C++ has no Event codec.
pub const EVENT_KINDS: [&str; 9] = [
    "event_trigger_pressed",
    "event_trigger_released",
    "event_hit",
    "event_kill",
    "event_kill_unknown",
    "event_mode_change",
    "event_controller_up",
    "event_controller_down",
    "event_run_ended",
];
pub const EVENT_LANGS: [&str; 2] = ["rust", "python"];

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

fn id(v: &Value, k: &str) -> u32 {
    u32::try_from(u(v, k)).unwrap_or_else(|_| panic!("{k} must fit in u32"))
}

fn u16_of(v: &Value, k: &str) -> u16 {
    u16::try_from(u(v, k)).unwrap_or_else(|_| panic!("{k} must fit in u16"))
}

fn s<'a>(v: &'a Value, k: &str) -> &'a str {
    v[k].as_str().unwrap_or_else(|| panic!("{k} must be a string"))
}

fn pos(v: &Value) -> Enu {
    match v["pos_enu"].as_array().map(Vec::as_slice) {
        Some([e, n, u]) => Enu {
            e: e.as_f64().expect("pos_enu e"),
            n: n.as_f64().expect("pos_enu n"),
            u: u.as_f64().expect("pos_enu u"),
        },
        _ => panic!("pos_enu must be [e, n, u]"),
    }
}

/// The variant of `all` whose `name()` is `v[k]` (names come from astrum-proto).
fn named<T: Copy>(all: &[T], name: fn(T) -> &'static str, v: &Value, k: &str) -> T {
    let want = s(v, k);
    all.iter().copied().find(|&x| name(x) == want).unwrap_or_else(|| panic!("bad {k} {want:?}"))
}

fn control_mode(v: &Value, k: &str) -> ControlMode {
    let all =
        [ControlMode::Uncommanded, ControlMode::Raw, ControlMode::Autopilot, ControlMode::Failsafe];
    named(&all, ControlMode::name, v, k)
}

fn mode_reason(v: &Value, k: &str) -> ModeReason {
    let all = [ModeReason::Command, ModeReason::LeaseExpired, ModeReason::ControllerLost];
    named(&all, ModeReason::name, v, k)
}

fn end_reason(v: &Value, k: &str) -> EndReason {
    let all =
        [EndReason::MaxTime, EndReason::EngagementDecided, EndReason::Fatal, EndReason::Stopped];
    named(&all, EndReason::name, v, k)
}

/// An event in the shared JSON shape (`{"tick", "sim_time_s", "kind", ...}`).
/// Panics unless the result's `EventKind::name()` is the JSON `kind`.
pub fn event(v: &Value) -> Event {
    let kind = match s(v, "kind") {
        "trigger-pressed" => {
            EventKind::TriggerPressed { aircraft_id: id(v, "aircraft_id"), ammo: u16_of(v, "ammo") }
        }
        "trigger-released" => EventKind::TriggerReleased {
            aircraft_id: id(v, "aircraft_id"),
            rounds_fired: u16_of(v, "rounds_fired"),
            ammo: u16_of(v, "ammo"),
        },
        "hit" => EventKind::Hit {
            shooter_id: id(v, "shooter_id"),
            target_id: id(v, "target_id"),
            damage: f(v, "damage") as f32,
            hit_points: f(v, "hit_points") as f32,
            pos_enu: pos(v),
        },
        "kill" => EventKind::Kill {
            victim_id: id(v, "victim_id"),
            killer_id: if v["killer_id"].is_null() { None } else { Some(id(v, "killer_id")) },
            pos_enu: pos(v),
        },
        "mode-change" => EventKind::ModeChange {
            aircraft_id: id(v, "aircraft_id"),
            from: control_mode(v, "from"),
            to: control_mode(v, "to"),
            reason: mode_reason(v, "reason"),
        },
        "controller-up" => EventKind::ControllerUp { aircraft_id: id(v, "aircraft_id") },
        "controller-down" => EventKind::ControllerDown { aircraft_id: id(v, "aircraft_id") },
        "run-ended" => EventKind::RunEnded {
            reason: end_reason(v, "reason"),
            message: s(v, "message").to_owned(),
        },
        other => panic!("bad event kind {other:?}"),
    };
    assert_eq!(kind.name(), s(v, "kind"), "EventKind::name() disagrees with values.json");
    Event { tick: u(v, "tick"), sim_time_s: f(v, "sim_time_s"), kind }
}
