//! Command semantics shared by the network bridge and the env, and the
//! autopilot's anti-windup.

use astrum_fdm::{Controls, FdmState};
use astrum_proto::{AutopilotTargets, Command, Control, RawActuators};
use astrum_sim::combat::Trigger;
use astrum_sim::components::AutopilotProfile;
use astrum_sim::pilot::{
    apply_command, fixed_wing, FixedWingGains, Integrators, Pilot, PilotMode, Refusal,
};

fn cmd(seq: u64, telemetry_tick: u64, control: Control) -> Command {
    Command { target_id: 1, seq, telemetry_tick, trigger: true, control }
}

fn raw(a: f64) -> Control {
    Control::Raw(RawActuators { aileron: a, elevator: f64::NAN, rudder: -3.0, throttle: 2.0 })
}

#[test]
fn raw_commands_are_clamped_and_set_the_trigger() {
    let mut p = Pilot::new(false);
    let mut t = Trigger(false);
    apply_command(&mut p, Some(&mut t), &cmd(1, 10, raw(0.4)), 10, Some(60)).unwrap();
    assert_eq!(
        p.mode,
        PilotMode::Raw(Controls { aileron: 0.4, elevator: 0.0, rudder: -1.0, throttle: 1.0 })
    );
    assert!(t.0);
    assert_eq!(p.last_command_tick, 10);
}

#[test]
fn stale_future_duplicate_and_non_finite_commands_are_refused() {
    let mut p = Pilot::new(false);
    let lease = Some(60);
    assert_eq!(
        apply_command(&mut p, None, &cmd(1, 11, raw(0.0)), 10, lease),
        Err(Refusal::FromTheFuture { telemetry_tick: 11, now: 10 })
    );
    assert_eq!(
        apply_command(&mut p, None, &cmd(1, 39, raw(0.0)), 100, lease),
        Err(Refusal::Stale { telemetry_tick: 39, now: 100 })
    );
    apply_command(&mut p, None, &cmd(5, 99, raw(0.0)), 100, lease).unwrap();
    assert_eq!(
        apply_command(&mut p, None, &cmd(5, 100, raw(0.1)), 100, lease),
        Err(Refusal::Duplicate { seq: 5 })
    );
    let bad = Control::Autopilot(AutopilotTargets {
        heading_rad: f64::NAN,
        alt_hae_m: 1.0,
        tas_mps: 1.0,
    });
    assert_eq!(apply_command(&mut p, None, &cmd(6, 100, bad), 100, lease), Err(Refusal::NonFinite));
    // A restarted sender (seq going backwards) is accepted.
    apply_command(&mut p, None, &cmd(1, 100, raw(0.2)), 100, lease).unwrap();
}

#[test]
fn saturated_altitude_loop_does_not_wind_up() {
    let g = FixedWingGains::for_profile(AutopilotProfile::Trainer);
    let mut i = Integrators::default();
    let low = FdmState { alt_hae_m: 0.0, tas_mps: 55.0, ..Default::default() };
    let far_above = AutopilotTargets { heading_rad: 0.0, alt_hae_m: 5000.0, tas_mps: 55.0 };
    // A long climb with the pitch command pinned at its limit.
    for _ in 0..120 * 600 {
        fixed_wing(&low, &far_above, &g, &mut i, 1.0 / 120.0);
    }
    assert_eq!(i.alt, 0.0, "no integration while saturated in the error's direction");
    // Arriving at the target: the pitch command responds immediately (no stored windup).
    let at = FdmState { alt_hae_m: 5000.0, tas_mps: 55.0, ..Default::default() };
    let c = fixed_wing(&at, &far_above, &g, &mut i, 1.0 / 120.0);
    assert!(
        c.elevator.abs() < 1e-9,
        "level pitch command at the target altitude, got {}",
        c.elevator
    );
}
