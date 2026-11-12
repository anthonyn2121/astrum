//! Command semantics shared by the network bridge and the env (ownership,
//! ordering, refusal bookkeeping), and the autopilot's anti-windup.

use astrum_fdm::{Controls, FdmState};
use astrum_proto::{
    AutopilotTargets, Command, CommandStatus, Control, ModeReason, MultirotorRaw, RawActuators,
};
use astrum_sim::combat::Trigger;
use astrum_sim::components::{AircraftState, AutopilotProfile, VehicleKind};
use astrum_sim::pilot::{
    apply_command, enter_failsafe, fixed_wing, FixedWingGains, Integrators, Pilot, PilotMode,
    Refusal,
};

const A: u64 = 0xA;
const B: u64 = 0xB;
const FW: VehicleKind = VehicleKind::FixedWing;

fn cmd(epoch: u64, seq: u64, telemetry_tick: u64, control: Control) -> Command {
    Command { target_id: 1, seq, telemetry_tick, trigger: true, epoch, control }
}

fn raw(a: f64) -> Control {
    Control::Raw(RawActuators { aileron: a, elevator: f64::NAN, rudder: -3.0, throttle: 2.0 })
}

fn multirotor() -> Control {
    Control::Multirotor(MultirotorRaw {
        roll_rate: 0.25,
        pitch_rate: -2.0,
        yaw_rate: f64::NAN,
        collective: 0.6,
    })
}

#[test]
fn raw_commands_are_clamped_set_the_trigger_and_claim_the_aircraft() {
    let mut p = Pilot::new(false);
    let mut t = Trigger(false);
    apply_command(&mut p, Some(&mut t), FW, &cmd(A, 7, 10, raw(0.4)), 10, Some(60)).unwrap();
    assert_eq!(
        p.mode,
        PilotMode::Raw(Controls { aileron: 0.4, elevator: 0.0, rudder: -1.0, throttle: 1.0 })
    );
    assert!(t.0);
    assert_eq!(p.last_command_tick, 10);
    assert_eq!(
        p.command_status(),
        CommandStatus { owner_epoch: A, last_seq: 7, ..Default::default() }
    );
}

#[test]
fn checks_refuse_in_order_and_are_counted() {
    let mut p = Pilot::new(false);
    let lease = Some(60);
    let apply = |p: &mut Pilot, c: Command, now| apply_command(p, None, FW, &c, now, lease);
    assert_eq!(apply(&mut p, cmd(0, 1, 10, raw(0.0)), 10), Err(Refusal::MissingEpoch));
    assert_eq!(
        apply(&mut p, cmd(A, 1, 11, raw(0.0)), 10),
        Err(Refusal::FromTheFuture { telemetry_tick: 11, now: 10 })
    );
    assert_eq!(
        apply(&mut p, cmd(A, 1, 39, raw(0.0)), 100),
        Err(Refusal::Stale { telemetry_tick: 39, now: 100 })
    );
    assert_eq!(p.owner, None, "a refused command claims nothing");
    apply(&mut p, cmd(A, 5, 99, raw(0.0)), 100).unwrap();
    // Another epoch is refused before anything else is looked at.
    assert_eq!(
        apply(&mut p, cmd(B, 1, 500, raw(0.0)), 100),
        Err(Refusal::NotOwner { epoch: B, owner: A })
    );
    assert_eq!(
        apply(&mut p, cmd(A, 5, 100, raw(0.1)), 100),
        Err(Refusal::OutOfOrder { seq: 5, last_seq: 5 })
    );
    assert_eq!(
        apply(&mut p, cmd(A, 4, 100, raw(0.1)), 100),
        Err(Refusal::OutOfOrder { seq: 4, last_seq: 5 }),
        "seq going backwards within the owner's epoch"
    );
    let bad = Control::Autopilot(AutopilotTargets {
        heading_rad: f64::NAN,
        alt_hae_m: 1.0,
        tas_mps: 1.0,
    });
    assert_eq!(apply(&mut p, cmd(A, 6, 100, bad), 100), Err(Refusal::NonFinite));
    assert_eq!(
        p.command_status(),
        CommandStatus {
            owner_epoch: A,
            last_seq: 5,
            refusals: 7,
            last_refusal: astrum_proto::Refusal::NonFinite,
            last_refused_epoch: A,
        }
    );
    apply(&mut p, cmd(A, 6, 100, raw(0.2)), 100).unwrap();
    assert_eq!(p.last_seq, 6);
}

#[test]
fn a_new_epoch_claims_the_aircraft_only_after_the_owner_lets_go() {
    let mut p = Pilot::new(false);
    let lease = Some(60);
    apply_command(&mut p, None, FW, &cmd(A, 50, 10, raw(0.1)), 10, lease).unwrap();
    // A restarted controller (new epoch, seq from 1) is refused while A owns it.
    assert_eq!(
        apply_command(&mut p, None, FW, &cmd(B, 1, 10, raw(0.2)), 10, lease),
        Err(Refusal::NotOwner { epoch: B, owner: A })
    );
    assert_eq!(p.command_status().last_refused_epoch, B);
    enter_failsafe(&mut p, None, &AircraftState::default(), ModeReason::LeaseExpired);
    assert_eq!((p.owner, p.last_seq), (None, 0));
    apply_command(&mut p, None, FW, &cmd(B, 1, 80, raw(0.2)), 80, lease).unwrap();
    assert_eq!((p.owner, p.last_seq), (Some(B), 1));
    assert_eq!(
        apply_command(&mut p, None, FW, &cmd(A, 51, 80, raw(0.1)), 80, lease),
        Err(Refusal::NotOwner { epoch: A, owner: B })
    );
}

#[test]
fn multirotor_demands_fly_a_multirotor_and_raw_does_not() {
    let mut p = Pilot::new(false);
    let mr = VehicleKind::Multirotor;
    assert_eq!(
        apply_command(&mut p, None, mr, &cmd(A, 1, 0, raw(0.0)), 0, None),
        Err(Refusal::WrongVehicleKind { kind: mr })
    );
    assert_eq!(p.last_refusal, astrum_proto::Refusal::WrongVehicleKind);
    apply_command(&mut p, None, mr, &cmd(A, 2, 0, multirotor()), 0, None).unwrap();
    assert_eq!(
        p.mode,
        PilotMode::Raw(Controls { aileron: 0.25, elevator: -1.0, rudder: 0.0, throttle: 0.6 })
    );
    let mut fw = Pilot::new(false);
    assert_eq!(
        apply_command(&mut fw, None, FW, &cmd(A, 1, 0, multirotor()), 0, None),
        Err(Refusal::WrongVehicleKind { kind: FW })
    );
    assert_eq!(fw.owner, None);
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
