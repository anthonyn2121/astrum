//! `ControlRequest::validate`: the value limits every control path (the sim's
//! queryable, the web API) shares.

use astrum_net::{speed_ok, ControlRequest, MIN_SPEED};

#[test]
fn requests_within_the_limits_pass_unchanged() {
    for ok in [
        ControlRequest::Pause,
        ControlRequest::Resume,
        ControlRequest::Step(1),
        ControlRequest::Step(u64::MAX),
        ControlRequest::Speed(0.0),
        ControlRequest::Speed(MIN_SPEED),
        ControlRequest::Speed(1e6),
    ] {
        assert_eq!(ok.validate(MIN_SPEED), Ok(ok));
    }
}

#[test]
fn zero_steps_and_out_of_range_speeds_are_refused_with_the_value() {
    assert_eq!(
        ControlRequest::Step(0).validate(MIN_SPEED),
        Err("step needs ticks >= 1, got 0".to_owned())
    );
    for bad in [MIN_SPEED / 2.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let err = ControlRequest::Speed(bad).validate(MIN_SPEED).unwrap_err();
        assert!(err.contains(&format!("got {bad}")) && err.contains(">= 0.001"), "{err}");
    }
    // The limit is the caller's.
    assert!(ControlRequest::Speed(2.0).validate(10.0).is_err());
    assert!(ControlRequest::Speed(10.0).validate(10.0).is_ok());
}

#[test]
fn the_speed_rule_is_zero_or_finite_at_or_above_the_limit() {
    for ok in [0.0, -0.0, MIN_SPEED, 1.0, 1e9] {
        assert!(speed_ok(ok, MIN_SPEED), "{ok}");
    }
    for bad in [MIN_SPEED * 0.999, -MIN_SPEED, f64::NAN, f64::INFINITY] {
        assert!(!speed_ok(bad, MIN_SPEED), "{bad}");
    }
}
