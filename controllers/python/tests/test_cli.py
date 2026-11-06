"""CLI precedence and validation: an ``ASTRUM_*`` environment value is checked
exactly like the flag it backs; numbers must be finite and in range; ``--id``
is required."""

import argparse

import pytest
from astrum_client import cli


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in (
        "ASTRUM_TARGET_ID",
        "ASTRUM_TARGET_SPD_MPS",
        "ASTRUM_TARGET_ALT_M",
        "ASTRUM_STARTUP_TIMEOUT_S",
        "ASTRUM_MAX_BANK_DEG",
    ):
        monkeypatch.delenv(key, raising=False)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    cli.add_connection_args(p)
    cli.add_reference_args(p)
    cli.add_limit_args(p)
    return p


def parse(*argv):
    return parser().parse_args(list(argv))


def test_zero_airspeed_is_accepted_from_flag_and_env(monkeypatch):
    assert parse("--id", "1", "--speed", "0").speed == 0.0
    monkeypatch.setenv("ASTRUM_TARGET_SPD_MPS", "0")
    assert parse("--id", "1").speed == 0.0


def test_env_values_are_validated_like_flags(monkeypatch):
    monkeypatch.setenv("ASTRUM_TARGET_SPD_MPS", "-5")
    with pytest.raises(SystemExit):
        parse("--id", "1")
    monkeypatch.setenv("ASTRUM_TARGET_SPD_MPS", "50")
    monkeypatch.setenv("ASTRUM_STARTUP_TIMEOUT_S", "0")
    with pytest.raises(SystemExit):
        parse("--id", "1")


def test_flag_overrides_env(monkeypatch):
    monkeypatch.setenv("ASTRUM_TARGET_ALT_M", "130")
    assert parse("--id", "1").alt == 130.0
    assert parse("--id", "1", "--alt", "200").alt == 200.0


def test_id_is_required_unless_set_in_env(monkeypatch):
    with pytest.raises(SystemExit) as e:
        parse()
    assert e.value.code == 2
    monkeypatch.setenv("ASTRUM_TARGET_ID", "4294967295")
    assert parse().id == 4294967295


@pytest.mark.parametrize("value", ["0", "-1", "4294967296", "1.5", "one"])
def test_id_out_of_range_is_a_usage_error(value):
    with pytest.raises(SystemExit) as e:
        parse("--id", value)
    assert e.value.code == 2


@pytest.mark.parametrize(
    "argv",
    [
        ("--heading", "nan"),
        ("--alt", "inf"),
        ("--speed", "-inf"),
        ("--telemetry-timeout", "nan"),
        ("--max-bank-deg", "120"),
        ("--max-bank-deg", "90"),
        ("--max-pitch-deg", "0"),
        ("--max-pitch-deg", "nan"),
    ],
)
def test_non_finite_or_out_of_range_numbers_are_usage_errors(argv):
    with pytest.raises(SystemExit) as e:
        parse("--id", "1", *argv)
    assert e.value.code == 2


def test_attitude_limits_inside_the_open_range_are_accepted():
    args = parse("--id", "1", "--max-bank-deg", "60", "--max-pitch-deg", "89.5")
    assert (args.max_bank_deg, args.max_pitch_deg) == (60.0, 89.5)
