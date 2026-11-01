"""CLI precedence and validation: an ``ASTRUM_*`` environment value is checked
exactly like the flag it backs, and the quad demo's zero airspeed is accepted."""

import argparse

import pytest
from astrum_client import cli


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    cli.add_connection_args(p)
    cli.add_reference_args(p)
    cli.add_limit_args(p)
    return p


def test_zero_airspeed_is_accepted_from_flag_and_env(monkeypatch):
    assert parser().parse_args(["--speed", "0"]).speed == 0.0
    monkeypatch.setenv("ASTRUM_TARGET_SPD_MPS", "0")
    assert parser().parse_args([]).speed == 0.0


def test_env_values_are_validated_like_flags(monkeypatch):
    monkeypatch.setenv("ASTRUM_TARGET_SPD_MPS", "-5")
    with pytest.raises(SystemExit):
        parser().parse_args([])
    monkeypatch.setenv("ASTRUM_TARGET_SPD_MPS", "50")
    monkeypatch.setenv("ASTRUM_STARTUP_TIMEOUT_S", "0")
    with pytest.raises(SystemExit):
        parser().parse_args([])


def test_flag_overrides_env(monkeypatch):
    monkeypatch.setenv("ASTRUM_TARGET_ALT_M", "130")
    assert parser().parse_args([]).alt == 130.0
    assert parser().parse_args(["--alt", "200"]).alt == 200.0
