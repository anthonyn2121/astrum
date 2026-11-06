"""run.py's law table: every law flies fixed-wing only, the table is valid, and
each law's flags parse through the shared runtime."""

import importlib.util
import os

import pytest
from astrum_client.runtime import check_laws, run_main
from astrum_client.wire import VehicleKind

RUN_PY = os.path.join(os.path.dirname(__file__), "..", "run.py")


def _load_run():
    spec = importlib.util.spec_from_file_location("astrum_optimal_run", RUN_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LAWS = _load_run().LAWS


def test_the_table_lists_every_law_and_each_is_fixed_wing_only():
    assert list(LAWS) == ["lqr", "mpc", "figure8", "formation"]
    check_laws(LAWS)
    for law in LAWS.values():
        assert law.vehicle_kinds == {VehicleKind.FIXED_WING}
    assert [name for name, law in LAWS.items() if law.needs_fleet] == ["formation"]


@pytest.mark.parametrize(
    ("law", "flag"),
    [
        ("lqr", "--max-bank-deg"),
        ("mpc", "--horizon-s"),
        ("figure8", "--fig-rx"),
        ("formation", "--leader"),
    ],
)
def test_help_shows_the_laws_flags_and_no_rate(capsys, law, flag):
    assert run_main(LAWS, ["--law", law, "--help"], prog="run.py", description="") == 0
    out = capsys.readouterr().out
    assert flag in out and "--rate-hz" not in out


@pytest.mark.parametrize(
    "argv",
    [
        ("--law", "pid", "--id", "1"),
        ("--law", "formation", "--id", "1", "--leader", "1"),
        ("--law", "formation", "--id", "2", "--ap-heading-gain", "0"),
        ("--law", "mpc", "--id", "1", "--horizon-s", "nan"),
        ("--law", "lqr", "--id", "1", "--rate-hz", "30"),
    ],
)
def test_usage_errors_exit_2(argv):
    assert run_main(LAWS, list(argv), prog="run.py", description="") == 2
