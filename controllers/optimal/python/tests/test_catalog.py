"""config/controllers.toml against the real programs: every Python entry's default
command line parses through the program's own argument parser, selects the law it
names, and declares the vehicle kinds that law flies (a renamed flag fails here)."""

import importlib.util
import os
import tomllib

import pytest
from astrum_client.runtime import _parse
from astrum_client.wire import VehicleKind

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")


def _laws(script):
    name = os.path.splitext(os.path.basename(script))[0]
    spec = importlib.util.spec_from_file_location(f"catalog_{name}", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LAWS


PROGRAMS = {
    script: _laws(os.path.join(ROOT, script))
    for script in ("controllers/python/example.py", "controllers/optimal/python/run.py")
}

with open(os.path.join(ROOT, "config", "controllers.toml"), "rb") as f:
    ENTRIES = [e for e in tomllib.load(f)["controller"] if e["id"].startswith("py-")]


def test_every_python_program_has_entries():
    assert {e["command"][1] for e in ENTRIES} == set(PROGRAMS)


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda e: e["id"])
def test_default_argv_parses_and_kinds_match(monkeypatch, entry):
    for key in [k for k in os.environ if k.startswith("ASTRUM_")]:
        monkeypatch.delenv(key)
    interpreter, script, *fixed = entry["command"]
    assert interpreter == "python3"
    # Aircraft 2: the formation entry's default leader is aircraft 1.
    argv = [*fixed, entry["id_flag"], "2"]
    for p in entry.get("param", []):
        argv += [p["flag"], str(p["default"])]
    laws = PROGRAMS[script]
    law, args = _parse(laws, argv, script, "")
    assert type(law) is laws[args.law]
    assert fixed == ["--law", args.law]
    kinds = {VehicleKind[k.upper().replace("-", "_")] for k in entry["kinds"]}
    assert law.vehicle_kinds == kinds
    assert law.needs_fleet == entry["needs_fleet"]
