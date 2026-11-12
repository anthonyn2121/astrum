"""Cross-language wire fixtures (``crates/astrum-proto/tests/fixtures``).

``values.json`` holds canonical values; each language writes
``<kind>.<lang>.bin`` and every language's test decodes every file present and
compares it to ``values.json`` exactly. The ``event_*`` kinds hold the shared
event JSON shape (:meth:`Event.as_dict`). Write the Python files with::

    python -m astrum_client.wire_fixture [--dir DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from astrum_client.wire import (
    AutopilotTargets,
    Command,
    CommandStatus,
    ControlMode,
    Event,
    MultirotorRaw,
    RawActuators,
    Refusal,
    TickEnd,
    VehicleKind,
    VehicleState,
    encode_command,
    encode_event,
    encode_telemetry,
    encode_tick_end,
    enum_member,
)

COMMAND_KINDS = ("command_raw", "command_autopilot", "command_multirotor")
KINDS = ("telemetry", *COMMAND_KINDS, "tick_end")
EVENT_KINDS = (
    "event_trigger_pressed",
    "event_trigger_released",
    "event_hit",
    "event_kill",
    "event_kill_unknown",
    "event_mode_change",
    "event_controller_up",
    "event_controller_down",
    "event_run_ended",
)
LANGS = ("rust", "cpp", "python")


def default_fixtures_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(
        os.path.join(here, "..", "..", "..", "crates", "astrum-proto", "tests", "fixtures")
    )


def load_values(fixtures_dir: str) -> dict:
    with open(os.path.join(fixtures_dir, "values.json"), encoding="utf-8") as f:
        return json.load(f)


def expected_telemetry(v: dict) -> VehicleState:
    pos, venu, vned = v["pos_enu"], v["vel_enu"], v["vel_ned"]
    att, rates, aero, combat = v["attitude"], v["rates"], v["aero"], v["combat"]
    cmd = v["command"]
    return VehicleState(
        aircraft_id=v["aircraft_id"],
        tick=v["tick"],
        sim_time_s=v["sim_time_s"],
        lat_deg=v["lat_deg"],
        lon_deg=v["lon_deg"],
        alt_hae_m=v["alt_hae_m"],
        e=pos["e"],
        n=pos["n"],
        u=pos["u"],
        vel_ned_n=vned["n"],
        vel_ned_e=vned["e"],
        vel_ned_d=vned["d"],
        vel_enu_e=venu["e"],
        vel_enu_n=venu["n"],
        vel_enu_u=venu["u"],
        roll=att["phi_rad"],
        pitch=att["theta_rad"],
        yaw=att["psi_rad"],
        p=rates["p_rad_s"],
        q=rates["q_rad_s"],
        r=rates["r_rad_s"],
        tas_mps=v["tas_mps"],
        alpha=aero["alpha_rad"],
        beta=aero["beta_rad"],
        nz_g=aero["nz_g"],
        mach=aero["mach"],
        fault_flags=v["fault_flags"],
        vehicle_kind=enum_member(VehicleKind, v["vehicle_kind"]),
        control_mode=enum_member(ControlMode, v["control_mode"]),
        team=combat["team"],
        hit_points=float(combat["hit_points"]),
        hit_points_max=float(combat["hit_points_max"]),
        alive=combat["alive"],
        ammo=combat["ammo"],
        control_period_s=v["control_period_s"],
        command=CommandStatus(
            owner_epoch=cmd["owner_epoch"],
            last_seq=cmd["last_seq"],
            refusals=cmd["refusals"],
            last_refusal=enum_member(Refusal, cmd["last_refusal"]),
            last_refused_epoch=cmd["last_refused_epoch"],
        ),
    )


def expected_command(v: dict) -> Command:
    ctrl = v["control"]
    if "raw" in ctrl:
        r = ctrl["raw"]
        control = RawActuators(r["aileron"], r["elevator"], r["rudder"], r["throttle"])
    elif "multirotor" in ctrl:
        m = ctrl["multirotor"]
        control = MultirotorRaw(m["roll_rate"], m["pitch_rate"], m["yaw_rate"], m["collective"])
    else:
        a = ctrl["autopilot"]
        control = AutopilotTargets(a["heading_rad"], a["alt_hae_m"], a["tas_mps"])
    return Command(v["target_id"], v["seq"], v["telemetry_tick"], v["trigger"], v["epoch"], control)


def expected_tick_end(v: dict) -> TickEnd:
    return TickEnd(v["tick"], float(v["sim_time_s"]))


def encode_kind(kind: str, values: dict) -> bytes:
    if kind == "telemetry":
        return encode_telemetry(expected_telemetry(values[kind]))
    if kind == "tick_end":
        return encode_tick_end(expected_tick_end(values[kind]))
    if kind in EVENT_KINDS:
        return encode_event(Event.from_dict(values[kind]))
    return encode_command(expected_command(values[kind]))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Write the Python wire fixtures (<kind>.python.bin).")
    p.add_argument("--dir", default=default_fixtures_dir(), help="fixtures directory")
    args = p.parse_args(argv)
    values = load_values(args.dir)
    for kind in KINDS + EVENT_KINDS:
        path = os.path.join(args.dir, f"{kind}.python.bin")
        with open(path, "wb") as f:
            f.write(encode_kind(kind, values))
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
