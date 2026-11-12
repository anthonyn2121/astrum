import dataclasses

import pytest
from astrum_client.runtime import StepInput
from astrum_client.wire import CommandStatus, ControlMode, VehicleKind, VehicleState

DT = 1.0 / 120.0

LEVEL_FLIGHT = VehicleState(
    aircraft_id=1,
    tick=1,
    sim_time_s=DT,
    lat_deg=37.6,
    lon_deg=-122.4,
    alt_hae_m=1500.0,
    e=0.0,
    n=0.0,
    u=1500.0,
    vel_ned_n=50.0,
    vel_ned_e=0.0,
    vel_ned_d=0.0,
    vel_enu_e=0.0,
    vel_enu_n=50.0,
    vel_enu_u=0.0,
    roll=0.0,
    pitch=0.0,
    yaw=0.0,
    p=0.0,
    q=0.0,
    r=0.0,
    tas_mps=50.0,
    alpha=0.03,
    beta=0.0,
    nz_g=1.0,
    mach=0.15,
    fault_flags=0,
    vehicle_kind=VehicleKind.FIXED_WING,
    control_mode=ControlMode.RAW,
    team=0,
    hit_points=100.0,
    hit_points_max=100.0,
    alive=True,
    ammo=0,
    control_period_s=DT,
    command=CommandStatus(),
)


@pytest.fixture
def make_state():
    """Factory: level flight north at 1500 m HAE / 50 m/s, overridable per field."""
    return lambda **overrides: dataclasses.replace(LEVEL_FLIGHT, **overrides)


@pytest.fixture
def make_input():
    """Factory: the StepInput of one state (no fleet, no events, 120 Hz)."""
    return lambda state, fleet=None: StepInput(state, fleet or {}, (), state.control_period_s)
