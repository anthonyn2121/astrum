//! Whether one aircraft of a scenario trims at its initial conditions. A trim
//! that fails to converge is fatal when the sim starts, so the studio checks
//! it before a scenario is saved or flown.

use crate::scenario::{AircraftSpec, Scenario};
use crate::systems::build_fdm;

/// Build `aircraft`'s FDM at `scenario`'s rate and seed, exactly as the sim
/// spawns it, when its class trims. An aircraft that does not trim is `Ok`
/// without being built. The error is [`build_fdm`]'s text (it names the
/// aircraft and its model).
pub fn check_aircraft(scenario: &Scenario, aircraft: &AircraftSpec) -> Result<(), String> {
    let class = aircraft.class().map_err(|e| e.to_string())?;
    if !class.trim {
        return Ok(());
    }
    build_fdm(aircraft, &class, 1.0 / scenario.sim.rate_hz, scenario.sim.seed).map(|_| ())
}
