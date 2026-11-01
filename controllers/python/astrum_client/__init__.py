"""astrum_client — the shared Python client for ASTRUM controllers.

Modules:
  * :mod:`astrum_client.wire`      schema load, safe Telemetry decode, Command encode
  * :mod:`astrum_client.transport` contract session, latest-value telemetry, command publisher
  * :mod:`astrum_client.loop`      the per-tick control loop with startup/staleness timeouts
  * :mod:`astrum_client.cli`       shared flags with ``ASTRUM_*`` env fallbacks
  * :mod:`astrum_client.mathutil`  ``wrap_pi`` / ``clamp`` / anti-windup integrator step
  * :mod:`astrum_client.hold`      the example raw hold law
  * :mod:`astrum_client.wire_fixture` cross-language wire fixtures (``python -m``)
"""
