"""astrum_client — the shared Python client for ASTRUM controllers (the contract
is controllers/README.md).

Modules:
  * :mod:`astrum_client.wire`      wire types, safe Telemetry/Event/TickEnd decode, Command encode
  * :mod:`astrum_client.transport` session, latest-value telemetry, events + TickEnd,
                                   command publisher
  * :mod:`astrum_client.loop`      the control loop and its outcomes / exit codes
  * :mod:`astrum_client.runtime`   ``Law``, ``StepInput``, ``Stop`` and ``run_main``
  * :mod:`astrum_client.cli`       shared flags with ``ASTRUM_*`` env fallbacks
  * :mod:`astrum_client.mathutil`  ``wrap_pi`` / ``clamp`` / anti-windup integrator step
  * :mod:`astrum_client.hold`      the example raw hold law
  * :mod:`astrum_client.wire_fixture` cross-language wire fixtures (``python -m``)
"""
