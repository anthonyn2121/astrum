#!/usr/bin/env python3
"""ASTRUM optimal-control controller (Python) — entry point.

    python controllers/optimal/python/run.py --law lqr --id 1
    python controllers/optimal/python/run.py --law mpc --id 1 --heading 90

Requires the ``astrum_client`` and ``astrum_optimal`` packages importable (the
dev image sets PYTHONPATH; elsewhere ``pip install controllers/python
controllers/optimal/python``). ``--law <name> --help`` lists every flag.
"""

import sys

from astrum_client.runtime import run_main
from astrum_optimal.laws.figure8 import Figure8Law
from astrum_optimal.laws.formation import FormationLaw
from astrum_optimal.laws.lqr import LqrLaw
from astrum_optimal.laws.mpc import MpcLaw

LAWS = {"lqr": LqrLaw, "mpc": MpcLaw, "figure8": Figure8Law, "formation": FormationLaw}

if __name__ == "__main__":
    sys.exit(
        run_main(LAWS, prog="run.py", description="ASTRUM optimal-control controller (Python).")
    )
