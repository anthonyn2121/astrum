#!/usr/bin/env python3
"""ASTRUM optimal-control controller (Python) — entry point.

    python controllers/optimal/python/run.py --law lqr --id 1
    python controllers/optimal/python/run.py --law mpc --heading 90

Requires the ``astrum_client`` and ``astrum_optimal`` packages importable (the
dev image sets PYTHONPATH; elsewhere ``pip install -e controllers/python -e
controllers/optimal/python``). ``--law <name> --help`` lists every flag.
"""

import sys

from astrum_optimal.runtime import main

if __name__ == "__main__":
    sys.exit(main())
