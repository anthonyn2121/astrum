#!/usr/bin/env python3
"""ASTRUM optimal-control controller (Python) — clean-API entry point.

Runs a control law from ``laws/`` against one aircraft over Zenoh. All the
transport/decoding lives in :mod:`astrum_control`; this just picks a law and
starts the runtime:

    python controllers/optimal/python/run.py --law lqr --id 1
    python controllers/optimal/python/run.py --law mpc --heading 90

Pick the law with ``--law <name>`` (a module in ``laws/``); every other flag
(shared plus any the law adds) is listed by ``--help``. As everywhere in ASTRUM,
each flag falls back to its ``ASTRUM_*`` env var (CLI arg > env > default).
"""

import argparse
import importlib
import os
import sys

# Make this directory importable so `astrum_control` and the `laws` package
# resolve no matter where the script is invoked from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import astrum_control as ac  # noqa: E402


def main():
    law_default = ac.env_str("ASTRUM_LAW", "lqr")

    # Pre-parse *only* --law (help suppressed) to learn which law is selected,
    # so we can import it and let it register its own flags before the real
    # parser handles -h/--help — otherwise `--law mpc --help` would fire help
    # during this pass and never show the law's options.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--law", default=law_default)
    known, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(
        description="ASTRUM optimal-control controller (Python).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--law", default=law_default,
                   help="control law to run (a module name in laws/, e.g. lqr or mpc)")
    ac.add_common_args(p)

    try:
        law_mod = importlib.import_module(f"laws.{known.law}")
    except ModuleNotFoundError as e:
        p.error(f"unknown --law '{known.law}': no laws/{known.law}.py ({e})")
    if hasattr(law_mod, "add_args"):
        law_mod.add_args(p)
    args = p.parse_args()

    law = law_mod.Law(args)
    ac.Runtime(args, law).run()


if __name__ == "__main__":
    main()
