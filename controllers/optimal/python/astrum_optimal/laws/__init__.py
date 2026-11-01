"""Control laws for the ASTRUM optimal-control controller.

Each public module exposes a ``Law`` class (subclassing
:class:`astrum_optimal.runtime.Law`) and, optionally, ``add_args(parser)`` to
register its own flags. Select one with ``run.py --law <name>``; modules whose
name starts with ``_`` are shared helpers, not laws.
"""
