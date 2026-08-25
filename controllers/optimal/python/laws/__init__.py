"""Control laws for the ASTRUM optimal-control controller.

Each module here exposes a ``Law`` class (subclassing
:class:`astrum_control.Law`) and, optionally, an ``add_args(parser)`` hook to
register its own CLI flags. Select one at runtime with ``run.py --law <name>``.
"""
