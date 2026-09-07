"""astrum-rl — from-scratch RL training for the ASTRUM dogfight learner.

Importing this package puts the embedded env (`astrum_env` .so + the
`astrum_dogfight` adapter) on `sys.path` (see `_paths`), so the rest of the
package can `from astrum_dogfight import ...` unconditionally.
"""

from __future__ import annotations

from . import _paths

_paths.ensure_on_path()

__all__ = ["_paths"]
