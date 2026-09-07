"""Put the embedded env on `sys.path` for this (separate) uv training env.

The training env is a standalone uv project, so it does *not* have the sim's
`astrum_env` extension module or the pure-Python `astrum_dogfight` adapter
installed. Rather than repackage them (they build without maturin — see
`crates/astrum-env/README.md`), we add them to `sys.path` at import time, exactly
like `run_gym_test.sh` does with `PYTHONPATH`:

- `astrum_dogfight`  — pure Python, lives at `crates/astrum-env/python/`.
- `astrum_env`       — the pyo3 cdylib, staged as `astrum_env.so`. Build + stage it
                       with `make rl-env`; it lands in `controllers/rl/vendor/`.
                       `abi3-py38` means the same `.so` loads under this env's
                       CPython. Override the dir with `ASTRUM_ENV_SO_DIR`.

Importing `astrum_rl` runs this, so `import astrum_dogfight` / `import astrum_env`
work everywhere — including AsyncVectorEnv subprocesses, which re-import the package.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# .../controllers/rl/astrum_rl/_paths.py → repo root is 3 levels up from this dir.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_ADAPTER_DIR = _REPO_ROOT / "crates" / "astrum-env" / "python"
_DEFAULT_SO_DIR = _REPO_ROOT / "controllers" / "rl" / "vendor"


def _prepend(path: Path) -> None:
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)


def ensure_on_path() -> None:
    """Idempotently prepend the adapter dir and the staged-`.so` dir to `sys.path`."""
    _prepend(_ADAPTER_DIR)
    so_dir = Path(os.environ.get("ASTRUM_ENV_SO_DIR", _DEFAULT_SO_DIR))
    _prepend(so_dir)


def require_astrum_env() -> None:
    """Import-check `astrum_env`, with a build hint if the `.so` hasn't been staged."""
    ensure_on_path()
    try:
        import astrum_env  # noqa: F401
    except ImportError as exc:  # pragma: no cover - operator-facing guidance
        raise ImportError(
            "astrum_env extension not found. Build and stage it first:\n"
            "    make rl-env\n"
            "(this compiles the pyo3 cdylib and copies it to controllers/rl/vendor/"
            "astrum_env.so), or point ASTRUM_ENV_SO_DIR at a dir containing it."
        ) from exc
