"""Import babyagi-inside's own modules from the pinned vendor/ checkout.

The harness never copies upstream code. `qfs` comes from the editable install of
vendor/babyagi-inside; its `tests` package is not part of that wheel, so the
checkout root is put on sys.path explicitly. vendor/qwenFS also ships `qfs` and
`tests` packages and is deliberately never importable: babyagi-inside's `qfs`
is the qwenFS-derived compiler these experiments build on.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
BABYAGI_INSIDE = ROOT / "vendor" / "babyagi-inside"
TRACES = BABYAGI_INSIDE / "traces"
SUITE_V3 = BABYAGI_INSIDE / "benchmarks" / "babyagi-v3.json"


class UpstreamError(RuntimeError):
    """vendor/babyagi-inside is missing, or an import resolved somewhere else."""


def ensure_importable() -> None:
    if not (BABYAGI_INSIDE / "qfs" / "__init__.py").is_file():
        raise UpstreamError("vendor/babyagi-inside is not checked out: "
                            "git submodule update --init --recursive")
    path = str(BABYAGI_INSIDE)
    if path not in sys.path:
        sys.path.insert(0, path)


def module(name: str) -> ModuleType:
    """Import an upstream module, e.g. "qfs.agent_evaluate" or "tests.test_agent_circuit"."""
    ensure_importable()
    loaded = importlib.import_module(name)
    origin = Path(loaded.__file__).resolve()
    if BABYAGI_INSIDE.resolve() not in origin.parents:
        raise UpstreamError(f"{name} resolved to {origin}, outside vendor/babyagi-inside")
    return loaded
