"""Load sglang/ scripts as importable modules.

They are standalone CLIs, not a package, so there is nothing to `import`.
Session-scoped: loading is cheap but neither module has import side effects
worth repeating.
"""
import importlib.util
import pathlib
import sys

import pytest

SGLANG = pathlib.Path(__file__).resolve().parents[1] / "sglang"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SGLANG / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def kv_reaper():
    return _load("kv_reaper")


@pytest.fixture(scope="session")
def bench_stream():
    return _load("bench_stream")


@pytest.fixture(scope="session")
def archive_script():
    """Path to archive_worker_log.sh -- a shell CLI, so it is run, not imported."""
    path = SGLANG / "archive_worker_log.sh"
    assert path.is_file(), f"missing {path}"
    return path
