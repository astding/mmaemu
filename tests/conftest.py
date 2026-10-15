"""Shared pytest configuration for mmaemu.

- Markers: gpu (needs a CUDA device; auto-skipped without one), slow (long; T1s), truth (needs $MMAEMU_TRUTH_DIR,
  the hardware ground truth; auto-skipped without it), profile_device("<profile>") (runs only on the profile's
  reference device; auto-skipped elsewhere with reason "profile device unavailable").
- The audited prototype package ``tcemu`` (and ``tcemu.*``) is blocked with a sys.meta_path finder: mmaemu tests
  must never exercise it (``tcemu_ada_truth``, the hardware ground truth, is not affected). The session fails if
  ``tcemu`` is in sys.modules at the end, however it got there.
- Fixtures: cuda_device, profile_device (a factory: profile_device(name) -> device), truth_dir.
- Legacy guard: the pre-v0.1 tests of the prototype (test_reference.py, test_cublas_ada.py, test_recorded_hw.py)
  imported ``mmaemu.api._legacy_reference``, a bootstrap shim that v0.1's api.py no longer provides. If such a file
  is present, it is reported as ONE counted skip instead of a collection error; while a shim exists it runs
  normally. None of these files ships with v0.1.

Nothing here initialises CUDA at import or collection time; torch.cuda is touched only when a gpu /
profile_device test (or fixture) is set up.
"""
from __future__ import annotations

import importlib
import importlib.abc
import os
import sys
from pathlib import Path

import pytest

BLOCKED_PACKAGE = "tcemu"


class TcemuBlocker(importlib.abc.MetaPathFinder):
    """Refuses to import the package ``tcemu`` and its submodules (not ``tcemu_ada_truth``)."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == BLOCKED_PACKAGE or fullname.startswith(BLOCKED_PACKAGE + "."):
            raise ModuleNotFoundError(
                f"import of {fullname!r} is blocked: mmaemu tests and tools must never use the audited tcemu "
                f"package (tests/conftest.py TcemuBlocker)", name=fullname)
        return None


def install_tcemu_blocker() -> TcemuBlocker:
    """Put a TcemuBlocker first on sys.meta_path (idempotent); returns the installed instance."""
    for f in sys.meta_path:
        if type(f).__name__ == "TcemuBlocker":
            return f
    blocker = TcemuBlocker()
    sys.meta_path.insert(0, blocker)
    return blocker


def tcemu_modules() -> list[str]:
    """Names of loaded modules that belong to the blocked package."""
    return sorted(m for m in sys.modules if m == BLOCKED_PACKAGE or m.startswith(BLOCKED_PACKAGE + "."))


install_tcemu_blocker()


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a CUDA device (skipped without one)")
    config.addinivalue_line("markers", "slow: long-running (T1s tier)")
    config.addinivalue_line("markers", "truth: needs the hardware ground truth ($MMAEMU_TRUTH_DIR; skipped without it)")
    config.addinivalue_line("markers", "profile_device(name): runs only on the reference device of profile `name` "
                                       "(skipped elsewhere: 'profile device unavailable')")


def pytest_sessionfinish(session, exitstatus):
    leaked = tcemu_modules()
    if leaked:
        sys.stderr.write(f"\nERROR: the blocked package 'tcemu' was imported during the session: {leaked}\n")
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


# --- helpers shared by markers and fixtures ------------------------------------------------------------------------

def _cuda_available() -> bool:
    import torch

    return torch.cuda.is_available()


def _profile_device_or_skip(name: str):
    import torch

    if not _cuda_available():
        pytest.skip(f"profile device unavailable: no CUDA device (profile {name!r})")
    device = torch.device("cuda", torch.cuda.current_device())
    try:
        from mmaemu.platform import get_profile

        matches = get_profile(name).matches_device(device)
    except (NotImplementedError, ImportError) as e:
        pytest.skip(f"profile device unavailable: profiles not implemented yet ({type(e).__name__}: {e})")
    if not matches:
        pytest.skip(f"profile device unavailable: {torch.cuda.get_device_name(device)} is not the device of "
                    f"profile {name!r}")
    return device


def _truth_dir_or_skip() -> Path:
    value = os.environ.get("MMAEMU_TRUTH_DIR")
    if not value:
        pytest.skip("hardware ground truth unavailable: MMAEMU_TRUTH_DIR is not set")
    path = Path(value)
    if not path.is_dir():
        pytest.fail(f"MMAEMU_TRUTH_DIR={value!r} is not a directory")
    sys.dont_write_bytecode = True  # never write __pycache__ into the read-only truth package
    return path


def pytest_runtest_setup(item):
    if item.get_closest_marker("gpu") is not None and not _cuda_available():
        pytest.skip("needs a CUDA device")
    if item.get_closest_marker("truth") is not None:
        _truth_dir_or_skip()
    mark = item.get_closest_marker("profile_device")
    if mark is not None and mark.args:
        _profile_device_or_skip(mark.args[0])


# --- legacy guard: superseded pre-v0.1 tests --------------------------------------------------------------------------

LEGACY_SHIM_IMPORT = "from mmaemu.api import _legacy_reference"
LEGACY_SHIM_TESTS = ("test_reference.py", "test_cublas_ada.py", "test_recorded_hw.py")
HERE = Path(__file__).resolve().parent


def _legacy_shim_removed() -> bool:
    """True only if mmaemu.api imports cleanly and has no _legacy_reference (a broken mmaemu.api is not hidden)."""
    try:
        api = importlib.import_module("mmaemu.api")
    except ImportError:
        return False
    return not hasattr(api, "_legacy_reference")


class SupersededLegacyModule(pytest.Module):
    """A legacy test module whose shim is gone: collected as one skip with the reason, never imported."""

    def collect(self):
        pytest.skip(f"superseded legacy test: mmaemu.api no longer provides _legacy_reference; "
                    f"{self.path.name} is a pre-v0.1 test of the prototype and should be deleted. Skipped by the "
                    f"legacy guard in tests/conftest.py; this file's coverage is not run",
                    allow_module_level=True)


def pytest_pycollect_makemodule(module_path, parent):
    path = Path(module_path)
    if path.name not in LEGACY_SHIM_TESTS or path.resolve().parent != HERE:
        return None
    try:
        imports_shim = any(ln.strip().startswith(LEGACY_SHIM_IMPORT)
                           for ln in path.read_text(encoding="utf-8").splitlines())
    except OSError:
        return None
    if imports_shim and _legacy_shim_removed():
        return SupersededLegacyModule.from_parent(parent, path=path)
    return None


# --- fixtures ------------------------------------------------------------------------------------------------------

@pytest.fixture
def cuda_device():
    """The current CUDA device (torch.device); skips the test without CUDA."""
    if not _cuda_available():
        pytest.skip("needs a CUDA device")
    import torch

    return torch.device("cuda", torch.cuda.current_device())


@pytest.fixture
def profile_device():
    """Factory: profile_device(name) -> the CUDA device if mmaemu.get_profile(name).matches_device(it), else the
    test is skipped with reason "profile device unavailable" (also if mmaemu.platform is unavailable)."""
    return _profile_device_or_skip


@pytest.fixture
def truth_dir() -> Path:
    """Path of the hardware ground truth ($MMAEMU_TRUTH_DIR); skips when unset. Sets sys.dont_write_bytecode."""
    return _truth_dir_or_skip()
