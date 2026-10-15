#!/usr/bin/env python
"""Host-clock gate: compare CLOCK_MONOTONIC (Python's perf_counter) with CLOCK_MONOTONIC_RAW.

On the WSL2 reference host the kernel tick is rewritten (adjtimex tick 10555-11000 instead of 10000), so
time.perf_counter runs 6-10% fast against external time; CUDA events and CLOCK_MONOTONIC_RAW agree with external
clocks within 0.06%. Benchmarks therefore time with CUDA events and record this ratio; any host-side timing must use
CLOCK_MONOTONIC_RAW, and is refused when |perf_counter / raw - 1| > tolerance.

  python tools/hostclock.py [seconds] [--tol 0.002] [--gate]     # prints JSON; --gate: exit 1 if out of tolerance
Port of the tcemu audit's synthesis/scripts/hostclock.py.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import sys
import time

DEFAULT_TOL = 0.002


class _Timex(ctypes.Structure):
    _fields_ = [("modes", ctypes.c_uint), ("offset", ctypes.c_long), ("freq", ctypes.c_long),
                ("maxerror", ctypes.c_long), ("esterror", ctypes.c_long), ("status", ctypes.c_int),
                ("constant", ctypes.c_long), ("precision", ctypes.c_long), ("tolerance", ctypes.c_long),
                ("time_sec", ctypes.c_long), ("time_usec", ctypes.c_long), ("tick", ctypes.c_long),
                ("pad", ctypes.c_byte * 256)]


def adjtimex() -> dict:
    """Read-only adjtimex(2): the kernel tick (us, nominal 10000) and frequency offset."""
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        tx = _Timex()
        tx.modes = 0  # read only
        rc = libc.adjtimex(ctypes.byref(tx))
        return {"adjtimex_rc": rc, "tick_us": tx.tick, "freq": tx.freq}
    except Exception as e:  # noqa: BLE001 - not Linux
        return {"adjtimex_error": f"{type(e).__name__}: {e}"}


def measure(seconds: float = 2.0) -> dict:
    """Elapsed CLOCK_MONOTONIC_RAW and perf_counter over `seconds` of sleep, and their ratio."""
    res = adjtimex()
    r0, m0 = time.clock_gettime(time.CLOCK_MONOTONIC_RAW), time.perf_counter()
    time.sleep(seconds)
    r1, m1 = time.clock_gettime(time.CLOCK_MONOTONIC_RAW), time.perf_counter()
    res.update(raw_s=r1 - r0, perf_counter_s=m1 - m0, ratio_perf_over_raw=(m1 - m0) / (r1 - r0))
    return res


def gate(seconds: float = 2.0, tol: float = DEFAULT_TOL) -> dict:
    """measure() plus ok = |ratio - 1| <= tol."""
    res = measure(seconds)
    res["tolerance"] = tol
    res["ok"] = abs(res["ratio_perf_over_raw"] - 1.0) <= tol
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("seconds", nargs="?", type=float, default=10.0)
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL)
    ap.add_argument("--gate", action="store_true", help="exit 1 if |perf/raw - 1| > tol")
    a = ap.parse_args(argv)
    res = gate(a.seconds, a.tol)
    print(json.dumps(res))
    return 0 if (res["ok"] or not a.gate) else 1


if __name__ == "__main__":
    sys.exit(main())
