"""Run one recorder step as ONE locked GPU run and log it to recordings/rtx4060_2026-10/gpu_runs.jsonl.

usage: python tools/record/gpu_run.py <item-label> -- <command...>
The command runs as  $MMAEMU_GPU_LOCK timeout 600 <command...>  (exclusive GPU lock, < 10 min). The
logged wall time includes waiting for the lock only if another job holds it; the time spent inside the lock is
measured by the child itself (first line 'LOCKED <epoch>' printed by the wrapper below).
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

# Command prefix that serialises GPU work on this host (MMAEMU_GPU_LOCK; default: flock on a file in the temp dir).
GPU = shlex.split(os.environ.get("MMAEMU_GPU_LOCK") or f"flock -w 3600 {tempfile.gettempdir()}/mmaemu-gpu.lock")


def main():
    argv = sys.argv[1:]
    item, sep = argv[0], argv.index("--")
    cmd = argv[sep + 1:]
    # inner: record the time the lock was acquired, then exec the command under timeout 600
    inner = ["bash", "-c", 'echo "LOCKED $(date +%s.%N)"; exec timeout 600 "$@"', "inner", *cmd]
    t_req = time.time()
    p = subprocess.Popen([*GPU, *inner], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         env=common.worker_env(None, None))
    t_lock = None
    for line in p.stdout:
        if line.startswith("LOCKED ") and t_lock is None:
            t_lock = float(line.split()[1])
            continue
        print(line, end="", flush=True)
    rc = p.wait()
    t_end = time.time()
    t0 = t_lock or t_req
    common.log_gpu_run(item, t0, t_end, rc == 0, note=f"rc={rc}; cmd={' '.join(cmd)[:300]}; "
                                                       f"waited_for_lock={round(t0 - t_req, 1)}s")
    print(f"[gpu_run] {item}: rc={rc}, {t_end - t0:.1f}s inside the lock")
    sys.exit(rc)


if __name__ == "__main__":
    main()
