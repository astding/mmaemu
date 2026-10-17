"""Run a live.py job file: one tools/record/live.py worker process per workspace size (ws -1 = torch default:
CUBLASLT_WORKSPACE_SIZE unset), with CUBLASLT_LOG_LEVEL=2 per-call capture. Output in PROBE_OUT:
<tag>_ws<ws>.npz / .json (live.py's worker format: per job lt config, full_sha256, inputs_cpu_eq_gpu, canary) and
<tag>_ws<ws>.lt.log. Run the same file twice with different tags to check determinism across processes.
usage: m4_run.py JOBS.json TAG"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pc

LIVE = pc.SRC / "tools" / "record" / "live.py"


def main():
    jobs = json.loads(Path(sys.argv[1]).read_text())
    tag = sys.argv[2]
    by_ws: dict = {}
    for j in jobs:
        by_ws.setdefault(int(j["ws"]), []).append(j)
    rcs = {}
    for ws, js in sorted(by_ws.items()):
        jp = pc.OUT / f"{tag}_ws{ws}.jobs.json"
        jp.write_text(json.dumps(js))
        env = dict(os.environ, PYTHONPATH=str(pc.SRC), PYTHONDONTWRITEBYTECODE="1", CUBLASLT_LOG_LEVEL="2",
                   CUBLASLT_LOG_FILE=str(pc.OUT / f"{tag}_ws{ws}.lt.log"))
        env.pop("CUBLASLT_WORKSPACE_SIZE", None)
        if ws >= 0:
            env["CUBLASLT_WORKSPACE_SIZE"] = str(ws // 1024)
        t0 = time.monotonic()
        try:
            r = subprocess.run([sys.executable, str(LIVE), "worker", "--jobs", str(jp),
                                "--out", str(pc.OUT / f"{tag}_ws{ws}.npz")], env=env, timeout=560)
            rcs[ws] = r.returncode
        except subprocess.TimeoutExpired:
            rcs[ws] = "timeout"
        print(f"{tag} ws={ws}: {len(js)} jobs rc={rcs[ws]} {time.monotonic() - t0:.1f}s", flush=True)
    print("return codes", rcs)
    sys.exit(0 if all(v == 0 for v in rcs.values()) else 1)


if __name__ == "__main__":
    main()
