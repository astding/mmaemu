"""Environment record (as the H100 probe's M7): GPU (name, SM count, capability, driver, clocks, ECC, full
nvidia-smi -q), torch / CUDA / cuBLASLt (version, library path and sha256), the source tree used (COMMIT, mmaemu.__file__), python packages, OS,
CPU. usage: env_record.py  (writes SWEEP_OUT/env.json, nvidia-smi-q.txt, pip-freeze.txt)"""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import platform
import subprocess
import sys
from pathlib import Path

import sc
import common  # noqa: E402  (tools/record/common.py of the source tree)


def run(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return r.stdout
    except Exception as e:  # noqa: BLE001
        return f"ERROR {e!r}"


def main():
    import torch
    import mmaemu
    from mmaemu.platform import cublaslt
    torch.cuda.init()
    torch._scaled_mm(torch.ones(16, 32, device="cuda").to(torch.float8_e4m3fn),
                     torch.ones(16, 32, device="cuda").to(torch.float8_e4m3fn).t(),
                     scale_a=torch.ones((), device="cuda"), scale_b=torch.ones((), device="cuda"),
                     out_dtype=torch.float32)
    torch.cuda.synchronize()
    lib = cublaslt.library_path()
    hw = common.hardware()
    p = torch.cuda.get_device_properties(0)
    env = dict(hardware=hw, torch=torch.__version__, cuda=torch.version.cuda,
               device=dict(name=p.name, capability=list(torch.cuda.get_device_capability(0)),
                           sm_count=p.multi_processor_count, total_memory=p.total_memory),
               cublaslt_version=cublaslt.version(), cublaslt_lib=lib,
               cublaslt_lib_sha256=hashlib.sha256(Path(lib).resolve().read_bytes()).hexdigest(),
               src=str(sc.SRC), src_commit=(sc.SRC / "COMMIT").read_text().strip() if (sc.SRC / "COMMIT").exists()
               else None, mmaemu_file=mmaemu.__file__, python=sys.version, executable=sys.executable,
               platform=platform.platform(), packages={d.metadata["Name"]: d.version for d in md.distributions()},
               nvidia_smi=run(["nvidia-smi", "--query-gpu=name,driver_version,pci.bus_id,clocks.sm,clocks.max.sm,"
                               "clocks.mem,temperature.gpu,power.limit,ecc.mode.current,persistence_mode,"
                               "compute_mode,mig.mode.current", "--format=csv"]),
               lscpu=run(["lscpu"]), uname=run(["uname", "-a"]), ldd_version=run(["ldd", "--version"])[:200])
    for name in ("transformer_engine", "transformer_engine_cu12", "transformer_engine_torch"):
        try:
            env[f"{name}_version"] = md.version(name)
        except md.PackageNotFoundError:
            env[f"{name}_version"] = None
    sc.OUT.mkdir(parents=True, exist_ok=True)
    (sc.OUT / "env.json").write_text(json.dumps(env, indent=1, default=str) + "\n")
    (sc.OUT / "nvidia-smi-q.txt").write_text(run(["nvidia-smi", "-q"]))
    (sc.OUT / "pip-freeze.txt").write_text("\n".join(f"{k}=={v}" for k, v in sorted(env["packages"].items())) + "\n")
    print(json.dumps({k: env[k] for k in ("torch", "cuda", "device", "cublaslt_version", "cublaslt_lib",
                                          "src_commit", "mmaemu_file")}, indent=1))


if __name__ == "__main__":
    main()
