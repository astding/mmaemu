"""Preflight for the RTX 4060 runs on the test host: assert that hardware and software match the reference environment of
the hardware recordings, and record everything that differs from it. Exit 1 on any failed assertion.

usage (from ~/mmaemu-4060): PYTHONPATH=src venv/bin/python preflight_4060.py --out <dir>
"""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

HOME = Path.home() / "mmaemu-4060"
EXPECT = dict(commit="da9edd4d77999acc31edc8b4fdd541993e97cfb6", python="3.12.2", torch="2.7.1+cu128", cuda="12.8",
              gpu="NVIDIA GeForce RTX 4060", capability=[8, 9], sms=24, cublaslt="12.8.3.14",
              truth_manifest="be1270ed64432a1311ab5e2c98ccb69028f72e3424ad4c6148d4cd468fba6695",
              driver_branch="570.", profile="ada-rtx4060-torch")
checks, info = [], {}


def check(name, ok, got, want=None):
    checks.append(dict(check=name, ok=bool(ok), got=got, want=want))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {got}" + ("" if ok or want is None else f"  (want {want})"))


def sh(cmd, cwd=None):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    return r.returncode, (r.stdout + r.stderr).strip()


def lock(path):
    d = {}
    for line in Path(path).read_text().splitlines():
        if "==" in line and not line.startswith("#"):
            n, v = line.split("==", 1)
            d[n.strip().lower()] = v.strip()
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    # host (recorded; not asserted, except the driver branch)
    info["host"] = dict(kernel=platform.release(), os=sh("grep PRETTY_NAME /etc/os-release")[1],
                        cpu=sh("grep -m1 'model name' /proc/cpuinfo")[1].split(":", 1)[-1].strip())
    _, smi = sh("nvidia-smi --query-gpu=name,driver_version,pci.bus_id,memory.total,power.limit,clocks.max.sm,"
                "pcie.link.gen.current,pcie.link.width.current,vbios_version --format=csv,noheader")
    info["nvidia_smi"] = smi
    info["kernel_module"] = sh("head -1 /proc/driver/nvidia/version")[1]
    info["driver_patch"] = sh("cat driver/nv-pci-exclude-bars.patch", cwd=HOME)[1]
    driver = smi.split(",")[1].strip() if "," in smi else smi
    check("driver branch R570 (reference: Windows 572.16 via WSL2)", driver.startswith(EXPECT["driver_branch"]), driver)

    # python and packages
    check("python", platform.python_version() == EXPECT["python"], platform.python_version(), EXPECT["python"])
    ref = lock(HOME / "constraints.txt")
    rc, freeze = sh(f"{HOME}/bin/uv pip freeze --python {sys.executable}")
    got = {}
    for line in freeze.splitlines():
        if "==" in line:
            n, v = line.split("==", 1)
            got[n.strip().lower()] = v.strip()
    bad = {n: (v, ref.get(n)) for n, v in got.items() if ref.get(n) != v}
    check(f"{len(got)} installed packages == reference lock", rc == 0 and not bad, bad or "all equal")
    info["packages"] = got

    # code
    commit = (HOME / "src" / "COMMIT").read_text().strip()
    check("release commit", commit == EXPECT["commit"], commit, EXPECT["commit"])
    for name, cmd in (("fatbins current (build_fatbin --check)", "tools/build_fatbin.py --check"),
                      ("fatbins == manifest sha256 (check_dist --verify-manifest)", "tools/check_dist.py --verify-manifest")):
        rc, out = sh(f"PYTHONDONTWRITEBYTECODE=1 {sys.executable} {cmd}", cwd=HOME / "src")
        check(name, rc == 0, out.splitlines()[-1] if out else rc)
    truth = hashlib.sha256((HOME / "truth" / "manifest.json").read_bytes()).hexdigest()
    check("truth manifest sha256", truth == EXPECT["truth_manifest"], truth[:16] + "...")

    # torch, device, cuBLASLt, profile
    import torch
    check("torch", torch.__version__ == EXPECT["torch"], torch.__version__, EXPECT["torch"])
    check("torch CUDA", torch.version.cuda == EXPECT["cuda"], torch.version.cuda, EXPECT["cuda"])
    check("one visible CUDA device", torch.cuda.device_count() == 1, torch.cuda.device_count())
    p = torch.cuda.get_device_properties(0)
    check("GPU name", p.name == EXPECT["gpu"], p.name, EXPECT["gpu"])
    check("capability", [p.major, p.minor] == EXPECT["capability"], [p.major, p.minor], EXPECT["capability"])
    check("SM count", p.multi_processor_count == EXPECT["sms"], p.multi_processor_count, EXPECT["sms"])
    import mmaemu
    from mmaemu.platform import cublaslt
    check("mmaemu imported from src/", str(HOME / "src") in mmaemu.__file__, mmaemu.__file__)
    check("cuBLASLt version", cublaslt.version() == EXPECT["cublaslt"], cublaslt.version(), EXPECT["cublaslt"])
    info["cublaslt_library"] = cublaslt.library_path()
    check("cuBLASLt loaded from the venv", str(HOME / "venv") in info["cublaslt_library"], info["cublaslt_library"])
    prof = mmaemu.get_profile(EXPECT["profile"])
    check(f"profile {EXPECT['profile']} matches this device", prof.matches_device(torch.device("cuda:0")), "matches")
    rc, out = sh(f"PYTHONDONTWRITEBYTECODE=1 {sys.executable} -m mmaemu info", cwd=HOME / "src")
    info["mmaemu_info"] = out
    check("python -m mmaemu info", rc == 0, f"rc={rc}")

    ok = all(c["ok"] for c in checks)
    (a.out / "preflight.json").write_text(json.dumps(dict(ok=ok, expect=EXPECT, checks=checks, info=info), indent=1) + "\n")
    (a.out / "mmaemu_info.txt").write_text(info["mmaemu_info"] + "\n")
    print(f"\npreflight: {'PASS' if ok else 'FAIL'} ({sum(c['ok'] for c in checks)}/{len(checks)}); {a.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
