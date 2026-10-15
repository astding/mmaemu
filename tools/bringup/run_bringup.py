"""G0-G10 hardware bring-up. Run outside the GPU lock; each GPU command takes its own lock.

The GPU lock is a command prefix (--gpu, default $MMAEMU_GPU_LOCK, or $GPU, or flock on $TMPDIR/mmaemu-gpu.lock).
G9 needs an evo2-anywhere checkout (--evo2-root / $EVO2_ROOT) and the Ada Evo 2 1B golden logits
(--golden / $MMAEMU_EVO2_GOLDEN; maintainer data, not published).

--dry-run executes the portable gates and skips native architecture research.
--from resumes at a gate in a new output directory; existing runs are never overwritten.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GATES = [f"G{i}" for i in range(11)]
NATIVE = {"G4", "G5", "G6", "G7", "G9"}
PY = sys.executable
CPU = "not gpu and not slow and not truth and not profile_device"
DEFAULT_LOCK = f"flock -w 3600 {tempfile.gettempdir()}/mmaemu-gpu.lock"


def _lock_file(prefix: list[str]) -> Path:
    """The file the GPU-lock prefix locks: flock's lock-file argument, else <wrapper dir>/.gpu.lock (the convention of
    a lock wrapper script such as `exec flock -w 3600 <dir>/.gpu.lock "$@"`)."""
    if Path(prefix[0]).name == "flock":
        opts_with_value = {"-w", "--timeout", "-E", "--conflict-exit-code"}
        args, skip = [], False
        for a in prefix[1:]:
            if skip:
                skip = False
            elif a in opts_with_value:
                skip = True
            elif not a.startswith("-"):
                args.append(a)
        return Path(args[0]).resolve() if args else Path("/nonexistent")
    return (Path(prefix[0]).parent / ".gpu.lock").resolve()


class Bringup:
    def __init__(self, args):
        self.args = args
        self.env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1")
        self.env.setdefault("OMP_NUM_THREADS", "4")
        self.env.setdefault("MKL_NUM_THREADS", "4")
        self.gpu = shlex.split(args.gpu)
        lock = _lock_file(self.gpu)
        self.inherited_lock = any(p.resolve() == lock for p in Path("/proc/self/fd").glob("*")
                                  if p.exists())
        if self.inherited_lock:
            self.env["GPU"] = self.env["MMAEMU_GPU_LOCK"] = shutil.which("env") or "/usr/bin/env"
        self.rows = []
        self.out = None

    def call(self, gate, name, argv, *, gpu=True, env=None, accepted=(0,)):
        target = self.out / gate
        target.mkdir(exist_ok=True)
        command = (self.gpu if gpu and not self.inherited_lock else []) + ["timeout", "580", *argv]
        t0 = time.monotonic()
        with (target / f"{name}.log").open("x") as log:
            proc = subprocess.run(command, cwd=ROOT, env=self.env | (env or {}), stdout=log,
                                  stderr=subprocess.STDOUT)
        row = dict(name=name, command=argv, gpu=gpu, returncode=proc.returncode,
                   seconds=round(time.monotonic() - t0, 2), log=str(target / f"{name}.log"))
        self.rows[-1]["commands"].append(row)
        print(f"{gate} {name}: rc={proc.returncode}, {row['seconds']} s", flush=True)
        if proc.returncode not in accepted:
            raise RuntimeError(f"{gate}/{name} failed; see {row['log']}")

    def python(self, gate, name, script, *argv, **kwargs):
        self.call(gate, name, [PY, script, *argv], **kwargs)

    def stage(self, gate):
        out = self.out / gate
        out.mkdir(exist_ok=True)
        env = {"BRINGUP_OUT": str(out), "BRINGUP_CACHE": str(self.out / "cache"),
               "TORCH_CUDA_ARCH_LIST": self.arch}
        kit = "tools/bringup/kits/"
        if gate == "G0":
            self.call(gate, "info", [PY, "-m", "mmaemu", "info", "--json", str(out / "info.json")])
            self.python(gate, "hostclock", "tools/hostclock.py", gpu=False, accepted=(0, 1))
        elif gate == "G1":
            self.call(gate, "cpu", [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests", "-m", CPU],
                      gpu=False, env={"CUDA_VISIBLE_DEVICES": ""})
            self.call(gate, "t3", ["bash", "tools/run_gpu_tests.sh", "--out", str(out / "t3")],
                      gpu=False, env={"PYTHON": PY})
        elif gate == "G2":
            self.call(gate, "selftest", [PY, "-m", "mmaemu", "selftest", "--force", "--json", str(out / "selftest.json")])
        elif gate == "G3":
            self.call(gate, "independence", [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                                           "tests/test_device_independence.py"],
                      env={"MMAEMU_ALLOW_GATED_KERNELS": "1", "MMAEMU_DI_FULL": "cuda:v1,cuda:v4,cuda:v9"})
        elif gate == "G4":
            for arch, kind in (("12.0", []), ("12.0a", ["--kind"])):
                argv = ["--sets", "MM_sm120", "MM_ada", "TBX_B200", "TBX_H100c0", "--arch", arch, *kind]
                self.python(gate, f"compile-{arch}", kit + "c6/run_kit.py", *argv, "--compile-only", gpu=False, env=env)
                self.python(gate, f"mma-{arch}", kit + "c6/run_kit.py", *argv, "--out", str(out / f"mma-{arch}.json"), env=env)
            self.python(gate, "interleave", kit + "interleave/interleave_probe.py", env=env, gpu=False)
        elif gate == "G5":
            self.python(gate, "cublas-kit", kit + "c6/run_cublas_kit.py", "--set", "MM_cublas64_sm120",
                        "--out", str(out / "cublas-kit.json"), env=env)
            self.python(gate, "native-vs-ada", "tools/bringup/legacy/g1_cublas.py", "--out", str(out / "native-vs-ada.json"),
                        env=env | {"CUBLASLT_LOG_LEVEL": "2"}, accepted=(0, 1))
            native_report = json.loads((out / "native-vs-ada.json").read_text())
            if native_report.get("verdict") not in ("PASS", "MISMATCH") or not native_report.get("rows"):
                raise RuntimeError("G5 native/Ada probe did not produce a valid numerical report")
            # Exit 1 here is a recorded native/Ada numerical difference, not an
            # emulator conformance failure; probe errors (exit 2) still stop.
        elif gate == "G6":
            self.python(gate, "heuristics", "tools/record_heuristics.py", "--grid", "c5,evo2,evo2_small_m",
                        "--ws", "0,1,4,32", "--out", str(out / "heuristics.npz"))
        elif gate == "G7":
            self.python(gate, "specials", kit + "zero_specials/scripts/zero_specials.py", env=env)
            special_rows = json.loads((out / "zero_specials.json").read_text())["rows"]
            if any(r["model_cpu_bits"] != r["model_gpu_bits"] for r in special_rows if "hw_error" not in r):
                raise RuntimeError("G7: reference CPU/GPU special-value bits differ")
            self.python(gate, "fp8-specials", "tools/bringup/fp8_specials.py", "--out", str(out / "fp8-specials.json"))
            self.python(gate, "ties", kit + "epilogue_ties/record_epilogue_ties.py", str(out / "ties.pt"), env=env)
            self.call(gate, "ties-replay", [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                                           kit + "epilogue_ties/test_epilogue_ties.py"],
                      gpu=False, env={"EPI_TIES": str(out / "ties.pt")})
        elif gate == "G8":
            self.python(gate, "compile-window", kit + "window/build.py", "epass_raw", gpu=False, env=env)
            self.python(gate, "window", kit + "window/hw_collect.py", env=env)
            self.python(gate, "analysis", kit + "window/analyse.py", "hw", gpu=False, env=env)
            self.call(gate, "design-a", [PY, "-c",
                "import json,torch; from dataclasses import asdict; "
                "from mmaemu.backends.cuda.fast import gates,specs; "
                "print(json.dumps(asdict(gates.design_a(specs.TC3,torch.device('cuda',0))),default=str))"])
            self.python(gate, "compile-b3soft", kit + "b3soft/scripts/b3soft.py", "build", self.arch.replace(".", ""),
                        gpu=False, env=env)
            self.python(gate, "b3soft", kit + "b3soft/scripts/b3soft.py", "probe", env=env)
        elif gate == "G9":
            if not self.args.evo2_root:
                raise RuntimeError("G9 needs an evo2-anywhere checkout: pass --evo2-root or set EVO2_ROOT")
            if not self.args.golden or not Path(self.args.golden).is_file():
                raise RuntimeError(f"G9 needs the Ada Evo 2 1B golden logits: pass --golden or set MMAEMU_EVO2_GOLDEN "
                                   f"(now {self.args.golden!r})")
            consumer = Path(self.args.evo2_root).resolve()
            bench = consumer / "benchmarks/bench.py"
            if not bench.is_file():
                raise RuntimeError(f"consumer benchmark missing: {bench}; pass --evo2-root")
            self.call(gate, "gemm-truth", [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                                         "tests/test_device_independence.py"])
            for mode in ("native", "emulate", "compat", "auto"):
                self.python(gate, mode, str(bench), "--model", "evo2_1b_base", "--device", "gpu", "--source", "blocks",
                            "--fp8-mode", mode, "--windows", "all", "--threads", "4", "--out", str(out / mode),
                            env={"PYTHONPATH": f"{consumer / 'src'}:{ROOT}", "EVO2_ANYWHERE_ALLOW_UNVALIDATED_NATIVE": "1"})
            self.call(gate, "golden-comparison", [PY, "-c",
                "import json,sys,numpy as np; from pathlib import Path; "
                "sys.path.insert(0,sys.argv[1]); from compare import compare; "
                "golden=Path(sys.argv[2]); root=Path(sys.argv[3]); G=np.load(golden); rows={}; "
                "\nfor mode in ('native','emulate','compat','auto'):\n"
                " p=next((root/mode).glob('*.npz')); T=np.load(p); assert set(G.files)==set(T.files);\n"
                " for k in G.files:\n"
                "  assert G[k].shape==T[k].shape and G[k].dtype==T[k].dtype, (mode,k);\n"
                "  if k.endswith('/ids'): assert np.array_equal(G[k],T[k]), (mode,k);\n"
                " rows[mode]=compare(golden,p,strict=False,log=lambda _:None);\n"
                "(root/'golden-comparison.json').write_text(json.dumps(rows,indent=2));\n",
                str(consumer / "benchmarks"), self.args.golden, str(out)], gpu=False)
        elif gate == "G10":
            for shape in ("1b", "7b", "40b"):
                self.python(gate, shape, "tools/bench_kernels.py", "--kernels", "v1,v4,v9", "--shape", shape,
                            "--M", "2048", "--out", str(out / f"{shape}.json"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    select = ap.add_mutually_exclusive_group()
    select.add_argument("--stage", choices=GATES)
    select.add_argument("--from", dest="from_gate", choices=GATES)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force-fail", choices=GATES)
    ap.add_argument("--out", help="new run directory (must not exist)")
    ap.add_argument("--gpu", default=os.environ.get("MMAEMU_GPU_LOCK") or os.environ.get("GPU") or DEFAULT_LOCK,
                    help="GPU-lock command prefix")
    ap.add_argument("--evo2-root", default=os.environ.get("EVO2_ROOT"), help="evo2-anywhere checkout (G9)")
    ap.add_argument("--golden", default=os.environ.get("MMAEMU_EVO2_GOLDEN"),
                    help="Ada Evo 2 1B golden logits, evo2_1b_base_real-te-fp8.npz (G9)")
    args = ap.parse_args(argv)
    runner = Bringup(args)
    meta_cmd = [PY, "-c", "import json,shutil; from tools.record.common import hardware; "
                "print(json.dumps(dict(hardware(),nvcc=shutil.which('nvcc'))))"]
    if not runner.inherited_lock:
        meta_cmd = [*runner.gpu, "timeout", "60", *meta_cmd]
    meta = subprocess.run(meta_cmd, cwd=ROOT, env=runner.env, capture_output=True, text=True, timeout=90)
    if meta.returncode:
        print(f"G0 metadata failed: {meta.stderr[-1000:]}", file=sys.stderr)
        return 1
    hw = json.loads(meta.stdout.strip().splitlines()[-1])
    if hw.get("arch") not in ({"sm_89", "sm_120"} if args.dry_run else {"sm_120"}):
        print(f"G0 unsupported device for this run: {hw}", file=sys.stderr)
        return 1
    runner.arch = "12.0" if hw["arch"] == "sm_120" else "8.9"
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    runner.out = (Path(args.out).resolve() if args.out else
                  ROOT / "results/bringup" / f"{hw['gpu'].replace(' ', '-')}-{hw['arch']}-{stamp}")
    runner.out.mkdir(parents=True, exist_ok=False)
    stages = [args.stage] if args.stage else GATES[GATES.index(args.from_gate or "G0"):]
    report = dict(format="mmaemu-bringup/1", hardware=hw, dry_run=args.dry_run,
                  inherited_lock=runner.inherited_lock, stages=runner.rows)
    rc = 0
    for gate in stages:
        row = dict(gate=gate, status="RUNNING", commands=[])
        runner.rows.append(row)
        try:
            if args.force_fail == gate:
                raise RuntimeError(f"{gate}: forced failure")
            if args.dry_run and gate in NATIVE:
                row.update(status="SKIP", reason="native sm_120 measurements omitted in --dry-run")
            else:
                runner.stage(gate)
                row["status"] = "RECORDED" if gate in NATIVE or gate == "G8" else "PASS"
        except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
            row.update(status="FAIL", reason=str(exc))
            print(str(exc), file=sys.stderr)
            rc = 1
        report["verdict"] = "FAIL" if rc else "PASS"
        (runner.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        if rc:
            break
    print(f"{report['verdict']}: {runner.out / 'summary.json'}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
