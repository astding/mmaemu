"""Shared helpers of the RTX 4060 recorders (tools/record/r*.py).

- paths: the repo, the recording root recordings/rtx4060_2026-10 (truth layout: <root>/data/<group>/<name>.json +
  .npz, <root>/manifest.json), the read-only truth package ($MMAEMU_TRUTH_DIR);
- the truth package's writer (tcemu_ada_truth.format.RecordingWriter), imported read-only with bytecode writing off;
- live hardware/software metadata;
- per-call cuBLASLt capture: CUBLASLT_LOG_LEVEL=2 Trace lines delivered synchronously through
  cublasLtLoggerSetCallback (the same lines also go to CUBLASLT_LOG_FILE), parsed with the regex logic of
  the tcemu audit's verify-c1-claims-0/parse_lt.py (+ workspace size, as its build/gpu/ltconfig.py);
- subprocess runner (one subprocess per workspace size: torch caches CUBLASLT_WORKSPACE_SIZE at first use);
- the GPU-run log (gpu_runs.jsonl in the recording root; every locked run with wall seconds).
Nothing here imports tcemu.
"""
from __future__ import annotations

import ctypes as C
import json
import os
import platform as _platform
import re
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

REPO = Path(__file__).resolve().parents[2]
REC_ROOT = REPO / "recordings" / "rtx4060_2026-10"
REC_DATA = REC_ROOT / "data"
# The hardware ground truth (not published; maintainer host only): $MMAEMU_TRUTH_DIR. Without it the truth-based
# steps stop with a message (truth_format / truth_loader); paths below it then simply do not exist.
TRUTH = Path(os.environ.get("MMAEMU_TRUTH_DIR") or "MMAEMU_TRUTH_DIR-is-not-set")
# Staging area of the recorders before they write into recordings/: $MMAEMU_RECORD_STAGE (default build/record-stage).
STAGE = Path(os.environ.get("MMAEMU_RECORD_STAGE") or REPO / "build" / "record-stage")
MAX_FILE_BYTES = 15e6          # every npz part stays well below the 20 MB per-file limit
NVSMI = "/usr/lib/wsl/lib/nvidia-smi" if Path("/usr/lib/wsl/lib/nvidia-smi").exists() else "nvidia-smi"


def _cublaslt_header() -> Path:
    """cublasLt.h of a CUDA 12.8 toolkit (tile/stage enum names): $MMAEMU_CUBLASLT_HEADER, else under $CUDA_HOME."""
    if os.environ.get("MMAEMU_CUBLASLT_HEADER"):
        return Path(os.environ["MMAEMU_CUBLASLT_HEADER"])
    home = Path(os.environ.get("CUDA_HOME") or "/usr/local/cuda")
    for sub in ("include", "targets/x86_64-linux/include", "targets/sbsa-linux/include"):
        if (home / sub / "cublasLt.h").exists():
            return home / sub / "cublasLt.h"
    return home / "include" / "cublasLt.h"


LT_HEADER = _cublaslt_header()

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _require_truth() -> None:
    if not TRUTH.is_dir():
        raise SystemExit("this step needs the hardware ground truth: set MMAEMU_TRUTH_DIR to its directory "
                         f"(now {os.environ.get('MMAEMU_TRUTH_DIR')!r})")


def truth_format():
    """The truth package's format module (writer, sha256 helpers), imported read-only."""
    _require_truth()
    if str(TRUTH) not in sys.path:
        sys.path.append(str(TRUTH))
    from tcemu_ada_truth import format as F  # noqa: PLC0415
    return F


def truth_loader():
    _require_truth()
    if str(TRUTH) not in sys.path:
        sys.path.append(str(TRUTH))
    from tcemu_ada_truth import loader as L  # noqa: PLC0415
    return L


# ---- metadata -----------------------------------------------------------------------------------------------------
def hardware(device: int = 0) -> dict:
    """Live hardware/software metadata (never the writer's HW_RTX4060 default)."""
    import torch  # noqa: PLC0415
    from mmaemu.platform import cublaslt  # noqa: PLC0415
    hw = {"gpu": None, "arch": None, "sm_count": None, "driver": None, "cuda": torch.version.cuda,
          "cublaslt": None, "cublaslt_lib": None, "torch": torch.__version__, "python": sys.version.split()[0],
          "os": f"{_platform.system()} {_platform.release()}"}
    try:
        hw["gpu"] = cublaslt.device_name(device)
        cc = cublaslt.device_capability(device)
        hw["arch"] = f"sm_{cc[0]}{cc[1]}"
        hw["sm_count"] = cublaslt.device_sm_count(device)
        hw["cublaslt"] = cublaslt.version()
        hw["cublas"] = hw["cublaslt"]
        hw["cublaslt_lib"] = cublaslt.library_path()
    except Exception as e:  # noqa: BLE001
        hw["error"] = repr(e)
    try:
        q = subprocess.run([NVSMI, "--query-gpu=name,driver_version,clocks.sm,temperature.gpu",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout.strip()
        hw["nvidia_smi"] = q
        hw["driver"] = q.split(",")[1].strip() if q else None
    except Exception as e:  # noqa: BLE001
        hw["nvidia_smi_error"] = repr(e)
    return hw


# ---- cuBLASLt trace parsing (parse_lt.py logic) --------------------------------------------------------------------
PAT = re.compile(r"\[cublasLtMatmul\] .*?Adesc=\[type=(\S+) rows=(\d+) cols=(\d+) ld=(\d+)[^\]]*\].*?"
                 r"Bdesc=\[type=(\S+) rows=(\d+) cols=(\d+) ld=(\d+)[^\]]*\].*?"
                 r"Cdesc=\[type=(\S+) rows=(\d+) cols=(\d+) ld=(\d+)[^\]]*\].*?"
                 r"Ddesc=\[type=(\S+) rows=(\d+) cols=(\d+) ld=(\d+)[^\]]*\] computeDesc=\[([^\]]*)\] "
                 r"algo=\[([^\]]*)\](?:.*?workSpaceSizeInBytes=(\d+))?")


def _tile_ids() -> dict:
    ids = {}
    if LT_HEADER.exists():
        t = LT_HEADER.read_text()
        for m in re.finditer(r"CUBLASLT_(MATMUL_(?:TILE|STAGES)_\w+)\s*=\s*(\d+),", t):
            ids[m.group(1)] = int(m.group(2))
    return ids


_TILE_IDS = None


def parse_trace(line: str) -> dict | None:
    """One [Trace][cublasLtMatmul] line (or the callback message prefixed with '[cublasLtMatmul] ') -> dict."""
    global _TILE_IDS
    m = PAT.search(line)
    if not m:
        return None
    g = m.groups()
    A = dict(type=g[0], rows=int(g[1]), cols=int(g[2]), ld=int(g[3]))
    B = dict(type=g[4], rows=int(g[5]), cols=int(g[6]), ld=int(g[7]))
    Cd = dict(type=g[8], rows=int(g[9]), cols=int(g[10]), ld=int(g[11]))
    D = dict(type=g[12], rows=int(g[13]), cols=int(g[14]), ld=int(g[15]))
    comp, algo, wsb = g[16], g[17], g[18]
    f = dict(re.findall(r"(\w+)=(\S+)", algo))
    if _TILE_IDS is None:
        _TILE_IDS = _tile_ids()
    out = dict(Adesc=A, Bdesc=B, Cdesc=Cd, Ddesc=D, computeDesc=comp, lt_algo_string=algo,
               fast="fastAccumulationFlag=1" in comp, d_scale="dScalePointer" in comp,
               transa="transa=OP_T" in comp, transb="transb=OP_T" in comp,
               algo=int(f.get("algoId", -1)), tile=f.get("tile"), stages=f.get("stages"),
               tile_id=_TILE_IDS.get(f.get("tile", ""), None), stages_id=_TILE_IDS.get(f.get("stages", ""), None),
               numSplitsK=int(f.get("numSplitsK", 1)), customOption=int(f.get("customOption", 0)),
               reductionScheme=f.get("reductionScheme"), ctaSwizzling=int(f.get("ctaSwizzling", 0)),
               workspace_bytes=int(wsb) if wsb else None)
    # torch._scaled_mm(a[M,K], b[K,N]): A = mat2 (K x N, transa = T), B = mat1 (K x M), D = N x M
    out["MNK"] = (B["cols"], A["cols"], A["rows"]) if out["transa"] else None
    return out


def splitk_of(cfg: dict | None) -> dict:
    """Logged config -> the truth format's params.splitk ({"S": 1} or {"S", "scheme"}); customOption 0/2 = serial,
    1 = pairwise (splitk_model.scheme_of_custom_option)."""
    if cfg is None:
        return {"S": 1, "source": "no cuBLASLt matmul call logged"}
    S = int(cfg["numSplitsK"])
    if S <= 1:
        return {"S": 1, "source": "logged cuBLASLt config (CUBLASLT_LOG_LEVEL=2)"}
    co = int(cfg["customOption"])
    scheme = {0: "serial", 2: "serial", 1: "pairwise"}.get(co, "unknown")
    return {"S": S, "scheme": scheme, "source": "logged cuBLASLt config (CUBLASLT_LOG_LEVEL=2)"}


def cublas_params(cfg: dict | None) -> dict | None:
    if cfg is None:
        return None
    keep = ("algo", "tile", "stages", "tile_id", "stages_id", "numSplitsK", "customOption", "reductionScheme",
            "ctaSwizzling", "workspace_bytes", "lt_algo_string", "d_scale")
    out = {k: cfg[k] for k in keep}
    out["Cdesc_type"] = cfg["Cdesc"]["type"]
    out["Ddesc_type"] = cfg["Ddesc"]["type"]
    out["source"] = "CUBLASLT_LOG_LEVEL=2 trace of this call (logger callback)"
    return out


class LtCapture:
    """Synchronous capture of cuBLASLt log messages through cublasLtLoggerSetCallback on the library torch loaded.
    CUBLASLT_LOG_LEVEL must be >= 2 in the environment (the file log keeps working)."""

    def __init__(self):
        import torch  # noqa: PLC0415,F401  (make sure torch's libcublasLt is mapped first)
        from mmaemu.platform import cublaslt  # noqa: PLC0415
        self.msgs: list[tuple[int, str, str]] = []
        self._lib = C.CDLL(cublaslt.library_path())
        CB = C.CFUNCTYPE(None, C.c_int, C.c_char_p, C.c_char_p)
        self._cb = CB(self._on)
        r = self._lib.cublasLtLoggerSetCallback(self._cb)
        if r != 0:
            raise RuntimeError(f"cublasLtLoggerSetCallback failed: {r}")

    def _on(self, level, fn, msg):
        self.msgs.append((int(level), (fn or b"").decode(errors="replace"), (msg or b"").decode(errors="replace")))

    def mark(self) -> int:
        return len(self.msgs)

    def matmuls_since(self, mark: int) -> list[dict]:
        out = []
        for lvl, fn, msg in self.msgs[mark:]:
            if fn == "cublasLtMatmul" and "Adesc=[" in msg:
                p = parse_trace("[cublasLtMatmul] " + msg)
                if p is not None:
                    out.append(p)
        return out

    def other_since(self, mark: int) -> list[str]:
        return [f"[{lvl}][{fn}] {msg}"[:300] for lvl, fn, msg in self.msgs[mark:]
                if not (fn == "cublasLtMatmul" and "Adesc=[" in msg) and lvl in (1, 3)]


# ---- subprocess runner + GPU run log -------------------------------------------------------------------------------
def worker_env(ws_bytes: int | None, log_file: Path | None) -> dict:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(REPO)
    if ws_bytes is not None:
        if ws_bytes % 1024:
            raise ValueError("CUBLASLT_WORKSPACE_SIZE is in KiB")
        env["CUBLASLT_WORKSPACE_SIZE"] = str(ws_bytes // 1024)
    if log_file is not None:
        env["CUBLASLT_LOG_LEVEL"] = "2"
        env["CUBLASLT_LOG_FILE"] = str(log_file)
    return env


def run_worker(argv: list[str], ws_bytes: int | None, log_file: Path | None, timeout: float = 540) -> int:
    t0 = time.monotonic()
    r = subprocess.run([sys.executable, *argv], env=worker_env(ws_bytes, log_file), timeout=timeout)
    print(f"  worker {' '.join(argv[:3])} ws={ws_bytes} -> rc={r.returncode} ({time.monotonic() - t0:.1f}s)",
          flush=True)
    return r.returncode


def log_gpu_run(item: str, t_start: float, t_end: float, ok: bool, note: str = "") -> None:
    REC_ROOT.mkdir(parents=True, exist_ok=True)
    with open(REC_ROOT / "gpu_runs.jsonl", "a") as f:
        f.write(json.dumps(dict(item=item, start=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_start)),
                                end=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_end)),
                                wall_seconds=round(t_end - t_start, 1), ok=ok, note=note)) + "\n")


# ---- bit helpers ---------------------------------------------------------------------------------------------------
def tbits(t):
    """torch tensor -> numpy unsigned bit patterns (contiguous copy on CPU)."""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    t = t.detach().cpu().contiguous()
    it = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()]
    nt = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[t.element_size()]
    return t.view(it).numpy().view(nt).copy()


def f32bits(x: float) -> int:
    import numpy as np  # noqa: PLC0415
    return int(np.array(x, dtype=np.float32).view(np.uint32))


DTYPE_NAMES = {"float32": "float32", "bfloat16": "bfloat16", "float16": "float16", "float8_e4m3fn": "float8_e4m3fn",
               "float8_e5m2": "float8_e5m2", "float8_e4m3fnuz": "float8_e4m3fnuz",
               "float8_e5m2fnuz": "float8_e5m2fnuz", "float64": "float64"}


def tdtype(name: str):
    import torch  # noqa: PLC0415
    return getattr(torch, name)


def fmt_of(name: str) -> str:
    return {"float8_e4m3fn": "e4m3", "float8_e5m2": "e5m2", "float8_e4m3fnuz": "e4m3fnuz",
            "float8_e5m2fnuz": "e5m2fnuz"}.get(name, name)


def err_text(e: BaseException) -> str:
    first = (str(e).strip().splitlines() or [""])[0]
    return f"{type(e).__name__}: {first}"[:500]
