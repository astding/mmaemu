"""MmaemuTruthModel: a replay model for the hardware truth (tcemu_ada_truth.replay, $MMAEMU_TRUTH_DIR) that predicts
every recording with mmaemu. Repo tool only; nothing here ships in the wheel.

Rules (each one is checked by tools/replay_truth.py or tests/test_truth_full.py):
- mma outputs -> mmaemu.model.dot (preset "ada"); HMMA and fp16-accumulator mma are n/a, as in FixedModel.
- gemm outputs -> mmaemu, with the recording's own split-K passed explicitly (never splitk="auto"):
    * argument validation: mmaemu.explain() on a call with the HARDWARE call's shape and dtypes (zero operands);
      for torch._scaled_mm-level outputs ReferenceRejectsError -> ModelRejects; for every level
      UnsupportedFeatureError -> NotApplicable. Exceptions are mapped by EXACT class; anything else propagates.
      (`except RuntimeError` would turn UnsupportedFeatureError, a NotImplementedError and so a RuntimeError, into a
      hardware rejection; it is never used here.)
    * n/a exactly as FixedModel: torch row-wise stride bug, split-K "unknown" or an unknown reduction scheme (the
      replay itself marks cuBLASLt workspace dumps n/a);
    * small GEMMs (M*N*K <= 3e7): mmaemu.scaled_mm (when the stored operands form a valid torch call) or
      mmaemu.accumulate + mmaemu.model.epilogue.apply, on the CPU reference backend;
    * large GEMMs: mmaemu's CUDA backend when the adapter was built with device="cuda" (caller holds the GPU lock),
      otherwise mmaemu's own plan (make_plan / execute_plan / epilogue.apply) with a ChainFn built on the truth's
      AVX-512 oracle tcemu_ada_truth.fastacc.chain_acc. Each such case is cross-checked against the mmaemu
      reference on 2 rows x 32 columns (ctx.checks, summed by the replay as fastacc_checks).
- GemmCtx.acc / _acc_ref / _acc_fast are never called (they import tcemu.reference).
- fastacc.lib() / fastacc.available() are never called (lib() recompiles into $TRUTH/build/lib when the shipped .so is
  older than its source). The adapter ctypes-loads $TRUTH/build/lib/libfastref3.so if it is at least as new as
  csrc/fastref3.c, else a private build in $MMAEMU_CACHE_DIR/truth_fastref3/, and sets fastacc._lib to it with the
  argtypes of fastacc.lib().
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402
import torch  # noqa: E402

import mmaemu  # noqa: E402
from mmaemu import errors as E  # noqa: E402
from mmaemu.model import epilogue as mm_epilogue  # noqa: E402
from mmaemu.model import plan as mm_plan  # noqa: E402

REF_LIMIT = 3e7                    # M*N*K above which the large-GEMM path is used (as in the truth replay)
SAMPLE_ROWS, SAMPLE_COLS = 2, 32
MMA_ELEMS = 1 << 23                # operand elements per mmaemu.model.dot call (bounds float64 temporaries)
FP8 = ("e4m3", "e5m2")
TD = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16,
      "float8_e4m3fn": torch.float8_e4m3fn, "float8_e5m2": torch.float8_e5m2}
TD_FP8 = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}
ONE = 0x3F800000


def truth_dir() -> Path:
    d = os.environ.get("MMAEMU_TRUTH_DIR")
    if not d:
        raise RuntimeError("MMAEMU_TRUTH_DIR is not set")
    return Path(d).resolve()


def cache_dir() -> Path:
    d = os.environ.get("MMAEMU_CACHE_DIR")
    if d:
        return Path(d)
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "mmaemu"


def truth_modules():
    """(models, fastacc) of the truth package in $MMAEMU_TRUTH_DIR (imported read-only, no bytecode)."""
    root = truth_dir()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from tcemu_ada_truth import fastacc, models  # noqa: WPS433
    if Path(models.__file__).resolve().parents[1] != root:
        raise RuntimeError(f"tcemu_ada_truth imported from {models.__file__}, not from {root}")
    return models, fastacc


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_fastref(fastacc) -> dict:
    """ctypes-load the AVX-512 chain oracle WITHOUT fastacc.lib(); set fastacc._lib. Returns provenance."""
    src = Path(fastacc.SRC)
    shipped = Path(fastacc.LIB)
    if shipped.exists() and shipped.stat().st_mtime >= src.stat().st_mtime:
        path, how = shipped, "shipped (not older than its source)"
    else:
        d = cache_dir() / "truth_fastref3"
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"libfastref3-{sha256_file(src)[:16]}.so"
        how = "private build in MMAEMU_CACHE_DIR"
        if not path.exists():
            with tempfile.TemporaryDirectory(dir=d) as td:
                tmp = Path(td) / path.name
                subprocess.check_call(["gcc", "-O3", "-march=native", "-fopenmp", "-ffp-contract=off", "-fPIC",
                                       "-shared", str(src), "-o", str(tmp)])
                shutil.move(str(tmp), str(path))
    lib = ctypes.CDLL(str(path))
    lib.chain_gemm_avx.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int] * 10 + [ctypes.c_void_p]
    fastacc._lib = lib
    return dict(path=str(path), how=how, sha256=sha256_file(path), source=str(src), source_sha256=sha256_file(src))


def tbits(t: torch.Tensor) -> np.ndarray:
    t = t.detach().cpu().contiguous()
    it = {1: torch.uint8, 2: torch.int16, 4: torch.int32}[t.element_size()]
    nt = {1: np.uint8, 2: np.uint16, 4: np.uint32}[t.element_size()]
    return t.view(it).numpy().view(nt)


def f32(bits: int, device="cpu") -> torch.Tensor:
    return torch.from_numpy(np.array([bits], np.uint32).view(np.int32).copy()).view(torch.float32) \
        .reshape(()).to(device)


def codes(a: np.ndarray, fmt: str, device="cpu") -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.uint8).copy()).view(TD_FP8[fmt]).to(device)


def splitk_of(p):
    sk = p.get("splitk")
    if isinstance(sk, dict) and sk.get("S") and int(sk["S"]) > 1:
        if sk.get("scheme") not in ("serial", "pairwise"):
            return ("na", f"split-K S={sk['S']} with unknown reduction scheme")
        return (int(sk["S"]), sk["scheme"])
    if sk == "unknown":
        return ("na", "split-K configuration not recorded")
    return (1, "none")


def splitk_arg(sk):
    return 1 if sk[0] == 1 else (int(sk[0]), sk[1])


def bias_name(p):
    return p.get("bias_input") or "bias"


class MmaemuTruthModel:
    name = "mmaemu"

    def __init__(self, *, device: str = "cpu", cuda_backend: str = "auto", ref_limit: float = REF_LIMIT,
                 load_fastacc: bool = True):
        self.models, self.fastacc = truth_modules()
        self.NotApplicable, self.ModelRejects = self.models.NotApplicable, self.models.ModelRejects
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested but CUDA is unavailable")
        self.cuda_backend = cuda_backend
        self.ref_limit = ref_limit
        self.fastref = load_fastref(self.fastacc) if (load_fastacc and self.device.type == "cpu") else None
        self.current = None              # set by replay_truth's eval_output hook: the truth Output being replayed
        self.sources = defaultdict(Counter)        # recording id -> accumulator source -> outputs
        self.na_log = []                           # every NotApplicable / ModelRejects, by output id
        self.description = ("mmaemu " + getattr(mmaemu, "__version__", "?") + " via tools/truth_adapter.py: "
                            "mma -> mmaemu.model.dot; gemm -> mmaemu.explain (validation on the hardware call "
                            "shape) + scaled_mm / accumulate + epilogue.apply with the recorded split-K; large "
                            f"GEMMs: {'CUDA backend' if self.device.type == 'cuda' else 'mmaemu plan + fastacc ChainFn'}")

    # -- helpers ---------------------------------------------------------------------------------------------------
    def _rid(self):
        o = self.current
        return o.rec.id if o is not None else "?"

    def _oid(self):
        o = self.current
        return f"{o.rec.id}/{o.case.id}/{o.name}" if o is not None else "?"

    def _na(self, reason, category="fixed-set"):
        self.na_log.append(dict(id=self._oid(), reason=reason, category=category,
                                values=int(np.prod(self.current.shape)) if (self.current is not None and
                                                                             self.current.shape) else 0,
                                hash_only=bool(self.current is not None and not self.current.stored
                                               and not self.current.hw_error),
                                hw_error=bool(self.current is not None and self.current.hw_error)))
        raise self.NotApplicable(reason)

    # -- mma -------------------------------------------------------------------------------------------------------
    def mma(self, a, b, c, p):
        fa, fb = p.get("fmt_a", "e4m3"), p.get("fmt_b", "e4m3")
        if fa not in FP8 or fb not in FP8:
            self._na("HMMA bf16/fp16 not modelled by tcemu")
        if p.get("acc", "f32") != "f32":
            self._na("FP8 mma with fp16 accumulator not modelled by tcemu")
        self.sources[self._rid()]["mmaemu.model.dot"] += 1
        cf = torch.from_numpy(np.ascontiguousarray(c).view(np.int32).copy()).view(torch.float32)
        try:
            if p.get("layout", "vector") == "tile":
                T, mi, K = a.shape
                nj = b.shape[1]
                out = np.empty((T, mi, nj), np.uint32)
                step = max(1, MMA_ELEMS // (mi * nj * K))
                for t0 in range(0, T, step):
                    at, bt = a[t0:t0 + step], b[t0:t0 + step]
                    n = at.shape[0]
                    A = np.broadcast_to(at[:, :, None, :], (n, mi, nj, K)).reshape(-1, K)
                    B = np.broadcast_to(bt[:, None, :, :], (n, mi, nj, K)).reshape(-1, K)
                    d = mmaemu.model.dot(codes(A, fa), codes(B, fb), cf[t0:t0 + step].reshape(-1), preset="ada")
                    out[t0:t0 + step] = tbits(d).reshape(n, mi, nj)
                return out
            n, K = a.shape
            out = np.empty((n,), np.uint32)
            step = max(1, MMA_ELEMS // max(K, 1))
            for i0 in range(0, n, step):
                d = mmaemu.model.dot(codes(a[i0:i0 + step], fa), codes(b[i0:i0 + step], fb), cf[i0:i0 + step],
                                     preset="ada")
                out[i0:i0 + step] = tbits(d)
            return out
        except E.UnsupportedFeatureError as e:
            if type(e) is not E.UnsupportedFeatureError:
                raise
            self._na(f"mmaemu: UnsupportedFeatureError: {e}", "mmaemu-unsupported")

    # -- gemm ------------------------------------------------------------------------------------------------------
    def _call_shape(self, ctx, p):
        hs = p.get("hw_shape")
        if hs:
            return [int(x) for x in hs]
        o = self.current
        hc = (o.case.params.get("hw_call_shape") if o is not None else None) or p.get("hw_call_shape")
        if hc:
            return [int(hc["M"]), int(hc["N"]), int(hc["K"])]
        a, b = ctx.inp["a"], ctx.inp["b"]
        return [a.shape[0], b.shape[0], a.shape[1]]

    def _scale_bits(self, ctx, p, name):
        v = p.get(name)
        if v is None and name in ctx.inp:
            arr = ctx.inp[name]
            if arr.size == 1:
                return int(arr.reshape(-1)[0])
        return ONE if v is None else int(v)

    def _validate(self, ctx, p, level):
        """mmaemu.explain() on the hardware call (zero operands of its shape and dtypes)."""
        fa, fb = p.get("fmt_a", "e4m3"), p.get("fmt_b", "e4m3")
        M, N, K = self._call_shape(ctx, p)
        a = torch.zeros(M, K, dtype=torch.uint8).view(TD_FP8[fa])
        b = torch.zeros(N, K, dtype=torch.uint8).view(TD_FP8[fb])
        if p.get("scaling", "tensorwise") == "rowwise":
            sa, sb = torch.ones(M, 1), torch.ones(1, N)
        else:
            sa, sb = f32(self._scale_bits(ctx, p, "scale_a")), f32(self._scale_bits(ctx, p, "scale_b"))
        bias = None
        if bias_name(p) in ctx.inp:
            bias = torch.zeros(N, dtype=TD[p.get("bias_dtype", "bfloat16")])
        sr = None if p.get("scale_result") is None else f32(int(p["scale_result"]))
        plan = mmaemu.explain(a, b.t(), sa, sb, bias, sr, TD[p["out_dtype"]], bool(p.get("fast_accum", True)),
                              splitk=1)
        v = plan.validation
        if v == "ok":
            return
        cls, msg = v
        torch_level = level.startswith("torch._scaled_mm")
        if cls == "UnsupportedFeatureError":
            cat = ("row-wise" if p.get("scaling", "tensorwise") == "rowwise"
                   else ("bias" if torch_level else "fused-bias") if bias is not None else "other")
            self._na(f"mmaemu: UnsupportedFeatureError: {msg}", cat)
        if cls == "ReferenceRejectsError":
            if torch_level:
                self.na_log.append(dict(id=self._oid(), reason=f"ModelRejects: {msg}", category="rejects",
                                        values=0, hash_only=False,
                                        hw_error=bool(self.current is not None and self.current.hw_error)))
                raise self.ModelRejects(f"mmaemu: ReferenceRejectsError: {msg}")
            return          # cuBLASLt / TE / emulator level: torch's argument rules do not apply; compute
        raise RuntimeError(f"{self._oid()}: mmaemu.explain() returned an unexpected validation {v!r}")

    def gemm(self, ctx, p):
        if "torch_rowwise_stride_bug" in set(p.get("tags") or []):
            self._na("torch 2.7.1 row-wise stride bug (C4-F11): the hardware output is not the GEMM of the stored "
                     "operands")
        level = str(p.get("level") or (self.current.rec.level if self.current is not None else "") or "")
        self._validate(ctx, p, level)
        sk = splitk_of(p)
        if sk[0] == "na":
            self._na(sk[1])
        fa, fb = p.get("fmt_a", "e4m3"), p.get("fmt_b", "e4m3")
        od = TD[p["out_dtype"]]
        fast = bool(p.get("fast_accum", True))
        sa, sb = f32(self._scale_bits(ctx, p, "scale_a")), f32(self._scale_bits(ctx, p, "scale_b"))
        sr = None if p.get("scale_result") is None else f32(int(p["scale_result"]))
        A, B = ctx.A, ctx.B
        M, N, K = A.shape[0], B.shape[0], A.shape[1]
        diag = p.get("out_select") == "diagonal"
        rid = self._rid()
        try:
            if M * N * K <= self.ref_limit:
                torch_call = (level.startswith("torch._scaled_mm") and not diag and N % 16 == 0 and K % 16 == 0
                              and K > 0)
                if torch_call:
                    y = mmaemu.scaled_mm(codes(A, fa), codes(B, fb).t(), sa, sb, None, sr, od, fast,
                                         splitk=splitk_arg(sk), backend="reference")
                    src = "mmaemu.scaled_mm(reference)"
                else:
                    acc = mmaemu.accumulate(codes(A, fa), codes(B, fb), use_fast_accum=fast, splitk=splitk_arg(sk),
                                            backend="reference")
                    y = mm_epilogue.apply(acc, sa, sb, out_dtype=od, scale_result=sr)
                    src = "mmaemu.accumulate(reference)+epilogue"
            else:
                if self.device.type == "cuda":
                    acc = mmaemu.accumulate(codes(A, fa, self.device), codes(B, fb, self.device),
                                            use_fast_accum=fast, splitk=splitk_arg(sk),
                                            backend=self.cuda_backend).cpu()
                    src = f"mmaemu.accumulate(cuda:{self.cuda_backend})+epilogue"
                else:
                    acc = self._acc_fastacc(A, B, fa, fb, fast, sk)
                    src = "mmaemu.plan+fastacc ChainFn+epilogue"
                self._crosscheck(ctx, A, B, fa, fb, fast, sk, acc)
                y = mm_epilogue.apply(acc, sa, sb, out_dtype=od, scale_result=sr)
        except E.UnsupportedFeatureError as e:
            if type(e) is not E.UnsupportedFeatureError:
                raise
            self._na(f"mmaemu: UnsupportedFeatureError: {e}", "mmaemu-unsupported")
        except E.ReferenceRejectsError as e:
            if type(e) is not E.ReferenceRejectsError:
                raise
            raise self.ModelRejects(f"mmaemu: ReferenceRejectsError at compute time: {e}")
        ctx.backend = src
        self.sources[rid][src] += 1
        return tbits(y)

    def _acc_fastacc(self, A, B, fa, fb, fast, sk):
        """mmaemu's plan (sanitise, gather, chains, fp32 combination, IEEE patch) with fastacc as the chain."""
        def chain_fn(a_u8: torch.Tensor, b_u8: torch.Tensor) -> torch.Tensor:
            r = self.fastacc.chain_acc(a_u8.cpu().numpy(), b_u8.cpu().numpy(), fa, fb, restart=0)
            return torch.from_numpy(r)
        plan = mm_plan.make_plan(A.shape[1], fast=fast, S=int(sk[0]), scheme=sk[1], preset="ada")
        a_u8 = torch.from_numpy(np.ascontiguousarray(A, dtype=np.uint8).copy())
        b_u8 = torch.from_numpy(np.ascontiguousarray(B, dtype=np.uint8).copy())
        return mm_plan.execute_plan(plan, a_u8, b_u8, fmt_a=fa, fmt_b=fb, chain_fn=chain_fn)

    def _crosscheck(self, ctx, A, B, fa, fb, fast, sk, acc):
        """Bitwise check of the large-GEMM accumulator against the mmaemu reference on a row/column sample."""
        M, N = A.shape[0], B.shape[0]
        ri = sorted({0, M - 1})[:SAMPLE_ROWS]
        ci = np.unique(np.linspace(0, N - 1, min(N, SAMPLE_COLS)).astype(int))
        ref = mmaemu.accumulate(codes(A[ri], fa), codes(B[ci], fb), use_fast_accum=fast, splitk=splitk_arg(sk),
                                backend="reference")
        got = acc[ri][:, torch.from_numpy(ci)]
        rb, gb = tbits(ref), tbits(got)
        ctx.checks.append(dict(case=ctx.label, fast=fast, S=int(sk[0]), scheme=sk[1], n=int(rb.size),
                               mismatches=int((rb != gb).sum())))
