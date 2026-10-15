#!/usr/bin/env python3
"""Build tests/data/hw_mini/ -- the committed RTX 4060 hardware subset replayed by CI tier T1 (docs/testing.md).

usage (from the repo root; the truth package is only READ, with bytecode writing disabled):
    PYTHONDONTWRITEBYTECODE=1 python tools/export_hw_mini.py --truth $MMAEMU_TRUTH_DIR \
        [--ws2 recordings/rtx4060_2026-10] [--out tests/data/hw_mini] [--dry-run]

Selection (every item is listed in the coverage report, tests/data/hw_mini/COVERAGE.txt + coverage.json):
  one       >= 1 output of every non-table truth recording that mmaemu v0.1 can evaluate (in scope or an expected
            error); recordings with no such output are listed as out of scope with the reason;
  hw_error  every hw_error output that the audit's FixedModel reproduces (113): replayed as a call on zero
            operands of the hardware call's shape and dtypes, expected mmaemu error by exact class;
  combo     for every hardware-replayed (fast, S, scheme, out_dtype, fmt_a, fmt_b), the smallest recorded case
            (M*N*K <= 3e7 preferred), restricted to a few output rows/columns (the split-K partition depends on K
            only, so any row/column subset of a recorded output is reproduced by the GEMM of the subset operands);
  c2fam     the c2 T_* and S16_* families (>= 4096 vectors each; all of S16_zeros);
  epilogue  every epilogue.* gemm case whose stored output has M*N <= 65,536 (all its in-scope outputs);
  project   misc.tcemu_tests_recorded_hw in full (it replaces tests/data/hw/*.pt);
  evo2      rows 0..31 of 3 Evo 2 1B projections (evo2.evo2_1b_w2k calls 0, 10, 20; a column subset, see
            --evo2-cols, because a full weight is 11 MB);
  extra     a few small special-value outputs (c3.gemm_table, scope-boundary cases: bias, row-wise);
  ws2       from the pre-swap recordings (--ws2, if it exists): R1 in full, >= 1 case per R4 combination, >= 1 output of
            every other recording.
The exporter fails if the directory would exceed --budget-mb (default 20, as `du` counts MiB).

Format (mmaemu-hwmini/1): index.json + <group>.npz shards (allow_pickle=False, raw bit patterns only, arrays
deduplicated by content). See docs/testing.md. Expected mmaemu error classes (scope rules of docs/semantics.md):
  hardware/torch rejected the call (FixedModel reproduces it) -> ReferenceRejectsError
  bias given and accepted by torch, row-wise scales, requires_grad under grad mode -> UnsupportedFeatureError
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import time
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True          # never write __pycache__ into the truth tree

import numpy as np  # noqa: E402

FORMAT = "mmaemu-hwmini/1"
REPO = Path(__file__).resolve().parents[1]


def _repo_rel(p: Path) -> str:
    """repo-relative for a path in the repository (no host path in the test data), the path itself otherwise."""
    try:
        return Path(p).resolve().relative_to(REPO).as_posix()
    except ValueError:
        return str(p)


ONE = 0x3F800000
FP8 = ("e4m3", "e5m2")
ITEM = {"uint8": 1, "uint16": 2, "uint32": 4, "int64": 8, "uint64": 8, "int32": 4, "int16": 2, "int8": 1}
OUT_DTYPES = ("float32", "bfloat16", "float16", "float8_e4m3fn", "float8_e5m2")


# ---------------------------------------------------------------------------------------------------------------
# truth access (read-only)
def import_loader(root: Path):
    root = Path(root).resolve()
    if not (root / "tcemu_ada_truth" / "loader.py").exists():
        raise SystemExit(f"{root}: not a tcemu_ada_truth checkout")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from tcemu_ada_truth import loader  # noqa: WPS433
    if Path(loader.__file__).resolve().parents[1] != root:
        raise SystemExit(f"imported tcemu_ada_truth from {loader.__file__}, expected {root}")
    return loader


def load_fastref_readonly(fastacc) -> str:
    """ctypes-load the truth's fastref3 library WITHOUT fastacc.lib() (which may recompile into the truth tree):
    the shipped .so only if it is not older than its source, else a private build in $MMAEMU_CACHE_DIR."""
    import ctypes
    import shutil
    import subprocess
    import tempfile
    src, shipped = Path(fastacc.SRC), Path(fastacc.LIB)
    if shipped.exists() and shipped.stat().st_mtime >= src.stat().st_mtime:
        path = shipped
    else:
        cache = Path(os.environ.get("MMAEMU_CACHE_DIR") or Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
                     / "mmaemu") / "truth_fastref3"
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / f"libfastref3-{sha256_file(src)[:16]}.so"
        if not path.exists():
            with tempfile.TemporaryDirectory(dir=cache) as td:
                tmp = Path(td) / path.name
                subprocess.check_call(["gcc", "-O3", "-march=native", "-fopenmp", "-ffp-contract=off", "-fPIC",
                                       "-shared", str(src), "-o", str(tmp)])
                shutil.move(str(tmp), str(path))
    lib = ctypes.CDLL(str(path))
    lib.chain_gemm_avx.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_int] * 10 + [ctypes.c_void_p]
    fastacc._lib = lib
    return str(path)


def portable_path(p: Path, var: str) -> str:
    """`$VAR/<relative path>` for a path under the directory named by the environment variable `var`, so that no
    host path enters the committed test data; the path itself otherwise."""
    root = os.environ.get(var)
    try:
        rel = Path(p).resolve().relative_to(Path(root).resolve()) if root else None
    except ValueError:
        rel = None
    if rel is None:
        return str(p)
    return f"${var}" if rel == Path(".") else f"${var}/{rel.as_posix()}"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    if a.dtype.byteorder == ">":
        a = a.astype(a.dtype.newbyteorder("<"))
    return hashlib.sha256(a.tobytes()).hexdigest()


# ---------------------------------------------------------------------------------------------------------------
# semantics of a truth output (mirrors tcemu_ada_truth.models.FixedModel; see docs/semantics.md for the mmaemu scope)
def bias_name(p):
    return p.get("bias_input") or "bias"


def splitk_of(p):
    """(S, scheme) | "unknown" | ("bad", reason)."""
    sk = p.get("splitk")
    if isinstance(sk, dict) and sk.get("S") and int(sk["S"]) > 1:
        if sk.get("scheme") not in ("serial", "pairwise"):
            return ("bad", f"split-K S={sk['S']} with unknown reduction scheme")
        return (int(sk["S"]), sk["scheme"])
    if sk == "unknown":
        return "unknown"
    return (1, "none")


def fixed_rejects(p, has_bias: bool):
    """The FixedModel's ModelRejects predicate (torch/cuBLAS argument rejections) -> message or None."""
    fa, fb, od = p.get("fmt_a", "e4m3"), p.get("fmt_b", "e4m3"), p.get("out_dtype")
    if fa == fb == "e5m2":
        return "torch._scaled_mm: e5m2 x e5m2 is not supported"
    if od == "float8_e5m2" and fa == fb == "e4m3":
        return "cuBLAS 12.8.3 / sm_89 rejects e4m3 x e4m3 -> e5m2"
    if has_bias and p.get("scaling", "tensorwise") == "tensorwise":
        bd = p.get("bias_dtype", "bfloat16")
        if od == "float32":
            return "torch._scaled_mm: bias is not supported when out_dtype is float32"
        if bd == "float32":
            return "torch._scaled_mm: bias must be Half or BFloat16"
        if od in ("bfloat16", "float16") and bd != od:
            return "torch._scaled_mm: bias dtype must equal the bf16/fp16 out_dtype"
    return None


def classify_output(rec, case, out):
    """-> dict(status=..., reason=..., error=...) with status in
       'bits'      in scope, hardware bits stored
       'hash'      in scope, hash-only (not exported)
       'error'     hardware call raised and FixedModel reproduces it -> expected ReferenceRejectsError
       'scope_out' hardware ran it; mmaemu v0.1 raises UnsupportedFeatureError (bias / row-wise)
       'na'        no semantics (same n/a set as FixedModel, plus cuBLASLt configuration validity)"""
    p = out.params
    if rec.kind == "mma":
        if p.get("fmt_a", "e4m3") not in FP8 or p.get("fmt_b", "e4m3") not in FP8:
            return dict(status="na", reason="HMMA bf16/fp16 (not FP8)")
        if p.get("acc", "f32") != "f32":
            return dict(status="na", reason="FP8 mma with fp16 accumulator")
        if out.rows:
            return dict(status="na", reason="row subset of an mma recording")
        return dict(status="bits" if out.stored else "hash")
    ins = case.spec["inputs"]
    if p.get("output_kind") == "workspace_dump":
        return dict(status="na", reason="raw cuBLASLt workspace dump (not a GEMM result)")
    if "torch_rowwise_stride_bug" in set(p.get("tags") or []):
        return dict(status="na", reason="torch 2.7.1 row-wise stride bug (C4-F11)")
    has_bias = bias_name(p) in ins
    rej = fixed_rejects(p, has_bias)
    if out.hw_error:
        if rej:
            return dict(status="error", error="ReferenceRejectsError", reason=rej)
        return dict(status="na", reason="hardware rejected a cuBLASLt configuration (config validity not modelled)")
    if rej:
        return dict(status="na", reason=f"FixedModel rejects but hardware ran: {rej}")   # must not happen
    if p.get("scaling", "tensorwise") == "rowwise":
        return dict(status="scope_out", error="UnsupportedFeatureError", reason="row-wise scales")
    if has_bias:
        return dict(status="scope_out", error="UnsupportedFeatureError", reason="bias")
    sk = splitk_of(p)
    if sk == "unknown":
        return dict(status="na", reason="split-K configuration not recorded")
    if sk[0] == "bad":
        return dict(status="na", reason=sk[1])
    return dict(status="bits" if out.stored else "hash")


def level_of(rec, out):
    return str(out.params.get("level") or rec.level or "")


def gemm_view(rec, case, out):
    """Index maps from output coordinates to stored operand rows: (arow [n_out_rows], bcol [n_out_cols] | None for
    diagonal, diagonal flag)."""
    p = out.params
    ins = case.spec["inputs"]
    M = ins["a"]["shape"][0]
    N = ins["b"]["shape"][0]
    sub = p.get("out_subset")
    diag = p.get("out_select") == "diagonal"
    if diag:
        n = int(np.prod(out.shape))
        return np.arange(n), np.arange(n), True
    if sub:
        inp = case.inputs()
        r = np.asarray(inp[sub["rows"]], dtype=np.int64) if sub.get("rows") else np.arange(M)
        c = np.asarray(inp[sub["cols"]], dtype=np.int64) if sub.get("cols") else np.arange(N)
        return r, c, False
    if out.rows:
        r0, r1 = out.rows
        return np.arange(r0, r1), np.arange(N), False
    return np.arange(M), np.arange(N), False


def scale_bits(case, p, name):
    v = p.get(name)
    if v is None and name in case.spec["inputs"] and p.get("scaling", "tensorwise") == "tensorwise":
        arr = case.inputs()[name]
        if arr.size == 1:
            return int(arr.reshape(-1)[0])
    return ONE if v is None else int(v)


def call_shape(rec, case, out):
    p = out.params
    hs = p.get("hw_shape")
    if hs:
        return [int(x) for x in hs]
    hc = case.params.get("hw_call_shape") or p.get("hw_call_shape")
    if hc:
        return [int(hc["M"]), int(hc["N"]), int(hc["K"])]
    ins = case.spec["inputs"]
    return [ins["a"]["shape"][0], ins["b"]["shape"][0], ins["a"]["shape"][1]]


def call_spec(rec, case, out):
    """How the hardware call passed its arguments (so that API-default mutants are observable)."""
    p = out.params
    spec = {"out_dtype": True, "use_fast_accum": True, "out_fill": None}
    ek = p.get("explicit_kwargs")
    if ek is not None:
        spec["out_dtype"] = "out_dtype" in ek
        spec["use_fast_accum"] = "use_fast_accum" in ek
    cv = p.get("call_variant")
    if cv == "default_all":
        spec["out_dtype"] = spec["use_fast_accum"] = False
    elif cv in ("default_fa", "default_fa_bf16"):
        spec["use_fast_accum"] = False
    if p.get("out_prefill") is not None:
        spec["out_fill"] = float(np.array([int(p["out_prefill"])], np.uint32).view(np.float32)[0])
    obp = p.get("out_buffer_prefill")
    if obp not in (None, "none", "fresh"):
        spec["out_fill"] = float(obp)
    if spec["out_fill"] is not None and not np.isfinite(spec["out_fill"]):
        spec["out_fill"] = str(spec["out_fill"])        # "nan" / "inf" / "-inf" (JSON-safe)
    return {k: v for k, v in spec.items() if v != CALL_DEFAULTS[k]}


# hw_mini call-spec defaults (omitted from index.json): out_dtype and use_fast_accum passed as keywords, scale_result
# passed iff params.scale_result_bits is not null, no bias, no out= buffer.
CALL_DEFAULTS = {"out_dtype": True, "use_fast_accum": True, "out_fill": None, "bias": None}


# ---------------------------------------------------------------------------------------------------------------
class Exporter:
    def __init__(self, truth: Path, ws2: Path | None, args):
        self.truth = Path(truth).resolve()
        self.L = import_loader(self.truth)
        self.ws2 = Path(ws2).resolve() if ws2 else None
        self.args = args
        self.arrays = OrderedDict()            # (shard, key) -> ndarray
        self.array_meta = {}                   # "shard/key" -> {sha256, dtype, shape}
        self.by_content = {}                   # (shard, sha, dtype, shape) -> key
        self.cases = []
        self.case_ids = set()
        self.cov = defaultdict(lambda: dict(kind=None, status=None, outputs_total=0, outputs_included=0,
                                            values_included=0, selectors=Counter(), reasons=Counter(), na=Counter()))
        self.macs = 0
        self.notes = []
        self.discrimination = []
        self.fast_chain = self._fast_chain()

    def _fast_chain(self):
        """The truth's AVX-512 chain oracle (loaded as tools/truth_adapter.py does: never fastacc.lib()) as a
        pairwise chain for tools/mutants/discriminate.py; None if unavailable (the torch model is used)."""
        try:
            from tcemu_ada_truth import fastacc  # noqa: WPS433
            fastacc_info = load_fastref_readonly(fastacc)
        except Exception as e:  # noqa: BLE001 -- optional speed-up only
            self.notes.append(f"fastacc unavailable for the discrimination model ({type(e).__name__}: {e}); torch model")
            return None
        info = str(fastacc_info).replace(str(self.truth), portable_path(self.truth, "MMAEMU_TRUTH_DIR"))
        self.notes.append(f"discrimination chains: truth fastacc {info}")

        def chain(a, b, fa, fb):
            r = np.empty(a.shape[0], np.float32)
            for i in range(a.shape[0]):
                r[i] = fastacc.chain_acc(a[i:i + 1], b[i:i + 1], fa, fb, restart=0)[0, 0]
            return r
        return chain

    # -- arrays ------------------------------------------------------------------------------------------------
    def put(self, shard: str, a: np.ndarray):
        a = np.ascontiguousarray(a)
        sha = sha256_array(a)
        ck = (shard, sha, str(a.dtype), a.shape)
        if ck in self.by_content:
            return [shard + ".npz", self.by_content[ck]]
        key = f"{a.dtype}_{'x'.join(map(str, a.shape))}_{sha[:16]}"
        self.by_content[ck] = key
        self.arrays[(shard, key)] = a
        self.array_meta[f"{shard}.npz/{key}"] = dict(sha256=sha, dtype=str(a.dtype), shape=list(a.shape))
        return [shard + ".npz", key]

    @staticmethod
    def shard_of(rid: str, src: str):
        g = rid.split(".", 1)[0]
        return f"ws2_{g}" if src == "ws2" else g

    def add_case(self, c: dict):
        base = c["id"]
        i = 1
        while c["id"] in self.case_ids:
            i += 1
            c["id"] = f"{base}#{i}"
        self.case_ids.add(c["id"])
        self.cases.append(c)

    # -- builders ------------------------------------------------------------------------------------------------
    def add_mma(self, src, rec, case, out, *, tiles=None, vecs=None, why):
        p = out.params
        ins = case.inputs()
        hw = out.bits()
        layout = p.get("layout", "vector")
        a, b, c = ins["a"], ins["b"], ins["c"]
        if layout == "tile":
            sel = np.arange(a.shape[0]) if tiles is None else np.asarray(tiles)
            a, b, c, hw = a[sel], b[sel], c[sel], hw[sel]
            n_out = hw.size
            K = a.shape[-1]
            src_sel = dict(tiles=[int(x) for x in sel]) if tiles is not None else None
        else:
            sel = np.arange(a.shape[0]) if vecs is None else np.asarray(vecs)
            a, b, c, hw = a[sel], b[sel], c[sel], hw[sel]
            n_out = hw.size
            K = a.shape[-1]
            src_sel = dict(vectors=[int(x) for x in sel]) if vecs is not None else None
        sh = self.shard_of(rec.id, src)
        rid = rec.id if src == "truth" else f"ws2:{rec.id}"
        self.add_case(dict(
            id=f"{rid}/{case.id}/{out.name}" + ("" if src_sel is None else f"@{why}"),
            truth_id=rid, kind="mma", why=why,
            src=dict(case=case.id, output=out.name) | ({} if src_sel is None else dict(sel=src_sel)),
            params=dict(fmt_a=p.get("fmt_a", "e4m3"), fmt_b=p.get("fmt_b", "e4m3"), layout=layout, preset="ada",
                        K=int(K)),
            inputs=dict(a=self.put(sh, a), b=self.put(sh, b), c=self.put(sh, c)),
            expect=dict(bits=self.put(sh, hw), dtype="float32", shape=list(hw.shape), sha256=sha256_array(hw))))
        self.macs += n_out * K
        self._cov(rid, rec, why, n_out)

    def gemm_params(self, rec, case, out):
        p = out.params
        sk = splitk_of(p)
        S, scheme = (sk if isinstance(sk, tuple) and sk[0] != "bad" else (1, "none"))
        return dict(fmt_a=p.get("fmt_a", "e4m3"), fmt_b=p.get("fmt_b", "e4m3"), out_dtype=p.get("out_dtype"),
                    fast=bool(p.get("fast_accum", True)), scale_a_bits=scale_bits(case, p, "scale_a"),
                    scale_b_bits=scale_bits(case, p, "scale_b"),
                    scale_result_bits=None if p.get("scale_result") is None else int(p["scale_result"]),
                    splitk=dict(S=int(S), scheme=scheme))

    def add_gemm_bits(self, src, rec, case, out, *, rows=None, cols=None, why, force_accumulate=False):
        """rows/cols: indices in OUTPUT coordinates (None = all)."""
        ins = case.inputs()
        hw = out.bits()
        arow, bcol, diag = gemm_view(rec, case, out)
        if diag:
            I = np.arange(hw.shape[0]) if rows is None else np.asarray(rows)
            a_sel, b_sel, exp = ins["a"][arow[I]], ins["b"][bcol[I]], hw[I]
        else:
            I = np.arange(hw.shape[0]) if rows is None else np.asarray(rows)
            J = np.arange(hw.shape[1]) if cols is None else np.asarray(cols)
            a_sel, b_sel = ins["a"][arow[I]], ins["b"][bcol[J]]
            exp = hw[np.ix_(I, J)]
        K = a_sel.shape[1]
        prm = self.gemm_params(rec, case, out)
        lv = level_of(rec, out)
        n_cols = b_sel.shape[0]
        api = "scaled_mm" if (lv.startswith("torch._scaled_mm") and not diag and not force_accumulate
                              and n_cols % 16 == 0 and K % 16 == 0 and K > 0 and a_sel.shape[0] >= 1) else "accumulate"
        prm["api"] = api
        if diag:
            prm["out_select"] = "diagonal"
        if api == "scaled_mm":
            prm["call"] = call_spec(rec, case, out)
        sh = self.shard_of(rec.id, src)
        rid = rec.id if src == "truth" else f"ws2:{rec.id}"
        subsel = None
        if rows is not None or cols is not None:
            subsel = dict(rows=None if rows is None else [int(x) for x in np.asarray(rows)],
                          cols=None if (cols is None or diag) else [int(x) for x in np.asarray(cols)])
        self.add_case(dict(
            id=f"{rid}/{case.id}/{out.name}" + ("" if subsel is None else f"@{why}"),
            truth_id=rid, kind="gemm", why=why,
            src=dict(case=case.id, output=out.name) | ({} if subsel is None else dict(sel=subsel)),
            params=prm,
            inputs=dict(a=self.put(sh, a_sel), b=self.put(sh, b_sel)),
            expect=dict(bits=self.put(sh, exp), dtype=prm["out_dtype"], shape=list(exp.shape),
                        sha256=sha256_array(exp))))
        n_out = exp.size
        self.macs += (n_out if diag else a_sel.shape[0] * b_sel.shape[0]) * K
        self._cov(rid, rec, why, n_out)

    def add_gemm_error(self, src, rec, case, out, *, error, reason, why, scope):
        """A call expected to raise `error` (exact mmaemu class). Operands are zeros of the hardware call's shape and
        dtypes (the rejection rules depend on dtypes, flags and shapes only); bias = zeros [N] of bias_dtype."""
        p = out.params
        prm = self.gemm_params(rec, case, out)
        prm["api"] = "scaled_mm"
        prm["call"] = call_spec(rec, case, out)
        prm.pop("level", None)
        prm["zero_operands"] = True
        M, N, K = call_shape(rec, case, out)
        prm["call_shape"] = [M, N, K]
        ins = case.spec["inputs"]
        if bias_name(p) in ins:
            prm["call"]["bias"] = "positional"
            prm["bias_dtype"] = p.get("bias_dtype", "bfloat16")
        if p.get("scaling", "tensorwise") == "rowwise":
            prm["scaling"] = "rowwise"
            # row-wise scales: [M,1] and [1,N] fp32 ones (values do not matter for the scope check)
        rid = rec.id if src == "truth" else f"ws2:{rec.id}"
        self.add_case(dict(
            id=f"{rid}/{case.id}/{out.name}", truth_id=rid, kind="gemm", why=why,
            src=dict(case=case.id, output=out.name) | ({"hw_error": out.hw_error[:160]} if out.hw_error else {}),
            params=prm, inputs={}, expect=dict(error=error), scope=scope, reason=reason))
        self._cov(rid, rec, why, 0, outputs=1)

    def _cov(self, rid, rec, why, n_values, outputs=1):
        c = self.cov[rid]
        c["kind"] = rec.kind
        c["outputs_included"] += outputs
        c["values_included"] += int(n_values)
        c["selectors"][why] += 1

    # -- selection -----------------------------------------------------------------------------------------------
    def census(self, src="truth", root=None):
        """Load every recording and classify every output."""
        L = self.L
        entries = L.list_recordings(root) if root else L.list_recordings()
        recs = []
        for e in entries:
            rec = L.load(e["id"], root=root) if root else L.load(e["id"])
            rid = rec.id if src == "truth" else f"ws2:{rec.id}"
            cv = self.cov[rid]
            cv["kind"] = rec.kind
            if rec.kind == "table":
                cv["status"] = "table"
                recs.append((rec, []))
                continue
            outs = []
            for case in rec.cases:
                for out in case.outputs:
                    cl = classify_output(rec, case, out)
                    cv["outputs_total"] += 1
                    if cl["status"] == "na":
                        cv["na"][cl["reason"]] += 1
                    outs.append((case, out, cl))
            st = Counter(cl["status"] for _, _, cl in outs)
            cv["output_status"] = dict(st)
            if st["bits"] or st["error"] or st["scope_out"]:
                cv["status"] = "in_scope" if (st["bits"] or st["error"]) else "scope_out_only"
            elif st["hash"]:
                cv["status"] = "hash_only"
            else:
                cv["status"] = "out_of_scope"
            recs.append((rec, outs))
        return recs

    @staticmethod
    def in_bytes(case):
        return sum(int(np.prod(s["shape"])) * ITEM[s["dtype"]] for s in case.spec["inputs"].values())

    def small_subset(self, rec, case, out, max_rows=4, max_cols=16):
        """Rows/cols (output coordinates) for a cheap but representative subset of a gemm output."""
        hw_shape = out.shape
        if out.params.get("out_select") == "diagonal":
            n = hw_shape[0]
            return (None if n <= max_rows * max_cols else np.linspace(0, n - 1, max_rows * max_cols).astype(int)), None
        M, N = hw_shape
        rows = None if M <= max_rows else np.arange(max_rows)
        cols = None if N <= max_cols else np.arange(max_cols)
        return rows, cols

    def select_truth(self, recs):
        a = self.args
        done_one = set()
        # ---- hw_error (113) ----------------------------------------------------------------------------------
        n_err = 0
        for rec, outs in recs:
            for case, out, cl in outs:
                if cl["status"] == "error":
                    self.add_gemm_error("truth", rec, case, out, error=cl["error"], reason=cl["reason"],
                                        why="hw_error", scope="in")
                    n_err += 1
                    done_one.add(rec.id)
        self.n_hw_error = n_err
        # ---- project cases --------------------------------------------------------------------------------------
        for rec, outs in recs:
            if rec.id == "misc.tcemu_tests_recorded_hw":
                for case, out, cl in outs:
                    if cl["status"] == "bits":
                        M, N = out.shape
                        rows = None if M <= a.project_rows else np.arange(a.project_rows)
                        cols = None if N <= a.project_cols else self.spread_cols(N, a.project_cols)
                        self.add_gemm_bits("truth", rec, case, out, rows=rows, cols=cols, why="project")
                done_one.add(rec.id)
        # ---- c2 families ---------------------------------------------------------------------------------------
        for rec, outs in recs:
            m = re.match(r"c2\.(T_|S16_)", rec.id)
            if not m:
                continue
            first = {}
            for case, out, cl in outs:
                if cl["status"] == "bits" and case.id not in first:
                    first[case.id] = (case, out)
            if rec.id == "c2.S16_zeros":
                for case, out in first.values():
                    self.add_mma("truth", rec, case, out, why="c2fam")
            else:
                per = -(-a.c2_vectors // (128 * len(first)))           # tiles per case
                for case, out in first.values():
                    T = case.spec["inputs"]["a"]["shape"][0]
                    tiles = np.unique(np.linspace(0, T - 1, min(T, per)).astype(int))
                    self.add_mma("truth", rec, case, out, tiles=tiles, why="c2fam")
            done_one.add(rec.id)
        # ---- epilogue.* with M*N <= 65,536 -----------------------------------------------------------------------
        for rec, outs in recs:
            if not rec.id.startswith("epilogue.") or rec.kind != "gemm":
                continue
            by_case = defaultdict(list)
            for case, out, cl in outs:
                if cl["status"] == "bits" and int(np.prod(out.shape)) <= a.epilogue_mn:
                    by_case[case.id].append((case, out))
            for cid, co in by_case.items():
                case = co[0][0]
                ins = case.spec["inputs"]
                op_bytes = (ins["a"]["shape"][0] + ins["b"]["shape"][0]) * ins["a"]["shape"][1]
                rows = cols = None
                if op_bytes > a.epilogue_op_bytes:
                    rows, cols = self.special_subset([o for _, o in co], a.epilogue_rows, a.epilogue_cols)
                for case, out in co:
                    if rows is not None and len(out.shape) != 2:
                        continue
                    self.add_gemm_bits("truth", rec, case, out, rows=rows, cols=cols, why="epilogue")
                done_one.add(rec.id)
        # ---- Evo 2 1B rows 0..31 ---------------------------------------------------------------------------------
        for rec, outs in recs:
            if rec.id != "evo2.evo2_1b_w2k":
                continue
            calls = [f"call{int(x):02d}" for x in a.evo2_calls.split(",")]
            for case, out, cl in outs:
                if case.id in calls and cl["status"] == "bits" and out.name in ("acc", "y_native", "y_te"):
                    self.add_gemm_bits("truth", rec, case, out, rows=np.arange(32), cols=np.arange(a.evo2_cols),
                                       why="evo2", force_accumulate=out.name == "y_te")
            done_one.add(rec.id)
        # ---- split-K / accumulation combinations -------------------------------------------------------------------
        cands = defaultdict(list)
        for rec, outs in recs:
            if rec.kind != "gemm" or rec.id.startswith("emulator."):
                continue
            for case, out, cl in outs:
                if cl["status"] != "bits":
                    continue
                p = out.params
                sk = splitk_of(p)
                key = (bool(p.get("fast_accum", True)), int(sk[0]), sk[1], p["out_dtype"], p.get("fmt_a", "e4m3"),
                       p.get("fmt_b", "e4m3"))
                ins = case.spec["inputs"]
                M, K = ins["a"]["shape"]
                N = ins["b"]["shape"][0]
                cands[key].append((K, M, N, ins["a"]["sha256"], ins["b"]["sha256"], rec, case, out))
        self.combos = {}
        chosen = set()
        R, C = a.combo_rows, a.combo_cols
        for key in sorted(cands, key=lambda k: (min(x[0] for x in cands[k]), str(k))):
            v = cands[key]

            def cost(x):
                K, M, N, sa, sb = x[:5]
                big = M * N * K > 3e7
                ca = 0 if ("a", sa) in chosen else K * min(R, M)
                cb = 0 if ("b", sb) in chosen else K * min(C, N)
                return (big, ca + cb, K, M * N * K, x[5].id, x[6].id, x[7].name)
            best = min(v, key=cost)
            K, M, N, sa, sb, rec, case, out = best
            chosen |= {("a", sa), ("b", sb)}
            hw_shape = out.shape
            if out.params.get("out_select") == "diagonal":
                cand_rows, cand_cols = np.arange(min(hw_shape[0], 16)), None
            else:
                cand_rows = np.arange(min(hw_shape[0], a.combo_cand_rows))
                cand_cols = self.spread_cols(hw_shape[1], a.combo_cand_cols)
            if R == 1 and C == 1:
                rows, cols = self.pick_1x1(rec, case, out, cand_rows, cand_cols)
            else:
                rows, cols = cand_rows[:R], (None if cand_cols is None else cand_cols[:C])
            self.add_gemm_bits("truth", rec, case, out, rows=rows, cols=cols, why="combo")
            self.combos[str(key)] = dict(truth_id=rec.id, case=case.id, output=out.name, M=M, N=N, K=K,
                                         S=key[1], scheme=key[2])
            done_one.add(rec.id)
        # ---- c3.gemm_table / scope-boundary extras -------------------------------------------------------------
        for rec, outs in recs:
            if rec.id == "c3.gemm_table":
                for case, out, cl in outs:
                    if cl["status"] == "bits" and out.name.startswith("s00_"):
                        self.add_gemm_bits("truth", rec, case, out, why="extra")
                done_one.add(rec.id)
        seen_scope = Counter()
        for rec, outs in recs:
            for case, out, cl in outs:
                if cl["status"] == "scope_out" and seen_scope[(rec.id, cl["reason"])] < 1:
                    self.add_gemm_error("truth", rec, case, out, error=cl["error"], reason=cl["reason"],
                                        why="scope", scope="out")
                    seen_scope[(rec.id, cl["reason"])] += 1
        # ---- >= 1 output per remaining recording ---------------------------------------------------------------
        for rec, outs in recs:
            if rec.kind == "table" or rec.id in done_one:
                continue
            bits = [(case, out) for case, out, cl in outs if cl["status"] == "bits"]
            if not bits:
                if any(cl["status"] == "scope_out" for _, _, cl in outs):
                    done_one.add(rec.id)
                continue
            if rec.kind == "mma":
                case, out = min(bits, key=lambda co: self.in_bytes(co[0]))
                layout = out.params.get("layout", "vector")
                n = case.spec["inputs"]["a"]["shape"][0]
                if layout == "tile":
                    nt = 1 if case.spec["inputs"]["a"]["shape"][-1] >= 512 else 4
                    self.add_mma("truth", rec, case, out, tiles=np.unique(np.linspace(0, n - 1, min(n, nt)).astype(int)),
                                 why="one")
                else:
                    self.add_mma("truth", rec, case, out,
                                 vecs=None if n <= a.one_vectors else np.unique(np.linspace(0, n - 1, a.one_vectors).astype(int)),
                                 why="one")
            else:
                def bcost(co):
                    case, out = co
                    ins = case.spec["inputs"]
                    K = ins["a"]["shape"][1]
                    M, N = (out.shape + (1,))[:2] if len(out.shape) == 1 else out.shape
                    return (K * (min(M, 4) + min(N, 16)), K, str(case.id), out.name)
                case, out = min(bits, key=bcost)
                K = case.spec["inputs"]["a"]["shape"][1]
                r = 4 if K <= 2048 else (2 if K <= 4096 else 1)
                c = 16 if K <= 2048 else 8
                rows, cols = self.small_subset(rec, case, out, max_rows=r, max_cols=c)
                self.add_gemm_bits("truth", rec, case, out, rows=rows, cols=cols, why="one")
            done_one.add(rec.id)

    def pick_1x1(self, rec, case, out, cand_rows, cand_cols):
        """(rows, cols) of the single output position (among the candidates) that the most rival plans get wrong
        (tools/mutants/discriminate.py); also checks that the recorded plan reproduces the hardware there."""
        sys.path.insert(0, str(REPO / "tools" / "mutants"))
        import discriminate as D  # noqa: WPS433
        if D.CHAIN is None and self.fast_chain is not None:
            D.CHAIN = self.fast_chain
        ins = case.inputs()
        hw = out.bits()
        arow, bcol, diag = gemm_view(rec, case, out)
        prm = self.gemm_params(rec, case, out)
        if diag:
            cand = [(int(i), int(i)) for i in cand_rows]
            pairs = [(int(arow[i]), int(bcol[i])) for i, _ in cand]
            hwb = [hw[i] for i, _ in cand]
        else:
            cand = [(int(i), int(j)) for i in cand_rows for j in cand_cols]
            pairs = [(int(arow[i]), int(bcol[j])) for i, j in cand]
            hwb = [hw[i, j] for i, j in cand]
        k, det = D.choose(ins["a"], ins["b"], pairs, hwb, fa=prm["fmt_a"], fb=prm["fmt_b"], fast=prm["fast"],
                          S=prm["splitk"]["S"], scheme=prm["splitk"]["scheme"], sa=prm["scale_a_bits"],
                          sb=prm["scale_b_bits"], sr=prm["scale_result_bits"], od=prm["out_dtype"])
        self.discrimination.append(dict(id=f"{rec.id}/{case.id}/{out.name}", **det,
                                        position=None if k is None else cand[k]))
        if len(self.discrimination) % 50 == 0:
            print(f"  discrimination: {len(self.discrimination)} picks", flush=True)
        if k is None:
            k = 0
        i, j = cand[k]
        return np.array([i]), (None if diag else np.array([j]))

    @staticmethod
    def special_mask(bits: np.ndarray, out_dtype: str) -> np.ndarray:
        """Outputs whose hardware value is 'special' for the epilogue: +-0, NaN, +-Inf, subnormal, fp8 saturation."""
        b = bits.astype(np.int64)
        if out_dtype in ("float8_e4m3fn", "float8_e5m2"):
            m = b & 0x7F
            if out_dtype == "float8_e4m3fn":
                return (m == 0) | (m == 0x7F) | (m == 0x7E) | ((m >> 3) == 0)
            return (m == 0) | (m >= 0x7B) | ((m >> 2) == 0)
        em, mm = {"float32": (0x7F800000, 0x007FFFFF), "bfloat16": (0x7F80, 0x007F), "float16": (0x7C00, 0x03FF)}[out_dtype]
        e = b & em
        return (e == 0) | (e == em)

    def special_subset(self, outs, R, C):
        """Rows/cols (output coordinates) concentrating the special outputs of all `outs` of one case."""
        shapes = {tuple(o.shape) for o in outs}
        if len(shapes) != 1 or len(next(iter(shapes))) != 2:
            o = outs[0]
            return self.small_subset(None, None, o, R, C)
        M, N = next(iter(shapes))
        score = np.zeros((M, N), np.int64)
        for o in outs:
            score += self.special_mask(o.bits(), o.params["out_dtype"])
        rs = score.sum(1)
        rows = np.sort(np.argsort(-rs, kind="stable")[:min(R, M)])
        cs = score[rows].sum(0)
        cols = np.sort(np.argsort(-cs, kind="stable")[:min(C, N)])
        return rows, cols

    @staticmethod
    def spread_cols(N, C):
        if N <= C:
            return np.arange(N)
        return np.unique(np.linspace(0, N - 1, C).astype(int))

    # -- pre-swap recordings ------------------------------------------------------------------------------------------
    def select_ws2(self):
        """The pre-swap recordings R1-R7 (recordings/rtx4060_2026-10; format tcemu_ada_truth/1 under --ws2):
        - recordings whose cases carry an exact call description (params.call: R1 API parity grid, R6 zero dims):
          every case and output, replayed as the exact call (layouts, offsets, scale dtypes/shapes, bias, out=, grad);
        - R4 (unverified split combinations): one case per recorded combination, as recorded (the recorder already
          stores discriminating row/column subsets);
        - every other gemm recording: --ws2-per-recording outputs (default 1; R2's fp8-output verification is tiny,
          and kept in full)."""
        if self.ws2 is None or not (self.ws2 / "manifest.json").exists():
            self.notes.append(f"pre-swap recordings not found at {self.ws2}: no ws2 selection")
            return
        recs = self.census("ws2", root=self.ws2)
        r4 = re.compile(self.args.ws2_r4_regex, re.I)
        for rec, outs in recs:
            if rec.kind == "table":
                continue
            rid = f"ws2:{rec.id}"
            if rec.cases and "call" in rec.cases[0].params:
                for case in rec.cases:
                    self.add_gemm_call(rec, case)
                continue
            if r4.search(rec.id):
                by = defaultdict(list)
                for case, out, cl in outs:
                    if cl["status"] == "bits":
                        k = json.dumps(case.params.get("combo") or [bool(out.params.get("fast_accum", True)),
                                                                      str(out.params.get("splitk")),
                                                                      out.params.get("out_dtype")])
                        by[k].append((case, out))
                for k, co in by.items():
                    case, out = sorted(co, key=lambda x: (x[0].params.get("data", {}).get("kind") != "adv_split",
                                                          str(x[0].id)))[0]
                    M, N = out.shape
                    rows, cols = self.pick_1x1(rec, case, out, np.arange(M), np.arange(N))
                    self.add_gemm_bits("ws2", rec, case, out, rows=rows, cols=cols, why="ws2_r4")
                self.notes.append(f"{rid}: {len(by)} combinations, one output of one recorded case each "
                                  "(the most discriminating position)")
                continue
            bits = sorted(((case, out) for case, out, cl in outs if cl["status"] == "bits"),
                          key=lambda co: (self.in_bytes(co[0]), str(co[0].id)))
            for case, out in bits[:self.args.ws2_per_recording]:
                self.add_gemm_bits("ws2", rec, case, out, why="ws2_one")

    # R1 / R6 exact-call cases ------------------------------------------------------------------------------------
    def add_gemm_call(self, rec, case):
        call = dict(case.params["call"])
        ins = case.inputs()
        sh = self.shard_of(rec.id, "ws2")
        rid = f"ws2:{rec.id}"
        outs = []
        for out in case.outputs:
            p = out.params
            eff = dict(call, **(p.get("call_diff") or {}))
            e = dict(name=out.name, out_dtype=p.get("out_dtype"), fast=bool(p.get("fast_accum", True)))
            if "out_dtype_arg" in p:
                e["out_dtype_passed"] = False
            if p.get("call_diff"):
                e["call_diff"] = compact_call(p["call_diff"], partial=True)
            sk = splitk_of(p)
            if isinstance(sk, tuple) and sk[0] not in (1, "bad"):
                e["splitk"] = [int(sk[0]), sk[1]]
            g = eff.get("grad") or {}
            rowwise = any((eff.get(w) or {}).get("value") == "rowwise" for w in ("scale_a", "scale_b"))
            if out.hw_error:
                e["expect"] = "TypeError" if out.hw_error.startswith("TypeError") else "ReferenceRejectsError"
            elif g.get("mode") == "enabled" and g.get("requires_grad"):
                e["expect"] = "UnsupportedFeatureError"
                e["scope"] = "out:grad"
            elif eff.get("bias"):
                e["expect"] = "UnsupportedFeatureError"
                e["scope"] = "out:bias"
            elif rowwise:
                e["expect"] = "UnsupportedFeatureError"
                e["scope"] = "out:row-wise"
            elif sk == "unknown" or (isinstance(sk, tuple) and sk[0] == "bad"):
                self.cov[rid]["na"]["split-K configuration not recorded"] += 1
                continue
            elif out.stored:
                hw = out.bits()
                e["expect"] = dict(bits=self.put(sh, hw), shape=list(hw.shape), sha256=sha256_array(hw))
                M, N, K = call["M"], call["N"], call["K"]
                self.macs += M * N * K
            else:
                continue
            outs.append(e)
        if not outs:
            return
        self.add_case(dict(id=f"{rid}/{case.id}", truth_id=rid, kind="gemm_call", why="ws2_" + rec.id.split(".")[0],
                           src=dict(case=case.id), call=compact_call(call),
                           inputs=dict(a=self.put(sh, ins["a"]), b=self.put(sh, ins["b"])), outputs=outs))
        self._cov(rid, rec, "ws2_" + rec.id.split(".")[0], sum(int(np.prod(o["expect"]["shape"])) for o in outs
                                                              if isinstance(o["expect"], dict)), outputs=len(outs))

    # -- output --------------------------------------------------------------------------------------------------
    def shard_bytes(self):
        """Compressed bytes per shard (exactly what write() produces)."""
        out = {}
        for shard in sorted({s for s, _ in self.arrays}):
            buf = io.BytesIO()
            np.savez_compressed(buf, **{k: v for (s, k), v in self.arrays.items() if s == shard})
            out[shard + ".npz"] = buf.getvalue()
        return out

    def grouped_cases(self):
        """Gemm bit cases that share a truth case and identical operands become one case with an `outputs` list
        (parameters common to all outputs stay at case level): the index stays small."""
        out, groups, ids = [], {}, Counter()
        for c in self.cases:
            if c["kind"] != "gemm" or "error" in c.get("expect", {}):
                out.append(c)
                continue
            key = (c["truth_id"], c["src"]["case"], json.dumps(c["inputs"], sort_keys=True))
            g = groups.get(key)
            if g is None:
                base = f"{c['truth_id']}/{c['src']['case']}"
                ids[base] += 1
                g = dict(id=base if ids[base] == 1 else f"{base}#{ids[base]}", truth_id=c["truth_id"], kind="gemm",
                         src=dict(case=c["src"]["case"]), inputs=c["inputs"], outputs=[])
                groups[key] = g
                out.append(g)
            o = dict(name=c["src"]["output"], why=c["why"], params=c["params"], expect=c["expect"])
            if c["src"].get("sel"):
                o["sel"] = c["src"]["sel"]
            g["outputs"].append(o)
        for g in groups.values():
            outs = g["outputs"]
            common = {k: v for k, v in outs[0]["params"].items()
                      if all(k in o["params"] and o["params"][k] == v for o in outs)}
            g["params"] = common
            for o in outs:
                o["params"] = {k: v for k, v in o["params"].items() if k not in common}
        return out

    def write(self, out_dir: Path, dry_run: bool):
        blobs = self.shard_bytes()
        total = sum(len(b) for b in blobs.values())
        manifest_sha = sha256_file(self.truth / "manifest.json")
        index = OrderedDict(
            format=FORMAT, created_by="tools/export_hw_mini.py", truth_manifest_sha256=manifest_sha,
            sources=[dict(name="truth", root=portable_path(self.truth, "MMAEMU_TRUTH_DIR"), manifest_sha256=manifest_sha)]
            + ([dict(name="ws2", root=_repo_rel(self.ws2),
                     manifest_sha256=sha256_file(self.ws2 / "manifest.json") if (self.ws2 / "manifest.json").exists()
                     else None)] if self.ws2 and self.ws2.exists() else []),
            selection=dict(c2_vectors=self.args.c2_vectors, epilogue_mn=self.args.epilogue_mn,
                           epilogue_op_bytes=self.args.epilogue_op_bytes, epilogue_rows=self.args.epilogue_rows,
                           epilogue_cols=self.args.epilogue_cols,
                           combo_rows=self.args.combo_rows, combo_cols=self.args.combo_cols,
                           combo_cand_rows=self.args.combo_cand_rows, combo_cand_cols=self.args.combo_cand_cols,
                           project_rows=self.args.project_rows, project_cols=self.args.project_cols,
                           ws2_per_recording=self.args.ws2_per_recording,
                           evo2_calls=self.args.evo2_calls, evo2_cols=self.args.evo2_cols,
                           one_vectors=self.args.one_vectors),
            shards={k: dict(sha256=hashlib.sha256(v).hexdigest(), bytes=len(v)) for k, v in sorted(blobs.items())},
            call_defaults=CALL_DESC_DEFAULTS,
            cases=self.grouped_cases())
        index_bytes = json.dumps(index, separators=(",", ":")).encode()
        cov = self.coverage(total + len(index_bytes))
        cov_bytes = json.dumps(cov, indent=1).encode()
        cov_txt = self.coverage_text(cov).encode()
        grand = total + len(index_bytes) + len(cov_bytes) + len(cov_txt)
        budget = int(self.args.budget_mb * 1024 * 1024)
        print(f"shards {len(blobs)}: {total / 1e6:.2f} MB compressed; index {len(index_bytes) / 1e6:.2f} MB; "
              f"total {grand / 1e6:.2f} MB (budget {budget / 1e6:.2f} MB); cases {len(self.cases)}; "
              f"est. MACs {self.macs / 1e9:.2f} G")
        for k, v in sorted(blobs.items(), key=lambda kv: -len(kv[1])):
            print(f"  {k:24s} {len(v) / 1e6:8.3f} MB")
        if grand > budget:
            raise SystemExit(f"hw_mini would be {grand / 1e6:.2f} MB > budget {budget / 1e6:.2f} MB")
        if dry_run:
            return cov
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob("*.npz"):
            if old.name not in blobs:
                old.unlink()
        for k, v in blobs.items():
            (out_dir / k).write_bytes(v)
        (out_dir / "index.json").write_bytes(index_bytes)
        (out_dir / "coverage.json").write_bytes(cov_bytes)
        (out_dir / "COVERAGE.txt").write_bytes(cov_txt)
        print("wrote", out_dir)
        return cov

    def coverage(self, nbytes):
        recs = {}
        for rid, c in sorted(self.cov.items()):
            st = c["status"]
            if rid.startswith("ws2:") and c["outputs_included"] > 0:
                st = "in_scope"
            if st in ("in_scope",) and c["outputs_included"] == 0:
                st = "MISSING"
            recs[rid] = dict(kind=c["kind"], status=st, outputs_total=c["outputs_total"],
                             outputs_included=c["outputs_included"], values_included=c["values_included"],
                             output_status=c.get("output_status", {}), selectors=dict(c["selectors"]),
                             na_reasons=dict(c["na"]))
        missing = [r for r, c in recs.items() if c["status"] == "MISSING"]
        return dict(format=FORMAT, built=time.strftime("%Y-%m-%d %H:%M:%S"), bytes=nbytes,
                    est_macs=self.macs, cases=len(self.cases), hw_error_cases=self.n_hw_error,
                    combos=self.combos, n_combos=len(self.combos), missing_in_scope=missing, notes=self.notes,
                    discrimination=self.discrimination,
                    recordings=recs)

    @staticmethod
    def coverage_text(cov):
        lines = [f"hw_mini coverage ({cov['format']}, built {cov['built']})",
                 f"cases {cov['cases']}; hw_error cases {cov['hw_error_cases']}; split/accumulation combinations "
                 f"{cov['n_combos']}; estimated reference MACs {cov['est_macs'] / 1e9:.2f} G; bytes {cov['bytes']}",
                 f"in-scope recordings missing: {cov['missing_in_scope'] or 'none'}", ""]
        for n in cov["notes"]:
            lines.append("NOTE " + n)
        lines.append(f"{'recording':44s} {'kind':5s} {'status':16s} {'incl':>6s} {'of':>6s} {'values':>10s}  selectors / n/a")
        for rid, c in cov["recordings"].items():
            sel = ",".join(f"{k}:{v}" for k, v in sorted(c["selectors"].items()))
            na = "; ".join(f"{k} ({v})" for k, v in c["na_reasons"].items())
            lines.append(f"{rid:44s} {str(c['kind']):5s} {str(c['status']):16s} {c['outputs_included']:6d} "
                         f"{c['outputs_total']:6d} {c['values_included']:10d}  {sel}{('  n/a: ' + na) if na else ''}")
        lines.append("")
        lines.append("combinations (fast, S, scheme, out_dtype, fmt_a, fmt_b) -> recording/case/output (M, N, K):")
        for k, v in cov["combos"].items():
            lines.append(f"  {k}: {v['truth_id']}/{v['case']}/{v['output']} ({v['M']}, {v['N']}, {v['K']})")
        return "\n".join(line.rstrip() for line in lines) + "\n"


# Defaults of the exact-call description of the R1/R6 cases (tools/record/r1_api_grid.py); compact_call() drops
# them from index.json and tests/test_hw_mini.py restores them from index["call_defaults"].
CALL_DESC_DEFAULTS = {
    "mat": {"layout": "std", "ld_pad": 0, "offset": 0},
    "scale_a": {"dtype": "float32", "shape": [], "value": 0.3},
    "scale_b": {"dtype": "float32", "shape": [], "value": 1.7},
    "bias": None, "scale_result": None, "out": None, "grad": None, "positional_bias": False,
}


def compact_call(call: dict, partial: bool = False) -> dict:
    out = {}
    for k, v in call.items():
        if k in ("mat1", "mat2") and isinstance(v, dict):
            out[k] = {kk: vv for kk, vv in v.items() if CALL_DESC_DEFAULTS["mat"].get(kk, object()) != vv}
        elif k in ("scale_a", "scale_b") and v == CALL_DESC_DEFAULTS[k] and not partial:
            continue
        elif k in CALL_DESC_DEFAULTS and CALL_DESC_DEFAULTS[k] is None and v is None and not partial:
            continue
        elif k == "positional_bias" and v is False and not partial:
            continue
        else:
            out[k] = v
    return out




def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth", default=os.environ.get("MMAEMU_TRUTH_DIR", ""),
                    help="the (unpublished) hardware ground truth checkout; default $MMAEMU_TRUTH_DIR")
    ap.add_argument("--ws2", default=str(REPO / "recordings" / "rtx4060_2026-10"))
    ap.add_argument("--out", default=str(REPO / "tests" / "data" / "hw_mini"))
    ap.add_argument("--budget-mb", type=float, default=19.5, help="hard limit in MiB (spec: 20; keep a margin)")
    ap.add_argument("--c2-vectors", type=int, default=4096)
    ap.add_argument("--epilogue-mn", type=int, default=65536)
    ap.add_argument("--epilogue-op-bytes", type=int, default=48 * 1024,
                    help="epilogue cases with larger operands keep only the rows/cols richest in special outputs")
    ap.add_argument("--epilogue-rows", type=int, default=4)
    ap.add_argument("--epilogue-cols", type=int, default=16)
    ap.add_argument("--combo-rows", type=int, default=1)
    ap.add_argument("--combo-cols", type=int, default=1)
    ap.add_argument("--combo-cand-rows", type=int, default=2, help="candidate rows for the discriminating 1x1 pick")
    ap.add_argument("--combo-cand-cols", type=int, default=8, help="candidate cols for the discriminating 1x1 pick")
    ap.add_argument("--project-rows", type=int, default=8)
    ap.add_argument("--project-cols", type=int, default=16)
    ap.add_argument("--evo2-calls", default="0,10,20")
    ap.add_argument("--evo2-cols", type=int, default=128)
    ap.add_argument("--one-vectors", type=int, default=512)
    ap.add_argument("--ws2-r4-regex", default=r"^r4[._]")
    ap.add_argument("--ws2-per-recording", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if not a.truth:
        raise SystemExit("export_hw_mini: give --truth or set MMAEMU_TRUTH_DIR (the unpublished hardware ground truth)")
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        print("note: PYTHONDONTWRITEBYTECODE is not 1; sys.dont_write_bytecode is set in-process instead")
    t0 = time.monotonic()
    import torch
    torch.set_num_threads(a.threads)
    ex = Exporter(Path(a.truth), Path(a.ws2) if a.ws2 else None, a)
    recs = ex.census("truth")
    ex.select_truth(recs)
    ex.select_ws2()
    cov = ex.write(Path(a.out), a.dry_run)
    print(f"done in {time.monotonic() - t0:.1f} s; in-scope recordings missing: {cov['missing_in_scope'] or 'none'}")
    if cov["missing_in_scope"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
