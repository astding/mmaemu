"""Split-K resolution: which (S, reduction scheme) cuBLAS uses for a GEMM on the profile's reference platform.

    resolve_splitk(M, N, K, *, a_dtype, b_dtype, out_dtype, use_fast_accum, profile=None, splitk="auto",
                   layout_std=True, strict=None) -> SplitDecision

Resolution order (docs/splitk_tables.md):
  1. explicit: splitk is 1, a SplitK or an (S, scheme) tuple. An int > 1 without a scheme raises ValueError.
  2. profile.workspace_bytes == 0: S = 1 ("workspace0"; 0 split rows among 209,712 workspace-0 queries).
  3. layout_std and the key (M, N, K, fmt_a, fmt_b, out, fast, ws) is in the profile's table: the recorded
     (numSplitsK, customOption); customOption 0 or 2 -> "serial", 1 -> "pairwise" ("table").
  4. the current CUDA device is the profile's reference platform (profile.matches_device): live cuBLASLt heuristic
     query through mmaemu.platform.cublaslt with the profile's workspace and alignment ("live-query").
  5. outside the split envelope of the setting (ws, out, fast, fmt pair) -- M*N above the largest M*N that split,
     or K below the smallest K that split -- and not strict: S = 1 ("envelope"; an inference, said in the note).
     A setting absent from the table uses the union envelope of the same (ws, fast, fmt pair) over output dtypes.
  6. otherwise SplitKUnknownError, naming the setting and the envelope, with three ways out.
Zero-size GEMMs (M, N or K = 0) resolve to S = 1 before step 3 (nothing is accumulated; r6.zero_dims).
Profiles without a table (hopper, b200) have no envelope: "auto" always raises.
UnverifiedSplitWarning is emitted once per combination when the (fast, S, scheme, out, fmt pair) combination was
never replayed on hardware (verified_split_combos.json), or when S > 1 on a profile whose splitk_maturity is
"experimental". strict defaults to $MMAEMU_SPLITK_STRICT == "1".

Package data (mmaemu/platform/data, written by tools/export_splitk_table.py): <table_id>.npz (columns M N K fa fb out
fast ws splitK custom algo tile stages reduction swizzle; align dropped after asserting invariance), <table_id>.json
(sources + sha256, per-setting envelopes; the npz sha256 is checked at load), verified_split_combos.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from .profiles import Profile, get_profile

try:  # mmaemu's error hierarchy; stand-ins with the same bases when this module is used on its own
    from mmaemu.errors import SplitKUnknownError, UnverifiedSplitWarning  # type: ignore
except Exception:  # noqa: BLE001
    class _MmaemuError(Exception):
        pass

    class SplitKUnknownError(_MmaemuError, LookupError):  # type: ignore[no-redef]
        pass

    class UnverifiedSplitWarning(UserWarning):  # type: ignore[no-redef]
        pass

__all__ = ["SplitK", "SplitDecision", "resolve_splitk", "load_table", "SplitTable", "SplitKUnknownError",
           "UnverifiedSplitWarning", "lookup"]

DATA = Path(__file__).resolve().parent / "data"
Scheme = Literal["none", "serial", "pairwise"]
FMTS = {"float8_e4m3fn": "e4m3", "float8_e5m2": "e5m2", "e4m3": "e4m3", "e5m2": "e5m2"}
OUT_CODES = {"float32": 0, "bfloat16": 1, "float16": 2, "float8_e4m3fn": 3, "float8_e5m2": 4}
FMT_CODES = {"e4m3": 0, "e5m2": 1}
_KEY_LIMIT = 1 << 21


@dataclass(frozen=True)
class SplitK:
    """An explicit split: S splits (64-wide k-tiles t mod S) combined with `scheme`. Also accepted as (S, scheme)."""
    S: int = 1
    scheme: Scheme = "none"

    def __post_init__(self):
        if not isinstance(self.S, (int, np.integer)) or isinstance(self.S, bool) or self.S < 1:
            raise ValueError(f"SplitK.S must be an int >= 1, got {self.S!r}")
        if self.S == 1 and self.scheme != "none":
            raise ValueError(f"S = 1 has no reduction scheme (got {self.scheme!r}); use scheme='none'")
        if self.S > 1 and self.scheme not in ("serial", "pairwise"):
            raise ValueError(f"S = {self.S} requires scheme 'serial' or 'pairwise', got {self.scheme!r}")


@dataclass(frozen=True)
class SplitDecision:
    S: int
    scheme: Scheme
    source: Literal["explicit", "workspace0", "table", "live-query", "envelope"]
    verified: bool                 # (fast, S, scheme, out, fmt pair) replayed on hardware
    table_id: str | None
    note: str

    @property
    def splitk(self) -> SplitK:
        return SplitK(self.S, self.scheme)


def _name(dt) -> str | None:
    if dt is None:
        return None
    s = str(dt)
    return s.split(".")[-1] if s.startswith("torch.") else s


def _fmt(dt, what: str) -> str:
    s = _name(dt)
    if s not in FMTS:
        raise TypeError(f"{what} must be float8_e4m3fn or float8_e5m2 (or 'e4m3'/'e5m2'), got {dt!r}")
    return FMTS[s]


def _scheme_of(custom: int) -> str:
    return "pairwise" if int(custom) == 1 else "serial"


class SplitTable:
    """One exported split-K table (sha256-checked at load)."""

    def __init__(self, table_id: str, data_dir: Path = DATA):
        self.id = table_id
        meta_p, npz_p = data_dir / f"{table_id}.json", data_dir / f"{table_id}.npz"
        self.meta = json.loads(meta_p.read_text())
        raw = npz_p.read_bytes()
        got = hashlib.sha256(raw).hexdigest()
        if got != self.meta["npz_sha256"]:
            raise RuntimeError(f"{npz_p}: sha256 {got} != {self.meta['npz_sha256']} recorded in {meta_p.name}")
        import io
        z = np.load(io.BytesIO(raw), allow_pickle=False)
        cols = {k: z[k] for k in z.files}
        self.ws_values = list(self.meta["codes"]["ws"])
        skey = (cols["ws"].astype(np.int64) << 16) | (cols["out"].astype(np.int64) << 8) | \
            (cols["fast"].astype(np.int64) << 4) | (cols["fa"].astype(np.int64) << 2) | cols["fb"].astype(np.int64)
        rkey = (cols["M"].astype(np.int64) << 42) | (cols["N"].astype(np.int64) << 21) | cols["K"].astype(np.int64)
        self._rows: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for s in np.unique(skey):
            idx = np.nonzero(skey == s)[0]
            o = np.argsort(rkey[idx], kind="stable")
            self._rows[int(s)] = (rkey[idx][o], idx[o])
        self._cols = cols
        self._settings = {}
        for st in self.meta["settings"]:
            k = (int(st["ws"]), st["out"], bool(st["fast"]), st["fmt_a"], st["fmt_b"])
            self._settings[k] = st
        vc = json.loads((data_dir / "verified_split_combos.json").read_text())
        self.verified = {(bool(c["fast"]), int(c["S"]), c["scheme"], c["out"], c["fmt_pair"])
                         for c in vc["combos"] if c["verified"]}

    def lookup(self, M, N, K, fmt_a, fmt_b, out, fast, ws) -> dict | None:
        if ws not in self.ws_values or out not in OUT_CODES or max(M, N, K) >= _KEY_LIMIT or min(M, N, K) < 0:
            return None
        s = (self.ws_values.index(ws) << 16) | (OUT_CODES[out] << 8) | (int(bool(fast)) << 4) | \
            (FMT_CODES[fmt_a] << 2) | FMT_CODES[fmt_b]
        e = self._rows.get(s)
        if e is None:
            return None
        k = (int(M) << 42) | (int(N) << 21) | int(K)
        i = int(np.searchsorted(e[0], k))
        if i >= len(e[0]) or int(e[0][i]) != k:
            return None
        j = int(e[1][i])
        return {c: int(v[j]) for c, v in self._cols.items()}

    def setting(self, ws, out, fast, fmt_a, fmt_b) -> dict | None:
        return self._settings.get((int(ws), out, bool(fast), fmt_a, fmt_b))

    def envelope(self, ws, out, fast, fmt_a, fmt_b) -> tuple[dict | None, str]:
        """(envelope dict with rows / split_rows / max_split_MN / min_split_K, how it was obtained) or (None, why)."""
        st = self.setting(ws, out, fast, fmt_a, fmt_b)
        if st is not None:
            return st, "setting"
        sts = [v for (w, o, f, a, b), v in self._settings.items() if (w, f, a, b) == (int(ws), bool(fast), fmt_a,
                                                                                         fmt_b)]
        if not sts:
            return None, f"no table rows for ws={ws}, fast={fast}, {fmt_a}x{fmt_b}"
        sp = [v for v in sts if v["split_rows"]]
        env = dict(rows=sum(v["rows"] for v in sts), split_rows=sum(v["split_rows"] for v in sts),
                   max_split_MN=max((v["max_split_MN"] for v in sp), default=None),
                   min_split_K=min((v["min_split_K"] for v in sp), default=None))
        return env, f"union over output dtypes {sorted(v['out'] for v in sts)} (setting {out} absent)"


_TABLES: dict[str, SplitTable] = {}
_LOCK = threading.Lock()
_WARNED: set = set()


def load_table(table_id: str) -> SplitTable:
    with _LOCK:
        if table_id not in _TABLES:
            _TABLES[table_id] = SplitTable(table_id)
        return _TABLES[table_id]


def lookup(M, N, K, *, a_dtype, b_dtype, out_dtype, use_fast_accum, profile=None) -> dict | None:
    """The table row (or None) of the profile's table for this GEMM."""
    p = get_profile(profile)
    if not p.splitk_table or p.workspace_bytes is None:
        return None
    return load_table(p.splitk_table).lookup(M, N, K, _fmt(a_dtype, "a_dtype"), _fmt(b_dtype, "b_dtype"),
                                             _name(out_dtype), use_fast_accum, p.workspace_bytes)


def _current_device():
    try:
        import torch
        if torch.cuda.is_initialized():
            return torch.cuda.current_device()
    except Exception:  # noqa: BLE001
        pass
    return 0


def _warn(p: Profile, fast, S, scheme, out, pair, verified, source):
    if verified and not (S > 1 and p.splitk_maturity == "experimental"):
        return
    key = (p.name, bool(fast), S, scheme, out, pair)
    if key in _WARNED:
        return
    _WARNED.add(key)
    why = ("this (fast, S, scheme, out, format pair) combination was never replayed on hardware"
           if not verified else f"split-K decisions of profile {p.name} are experimental")
    warnings.warn(f"mmaemu split-K S={S} scheme={scheme} ({'fast' if fast else 'slow'} accumulation, out {out}, "
                  f"{pair}, source {source}, profile {p.name}): {why}", UnverifiedSplitWarning, stacklevel=3)


def resolve_splitk(M, N, K, *, a_dtype, b_dtype, out_dtype, use_fast_accum, profile=None, splitk="auto",
                   layout_std=True, strict=None) -> SplitDecision:
    """Resolve the split-K decision for torch._scaled_mm(a[M,K], b[K,N]) on the profile's platform (see module
    docstring). out_dtype None means a_dtype (torch's default)."""
    p = get_profile(profile)
    fa, fb = _fmt(a_dtype, "a_dtype"), _fmt(b_dtype, "b_dtype")
    out = _name(out_dtype) or _name(a_dtype)
    if out in FMTS and out not in OUT_CODES:
        out = {"e4m3": "float8_e4m3fn", "e5m2": "float8_e5m2"}[out]
    if out not in OUT_CODES:
        raise TypeError(f"out_dtype must be one of {sorted(OUT_CODES)}, got {out_dtype!r}")
    fast = bool(use_fast_accum)
    pair = f"{fa}x{fb}"
    M, N, K = int(M), int(N), int(K)
    if strict is None:
        strict = os.environ.get("MMAEMU_SPLITK_STRICT") == "1"
    table = load_table(p.splitk_table) if p.splitk_table else None
    verified_set = table.verified if table is not None else set()

    # 1. explicit
    if not (isinstance(splitk, str) and splitk == "auto"):
        if isinstance(splitk, SplitK):
            sk = splitk
        elif isinstance(splitk, (int, np.integer)) and not isinstance(splitk, bool):
            if splitk > 1:
                raise ValueError(f"splitk={splitk} needs a reduction scheme: pass splitk=({splitk}, 'serial') or "
                                 f"({splitk}, 'pairwise')")
            sk = SplitK(int(splitk), "none")
        elif isinstance(splitk, tuple) and len(splitk) == 2:
            sk = SplitK(int(splitk[0]), splitk[1])
        else:
            raise TypeError(f"splitk must be 'auto', 1, a SplitK or an (S, scheme) tuple, got {splitk!r}")
        ver = sk.S == 1 or (fast, sk.S, sk.scheme, out, pair) in verified_set
        _warn(p, fast, sk.S, sk.scheme, out, pair, ver, "explicit")
        return SplitDecision(sk.S, sk.scheme, "explicit", ver, None, "given by the caller")
    # 2. workspace 0
    if p.workspace_bytes == 0:
        return SplitDecision(1, "none", "workspace0", True, p.splitk_table,
                             "cuBLASLt never splits without a workspace (0 split rows among 209,712 queries)")
    if p.workspace_bytes is None or table is None:
        raise SplitKUnknownError(
            f"profile {p.name!r} has no reference platform and no split-K table, so splitk='auto' cannot be "
            f"resolved for M={M} N={N} K={K}. Ways out: pass splitk=1 if the call is known to be unsplit, "
            f"splitk=(S, 'serial'|'pairwise') if the split is known, or record the GPU's heuristics with "
            f"tools/record_heuristics.py (docs/runbooks/sm120_bringup.md)")
    ws = p.workspace_bytes
    if min(M, N, K) == 0:
        return SplitDecision(1, "none", "envelope", True, p.splitk_table,
                             "zero-size GEMM: nothing is accumulated (r6.zero_dims)")
    # 3. table
    if layout_std:
        row = table.lookup(M, N, K, fa, fb, out, fast, ws)
        if row is not None:
            S = int(row["splitK"])
            scheme = "none" if S <= 1 else _scheme_of(row["custom"])
            ver = S == 1 or (fast, S, scheme, out, pair) in verified_set
            _warn(p, fast, S, scheme, out, pair, ver, "table")
            return SplitDecision(S, scheme, "table", ver, table.id,
                                 f"table row: algo {row['algo']}, tile {row['tile']}, numSplitsK {S}, customOption "
                                 f"{row['custom']}")
    # 4. live query on the profile device
    if layout_std and p.matches_device(_current_device()):
        from . import cublaslt
        r = cublaslt.heuristic(M, N, K, fmt_a=fa, fmt_b=fb, out_dtype=out, fast=fast, workspace_bytes=ws,
                               align=p.align, device=_current_device())
        if r["status"] == 0:
            S = int(r["splitK"])
            scheme = "none" if S <= 1 else _scheme_of(r["custom"])
            ver = S == 1 or (fast, S, scheme, out, pair) in verified_set
            _warn(p, fast, S, scheme, out, pair, ver, "live-query")
            return SplitDecision(S, scheme, "live-query", ver, None,
                                 f"live cuBLASLt heuristic on {p.gpu_name}: algo {r['algo']}, tile {r['tile']}, "
                                 f"numSplitsK {S}, customOption {r['custom']}")
    # 5. envelope
    env, how = table.envelope(ws, out, fast, fa, fb)
    setting = f"(ws={ws} B, out={out}, {'fast' if fast else 'slow'} accumulation, {pair})"
    if env is not None:
        mx, mk = env["max_split_MN"], env["min_split_K"]
        outside = (not env["split_rows"]) or M * N > mx or K < mk
        env_s = (f"split envelope of setting {setting} [{how}]: max split M*N = {mx}, min split K = {mk}, "
                 f"{env['split_rows']} of {env['rows']} rows split")
        if outside and not strict:
            return SplitDecision(1, "none", "envelope", True, table.id,
                                 f"inference: outside the {env_s} (M*N = {M * N}, K = {K})")
    else:
        env_s = f"no envelope for setting {setting}: {how}"
    reason = "strict mode (MMAEMU_SPLITK_STRICT=1 or strict=True)" if (env is not None and strict) else \
        "the shape is inside the split envelope but not in the table"
    raise SplitKUnknownError(
        f"split-K decision unknown for M={M} N={N} K={K} on profile {p.name!r}: {reason}; {env_s}. Ways out: pass "
        f"splitk=1 if the call is known to be unsplit; splitk=(S, 'serial'|'pairwise') if the split is known; or "
        f"record the shape on the reference GPU with tools/record_heuristics.py")
