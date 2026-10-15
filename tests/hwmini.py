"""Reader of the committed hardware subset tests/data/hw_mini/ (format "mmaemu-hwmini/1", docs/testing.md).

numpy only (no torch, no mmaemu, no truth package): the format is self-contained.

    from hwmini import HwMini
    hm = HwMini()                       # tests/data/hw_mini by default; checks shard sha256 on first use
    for case in hm.cases:               # dicts (see docs/testing.md for every field)
        a = hm.array(case["inputs"]["a"])           # raw bit patterns (uint8 FP8 codes, uint32 fp32, ...)
        exp = hm.expected(case)                       # ndarray of bits, or None when case["expect"]["error"]

Shards are .npz files (allow_pickle=False) holding raw bit patterns only, deduplicated by content. Every case's
expected array is checked against its recorded sha256 when it is read.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

FORMAT = "mmaemu-hwmini/1"
DEFAULT_DIR = Path(__file__).resolve().parent / "data" / "hw_mini"
BITS_DTYPE = {"float32": np.uint32, "bfloat16": np.uint16, "float16": np.uint16,
              "float8_e4m3fn": np.uint8, "float8_e5m2": np.uint8}


class HwMiniError(RuntimeError):
    pass


def sha256_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


class HwMini:
    def __init__(self, path: str | Path | None = None, verify: bool = True):
        self.dir = Path(path) if path else DEFAULT_DIR
        idx = self.dir / "index.json"
        if not idx.exists():
            raise HwMiniError(f"{idx} missing (build it with tools/export_hw_mini.py)")
        self.index = json.loads(idx.read_text())
        if self.index.get("format") != FORMAT:
            raise HwMiniError(f"{idx}: format {self.index.get('format')!r}, expected {FORMAT!r}")
        self.cases = self.index["cases"]
        self.verify = verify
        self._npz = {}

    @property
    def truth_manifest_sha256(self) -> str:
        return self.index["truth_manifest_sha256"]

    def _shard(self, name: str):
        if name not in self._npz:
            p = self.dir / name
            meta = self.index["shards"].get(name)
            if meta is None:
                raise HwMiniError(f"shard {name} not listed in index.json")
            if self.verify:
                h = hashlib.sha256(p.read_bytes()).hexdigest()
                if h != meta["sha256"]:
                    raise HwMiniError(f"{p}: sha256 {h} != index {meta['sha256']}")
            self._npz[name] = np.load(p, allow_pickle=False)
        return self._npz[name]

    def array(self, ref) -> np.ndarray:
        shard, key = ref
        return self._shard(shard)[key]

    def expected(self, case) -> np.ndarray | None:
        e = case["expect"]
        if "error" in e:
            return None
        a = self.array(e["bits"])
        if a.dtype != BITS_DTYPE[e["dtype"]] or list(a.shape) != list(e["shape"]):
            raise HwMiniError(f"{case['id']}: expected array {a.dtype}{a.shape} does not match {e['dtype']}{e['shape']}")
        if self.verify and sha256_array(a) != e["sha256"]:
            raise HwMiniError(f"{case['id']}: expected bits sha256 mismatch")
        return a

    def bits(self, expect: dict, label: str = "") -> np.ndarray:
        """Expected bits of one output entry {"bits": [shard, key], "shape", "sha256"[, "dtype"]}."""
        a = self.array(expect["bits"])
        if list(a.shape) != list(expect["shape"]):
            raise HwMiniError(f"{label}: expected array shape {list(a.shape)} != {expect['shape']}")
        if "dtype" in expect and a.dtype != BITS_DTYPE[expect["dtype"]]:
            raise HwMiniError(f"{label}: expected array dtype {a.dtype} does not match {expect['dtype']}")
        if self.verify and sha256_array(a) != expect["sha256"]:
            raise HwMiniError(f"{label}: expected bits sha256 mismatch")
        return a

    def by_truth_id(self) -> dict:
        out = {}
        for c in self.cases:
            out.setdefault(c["truth_id"], []).append(c)
        return out
