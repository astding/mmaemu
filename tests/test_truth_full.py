"""Truth-marker subset of T2: when MMAEMU_TRUTH_DIR is set, replay a representative subset of
the RTX 4060 hardware truth through tools/replay_truth.py (the same adapter and verdict as the full T2 gate, in a
subprocess with the tcemu import blocker) and require the partial verdict PASS against the audit's FixedModel report.

The full replay (all 173 recordings, about 80 min on CPU) is run by hand before a release:
    MMAEMU_TRUTH_DIR=... python tools/replay_truth.py --out docs/validation_logs/<date> --compare .../replay_fixed.json

Also unit-tests the adapter's exception mapping (exact classes; never `except RuntimeError`).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.truth
REPO = Path(__file__).resolve().parents[1]
SUBSET = ("^(c2\\.T_overflow|c2\\.S16_zeros|c3\\.instr_specials|c3\\.minimal_repros|c3\\.gemm_boundary|"
          "c3\\.v3_order_probe|c4\\.misc|c4\\.V7|c5\\.v4_torch_demo|c6\\.kit_MM_ada_sm89|c6\\.zero_sign_mma_4060|"
          "c6\\.kit_MM_cublas64_cublas_sm89|epilogue\\.tie_cases_a3_7|epilogue\\.zero_specials_a3_7|"
          "epilogue\\.zero_nan_v2_a1_5|misc\\.splitk_probe_c1_6|misc\\.outdtype_splitk_va1_0|misc\\.rowwise_c1_8|"
          "misc\\.tcemu_tests_recorded_hw|hmma\\.p2v3_epass|c3\\.instr_f16acc)$")


def truth():
    d = os.environ.get("MMAEMU_TRUTH_DIR")
    if not d:
        pytest.skip("MMAEMU_TRUTH_DIR not set")
    return Path(d)


def test_truth_subset_replay(tmp_path):
    t = truth()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=os.pathsep.join([str(REPO), str(t)] + ([os.environ["PYTHONPATH"]]
                                                                  if os.environ.get("PYTHONPATH") else [])))
    r = subprocess.run([sys.executable, str(REPO / "tools" / "replay_truth.py"), "--out", str(tmp_path),
                        "--compare", str(t / "replay" / "replay_fixed.json"), "--only", SUBSET, "--threads", "4"],
                       cwd=REPO, env=env, capture_output=True, text=True, timeout=1800)
    c = json.loads((tmp_path / "compare.json").read_text()) if (tmp_path / "compare.json").exists() else None
    assert r.returncode == 0 and c is not None and c["verdict"] == "PASS", \
        f"exit {r.returncode}\n{(tmp_path / 'compare.txt').read_text() if c else r.stdout[-3000:] + r.stderr[-3000:]}"
    assert c["checks"]["truth_unchanged"] and not c["checks"]["tcemu_in_sys_modules"]
    assert c["totals"]["x_zero_sign"] == 0 and c["totals"]["x_nan_payload"] == 0


def test_adapter_exception_mapping(monkeypatch):
    """ReferenceRejectsError -> ModelRejects, UnsupportedFeatureError -> NotApplicable, others propagate."""
    truth()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(REPO / "tools"))
    import truth_adapter as TA
    import mmaemu
    from mmaemu import errors as E

    m = TA.MmaemuTruthModel(load_fastacc=False)

    class Ctx:
        inp = {"a": __import__("numpy").zeros((16, 32), "uint8"), "b": __import__("numpy").zeros((16, 32), "uint8")}
        A, B = inp["a"], inp["b"]
        label, checks = "unit/0", []

    p = dict(fmt_a="e4m3", fmt_b="e4m3", out_dtype="float32", fast_accum=True, splitk={"S": 1})

    class Plan:
        def __init__(self, v):
            self.validation = v

    for cls_name, expected in (("ReferenceRejectsError", m.ModelRejects), ("UnsupportedFeatureError",
                                                                             m.NotApplicable)):
        monkeypatch.setattr(mmaemu, "explain", lambda *a, _v=(cls_name, "x"), **k: Plan(_v))
        with pytest.raises(expected):
            m.gemm(Ctx(), dict(p, level="torch._scaled_mm"))

    def boom(*a, **k):
        raise E.BackendUnavailableError("not a rejection")
    monkeypatch.setattr(mmaemu, "explain", lambda *a, **k: Plan("ok"))
    monkeypatch.setattr(mmaemu, "scaled_mm", boom)
    with pytest.raises(E.BackendUnavailableError):          # a RuntimeError that is neither: must propagate
        m.gemm(Ctx(), dict(p, level="torch._scaled_mm"))
    import ast
    tree = ast.parse((REPO / "tools" / "truth_adapter.py").read_text())
    calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert not calls & {"acc", "_acc_ref", "_acc_fast", "lib", "available"}, calls & {"acc", "_acc_ref", "lib"}
    handlers = [ast.unparse(h.type) for h in ast.walk(tree) if isinstance(h, ast.ExceptHandler) and h.type]
    assert all(h.split(".")[-1] not in ("RuntimeError", "Exception", "BaseException", "NotImplementedError")
               for h in handlers), handlers
