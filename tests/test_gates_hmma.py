"""Gates of the gated fast kernels: design_a (tc3) and window (v9).

CPU part (no GPU): the gates' property checks are fed CPU simulations of tensor-core HMMA models (the
Khattak-Mikaitis block-FMA family, port of the tcemu audit's verify-p3-tensorcore-exact-1/scripts/
sim_gate.py hmma_gen):
  - the Ada/Ampere fp16 model (8-term blocks, F = 24) passes design_a, and so does its RNE-final variant, which the
    p3 prototype check (p3 scripts/selftest.py:58) falsely rejected although tc3 is exact under it;
  - the Hopper / Blackwell model (16-term blocks, F = 25; the paper's model for sm_120) makes design_a reject with the
    named reason "design_a.property";
  - the published bf16 models pass the window gate, a range-limited mutant (alignment clamped at -60) fails it.
  The stored GEMM reference results of the gates are recomputed with the mmaemu reference.
GPU part: both gates pass on a validated device within 2 s; a forced gate failure falls back to the next kernel with
one BackendFallbackWarning; a capability outside validated_on is not selected unless MMAEMU_ALLOW_GATED_KERNELS=1;
a second process hits the verdict cache in < 50 ms without GPU work, and a corrupt cache file makes the gate re-run.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch

from mmaemu.backends.cuda.fast import gates, specs

# ------------------------------------------------------------------------------------- CPU HMMA models (simulation)


def _exp(x, emin=None):
    e = (torch.frexp(x)[1] - 1).double()
    if emin is not None:
        e = e.clamp_min(emin)
    return torch.where(x != 0, e, torch.full_like(x, -1e5))


def _p2(e):
    return torch.pow(2.0, e)


def hmma_gen(a, b, c, F, nfma, emin_in, late=False, align="rz", final="rz", emin_al=-133):
    """Block-FMA model of an fp16/bf16 HMMA (float64, exact): chained nfma-term blocks; products (and c unless
    late) aligned to the largest exponent (clamped at emin_al) and truncated (rz) / floored (rd) to F fraction bits;
    the sum normalised to fp32 by truncation (rz) or RNE; late: c added afterwards with an fp32 RNE addition.
    a, b [..., K], c [...]; returns float64 holding fp32 values.  (sim_gate.py hmma_gen.)"""
    acc = c
    rnd = torch.trunc if align == "rz" else torch.floor
    for k0 in range(0, a.shape[-1], nfma):
        aa, bb = a[..., k0:k0 + nfma], b[..., k0:k0 + nfma]
        p = aa * bb
        ep = torch.where(p != 0, _exp(aa, emin_in) + _exp(bb, emin_in), torch.full_like(p, -1e5))
        E = ep.amax(-1)
        if not late:
            E = torch.maximum(E, _exp(acc))
        E = E.clamp_min(emin_al)
        s = rnd(p * _p2(F - E)[..., None]).sum(-1)
        if not late:
            s = s + rnd(acc * _p2(F - E))
        d = s * _p2(E - F)
        if final == "rz":
            ed = _exp(d).clamp_min(-126)
            d = torch.where(d != 0, torch.trunc(d * _p2(23 - ed)) * _p2(ed - 23), d)
        else:
            d = d.float().double()
        if late:
            d = (d.float() + acc.float()).double()
        acc = d
    return acc


ADA_F16 = dict(F=24, nfma=8)                       # A100 / A2 / L40S / Ada (toolbox vectors; RTX 4060 hardware)
HOPPER_F16 = dict(F=25, nfma=16)                   # H100 / H200 / B200 / RTX PRO 6000 (paper: also sm_120)
ADA_F16_RNE = dict(F=24, nfma=8, final="rne")      # p3 prototype gate false-rejects this one
ADA_BF16 = dict(F=24, nfma=8, emin_al=-132)        # tc_models PUBLISHED (neab = 1)
HOPPER_BF16 = dict(F=25, nfma=16, emin_al=-133)    # tc_models PUBLISHED (neab = 2)
MUTANT_BF16 = dict(F=25, nfma=16, emin_al=-60)     # tc_models mutant_models(): violates the window property


def _bf16_vals(bits16):
    return torch.from_numpy(np.ascontiguousarray(bits16).astype(np.uint16).view(np.int16)).view(torch.bfloat16).double()


def _f32_bits(x64):
    return x64.float().contiguous().view(torch.int32).numpy()


class SimRunner:
    """A CPU stand-in for a GPU: the probes are evaluated with HMMA models; gemm returns the stored reference
    (only reached when the instruction-level stages pass)."""

    def __init__(self, f16=ADA_F16, bf16=ADA_BF16):
        self.f16, self.bf16 = f16, bf16
        self.gemm_calls = 0

    def hmma_f16_k8(self, a, b, c):
        A = torch.from_numpy(a)[:, :, None, :].expand(-1, 16, 8, 8)
        B = torch.from_numpy(b)[:, None, :, :].expand(-1, 16, 8, 8)
        return hmma_gen(A, B, torch.from_numpy(np.asarray(c, np.float64)), emin_in=-14, **self.f16).float().numpy()

    def _bf16(self, A, B, C):  # A [T,16,16], B [T,8,16] float64 values
        Ae = A[:, :, None, :].expand(-1, 16, 8, 16)
        Be = B[:, None, :, :].expand(-1, 16, 8, 16)
        return hmma_gen(Ae, Be, C, emin_in=-126, **self.bf16)

    def epass_raw(self, At, Bt):
        ea, eb = torch.from_numpy(gates.E4M3_EA[At]).double(), torch.from_numpy(gates.E4M3_EA[Bt]).double()
        A = torch.where(torch.from_numpy(gates.E4M3_NZ[At]), _p2(6 * ea - 7), torch.zeros_like(ea))
        B = torch.where(torch.from_numpy(gates.E4M3_NZ[Bt]), _p2(6 * eb - 7), torch.zeros_like(eb))
        C = torch.full((At.shape[0], 16, 8), 2.0 ** -86, dtype=torch.float64)
        return _f32_bits(self._bf16(A, B, C))

    def mma_bf16_raw(self, a16, b16, c):
        C = torch.from_numpy(np.asarray(c, np.float64))
        return _f32_bits(self._bf16(_bf16_vals(a16), _bf16_vals(b16), C))

    def gemm(self, spec, a, b):
        self.gemm_calls += 1
        name = next(n for n, *_ in gates.GEMM_CASES if gates.gemm_inputs(n)[0].shape == tuple(a.shape)
                    and np.array_equal(gates.gemm_inputs(n)[0], a.numpy()))
        return torch.from_numpy(gates.gemm_expected()[name].view(np.int32)).view(torch.float32)


# ----------------------------------------------------------------------------------------------- CPU-only tests
def test_specs_and_registration():
    from mmaemu.backends.cuda import spec as S
    assert {s.name for s in specs.SPECS} == {"v9", "tc3"}
    v9, tc3 = specs.V9, specs.TC3
    assert (v9.priority, tc3.priority) == (30, 40)
    assert v9.gated and tc3.gated and v9.validated_on == frozenset({(8, 9)}) == tc3.validated_on
    assert v9.gates == ("window",) and tc3.gates == ("design_a",)
    assert tc3.sass_archs == (80, 86, 89) and tc3.ptx_arch is None
    assert v9.sass_archs == (80, 86, 89, 90, 100, 120) and v9.ptx_arch is None
    assert v9.k_max == 1 << 24 and tc3.k_max == 1 << 24 and tc3.align_bytes == 16 and v9.align_bytes == 4
    assert v9.entry == "mmaemu_v9_e4m3e4m3" and tc3.entry == "mmaemu_tc3_e4m3e4m3"
    assert specs.PROBE_MODULES == {"probes": "probes.cu"} and S.PROBE_MODULES.get("probes") == "probes.cu"
    assert S.GATES["design_a"] is gates.design_a and S.GATES["window"] is gates.window
    assert getattr(gates.design_a, "version") == getattr(gates.window, "version") == gates.GATE_VERSION


def test_kernel_sources_are_device_only():
    from pathlib import Path
    csrc = Path(gates.__file__).resolve().parents[1] / "csrc"
    import re
    for f in ("v9.cu", "v9_common.cuh", "tc3.cu", "probes.cu"):
        code = re.sub(r"//[^\n]*", "", (csrc / f).read_text())          # code without comments
        incs = re.findall(r"#\s*include\s*[<\"]([^>\"]+)[>\"]", code)
        assert set(incs) <= {"v9_common.cuh", "stdint.h"}, (f, incs)   # NVRTC needs no CUDA / torch headers
        assert "TORCH_CHECK" not in code and "PYBIND11" not in code and "torch::" not in code, f
        assert "__host__" not in code and "<<<" not in code, f                # device-only: no host code / launches
    for f, entries in (("v9.cu", ["mmaemu_v9_e4m3e4m3"]), ("tc3.cu", ["mmaemu_tc3_e4m3e4m3"]),
                       ("probes.cu", ["mmaemu_probe_hmma_f16_k8", "mmaemu_probe_epass_raw", "mmaemu_probe_mma_bf16_raw"])):
        src = (csrc / f).read_text()
        for e in entries:
            assert 'extern "C" __global__' in src and e in src, (f, e)


def test_design_a_vector_domain():
    v = gates.design_a_vectors()
    E, n = v["E"], v["E"].size
    assert n >= 4096 and n == 16384
    nz = E >= -12
    assert E[nz].max() == 43                         # covers every E reachable by tc3 for K < 2^24 (<= 41)
    maxep, anynz = gates._pair_maxexp(v["codes_a"], v["codes_b"])
    off = (E - np.maximum(maxep, -12))[anynz]
    assert off.min() == 0 and off.max() >= 50 and (off <= 7).mean() > 0.4
    assert (~anynz).sum() > 0 and (E == -127).sum() > 0   # all-zero-product outputs incl. tc3's c for acc == +0


def test_design_a_accepts_ada_model():
    r = SimRunner(f16=ADA_F16, bf16=ADA_BF16)
    res = gates.design_a(specs.TC3, "cpu", runner=r)
    assert res.passed, res.reason
    assert res.details["property"]["mismatches"] == 0 and res.details["xpass"]["mismatches"] == 0
    assert r.gemm_calls == len(gates.GEMM_CASES)


def test_design_a_rejects_f25_16term_model():
    """The paper's Hopper/Blackwell fp16 HMMA (16-term blocks, F = 25) -- the model the paper gives sm_120."""
    r = SimRunner(f16=HOPPER_F16, bf16=HOPPER_BF16)
    res = gates.design_a(specs.TC3, "cpu", runner=r)
    assert not res.passed
    assert res.reason.startswith("design_a.property:"), res.reason
    assert res.details["property"]["mismatches"] > 1000
    assert r.gemm_calls == 0                    # rejected at instruction level, before the GEMM check


@pytest.mark.parametrize("model", [dict(F=24, nfma=8, late=True), dict(F=24, nfma=8, align="rd"),
                                   dict(F=25, nfma=8), dict(F=23, nfma=4), dict(F=26, nfma=16)],
                         ids=["ada_late_c", "ada_align_rd", "F25_n8", "v100_F23_n4", "F26_n16"])
def test_design_a_rejects_other_models(model):
    res = gates.design_a(specs.TC3, "cpu", runner=SimRunner(f16=model))
    assert not res.passed and res.reason.startswith("design_a.property:"), res.reason


def test_design_a_no_false_reject_rne_final():
    """p3 scripts/selftest.py:58 compared the same-sign half with the Ada truncation model and rejected an HMMA with
    RNE final normalisation; tc3 only uses the opposite-sign result, so the two-sign check accepts it."""
    r = SimRunner(f16=ADA_F16_RNE)
    v = gates.design_a_vectors()
    rp = r.hmma_f16_k8(v["a"], v["b"], v["cp"])
    rn = r.hmma_f16_k8(v["a"], v["b"], -v["cp"])
    ok, reason, det = gates.eval_design_a(v, rp, rn)
    assert ok, reason
    # the prototype's check on the same vectors: one sign per vector (alternating), same-sign half vs Ada model
    sgn = np.where((np.arange(v["cp"].size).reshape(v["cp"].shape) % 2) == 0, 1.0, -1.0)
    c = sgn * v["cp"]
    out = np.where(sgn > 0, rp, rn).astype(np.float64)
    A = torch.from_numpy(v["a"])[:, :, None, :].expand(-1, 16, 8, 8)
    B = torch.from_numpy(v["b"])[:, None, :, :].expand(-1, 16, 8, 8)
    model = hmma_gen(A, B, torch.from_numpy(c), emin_in=-14, **ADA_F16).numpy()
    S_u = v["want"]
    old_want = np.where(np.sign(c) != np.sign(S_u), c + S_u, model)
    assert (out != old_want).sum() > 0         # the old check would have rejected this (exact) device


@pytest.mark.parametrize("model", [ADA_BF16, HOPPER_BF16], ids=["ada_bf16", "hopper_bf16"])
def test_window_accepts_published_models(model):
    r = SimRunner(bf16=model)
    res = gates.window(specs.V9, "cpu", runner=r)
    assert res.passed, res.reason
    ep = res.details["epass"]
    assert ep["outputs"] >= 942080 and ep["ratio_min"] >= 1.0 and ep["ratio_max"] <= 17.0


def test_window_rejects_range_limited_mutant():
    res = gates.window(specs.V9, "cpu", runner=SimRunner(bf16=MUTANT_BF16))
    assert not res.passed and res.reason.startswith("window."), res.reason


def test_forced_failure_env(monkeypatch):
    monkeypatch.setenv("MMAEMU_TEST_FORCE_FAIL", "design_a")
    res = gates.design_a(specs.TC3, "cpu", runner=SimRunner())
    assert not res.passed and res.reason.startswith("forced")
    assert gates.window(specs.V9, "cpu", runner=SimRunner()).passed


def test_gemm_data_inputs_and_size():
    exp = gates.gemm_expected()   # raises if the generator no longer reproduces the recorded input sha256
    assert set(exp) == {c[0] for c in gates.GEMM_CASES}
    for name, kind, M, N, K in gates.GEMM_CASES:
        a, b = gates.gemm_inputs(name)
        assert a.shape == (M, K) and b.shape == (N, K) and exp[name].shape == (M, N)
        assert not np.isin(a, [0x7F, 0xFF]).any() and not np.isin(b, [0x7F, 0xFF]).any()
    assert gates.GEMM_DATA.stat().st_size < 400_000


def _reference_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.slow
def test_gemm_data_equals_reference():
    """Provenance of data/gate_gemm_v1.npz: every expected matrix equals the mmaemu reference (bitwise)."""
    from mmaemu.model.chain import chain_gemm
    from mmaemu.testing import assert_bitwise_equal
    dev = _reference_device()
    exp = gates.gemm_expected()
    for name, *_ in gates.GEMM_CASES:
        a, b = gates.gemm_inputs(name)
        ref = chain_gemm(torch.from_numpy(a).to(dev), torch.from_numpy(b).to(dev), fmt_a="e4m3", fmt_b="e4m3").cpu()
        assert_bitwise_equal(ref, torch.from_numpy(exp[name].view(np.int32)).view(torch.float32), msg=name)


# ------------------------------------------------------------------------------------------------------ GPU tests
def _cuda():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    return torch.device("cuda", torch.cuda.current_device())


def _arch(dev):
    cc = torch.cuda.get_device_capability(dev)
    return cc, cc[0] * 10 + cc[1]


@pytest.mark.gpu
def test_gates_on_device():
    """design_a and window on the real device: PASS within 2 s on validated hardware (module load excluded)."""
    from mmaemu.backends.cuda import modules
    dev = _cuda()
    cc, arch = _arch(dev)
    for mod, entry in (("probes", "mmaemu_probe_hmma_f16_k8"), ("v9", "mmaemu_v9_e4m3e4m3")) + \
            ((("tc3", "mmaemu_tc3_e4m3e4m3"),) if arch in specs.TC3.sass_archs else ()):
        modules.get_function(mod, entry, dev)           # module load is excluded from the gate budget
    rd = gates.design_a(specs.TC3, dev)
    rw = gates.window(specs.V9, dev)
    print(json.dumps(dict(device=torch.cuda.get_device_name(dev), cc=cc,
                          design_a=dict(passed=rd.passed, reason=rd.reason, seconds=round(rd.seconds, 3),
                                        stages={k: v for k, v in rd.details.items() if isinstance(v, dict)}),
                          window=dict(passed=rw.passed, reason=rw.reason, seconds=round(rw.seconds, 3),
                                      stages={k: v for k, v in rw.details.items() if isinstance(v, dict)})),
                     default=str))
    if tuple(cc) in specs.TC3.validated_on:
        assert rd.passed, rd.reason
        assert rd.seconds <= 2.0, rd.seconds
    elif arch not in specs.TC3.sass_archs:
        assert not rd.passed                            # no tc3 binary here, or the HMMA is not design-A capable
    if tuple(cc) in specs.V9.validated_on:
        assert rw.passed, rw.reason
        assert rw.seconds <= 2.0, rw.seconds


def _py(code: str, env_extra: dict, timeout=300):
    env = dict(os.environ)
    env.update(env_extra)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    r = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], env=env, capture_output=True, text=True,
                       timeout=timeout)
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


_SELECT = """
import json, warnings, torch
{patch}
from mmaemu.backends import ChainRequest, select
from mmaemu.errors import BackendFallbackWarning
req = ChainRequest(torch.device("cuda", torch.cuda.current_device()), "e4m3", "e4m3", "ada")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    b1 = select(req)
    b2 = select(req)
fb = [str(x.message) for x in w if issubclass(x.category, BackendFallbackWarning)]
print(json.dumps(dict(b1=b1.name, b2=b2.name, fallback_warnings=fb)))
"""


@pytest.mark.gpu
def test_forced_gate_failure_falls_back_once(tmp_path):
    dev = _cuda()
    cc, arch = _arch(dev)
    if tuple(cc) not in specs.TC3.validated_on:
        pytest.skip("tc3 is only auto-selected on validated capabilities")
    env = {"MMAEMU_CACHE_DIR": str(tmp_path / "cache")}
    base = _py(_SELECT.format(patch=""), env)
    assert base["b1"] == base["b2"] == "cuda:tc3" and base["fallback_warnings"] == [], base
    forced = _py(_SELECT.format(patch=""), dict(env, MMAEMU_TEST_FORCE_FAIL="design_a"))
    assert forced["b1"] == forced["b2"] == "cuda:v9", forced
    assert len(forced["fallback_warnings"]) == 1 and "design_a" in forced["fallback_warnings"][0], forced
    both = _py(_SELECT.format(patch=""), dict(env, MMAEMU_TEST_FORCE_FAIL="design_a,window"))
    assert both["b1"] not in ("cuda:tc3", "cuda:v9") and len(both["fallback_warnings"]) == 2, both


_PATCH_CC = "torch.cuda.get_device_capability = lambda *a, **k: (8, 6)"


@pytest.mark.gpu
def test_unvalidated_capability_needs_opt_in(tmp_path):
    dev = _cuda()
    cc, arch = _arch(dev)
    if tuple(cc) not in specs.TC3.validated_on:
        pytest.skip("the opt-in path is exercised on a device where the gates pass")
    env = {"MMAEMU_CACHE_DIR": str(tmp_path / "cache"), "MMAEMU_ALLOW_GATED_KERNELS": ""}
    off = _py(_SELECT.format(patch=_PATCH_CC), env)
    assert off["b1"] not in ("cuda:tc3", "cuda:v9"), off
    on = _py(_SELECT.format(patch=_PATCH_CC), dict(env, MMAEMU_ALLOW_GATED_KERNELS="1"))
    assert on["b1"] == "cuda:tc3", on


_CACHE_RUN = """
import json, time, torch
from mmaemu.backends.cuda import selftest, modules, spec as S
from mmaemu.backends.cuda.fast import specs, gates
dev = torch.device("cuda", torch.cuda.current_device())
torch.zeros(1, device=dev); torch.cuda.synchronize()
calls = []
if {forbid_gpu}:
    def _no(*a, **k):
        raise AssertionError("GPU work during a cached verdict lookup")
    modules.launch = _no
    gates.DeviceRunner = _no
for name in ("design_a", "window", "smoke"):
    fn = S.GATES[name]
    def wrapped(*a, _fn=fn, _n=name, **k):
        calls.append(_n)
        return _fn(*a, **k)
    wrapped.version = getattr(fn, "version", "0")
    S.GATES[name] = wrapped
out = {{}}
for sp in (specs.TC3, specs.V9):
    t = time.perf_counter()
    res = selftest.run_gates(sp, dev)
    out[sp.name] = dict(seconds=time.perf_counter() - t, passed=[r.passed for r in res], cached=[r.cached for r in res],
                        names=[r.name for r in res])
out["gate_calls"] = calls
print(json.dumps(out))
"""


@pytest.mark.gpu
def test_verdict_cache_second_process_and_corruption(tmp_path):
    dev = _cuda()
    cc, arch = _arch(dev)
    if tuple(cc) not in specs.TC3.validated_on:
        pytest.skip("needs a device where both gates pass")
    cache = tmp_path / "cache"
    env = {"MMAEMU_CACHE_DIR": str(cache)}
    first = _py(_CACHE_RUN.format(forbid_gpu=False), env)
    for k in ("tc3", "v9"):
        assert all(first[k]["passed"]) and not any(first[k]["cached"]), first
    assert {"design_a", "window"} <= set(first["gate_calls"])
    second = _py(_CACHE_RUN.format(forbid_gpu=True), env)
    for k in ("tc3", "v9"):
        assert all(second[k]["passed"]) and all(second[k]["cached"]), second
        assert second[k]["seconds"] < 0.050, second
    assert second["gate_calls"] == [], second
    # corrupt every cached verdict (anything that is not a compiled image) -> the gates run again and pass
    n = 0
    for p in cache.rglob("*"):
        if p.is_file() and p.suffix not in (".cubin", ".fatbin", ".ptx"):
            p.write_bytes(b"\x00{not json")
            n += 1
    assert n >= 2
    third = _py(_CACHE_RUN.format(forbid_gpu=False), env)
    for k in ("tc3", "v9"):
        assert all(third[k]["passed"]) and not any(third[k]["cached"]), third
    assert {"design_a", "window"} <= set(third["gate_calls"])
    print(json.dumps(dict(first=first, second=second, third=third)))
