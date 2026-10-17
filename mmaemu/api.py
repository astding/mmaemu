"""Public API: scaled_mm, accumulate, explain.

``scaled_mm`` reproduces torch._scaled_mm with tensor-wise scales as the reference profile's GPU computes it
(default profile ``hopper``: H100 PCIe, 114 SMs, cuBLASLt 12.8.3.14, torch 2.7.1), bit for bit, on any
device. Scope: FP8 (E4M3 / E5M2) inputs, tensor-wise scales, no bias, inference only. Split-K decisions come from the
profile (mmaemu.platform); calls the profile cannot decide raise instead of guessing.

Validation order (normative; parity tests check exception classes and raise/no-raise):
  1. Python argument types: TypeError.
  2. torch 2.7.1 _scaled_mm argument checks: ReferenceRejectsError. Then, as torch does, zero-size results return
     early (M == 0 or N == 0: empty; K == 0: all +0 for every scale, measured on the RTX 4060), for any out dtype
     and for every float8 dtype torch accepts (fnuz and e8m0 included). cuBLAS-level checks follow: operand layout
     (mat1 not column-major, mat2 column-major; a mat1 that is neither row- nor column-major is copied to
     row-major, as torch does), FP8 formats other than e4m3fn / e5m2, out dtype, 16-byte operand alignment, leading
     dimensions % 16 (measured on the RTX 4060 with torch 2.7.1 and cuBLASLt 12.8.3.14; the R1 recordings of
     recordings/rtx4060_2026-10 are the authority). An out= that is neither row- nor column-major is returned
     unchanged with a UserWarning: torch computes into a contiguous copy of it and never copies the result back.
  3. Profile rejects (e.g. e4m3 x e4m3 -> e5m2 on the ada and hopper profiles): ReferenceRejectsError.
  4. Scope: bias, row-wise scales, inputs requiring grad under grad mode: UnsupportedFeatureError.
  5. Preset maturity warning (ExperimentalPresetWarning).
  6. Split-K resolution (mmaemu.platform.splitk_table.resolve_splitk; SplitKUnknownError if undecidable).
  7. Backend selection and plan execution (mmaemu.model.plan).
  8. Shared epilogue (mmaemu.model.epilogue).
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field

import torch

from mmaemu.errors import MmaemuError, ReferenceRejectsError, UnsupportedFeatureError
from mmaemu.formats import fmt_of, is_fp8_dtype
from mmaemu.model import epilogue
from mmaemu.model.plan import execute_plan, make_plan
from mmaemu.model.presets import get_preset, maturity_warning, warn_once

__all__ = ["scaled_mm", "accumulate", "explain", "CallPlan"]

_E4, _E5 = torch.float8_e4m3fn, torch.float8_e5m2
_OUT_NAMES = {torch.float32: "float32", torch.bfloat16: "bfloat16", torch.float16: "float16",
              _E4: "float8_e4m3fn", _E5: "float8_e5m2"}
# torch 2.7.1 isFloat8Type: the dtypes _scaled_mm accepts before its zero-size early return
_TORCH_FP8 = {getattr(torch, n) for n in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz",
                                          "float8_e8m0fnu") if hasattr(torch, n)}


@dataclass
class CallPlan:
    """What a scaled_mm call would do (returned by :func:`explain`)."""
    validation: object = "ok"            # "ok" or (error class name, message)
    profile: str = ""
    preset: str = ""
    preset_maturity: str = ""
    split: object = None                 # SplitDecision | None
    backend: str = ""
    fallbacks: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


# ---------------------------------------------------------------------------------------------------------------
# helpers
def _ftz_canary() -> None:
    """Raise MmaemuError if this thread flushes fp32 subnormals (torch.set_flush_denormal(True)): the model's
    float64 -> fp32 roundings would then differ from the GPU."""
    x = torch.tensor([3 * 2.0 ** -149], dtype=torch.float64).to(torch.float32)
    y = torch.tensor([2.0 ** -130], dtype=torch.float32) * torch.tensor([0.5], dtype=torch.float32)
    if x.item() == 0.0 or y.item() == 0.0:
        raise MmaemuError("the CPU flushes subnormals to zero (torch.set_flush_denormal(True) is active); mmaemu's "
                          "CPU results would differ from the hardware. Call torch.set_flush_denormal(False).")


def _is_bool(v) -> bool:
    if isinstance(v, bool):
        return True
    try:
        import numpy as np
        return isinstance(v, np.bool_)
    except ImportError:  # pragma: no cover
        return False


def _out_name(dt) -> str:
    return _OUT_NAMES.get(dt, str(dt).replace("torch.", ""))


def _rule_matches(rule, fa: str, fb: str, od) -> bool:
    def eq(val, want):
        if val is None:
            return True
        v = str(val).replace("torch.", "")
        return v in want
    od_forms = {_out_name(od), {_E4: "e4m3", _E5: "e5m2"}.get(od, _out_name(od))}
    return eq(rule.fmt_a, {fa}) and eq(rule.fmt_b, {fb}) and eq(rule.out_dtype, od_forms)


def _misaligned(t: torch.Tensor) -> bool:
    if t.device.type == "cuda":
        return t.data_ptr() % 16 != 0
    return (t.storage_offset() * t.element_size()) % 16 != 0


def _row_major(t: torch.Tensor) -> bool:
    """[R, C] usable by cuBLASLt as row-major (dense contiguous, or unit column stride with ld >= C)."""
    return t.is_contiguous() or (t.stride(1) == 1 and t.stride(0) >= max(1, t.shape[1]))


def _col_major(t: torch.Tensor) -> bool:
    """[R, C] that torch 2.7.1's prepare_matrix_for_cublas passes to cuBLAS untransposed (not contiguous, unit row
    stride, ld >= R). Every other layout is passed transposed: row-major as is, anything else as a contiguous copy."""
    return not t.is_contiguous() and t.stride(0) == 1 and t.stride(1) >= max(1, t.shape[0])


def _profile_and_preset(profile, warn: bool):
    from mmaemu.platform.profiles import get_profile
    prof = get_profile(profile)
    return prof, get_preset(prof.preset, warn=warn)


def _maturity_message(prof, preset) -> str | None:
    """The maturity warning of the call: the preset's, else the profile's (hopper-experimental)."""
    from mmaemu.platform.profiles import profile_maturity_warning
    return maturity_warning(preset) or profile_maturity_warning(prof, preset.maturity == "validated")


def _emit_maturity(prof, preset) -> None:
    """Emit the ExperimentalPresetWarning of the preset (once per name) or of the profile (once per name)."""
    get_preset(prof.preset)
    from mmaemu.platform.profiles import profile_maturity_warning
    msg = profile_maturity_warning(prof, preset.maturity == "validated")
    if msg is not None:
        warn_once("profile:" + prof.name, msg, stacklevel=4)


def _select_backend(device: torch.device, fa: str, fb: str, preset_name: str, backend: str):
    from mmaemu.backends import ChainRequest, select
    if backend == "auto":
        backend = os.environ.get("MMAEMU_BACKEND", "auto") or "auto"
    req = ChainRequest(device=device, fmt_a=fa, fmt_b=fb, preset=preset_name)
    be = select(req, backend)
    return req, be


def _fallbacks(req, backend: str) -> list:
    """Why each backend tried before the chosen one was skipped for this call (its formats, preset and device)."""
    from mmaemu.backends.registry import explain_selection
    if backend == "auto":
        backend = os.environ.get("MMAEMU_BACKEND", "auto") or "auto"
    try:
        return explain_selection(req, backend)[1]
    except Exception as e:  # noqa: BLE001  (diagnostics only)
        return [f"backend selection unavailable: {type(e).__name__}: {e}"]


# ---------------------------------------------------------------------------------------------------------------
# step 1 + 2: argument types and torch 2.7.1 checks
def _check_types(input, mat2, scale_a, scale_b, bias, scale_result, out_dtype, use_fast_accum, out):
    for name, v in (("input", input), ("mat2", mat2), ("scale_a", scale_a), ("scale_b", scale_b)):
        if not isinstance(v, torch.Tensor):
            raise TypeError(f"scaled_mm(): argument '{name}' must be Tensor, not {type(v).__name__}")
    for name, v in (("bias", bias), ("scale_result", scale_result), ("out", out)):
        if v is not None and not isinstance(v, torch.Tensor):
            raise TypeError(f"scaled_mm(): argument '{name}' must be Tensor, not {type(v).__name__}")
    if out_dtype is not None and not isinstance(out_dtype, torch.dtype):
        raise TypeError(f"scaled_mm(): argument 'out_dtype' must be torch.dtype, not {type(out_dtype).__name__}")
    if type(use_fast_accum) is not bool:                  # torch's parser rejects numpy.bool_ too
        raise TypeError(f"scaled_mm(): argument 'use_fast_accum' must be bool, not {type(use_fast_accum).__name__}")


def _torch_checks(mat1, mat2, scale_a, scale_b, bias, scale_result, out_dtype, out) -> tuple:
    """torch 2.7.1 _scaled_mm_out_cuda argument checks; returns (scaling, out dtype)."""
    R = ReferenceRejectsError
    if mat1.dim() != 2:
        raise R("mat1 must be a matrix")
    if mat2.dim() != 2:
        raise R("mat2 must be a matrix")
    if mat1.shape[1] != mat2.shape[0]:
        raise R(f"mat1 and mat2 shapes cannot be multiplied ({mat1.shape[0]}x{mat1.shape[1]} and "
                f"{mat2.shape[0]}x{mat2.shape[1]})")
    M, N = mat1.shape[0], mat2.shape[1]
    if scale_a.dtype != torch.float32 or scale_b.dtype != torch.float32:
        raise R("Both scale_a and scale_b must be float (fp32) tensors.")
    if scale_a.numel() == 1 and scale_b.numel() == 1:
        scaling = "tensorwise"
    else:
        if scale_a.dim() != 2 or scale_b.dim() != 2:
            raise R("For non-TensorWise scaling, scale tensors must be 2-dimensional")
        if tuple(scale_a.shape) == (M, 1) and tuple(scale_b.shape) == (1, N):
            if not (scale_a.is_contiguous() and scale_b.is_contiguous()):
                raise R("Both scale_a and scale_b must be contiguous for RowWise scaling.")
            scaling = "rowwise"
        else:
            raise R("Invalid scaling configuration. For TensorWise scaling, both scales should be scalar.")
    if scale_result is not None and (scale_result.numel() != 1 or scale_result.dtype != torch.float32):
        raise R("scale_result must be a float scalar")
    if bias is not None and bias.numel() != N:
        raise R(f"Bias must be size {N} but got {bias.numel()}")
    if mat1.shape[1] % 16 != 0:
        raise R(f"Expected trailing dimension of mat1 to be divisible by 16 but got mat1 shape: "
                f"({mat1.shape[0]}x{mat1.shape[1]}).")
    if mat2.shape[0] % 16 != 0 or mat2.shape[1] % 16 != 0:
        raise R(f"mat2 shape ({mat2.shape[0]}x{mat2.shape[1]}) must be divisible by 16")
    if out is not None and out_dtype is not None and out_dtype != out.dtype:
        raise R("out_dtype must match output matrix type")
    od = out_dtype if out_dtype is not None else (out.dtype if out is not None else mat1.dtype)
    for name, t in (("mat1", mat1), ("mat2", mat2)):
        if t.dtype not in _TORCH_FP8:
            raise R(f"Expected {name} to be Float8 matrix got {t.dtype}")
    if mat1.dtype == _E5 and mat2.dtype == _E5:
        raise R("Multiplication of two Float8_e5m2 matrices is not supported")
    if bias is not None:
        if od == torch.float32:
            raise R("Bias is not supported when out_dtype is set to Float32")
        if bias.dtype not in (torch.bfloat16, torch.float16):
            raise R(f"Bias must be either Half or BFloat16, but got {bias.dtype}")
        if od == torch.bfloat16 and bias.dtype != torch.bfloat16:
            raise R(f"Bias must be BFloat16 to compute BFloat16 output, but got {bias.dtype}")
        if od == torch.float16 and bias.dtype != torch.float16:
            raise R(f"Bias must be Float16 to compute Half output, but got {bias.dtype}")
    dev = mat1.device
    for t in (out, mat2, bias, scale_a, scale_b, scale_result):
        if t is not None and t.device != dev:
            raise R(f"Expected all tensors to be on the same device, but found at least two devices, {dev} and "
                    f"{t.device}!")
    return scaling, od


def _cublas_checks(mat1, mat2, scaling, od) -> torch.Tensor:
    """Checks torch/cuBLASLt apply only for non-empty calls (after torch's zero-size early return). Returns mat1 as
    cuBLAS reads it: torch copies a mat1 that is neither row- nor column-major to a contiguous one."""
    R = ReferenceRejectsError
    if scaling == "rowwise" and od != torch.bfloat16:
        raise R("Only bf16 high precision output types are supported for row-wise scaling.")
    if _col_major(mat1) or not _row_major(mat2.t()):
        raise R("Only multiplication of row-major and column-major matrices is supported by cuBLASLt")
    if not _row_major(mat1):
        mat1 = mat1.contiguous()
    for name, t in (("mat1", mat1), ("mat2", mat2)):
        if not is_fp8_dtype(t.dtype):
            raise R(f"{name} is {t.dtype}: torch 2.7.1 fails on CUDA for FP8 formats other than float8_e4m3fn and "
                    f"float8_e5m2 (internal assert: cannot convert the ScalarType to a CUDA data type)")
    if od not in _OUT_NAMES:
        raise R(f"out dtype {od} is not supported by cuBLASLt FP8 GEMMs (CUBLAS_STATUS_NOT_SUPPORTED)")
    if _misaligned(mat1) or _misaligned(mat2):
        raise R("CUBLAS_STATUS_NOT_SUPPORTED when calling cublasLtMatmul (operand pointer not 16-byte aligned)")
    lda = mat1.stride(0) if mat1.shape[0] > 1 else mat1.shape[1]
    ldb = mat2.stride(1) if mat2.shape[1] > 1 else mat2.shape[0]
    if lda % 16 or ldb % 16:
        raise R("CUBLAS_STATUS_NOT_SUPPORTED when calling cublasLtMatmulAlgoGetHeuristic (leading dimension not a "
                "multiple of 16)")
    return mat1


def _profile_rejects(prof, fa: str, fb: str, od):
    for rule in getattr(prof, "rejects", ()):
        if _rule_matches(rule, fa, fb, od):
            raise ReferenceRejectsError(f"{rule.message} [profile {prof.name}]")


def _scope_checks(scaling, bias, tensors):
    if bias is not None:
        raise UnsupportedFeatureError("bias is not supported in mmaemu 0.1 (cuBLAS's fused-bias epilogue was "
                                      "measured only in an internal review of the prototype); add the bias "
                                      "yourself after scaled_mm")
    if scaling == "rowwise":
        raise UnsupportedFeatureError("row-wise scales are not supported in mmaemu 0.1 (sm_89 runs them through "
                                      "CUTLASS stream-K, whose split depends on the SM count)")
    if torch.is_grad_enabled() and any(t is not None and t.requires_grad for t in tensors):
        raise UnsupportedFeatureError("mmaemu is inference-only: an input requires grad while grad mode is enabled "
                                      "(use torch.no_grad() or detach the inputs)")


# ---------------------------------------------------------------------------------------------------------------
class _Prepared:
    __slots__ = ("early", "M", "N", "K", "fa", "fb", "od", "prof", "preset", "decision", "req", "be", "warnings",
                 "mat1", "out_unchanged")


def _prepare(input, mat2, scale_a, scale_b, bias, scale_result, out_dtype, use_fast_accum, out, profile, splitk,
             backend, *, dry: bool) -> _Prepared:
    from mmaemu.platform.splitk_table import resolve_splitk
    P = _Prepared()
    P.warnings = []
    _check_types(input, mat2, scale_a, scale_b, bias, scale_result, out_dtype, use_fast_accum, out)
    scaling, od = _torch_checks(input, mat2, scale_a, scale_b, bias, scale_result, out_dtype, out)
    P.M, P.K, P.N = input.shape[0], input.shape[1], mat2.shape[1]
    P.od = od
    P.prof, P.preset = _profile_and_preset(profile, warn=False)
    P.early = P.M == 0 or P.N == 0 or P.K == 0
    P.decision = P.req = P.be = P.fa = P.fb = None
    P.mat1 = input
    P.out_unchanged = False
    if P.early:
        return P
    input = P.mat1 = _cublas_checks(input, mat2, scaling, od)
    P.fa, P.fb = fmt_of(input.dtype), fmt_of(mat2.dtype)
    _profile_rejects(P.prof, P.fa, P.fb, od)
    _scope_checks(scaling, bias, (input, mat2, scale_a, scale_b, scale_result, bias))
    msg = _maturity_message(P.prof, P.preset)
    if msg is not None:
        P.warnings.append(msg)
    if not dry:
        _emit_maturity(P.prof, P.preset)              # emits the ExperimentalPresetWarning (once per name)
    # torch 2.7.1 keeps an out= of shape [M, N] (another shape is resized to a contiguous one); one that is neither
    # row- nor column-major is computed into a contiguous copy that is never copied back (RTX 4060: out unchanged).
    # Its result is discarded, so no split-K decision or backend is needed.
    if out is not None and tuple(out.shape) == (P.M, P.N) and not (_row_major(out) or _col_major(out)):
        P.out_unchanged = True
        P.warnings.append(f"out= with strides {tuple(out.stride())} is neither row- nor column-major: torch 2.7.1 "
                          f"computes into a contiguous copy and never copies it back, so out= is returned unchanged "
                          f"(mmaemu does the same)")
        return P
    layout_std = input.is_contiguous() and mat2.t().is_contiguous()
    # torch runs a row-major result as mat2^T x mat1^T, so the table is keyed by that problem. A column-major out=
    # (kept by torch when its shape is [M, N]) is computed as mat1 x mat2: cuBLASLt's heuristic then sees the
    # problem of a row-major call with shape (N, M, K) and the formats swapped (RTX 4060: 406 of 406 such calls).
    if out is not None and tuple(out.shape) == (P.M, P.N) and _col_major(out):
        key = (P.N, P.M, P.K, mat2.dtype, input.dtype)
    else:
        key = (P.M, P.N, P.K, input.dtype, mat2.dtype)
    P.decision = resolve_splitk(*key[:3], a_dtype=key[3], b_dtype=key[4], out_dtype=od,
                                use_fast_accum=bool(use_fast_accum), profile=P.prof, splitk=splitk,
                                layout_std=layout_std, dry=dry)
    P.req, P.be = _select_backend(input.device, P.fa, P.fb, P.preset.name, backend)
    return P


def _scaled_mm_impl(input, mat2, scale_a, scale_b, bias=None, scale_result=None, out_dtype=None,
                    use_fast_accum=False, *, out=None, profile=None, splitk="auto", backend="auto"):
    P = _prepare(input, mat2, scale_a, scale_b, bias, scale_result, out_dtype, use_fast_accum, out, profile, splitk,
                 backend, dry=False)
    if P.out_unchanged:
        warnings.warn(P.warnings[-1], UserWarning, stacklevel=3)
        return out
    dev = input.device
    if P.early:
        y = torch.zeros(P.M, P.N, dtype=P.od, device=dev)      # torch: empty result, or K == 0 -> out.zero_()
    else:
        if dev.type == "cpu":
            _ftz_canary()
        plan = make_plan(P.K, fast=bool(use_fast_accum), S=P.decision.S, scheme=P.decision.scheme,
                         preset=P.preset)
        acc = execute_plan(plan, P.mat1.view(torch.uint8), mat2.t().view(torch.uint8), fmt_a=P.fa, fmt_b=P.fb,
                           chain_fn=P.be.chain_fn(P.req))
        y = epilogue.apply(acc, scale_a, scale_b, out_dtype=P.od, scale_result=scale_result)
    if out is None:
        return y
    if tuple(out.shape) != (P.M, P.N):
        if out.numel() != 0:
            warnings.warn(f"An output with one or more elements was resized since it had shape {list(out.shape)}, "
                          f"which does not match the required output shape {[P.M, P.N]}.", UserWarning, stacklevel=3)
        out.resize_(P.M, P.N)
    out.copy_(y)
    return out


def scaled_mm(input, mat2, scale_a, scale_b, bias=None, scale_result=None, out_dtype=None, use_fast_accum=False,
              *, out=None, profile=None, splitk="auto", backend="auto"):
    """Emulate ``torch._scaled_mm`` (torch 2.7.1 signature, ``self`` exposed as ``input``) with the numerics of the
    profile's GPU, on any device.

    input [M, K] FP8 row-major; mat2 [K, N] FP8 column-major (e.g. ``w.t()`` for a row-major [N, K] weight);
    scale_a, scale_b: one-element fp32 tensors (tensor-wise scales). Returns [M, N] of ``out_dtype`` (default:
    input.dtype, as torch) on input.device; CPU tensors are allowed.

    Scope: tensor-wise scales, FP8 inputs and fp32 / bf16 / fp16 / e4m3 / e5m2 outputs, no bias (raises
    UnsupportedFeatureError), inference only. Bitwise equality with torch._scaled_mm is validated on the reference
    profiles' GPUs (hopper, hopper-te, hopper-ws0, hopper the default: measured on an H100 PCIe, 114 SMs; ada,
    ada-te, ada-ws0: measured on an RTX 4060, 24 SMs; both
    cuBLASLt 12.8.3.14, torch 2.7.1) for the split-K decisions the profile resolves; elsewhere the profile is the
    claim. See the module docstring for the validation order.

    profile: Profile or name (default $MMAEMU_PROFILE or "hopper"). splitk: "auto" (from the profile;
    SplitKUnknownError if it cannot decide), 1, SplitK or (S, scheme), scheme "serial"|"pairwise" on the ada
    profiles and "contiguous" on the hopper ones. backend: "auto" | "reference" | "cuda" | "cuda:v1" | "cuda:v4" |
    "cuda:v9" | "cuda:tc3" | "cuda:h1" | "cuda:h4" (each kernel serves one preset; docs/backends.md)."""
    return _scaled_mm_impl(input, mat2, scale_a, scale_b, bias, scale_result, out_dtype, use_fast_accum, out=out,
                           profile=profile, splitk=splitk, backend=backend)


def accumulate(a, b, *, use_fast_accum, profile=None, splitk="auto", backend="auto",
               out_dtype_for_split=torch.float32):
    """fp32 [M, N] accumulator of a [M, K] x b [N, K]^T before alpha, as the profile's cuBLAS computes it.

    a, b: float8_e4m3fn / float8_e5m2 tensors (any K, alignment or layout; K is zero padded internally). NaN is
    canonical. ``use_fast_accum`` is required (no default: a mismatched default silently changed 99.56% of outputs
    in the tcemu prototype). The split-K decision is keyed with ``out_dtype_for_split`` (cuBLAS picks splits per
    output dtype)."""
    for name, v in (("a", a), ("b", b)):
        if not isinstance(v, torch.Tensor):
            raise TypeError(f"accumulate(): argument '{name}' must be Tensor, not {type(v).__name__}")
    if not _is_bool(use_fast_accum):
        raise TypeError(f"accumulate(): use_fast_accum must be bool, not {type(use_fast_accum).__name__}")
    if not isinstance(out_dtype_for_split, torch.dtype):
        raise TypeError("accumulate(): out_dtype_for_split must be a torch.dtype")
    fa, fb = fmt_of(a.dtype), fmt_of(b.dtype)
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError(f"accumulate(): expected a [M, K] and b [N, K], got {tuple(a.shape)} and {tuple(b.shape)}")
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"accumulate(): K mismatch: a is {tuple(a.shape)}, b is {tuple(b.shape)}")
    if a.device != b.device:
        raise ValueError(f"accumulate(): a and b are on different devices ({a.device}, {b.device})")
    if torch.is_grad_enabled() and (a.requires_grad or b.requires_grad):
        raise UnsupportedFeatureError("mmaemu is inference-only: an input requires grad while grad mode is enabled")
    from mmaemu.platform.profiles import get_profile
    from mmaemu.platform.splitk_table import resolve_splitk
    prof = get_profile(profile)
    p = get_preset(prof.preset, warn=False)
    _emit_maturity(prof, p)
    M, K, N = a.shape[0], a.shape[1], b.shape[0]
    if M == 0 or N == 0 or K == 0:
        return torch.zeros(M, N, dtype=torch.float32, device=a.device)
    decision = resolve_splitk(M, N, K, a_dtype=a.dtype, b_dtype=b.dtype, out_dtype=out_dtype_for_split,
                              use_fast_accum=bool(use_fast_accum), profile=prof, splitk=splitk, layout_std=True)
    if a.device.type == "cpu":
        _ftz_canary()
    req, be = _select_backend(a.device, fa, fb, p.name, backend)
    plan = make_plan(K, fast=bool(use_fast_accum), S=decision.S, scheme=decision.scheme, preset=p)
    return execute_plan(plan, a.view(torch.uint8), b.view(torch.uint8), fmt_a=fa, fmt_b=fb,
                        chain_fn=be.chain_fn(req))


def explain(input, mat2, scale_a, scale_b, bias=None, scale_result=None, out_dtype=None, use_fast_accum=False,
            *, out=None, profile=None, splitk="auto", backend="auto") -> CallPlan:
    """Validate a scaled_mm call and report what it would do, without computing (same arguments as scaled_mm)."""
    cp = CallPlan()
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            P = _prepare(input, mat2, scale_a, scale_b, bias, scale_result, out_dtype, use_fast_accum, out, profile,
                         splitk, backend, dry=True)
            if not (P.early or P.out_unchanged):        # the planning checks of scaled_mm (scheme of the preset, ...)
                make_plan(P.K, fast=bool(use_fast_accum), S=P.decision.S, scheme=P.decision.scheme, preset=P.preset)
        cp.warnings = list(P.warnings) + [str(x.message) for x in w]
        cp.profile, cp.preset, cp.preset_maturity = P.prof.name, P.preset.name, P.preset.maturity
        cp.split = P.decision
        if P.early:
            cp.backend = "none (zero-size result)"
        elif P.out_unchanged:
            cp.backend = "none (out= is returned unchanged)"
        else:
            cp.backend = P.be.name
            cp.fallbacks = _fallbacks(P.req, backend)
    except Exception as e:  # noqa: BLE001  (explain reports instead of raising)
        cp.validation = (type(e).__name__, str(e))
        try:
            prof, pre = _profile_and_preset(profile, warn=False)
            cp.profile, cp.preset, cp.preset_maturity = prof.name, pre.name, pre.maturity
        except Exception:  # noqa: BLE001
            pass
    return cp


# torch.compile must never trace into mmaemu: Inductor/Triton <= 3.8 fp8 conversion double-rounds (found by the
# tcemu audit: 1,032,066 wrong codes). The wrappers keep the signatures (inspect follows __wrapped__).
scaled_mm = torch.compiler.disable(scaled_mm)
accumulate = torch.compiler.disable(accumulate)
explain = torch.compiler.disable(explain)
