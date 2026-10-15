"""The mutation matrix (T4). Every mutant changes ONE modelling decision in a temporary copy of the
package; the CPU tier must be killed by T0+T1, the GPU tier by T3. A surviving non-equivalent mutant blocks the
release. "identity" mutants inject an override with no change and must SURVIVE (they prove the injection is sound);
"control" mutants must be KILLED (they prove the harness can fail).

Edit kinds (applied by run.py to <tmp>/<file>):
  ("override", file, code)   append `code` at the end of the module (impl.py is available as mmaemu._mutation_impl)
  ("replace", file, old, new, count)   exact text replacement; the count is asserted (kernel sources)
  ("remove", path)           delete a file or directory of the copy (e.g. committed fatbins for kernel mutants)
"""
from __future__ import annotations

CHAIN = "mmaemu/model/chain.py"
PLAN = "mmaemu/model/plan.py"
EPI = "mmaemu/model/epilogue.py"
API = "mmaemu/api.py"
V1 = "mmaemu/backends/cuda/csrc/v1.cu"
NVRTC = "mmaemu/backends/cuda/nvrtc.py"
BIN = "mmaemu/backends/cuda/bin"


def chain(knobs):
    return [("override", CHAIN, "from mmaemu._mutation_impl import chain_overrides as _mut_co\n"
                                f"dot, chain_gemm = _mut_co(dot, chain_gemm, {knobs!r})\n")]


def plan(knobs):
    return [("override", PLAN, "from mmaemu._mutation_impl import plan_overrides as _mut_po\n"
                               f"make_plan = _mut_po(make_plan, GemmPlan, {knobs!r})\n")]


def execute(knobs):
    return [("override", PLAN, "from mmaemu._mutation_impl import execute_override as _mut_eo\n"
                               f"execute_plan = _mut_eo(execute_plan, {knobs!r})\n")]


def epi(knobs):
    return [("override", EPI, "from mmaemu._mutation_impl import epilogue_override as _mut_ep\n"
                              f"apply = _mut_ep(apply, {knobs!r})\n")]


def api(mode):
    return [("override", API, "from mmaemu._mutation_impl import api_override as _mut_api\n"
                              f"scaled_mm = _mut_api(scaled_mm, {{'mode': {mode!r}}})\n")]


# Where each mutant comes from (report labels only; nothing is read from these places). The audit scripts belong to
# the unpublished audit of the tcemu prototype; "v0.1 catalogue" mutants were added for v0.1.
C2 = "tcemu audit: verify-c2-instr-fuzz-0/mut/make_mutants.py"
C4 = "tcemu audit: verify-c4-cublas-fuzz-8/mutants.py"
A1 = "tcemu audit: verify-a1-reference-review-5/scripts/v3_mutation.py"
A3 = "tcemu audit: verify-a3-integration-review-5/build.py"
V01 = "v0.1 catalogue"

MINEXP_EVIDENCE = ("tcemu audit finding: 'min_exp is unobservable (-132, -133 or smaller give the same results)', "
                   "from its scripts v8_minexp_unconstrained.py and v8b_minexp_vs_hw.py; for an FP8 product the block "
                   "exponent is >= -32 and the fp32 result exponent is clamped at -126 before truncation, so only c "
                   "values below 2^-132 could observe it")

MUTANTS = [
    # ---- harness self-checks ---------------------------------------------------------------------------------
    dict(name="identity_chain", group="identity", tier="cpu", edits=chain({}), source="tools/mutants/impl.py",
         desc="re-implementation of dot/chain_gemm with no change (must survive)"),
    dict(name="identity_plan", group="identity", tier="cpu", edits=plan({}), source="tools/mutants/impl.py",
         desc="re-implementation of make_plan with no change (must survive)"),
    dict(name="identity_execute", group="identity", tier="cpu", edits=execute({}), source="tools/mutants/impl.py",
         desc="re-implementation of execute_plan with no change (must survive)"),
    dict(name="identity_epilogue", group="identity", tier="cpu", edits=epi({}), source="tools/mutants/impl.py",
         desc="re-implementation of epilogue.apply with no change (must survive)"),
    dict(name="identity_api", group="identity", tier="cpu", edits=api("identity"), source="tools/mutants/impl.py",
         desc="pass-through wrapper of scaled_mm (must survive)"),
    dict(name="control_sign_flip", group="control", tier="cpu", edits=epi({"sign_flip_nonzero": True}),
         source=A1, desc="every nonzero output negated (must be killed)"),
    # ---- instruction model (c2 make_mutants, c4 mutants, a1 v3_mutation) ------------------------------------------
    dict(name="sub_true_exp", group="model", tier="cpu", edits=chain({"emin_clamp": False}), source=C2,
         desc="subnormal inputs use their true exponent (no clamp at the format emin)"),
    dict(name="phantom_zero", group="model", tier="cpu", edits=chain({"zero_products_in_max": True}), source=C2,
         desc="zero products take part in the block maximum exponent"),
    dict(name="c_pass", group="model", tier="cpu", edits=chain({"allzero_c_pass": True}), source=C2,
         desc="an all-zero block returns c without truncating it"),
    dict(name="ieee_zero", group="model", tier="cpu", edits=chain({"ieee_zero": True, "m1": False}), source=C2,
         desc="exact-zero block results follow IEEE sign rules instead of +0"),
    dict(name="m1_off", group="model", tier="cpu", edits=chain({"m1": False}), source=A1,
         desc="c2-M1 off: a truncated-to-zero block result keeps the sign of the exact sum (-0)"),
    dict(name="min_exp_-126", group="model", tier="cpu", edits=chain({"min_exp": -126}), source=C2,
         desc="alignment exponent clamped at -126 instead of -132", equivalent_evidence=MINEXP_EVIDENCE),
    dict(name="min_exp_-140", group="model", tier="cpu", edits=chain({"min_exp": -140}), source=C4,
         desc="alignment exponent clamped at -140", equivalent_evidence=MINEXP_EVIDENCE),
    dict(name="nfma32", group="model", tier="cpu", edits=chain({"nfma": 32}), source=C4, desc="32-term blocks"),
    dict(name="frac12", group="model", tier="cpu", edits=chain({"frac": 12, "frac_out": 12}), source=C4,
         desc="12 fraction bits (positive control CTRL_frac12_ada of c2)"),
    dict(name="frac14", group="model", tier="cpu", edits=chain({"frac": 14, "frac_out": 14}), source=C4,
         desc="14 fraction bits"),
    dict(name="frac_out23", group="model", tier="cpu", edits=chain({"frac_out": 23}), source=C4,
         desc="block result kept at full fp32 precision"),
    dict(name="align_rne", group="model", tier="cpu", edits=chain({"align": "rne"}), source=C4,
         desc="aligned terms rounded to nearest even instead of truncated"),
    dict(name="norm_rne", group="model", tier="cpu", edits=chain({"norm": "rne"}), source=C4,
         desc="block result rounded to nearest even instead of truncated"),
    dict(name="c_exact", group="model", tier="cpu", edits=chain({"c_trunc": False}), source=C4,
         desc="c enters the block sum exactly (not truncated at alignment)"),
    dict(name="ed_noclamp", group="model", tier="cpu", edits=chain({"ed_clamp": None}), source=C4,
         desc="result exponent not clamped at -126 before normalisation"),
    dict(name="rival_exact_rn", group="model", tier="cpu", edits=chain({"model": "exact_rn"}), source=C4,
         desc="rival model: exact dot product rounded once to fp32"),
    dict(name="rival_fp32_fma_seq", group="model", tier="cpu", edits=chain({"model": "fp32_fma_seq"}), source=C4,
         desc="rival model: sequential fp32 FMA along K"),
    # ---- GEMM plan --------------------------------------------------------------------------------------------
    dict(name="split_contiguous", group="plan", tier="cpu", edits=plan({"split": "contiguous"}), source=V01,
         desc="split s owns a contiguous slice of k-tiles instead of the tiles t mod S = s"),
    dict(name="scheme_swapped", group="plan", tier="cpu", edits=plan({"swap_scheme": True}), source=V01,
         desc="serial <-> pairwise split-K reduction"),
    dict(name="slow_chunk64", group="plan", tier="cpu", edits=plan({"slow_chunk": 64}), source=C4,
         desc="slow accumulation restarts every 64 elements"),
    dict(name="slow_chunk256", group="plan", tier="cpu", edits=plan({"slow_chunk": 256}), source=C4,
         desc="slow accumulation restarts every 256 elements"),
    dict(name="slow_restart_global_tiles", group="plan", tier="cpu", edits=plan({"split_slow_global": True}),
         source=V01, desc="split-K slow restart counted on global 128-element tiles instead of the split's own"),
    dict(name="fast_restart2048", group="plan", tier="cpu", edits=plan({"fast_restart": 2048}), source=C4,
         desc="fast accumulation restarts every 2048 elements (fp32 adds)"),
    dict(name="fast_restart4096", group="plan", tier="cpu", edits=plan({"fast_restart": 4096}), source=C4,
         desc="fast accumulation restarts every 4096 elements (fp32 adds)"),
    dict(name="fast_restart1024", group="plan", tier="cpu", edits=plan({"fast_restart": 1024}), source=C4,
         desc="fast accumulation restarts every 1024 elements (fp32 adds)"),
    dict(name="slow_tail_head", group="plan", tier="cpu", edits=plan({"slow_tail": "head"}), source=C4,
         desc="the partial slow chunk sits at the start of K instead of the end"),
    dict(name="slow_chunk_sum_reversed", group="plan", tier="cpu", edits=plan({"reverse_chunks": True}),
         source=V01, desc="slow chunk results added in reverse order"),
    dict(name="slow_sum_fp64", group="plan", tier="cpu", edits=execute({"chunk_sum": "fp64"}), source=C4,
         desc="slow chunk results summed in float64, rounded once"),
    dict(name="slow_sum_pairwise", group="plan", tier="cpu", edits=execute({"chunk_sum": "pairwise"}), source=C4,
         desc="slow chunk results summed as an fp32 binary tree"),
    dict(name="alpha_before_reduction", group="plan", tier="cpu",
         edits=execute({"alpha_before_reduction": True}) + epi({}), source=V01,
         desc="alpha applied to each split-K partial before the reduction"),
    # ---- epilogue ---------------------------------------------------------------------------------------------
    dict(name="epi_single_round", group="epilogue", tier="cpu", edits=epi({"epi": "single"}), source=C4,
         desc="bf16/fp16 rounded once from the exact acc*alpha (no fp32 intermediate)"),
    dict(name="epi_sa_then_sb", group="epilogue", tier="cpu", edits=epi({"epi": "sa_sb"}), source=C4,
         desc="(acc*sa)*sb instead of acc*fp32(sa*sb)"),
    dict(name="epi_fp32_mul_add0", group="epilogue", tier="cpu", edits=epi({"epi": "fp32_mul_add0"}),
         source="tcemu audit (epilogue)", desc="fp32 acc*alpha + 0.0 instead of fma(acc, alpha, +0)"),
    dict(name="fp8_saturation_off", group="epilogue", tier="cpu", edits=epi({"saturate": False}), source=V01,
         desc="fp8 outputs not saturated (Inf/NaN on overflow)"),
    dict(name="scale_result_on_bf16", group="epilogue", tier="cpu", edits=epi({"sr_nonfp8": True}), source=V01,
         desc="scale_result also applied to non-fp8 outputs"),
    dict(name="scale_result_after_sat", group="epilogue", tier="cpu", edits=epi({"sr_after_sat": True}), source=V01,
         desc="scale_result applied after saturation"),
    dict(name="nan_payload_kept", group="epilogue", tier="cpu", edits=epi({"nan": "keep"}), source=V01,
         desc="NaN outputs not canonicalised"),
    dict(name="nan_negpayload", group="epilogue", tier="cpu", edits=epi({"nan": "negpayload"}), source=C4,
         desc="NaN outputs all-ones"),
    dict(name="zero_plus", group="epilogue", tier="cpu", edits=epi({"zero": "plus"}), source=C4,
         desc="every zero output +0"),
    dict(name="zero_flip", group="epilogue", tier="cpu", edits=epi({"zero": "flip"}), source=C4,
         desc="the sign of every zero output flipped (a1 v3 M2: -0 where +0)"),
    # ---- API ----------------------------------------------------------------------------------------------------
    dict(name="api_default_fast_accum", group="api", tier="cpu", edits=api("default_fast"), source=V01,
         desc="use_fast_accum defaults to True"),
    dict(name="api_default_out_fp32", group="api", tier="cpu", edits=api("default_fp32"), source=V01,
         desc="out_dtype defaults to float32"),
    dict(name="api_positional_bias_dropped", group="api", tier="cpu", edits=api("drop_positional_bias"),
         source=V01, desc="a positional bias is silently dropped"),
    dict(name="api_e4m3_e4m3_to_e5m2_allowed", group="api", tier="cpu", edits=api("allow_e4m3_e5m2_out"),
         source=V01, desc="e4m3 x e4m3 -> e5m2 computed instead of rejected"),
    # ---- kernels (GPU tier, T3): v1 source from csrc/ada_fp8_gemm.cu v1 (audit build.py anchors) ------------------
    dict(name="kernel_identity", group="identity", tier="gpu", edits=[("remove", BIN)], source=A3,
         desc="committed fatbins removed: the shipped sources are compiled by NVRTC (must survive)"),
    dict(name="kernel_rn", group="kernel", tier="gpu", source=A3, desc="trunc_to rounds to nearest",
         edits=[("remove", BIN), ("replace", V1, "__fadd_rz(x, m)", "__fadd_rn(x, m)", 1)]),
    dict(name="kernel_noacc", group="kernel", tier="gpu", source=A3,
         desc="the accumulator does not take part in the block alignment",
         edits=[("remove", BIN), ("replace", V1, "e = max(max(e, fexp(acc[i][j])), -100);", "e = max(e, -100);", 1)]),
    dict(name="kernel_rowtile", group="kernel", tier="gpu", source=A3,
         desc="outputs of thread blocks with blockIdx.y > 0 perturbed by one ulp",
         edits=[("remove", BIN), ("regex", V1, r"(C\[\(size_t\)gm \* N \+ gn\] = )", r"\1(blockIdx.y ? 1.0000001f : 1.f) * ", 1)]),
    dict(name="kernel_floor", group="kernel", tier="gpu", source=A3, desc="tail row tiles never launched",
         edits=[("remove", BIN), ("regex", "mmaemu/backends/cuda/kernels.py",
                                  r"-\(-m // spec\.tile_m\)", "(m // spec.tile_m)", 1)]),
    dict(name="kernel_nan_as_zero", group="kernel", tier="gpu", source="tcemu audit (NaN decode)",
         desc="NaN codes decoded as 0 (prototype v1) -- reached only if execute_plan's sanitisation is bypassed",
         edits=[("remove", BIN), ("regex", "mmaemu/model/specials.py", r"(def sanitize\([^)]*\)[^:]*:)",
                                  r"\1\n    return codes_u8, False", 1),
                ("replace", "mmaemu/backends/cuda/csrc/common.cuh",
                 "return ((v & 0x7f) == 0x7f) ? 0x7fffffffu",
                 "return ((v & 0x7f) == 0x7f) ? 0u", 1)]),
    dict(name="kernel_zero_lut", group="kernel", tier="gpu", source="tcemu audit (per-context tables)",
         desc="__constant__ decode table all zeros",
         edits=[("remove", BIN), ("regex", V1, r"(__constant__ uint32_t c_valbits\[256\] =\s*\{)[^}]*(\})",
                                  r"\1 0 \2", 1)]),
    dict(name="kernel_fmad_true", group="kernel", tier="gpu", source="v0.1 build flags (--fmad=false pinned)",
         desc="kernels compiled with --fmad=true",
         edits=[("remove", BIN), ("replace", NVRTC, "--fmad=false", "--fmad=true", None)]),
]

GROUPS_MUST_KILL = {"model", "plan", "epilogue", "api", "kernel", "control"}
