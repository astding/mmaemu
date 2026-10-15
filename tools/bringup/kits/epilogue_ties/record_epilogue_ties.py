"""Record discriminating epilogue cases on the GPU (proposed fix for F6). Run under /data/quant/tcemu_audit/gpu.sh.

Ada fast accumulation leaves 14 significant bits in acc, so for a tensor-wise alpha only 8192 significands exist and
whether any output can tell single from double rounding, or acc*(sa*sb) from (acc*sa)*sb, depends on alpha alone.
For each kind (single/order x bf16/fp16) we search NA alphas on the CPU that make some significand sensitive, run ONE
production-shape GEMM per alpha (M=16384 rows = all significands x 2 signs, K=1920 (4 nonzero), N=768) for fast and
slow accumulation, and keep the sensitive rows (+2 control rows) with the hardware outputs. The CPU test
test_epilogue_ties.py replays them against tcemu.reference.scaled_mm."""
import sys
import torch
from tools.bringup.checks import same_bits
from torch.profiler import profile, ProfilerActivity

OUT = sys.argv[1]   # vendored: the output path is required (never the audit dir)
NA = int(sys.argv[2]) if len(sys.argv) > 2 else 8
dev = "cuda:0"
E4 = torch.float8_e4m3fn
PE = {torch.bfloat16: (8, -126), torch.float16: (11, -14)}
K, N = 1920, 768


def round_sig(x, p, emin):
    e = (torch.frexp(x)[1] - 1).double()
    q = torch.exp2(torch.clamp(e - (p - 1), min=emin - (p - 1)))
    return torch.where(x == 0, x, torch.round(x / q) * q)


def alt_of(kind, acc, sa, sb, out):
    if kind == "single":
        return round_sig(acc.double() * (sa * sb).double(), *PE[out]).float().to(out)
    return ((acc * sa) * sb).to(out)


S = torch.arange(8192, 16384, dtype=torch.float64)
v = S.long()
digits = torch.stack([(v >> 12) * 16, (v >> 8) & 15, (v >> 4) & 15, v & 15], 1).double()
A = torch.zeros(16384, K, dtype=torch.float64); A[:8192, :4] = digits; A[8192:, :4] = -digits
B = torch.zeros(N, K, dtype=torch.float64)
B[:, :4] = torch.tensor([256.0, 256.0, 16.0, 1.0])[None, :] * torch.exp2(-(torch.arange(N) % 4).double())[:, None]
a8, b8 = A.to(E4), B.to(E4)
assert same_bits(a8.double(), A) and same_bits(b8.double(), B)
acc_exact = (torch.cat([S, -S])[:, None] * torch.exp2(-(torch.arange(N) % 4).double())[None, :]).float()
a8d, b8d = a8.to(dev), b8.to(dev)
one = torch.ones((), device=dev)
for fast in (True, False):
    acc = torch._scaled_mm(a8d, b8d.t(), scale_a=one, scale_b=one, out_dtype=torch.float32, use_fast_accum=fast).cpu()
    assert same_bits(acc, acc_exact), "accumulator is not s exactly"

cases, kernels = [], {}
g = torch.Generator().manual_seed(2026)
for kind in ("single", "order"):
    for out in (torch.bfloat16, torch.float16):
        found = 0
        while found < NA:
            sa = (torch.rand((), generator=g) * 0.05 + 0.001).float(); sb = (torch.rand((), generator=g) * 0.003 + 0.0001).float()
            dbl = (S.float() * (sa * sb)).to(out)
            if not bool((alt_of(kind, S.float(), sa, sb, out) != dbl).any()):
                continue
            found += 1
            model = (acc_exact * (sa * sb)).to(out)
            sens_rows = torch.nonzero((alt_of(kind, acc_exact, sa, sb, out) != model).any(1))[:, 0]
            rows = torch.cat([sens_rows[:6], torch.randint(0, 16384, (2,), generator=g)])
            for fast in (True, False):
                key = f"fast={fast},{out}"
                if key not in kernels:
                    with profile(activities=[ProfilerActivity.CUDA]) as prof:
                        torch._scaled_mm(a8d, b8d.t(), scale_a=one, scale_b=one, out_dtype=out, use_fast_accum=fast)
                        torch.cuda.synchronize()
                    kernels[key] = sorted({e.name for e in prof.events() if e.device_type.name == "CUDA"})
                hw = torch._scaled_mm(a8d, b8d.t(), scale_a=sa.to(dev), scale_b=sb.to(dev), out_dtype=out,
                                      use_fast_accum=fast).cpu()
                cases.append(dict(kind=kind, out_dtype=str(out).split(".")[-1], fast=fast, sa=sa.clone(), sb=sb.clone(),
                                  rows=rows.clone(), a=a8.view(torch.uint8)[rows].clone(), out=hw[rows].clone()))
                print(kind, out, fast, f"sa={sa.item():.9g} sb={sb.item():.9g}", "sensitive rows", len(sens_rows),
                      "hw==model(all rows)", bool(same_bits(hw.view(torch.int16), model.view(torch.int16))), flush=True)
env = dict(gpu=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
           torch=torch.__version__, cuda=torch.version.cuda, cublas=__import__("importlib.metadata").metadata.version("nvidia-cublas-cu12"),
           shape=[16384, K, N], kernels=kernels)
torch.save(dict(env=env, b=b8.view(torch.uint8).clone(), cases=cases), OUT)
print(env, flush=True)
print("saved", OUT, len(cases), "cases")
