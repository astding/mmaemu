"""Release-candidate review, round 2: envelope K extrapolation, outs without unit stride, fnuz/e8m0 zero size."""
import json, itertools
import torch
from mmaemu.platform import cublaslt
from mmaemu.platform.profiles import get_profile
from mmaemu.platform.splitk_table import load_table

dev = torch.device("cuda", 0)
res = {"a09": [], "misc": []}
OUTS = ["float32", "bfloat16", "float16", "float8_e4m3fn", "float8_e5m2"]
PAIRS = [("e4m3", "e4m3"), ("e4m3", "e5m2"), ("e5m2", "e4m3")]
for prof_name in ("ada-rtx4060-torch", "ada-rtx4060-te"):
    p = get_profile(prof_name)
    t = load_table(p.splitk_table)
    for od, fast, (fa, fb) in itertools.product(OUTS, (True, False), PAIRS):
        if od == "float8_e5m2" and (fa, fb) == ("e4m3", "e4m3"):
            continue
        env, how = t.envelope(p.workspace_bytes, od, fast, fa, fb)
        if env is None or not env["split_rows"]:
            continue
        mx = env["max_split_MN"]
        for M in (16, 64, 256, 1024):
            for f in (1.05, 2, 8):
                N = max(16, -(-int(mx * f / M) // 16) * 16)
                if M * N <= mx:
                    N += 16
                for K in (16384, 32768, 65536, 262144, 1048576):
                    r = cublaslt.heuristic(M, N, K, fmt_a=fa, fmt_b=fb, out_dtype=od, fast=fast,
                                           workspace_bytes=p.workspace_bytes, align=p.align, device=dev)
                    res["a09"].append(dict(profile=prof_name, ws=p.workspace_bytes, out=od, fast=fast, fa=fa, fb=fb,
                                           M=M, N=N, K=K, mx=mx, status=r["status"], S=r["splitK"], custom=r["custom"]))
s = torch.tensor(1.0, device=dev)
E4 = torch.float8_e4m3fn
a = (torch.randn(16, 64, device=dev) * 4).to(E4)
b = (torch.randn(32, 64, device=dev) * 4).to(E4)
o = torch.full((16, 64), float("nan"), dtype=torch.bfloat16, device=dev)[:, ::2]
y = torch._scaled_mm(a, b.t(), s, s, out_dtype=torch.bfloat16, out=o)
torch.cuda.synchronize()
ref = torch._scaled_mm(a, b.t(), s, s, out_dtype=torch.bfloat16)
res["misc"].append(dict(name="out step 2 (no unit stride), prefilled NaN", returned_is_out=y.data_ptr() == o.data_ptr(),
                        out_still_all_nan=bool(o.isnan().all()), out_equals_product=bool((o == ref).all()),
                        stride=list(o.stride())))
for dt in ("float8_e4m3fnuz", "float8_e5m2fnuz", "float8_e8m0fnu"):
    if not hasattr(torch, dt):
        res["misc"].append(dict(name=dt, error="no such dtype"))
        continue
    D = getattr(torch, dt)
    for M, K, N in ((0, 32, 16), (16, 0, 16), (16, 32, 0), (16, 32, 16)):
        x = torch.zeros(M, K, device=dev).to(D) if dt != "float8_e8m0fnu" else torch.ones(M, K, device=dev).to(D)
        w = torch.zeros(N, K, device=dev).to(E4)
        try:
            y = torch._scaled_mm(x, w.t(), s, s, out_dtype=torch.float32)
            torch.cuda.synchronize()
            r = dict(ok=True, shape=list(y.shape), any_nonzero=bool(y.any()) if y.numel() else None)
        except Exception as e:  # noqa: BLE001
            r = dict(ok=False, error=f"{type(e).__name__}: {str(e).splitlines()[0][:160]}")
        res["misc"].append(dict(name=f"{dt} mat1 M={M} K={K} N={N}", **r))
json.dump(res, open("probe3.json", "w"), indent=1)
a09 = res["a09"]
bad = [r for r in a09 if r["status"] == 0 and r["S"] > 1]
print("A-09 queries", len(a09), "status!=0", sum(r["status"] != 0 for r in a09), "split (S>1) outside envelope:", len(bad))
for r in bad[:15]:
    print("  ", r)
import collections
print("split by K:", collections.Counter(r["K"] for r in bad))
for m in res["misc"]:
    print(m)
