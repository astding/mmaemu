"""Does a padded leading dimension of out= change torch._scaled_mm's result on the RTX 4060? (column-major follow-up)"""
import json, sys
import torch
OUT = {0: torch.float32, 1: torch.bfloat16, 2: torch.float16, 3: torch.float8_e4m3fn, 4: torch.float8_e5m2}
FMT = {0: torch.float8_e4m3fn, 1: torch.float8_e5m2}
dev = torch.device("cuda")
def mk(r, c, fmt, seed, scale):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(r, c, generator=g) * scale).clamp(-448, 448).to(FMT[fmt])
def eq(x, y):
    u = {1: torch.uint8, 2: torch.int16, 4: torch.int32}[x.element_size()]
    return bool((x.contiguous().view(u) == y.contiguous().view(u)).all())
cases = json.load(open(sys.argv[1]))
res = {"row_pad_eq_row": 0, "col_pad_eq_col": 0, "n": 0, "row_pad_diff": [], "col_pad_diff": []}
for i, (key, dec, dec_sw) in enumerate(cases):
    M, N, K, fa, fb, o, f = key
    scale = 0.5 if o in (3, 4) else 4.0
    a, b = mk(M, K, fa, 2 * i, scale).to(dev), mk(N, K, fb, 2 * i + 1, scale).to(dev)
    s = torch.tensor(1.0, device=dev)
    kw = dict(out_dtype=OUT[o], use_fast_accum=bool(f))
    y_row = torch._scaled_mm(a, b.t(), s, s, **kw)
    y_col = torch._scaled_mm(a, b.t(), s, s, out=torch.empty(N, M, dtype=OUT[o], device=dev).t(), **kw)
    rp = torch.empty(M, N + 16, dtype=OUT[o], device=dev)[:, :N]
    cp = torch.empty(N, M + 16, dtype=OUT[o], device=dev)[:, :M].t()
    torch._scaled_mm(a, b.t(), s, s, out=rp, **kw)
    torch._scaled_mm(a, b.t(), s, s, out=cp, **kw)
    res["n"] += 1
    if eq(rp, y_row): res["row_pad_eq_row"] += 1
    else: res["row_pad_diff"].append(key)
    if eq(cp, y_col): res["col_pad_eq_col"] += 1
    else: res["col_pad_diff"].append(key)
print(json.dumps({k: (v if not isinstance(v, list) else v[:10]) for k, v in res.items()}))
print("row_pad_diff", len(res["row_pad_diff"]), "col_pad_diff", len(res["col_pad_diff"]))
