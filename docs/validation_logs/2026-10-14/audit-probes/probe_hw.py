"""Hardware probe for the release-candidate review: column-major out= vs split-K, strided mat1, zero sizes.

Runs torch._scaled_mm on the RTX 4060 only; emulation is compared offline (compare.py). Writes hw_out.npz + hw_out.json.
"""
import json, sys
import numpy as np
import torch

OUT = {0: torch.float32, 1: torch.bfloat16, 2: torch.float16, 3: torch.float8_e4m3fn, 4: torch.float8_e5m2}
FMT = {0: torch.float8_e4m3fn, 1: torch.float8_e5m2}
UINT = {1: torch.uint8, 2: torch.int16, 4: torch.int32}
dev = torch.device("cuda")
print(torch.__version__, torch.cuda.get_device_name(0), file=sys.stderr)


def bits(t):
    return t.contiguous().view(UINT[t.element_size()]).cpu().numpy()


def mk(r, c, fmt, seed, scale):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(r, c, generator=g) * scale).clamp(-448, 448).to(FMT[fmt])


arrays, meta = {}, {"a01": [], "misc": []}
cases = json.load(open(sys.argv[1]))
for i, (key, dec, dec_sw) in enumerate(cases):
    M, N, K, fa, fb, o, f = key
    scale = 0.5 if o in (3, 4) else 4.0
    a, b = mk(M, K, fa, 2 * i, scale), mk(N, K, fb, 2 * i + 1, scale)
    s = torch.tensor(1.0, device=dev)
    ag, bg = a.to(dev), b.to(dev)
    kw = dict(out_dtype=OUT[o], use_fast_accum=bool(f))
    y_row = torch._scaled_mm(ag, bg.t(), s, s, **kw)
    oc = torch.empty(N, M, dtype=OUT[o], device=dev).t()
    y_col = torch._scaled_mm(ag, bg.t(), s, s, out=oc, **kw)
    assert y_col.data_ptr() == oc.data_ptr() and y_col.stride() == (1, M)
    arrays[f"a01_{i}_row"], arrays[f"a01_{i}_col"] = bits(y_row), bits(y_col)
    arrays[f"a01_{i}_a"], arrays[f"a01_{i}_b"] = bits(a), bits(b)
    meta["a01"].append({"i": i, "key": key, "dec": dec, "dec_swapped": dec_sw,
                        "row_eq_col": bool((bits(y_row) == bits(y_col)).all())})
torch.cuda.synchronize()


def misc(name, fn):
    try:
        y = fn()
        torch.cuda.synchronize()
        arrays[f"misc_{name}"] = bits(y) if y.numel() else np.zeros(0, np.uint8)
        r = {"name": name, "ok": True, "dtype": str(y.dtype), "shape": list(y.shape), "stride": list(y.stride())}
    except Exception as e:  # noqa: BLE001
        r = {"name": name, "ok": False, "error": f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"}
    meta["misc"].append(r)


E4 = torch.float8_e4m3fn
s1 = torch.tensor(1.0, device=dev)
s2 = torch.tensor(2.0, device=dev)
K = 256
x = mk(1, K, 0, 1001, 4.0).to(dev)
big = mk(64, 2 * K, 0, 1002, 4.0).to(dev)
w = mk(48, K, 0, 1003, 4.0).to(dev)
arrays["misc_in_x"], arrays["misc_in_big"], arrays["misc_in_w"] = bits(x), bits(big), bits(w)
# A-03: mat1 neither row- nor column-major
misc("expand", lambda: torch._scaled_mm(x.expand(64, K), w.t(), s1, s1, out_dtype=torch.bfloat16))
misc("stride2", lambda: torch._scaled_mm(big[:, ::2], w.t(), s1, s1, out_dtype=torch.bfloat16))
# A-04: zero-size calls with an out dtype cuBLASLt does not support
for dt in ("float64", "int32"):
    misc(f"M0_{dt}", lambda dt=dt: torch._scaled_mm(big[:0, :K], w.t(), s1, s1, out_dtype=getattr(torch, dt)))
    misc(f"N0_{dt}", lambda dt=dt: torch._scaled_mm(big[:, :K], w[:0].t(), s1, s1, out_dtype=getattr(torch, dt)))
    misc(f"K0_{dt}", lambda dt=dt: torch._scaled_mm(big[:, :0], w[:, :0].t(), s1, s1, out_dtype=getattr(torch, dt)))
    misc(f"full_{dt}", lambda dt=dt: torch._scaled_mm(big[:, :K], w.t(), s1, s1, out_dtype=getattr(torch, dt)))
# A-05: K = 0 with non-unit scales and with a bias
misc("K0_scale2_bf16", lambda: torch._scaled_mm(big[:, :0], w[:, :0].t(), s2, s2, out_dtype=torch.bfloat16))
bias = torch.full((48,), 3.0, dtype=torch.bfloat16, device=dev)
misc("K0_bias_bf16", lambda: torch._scaled_mm(big[:, :0], w[:, :0].t(), s2, s1, bias=bias, out_dtype=torch.bfloat16))
misc("K0_e4m3_sr", lambda: torch._scaled_mm(big[:, :0], w[:, :0].t(), s1, s1, scale_result=s2, out_dtype=E4))

np.savez("hw_out.npz", **arrays)
json.dump(meta, open("hw_out.json", "w"), indent=1)
n = len(meta["a01"])
print("A-01 cases", n, "row==col on hardware:", sum(m["row_eq_col"] for m in meta["a01"]), "of", n)
for r in meta["misc"]:
    print(r)
