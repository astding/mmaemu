"""Offline: compare the RTX 4060 outputs of probe_hw.py with mmaemu (CPU, reference backend) as found on PYTHONPATH
(compare-8429385.txt: commit 8429385; compare-fixed.txt: commit 49f1ecf, the review's fixes). Needs hw_out.npz
(not kept: 648 MB)."""
import json, collections, warnings
import numpy as np
import torch
import mmaemu

warnings.simplefilter("ignore")
OUT = {0: torch.float32, 1: torch.bfloat16, 2: torch.float16, 3: torch.float8_e4m3fn, 4: torch.float8_e5m2}
FMT = {0: torch.float8_e4m3fn, 1: torch.float8_e5m2}
UINT = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
z = np.load("hw_out.npz")
meta = json.load(open("hw_out.json"))


def bits(t):
    return t.contiguous().view(UINT[t.element_size()]).numpy()


def sk(d):
    return 1 if d[0] == 1 else (d[0], d[1])


s = torch.tensor(1.0)
tally = collections.Counter()
rows = []
for m in meta["a01"]:
    i = m["i"]
    M, N, K, fa, fb, o, f = m["key"]
    a = torch.from_numpy(z[f"a01_{i}_a"]).view(FMT[fa])
    b = torch.from_numpy(z[f"a01_{i}_b"]).view(FMT[fb])
    kw = dict(out_dtype=OUT[o], use_fast_accum=bool(f), backend="reference")
    hw_row, hw_col = z[f"a01_{i}_row"], z[f"a01_{i}_col"]
    e_def = bits(mmaemu.scaled_mm(a, b.t(), s, s, **kw))
    oc = torch.empty(N, M, dtype=OUT[o]).t()
    e_col = bits(mmaemu.scaled_mm(a, b.t(), s, s, out=oc, **kw))
    e_sw = bits(mmaemu.scaled_mm(a, b.t(), s, s, splitk=sk(m["dec_swapped"]), **kw))
    r = dict(row_eq_def=bool((hw_row == e_def).all()), col_eq_mmaemu_col=bool((hw_col == e_col).all()),
             col_eq_swapped=bool((hw_col == e_sw).all()), row_eq_col=m["row_eq_col"])
    tally.update(k for k, v in r.items() if v)
    rows.append({"key": m["key"], "dec": m["dec"], "dec_swapped": m["dec_swapped"], **r,
                 "n_diff_col_vs_mmaemu": int((hw_col != e_col).sum()), "numel": int(hw_col.size)})
n = len(rows)
print(f"{n} cases (keys at ws 1 MiB whose swapped key records another split)")
for k in ("row_eq_def", "row_eq_col", "col_eq_mmaemu_col", "col_eq_swapped"):
    print(f"  {k:20s} {tally[k]:4d} of {n}")
bad = [r for r in rows if not r["col_eq_swapped"]]
print("cases where hardware col-major out != swapped-key emulation:", len(bad))
for r in bad[:10]:
    print("  ", r)
worst = max(rows, key=lambda r: r["n_diff_col_vs_mmaemu"])
print("largest mismatch, mmaemu today vs hardware (col-major out):", worst)
json.dump(rows, open("a01_rows.json", "w"), indent=0)

# misc
x = torch.from_numpy(z["misc_in_x"]).view(torch.float8_e4m3fn)
big = torch.from_numpy(z["misc_in_big"]).view(torch.float8_e4m3fn)
w = torch.from_numpy(z["misc_in_w"]).view(torch.float8_e4m3fn)
K = 256
for name, mat1 in (("expand", x.expand(64, K)), ("stride2", big[:, ::2])):
    try:
        y = mmaemu.scaled_mm(mat1, w.t(), s, s, out_dtype=torch.bfloat16, backend="reference")
        print(name, "mmaemu OK, equal to hardware:", bool((bits(y) == z[f"misc_{name}"]).all()))
    except Exception as e:  # noqa: BLE001
        print(name, "mmaemu raises", type(e).__name__, str(e).splitlines()[0][:160])
    y = mmaemu.scaled_mm(mat1.contiguous(), w.t(), s, s, out_dtype=torch.bfloat16, backend="reference")
    print(name, "mmaemu on .contiguous() equal to hardware:", bool((bits(y) == z[f"misc_{name}"]).all()))
for dt in ("float64", "int32"):
    for nm, args in ((f"M0_{dt}", (big[:0, :K], w.t())), (f"N0_{dt}", (big[:, :K], w[:0].t())),
                     (f"K0_{dt}", (big[:, :0], w[:, :0].t()))):
        try:
            y = mmaemu.scaled_mm(*args, s, s, out_dtype=getattr(torch, dt))
            print(nm, "mmaemu OK", y.dtype, tuple(y.shape))
        except Exception as e:  # noqa: BLE001
            print(nm, "mmaemu raises", type(e).__name__, str(e).splitlines()[0][:160])
s2 = torch.tensor(2.0)
bias = torch.full((48,), 3.0, dtype=torch.bfloat16)
for nm, fn in (("K0_scale2_bf16", lambda: mmaemu.scaled_mm(big[:, :0], w[:, :0].t(), s2, s2, out_dtype=torch.bfloat16)),
               ("K0_bias_bf16", lambda: mmaemu.scaled_mm(big[:, :0], w[:, :0].t(), s2, s, bias=bias,
                                                          out_dtype=torch.bfloat16)),
               ("K0_e4m3_sr", lambda: mmaemu.scaled_mm(big[:, :0], w[:, :0].t(), s, s, scale_result=s2,
                                                        out_dtype=torch.float8_e4m3fn))):
    hw = z[f"misc_{nm}"]
    y = bits(fn())
    print(nm, "hardware unique bits", np.unique(hw)[:5], "mmaemu unique", np.unique(y)[:5], "equal", bool((hw == y).all()))
