"""CPU analysis (no GPU).  usage: python analyse.py hw     -> results/analyse_hw.json
                                  python analyse.py sweep  -> results/analyse_sweep.json
hw:    (1) 4060 raw E-pass MMA outputs (v9 data path) vs. the exact sum and vs. the published Ada / Hopper-Blackwell
           bf16 models (bit-exact fp32 comparison), the v9 exponent mapping on the hardware values, d-field histogram;
       (2) 4060 raw bf16 MMA outputs on general inputs vs. the two published models (validates tc_models.py).
sweep: every published + hypothetical model (and range-limited mutants) on the E-pass vectors: does v9's exponent
       mapping (pbits64_raw / pbits64_hi) still return 2^E?"""
import json, sys, time
from pathlib import Path
import torch

import os as _os
from pathlib import Path as _Path
def _bringup_out():
    d = _Path(_os.environ.get("BRINGUP_OUT") or ".")
    d.mkdir(parents=True, exist_ok=True)
    return d
def _bringup_cache():
    d = _Path(_os.environ.get("BRINGUP_CACHE") or (_Path.home() / ".cache" / "mmaemu" / "bringup"))
    d.mkdir(parents=True, exist_ok=True)
    return d

W = Path(__file__).resolve().parent
sys.path.insert(0, str(W))
import tc_models as T  # noqa: E402
import vectors as V  # noqa: E402

I64 = torch.int64


def f32bits(d):
    return d.to(torch.float32).view(torch.int32).to(I64) & 0xFFFFFFFF


def epass_pairs(At, Bt):
    A, B = V.pairs_from_tiles(At.long(), Bt.long())
    return A, B


def hw_raw_to_rows(d):  # [T,16,8] int32 -> [T*128] (row-major (i, j)) matching pairs_from_tiles
    return d.reshape(-1).to(I64) & 0xFFFFFFFF


def bf16_terms(A, B):
    """A [T,16,16] int16 bits (rows i, k), B [T,8,16] (cols j, k) -> product terms for all (i, j): [T*128, 16]."""
    Tn = A.shape[0]
    a = (A.to(I64) & 0xFFFF)[:, :, None, :].expand(Tn, 16, 8, 16).reshape(-1, 16)
    b = (B.to(I64) & 0xFFFF)[:, None, :, :].expand(Tn, 16, 8, 16).reshape(-1, 16)
    def dec(x):
        s = (x >> 15) & 1; ef = (x >> 7) & 0xFF; m = x & 0x7F
        assert bool(((ef > 0) | (m == 0)).all()) and bool((ef < 255).all()), "no subnormal/special bf16 in the families"
        return s, 128 + m, ef - 127, ef > 0
    sa, ma, ea, nza = dec(a); sb, mb, eb, nzb = dec(b)
    S = (ma * mb) << 9          # prod_sig * 2^23, prod_sig = ma*mb/2^14 in [1, 4)
    e = ea + eb
    return S, e, sa ^ sb, nza & nzb


def run_hw():
    res = {}
    hw = torch.load(_bringup_out() / "hw_epass_raw.pt")
    models = dict(T.PUBLISHED)
    models["exact then rz"] = dict(kind="exact", frmode="rz")
    for name, blob in hw.items():
        A, B = epass_pairs(blob["At"], blob["Bt"])
        S, e, sg, nz, c, E = T.epass_terms(A, B)
        dh = hw_raw_to_rows(blob["d"])
        dhw = dh.to(torch.int32).view(torch.float32).to(torch.float64)
        okr, okh, df = T.v9_exponent_ok(dhw, E)
        r = dict(outputs=int(len(dh)), v9_map_fail_raw=int((~okr).sum()), v9_map_fail_hi=int((~okh).sum()),
                 dfield_hist={int(k): int(v) for k, v in zip(*torch.unique(df, return_counts=True))},
                 min_ratio_d_over_2pow=float((dhw / torch.ldexp(torch.ones_like(dhw), (6 * E - 14).double())).min()),
                 max_ratio_d_over_2pow=float((dhw / torch.ldexp(torch.ones_like(dhw), (6 * E - 14).double())).max()))
        for mn, m in models.items():
            dm = f32bits(T.mma_model(S, e, sg, nz, c, m))
            r[f"hw_vs[{mn}]_bit_mismatches"] = int((dm != dh).sum())
        res[f"epass_{name}"] = r
        print(name, json.dumps(r, indent=1), flush=True)
    hw = torch.load(_bringup_out() / "hw_bf16_raw.pt")
    for name, blob in hw.items():
        S, e, sg, nz = bf16_terms(blob["A"], blob["B"])
        c = blob["C"].reshape(-1).double()
        dh = hw_raw_to_rows(blob["d"])
        dhv = dh.to(torch.int32).view(torch.float32)
        r = dict(outputs=int(len(dh)))
        for mn, m in T.PUBLISHED.items():
            dmv = T.mma_model(S, e, sg, nz, c, m).to(torch.float32)
            r[f"hw_vs[{mn}]_value_mismatches"] = int((dmv != dhv).sum())
            r[f"hw_vs[{mn}]_bit_mismatches"] = int((f32bits(dmv.double()) != dh).sum())
        # exact (float64 dot can be inexact for wide spans; only used as a 'truncation happens' indicator)
        vals = torch.ldexp(S.double(), (e - 23).double()) * (1 - 2 * sg).double()
        ex = (torch.where(nz, vals, torch.zeros_like(vals)).sum(1) + c).to(torch.float32)
        r["hw_vs_float64_sum_rounded_to_fp32_value_mismatches"] = int((ex != dhv).sum())
        res[f"bf16_{name}"] = r
        print(name, json.dumps(r, indent=1), flush=True)
    json.dump(res, open(_bringup_out() / "analyse_hw.json", "w"), indent=1)


def run_sweep():
    t0 = time.time()
    sets = {}
    At, Bt = V.prims8_tiles()
    sets["prims8_tiles_all_pairs"] = epass_pairs(At, Bt)
    A, B, meta = V.structural()
    sets["structural_rows"] = (A, B)
    models = {}
    models.update(T.PUBLISHED); models.update(T.hypothetical_models()); models.update(T.mutant_models())
    res = {}
    for sn, (A, B) in sets.items():
        S, e, sg, nz, c, E = T.epass_terms(A, B)
        res[sn] = {}
        for mn, m in models.items():
            if sn == "prims8_tiles_all_pairs" and not (mn in T.PUBLISHED or "MUTANT" in mn or mn.startswith("exact")
                                                        or mn.startswith("seq") or mn.startswith("pairwise")
                                                        or ("nfma=16" in mn and "neab=2 " in mn)
                                                        or ("nfma=8" in mn and "neab=1 " in mn)
                                                        or "neab=-23" in mn or "14-bit" in mn or "late_c" in mn):
                continue
            d = T.mma_model(S, e, sg, nz, c, m)
            okr, okh, df = T.v9_exponent_ok(d, E)
            bad = ~(okr & okh)
            r = dict(outputs=int(len(d)), fail_raw=int((~okr).sum()), fail_hi=int((~okh).sum()),
                     dfield_min=int(df.min()), dfield_max=int(df.max()))
            if bad.any():
                r["fail_E_values"] = sorted(set(E[bad].tolist()))[:40]
            res[sn][mn] = r
        nmod = len(res[sn]); nfail = sum(1 for k, v in res[sn].items() if v["fail_raw"] or v["fail_hi"])
        print(f"{sn}: {len(A)} rows, {nmod} models, models with failures: "
              f"{[k for k, v in res[sn].items() if v['fail_raw'] or v['fail_hi']]}  ({time.time() - t0:.0f}s)", flush=True)
    json.dump(res, open(_bringup_out() / "analyse_sweep.json", "w"), indent=1)


if __name__ == "__main__":
    {"hw": run_hw, "sweep": run_sweep}[sys.argv[1]]()
