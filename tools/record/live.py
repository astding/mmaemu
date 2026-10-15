"""Live torch._scaled_mm calls with per-call cuBLASLt config capture (used by R2-verify, R4, R5).

A job is a dict:
  {"id", "M", "N", "K", "fa", "fb", "out" (dtype name), "fast" (bool), "ws" (bytes),
   "data": {"kind": "randn40" | "adv", "seed": int}, "scale_a", "scale_b" (python floats, rounded to fp32),
   "scale_result" (float or null), "sub": {"rows": [...], "cols": [...]} or null (null = store everything),
   "table": {...} (the table row the job was built from; stored as params.table_row), "tags": [...]}
Inputs are regenerated deterministically from (kind, seed, shape, formats) by make_inputs() (torch CPU generator),
so the CPU analysis (choice of discriminating rows/columns) and the GPU worker see the same bytes; the stored
recording keeps the raw bytes of the stored subset (no generators).

usage:
  python tools/record/live.py run --jobs JOBS.json --stage DIR      (GPU, under gpu_run.py: one subprocess per ws)
  python tools/record/live.py worker --jobs JOBS.json --out FILE    (internal; env carries the workspace + log level)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

FP8 = {"e4m3": "float8_e4m3fn", "e5m2": "float8_e5m2"}
FP8_MAX = {"e4m3": 448.0, "e5m2": 57344.0}


M32 = 0xFFFFFFFF
_C1, _C2, _C3, _C4 = 0x7FEB352D, 0x2C1B3C6D, 0x297A2D39, 0x632BE5AB   # all < 2^31: int64 products never overflow


def _h32(x):
    """32-bit integer mix (lowbias32-style) on int64 tensors holding values in [0, 2^32); exact on CPU and CUDA."""
    x = x ^ (x >> 16)
    x = (x * _C1) & M32
    x = x ^ (x >> 15)
    x = (x * _C2) & M32
    return x ^ (x >> 16)


def _key(seed: int, salt: int, stream: int) -> int:
    x = (seed * 0x01000193 + salt * 0x9E37 + stream * 0x3C6EF + 0x3C6EF372) & M32
    for _ in range(2):
        x ^= x >> 16
        x = (x * _C1) & M32
        x ^= x >> 15
        x = (x * _C2) & M32
        x ^= x >> 16
    return x


def _u32(seed, salt, stream, rows, cols):
    """uniform uint32 (as int64) [len(rows), len(cols)] from (seed, salt, stream, row, col)."""
    r = _h32((rows ^ _key(seed, salt, stream)) & M32)
    return _h32((r[:, None] + cols[None, :] * _C3 + stream * _C4) & M32)


_CDF = {}


def _randn_table(fmt: str, scale: float = 40.0):
    """(thresholds int64 [n], codes uint8 [n]) for sampling the code distribution of RNE_fp8(clamp(scale*N(0,1)))
    with a uniform uint32: code = codes[searchsorted(thresholds, u, right=True)]."""
    if (fmt, scale) not in _CDF:
        import math
        import torch
        dt = getattr(torch, FP8[fmt])
        allc = torch.arange(256, dtype=torch.uint8)
        vals = allc.view(dt).double()
        fin = torch.isfinite(vals)
        pos = [(float(v), int(c)) for v, c in zip(vals[fin].tolist(), allc[fin].tolist()) if v > 0]
        pos.sort()
        mx = FP8_MAX[fmt]
        # value ladder: -max .. -min_sub, -0, +0, +min_sub .. +max (zero split by sign of the normal draw)
        ladder = [(-v, c | 0x80) for v, c in reversed(pos)] + [(-0.0, 0x80), (0.0, 0x00)] + pos
        Phi = lambda x: 0.5 * (1.0 + math.erf(x / (scale * math.sqrt(2.0))))  # noqa: E731
        bounds = []
        for i in range(len(ladder) - 1):
            a, b = ladder[i][0], ladder[i + 1][0]
            bounds.append(0.0 if (a == 0.0 and b == 0.0) else (a + b) / 2)
        cdf = [Phi(x) if abs(x) < mx else (0.0 if x < 0 else 1.0) for x in bounds]
        thr = torch.tensor([min(M32, int(c * 2.0 ** 32)) for c in cdf], dtype=torch.int64)
        codes = torch.tensor([c for _, c in ladder], dtype=torch.uint8)
        _CDF[(fmt, scale)] = (thr, codes)
    return _CDF[(fmt, scale)]


def _assemble(fmt, sgn, e, m):
    """FP8 code from sign, unbiased exponent (normal range) and a 2-bit mantissa index (1, 1.25, 1.5, 1.75)."""
    import torch
    code = (sgn << 7) | ((e + 7) << 3) | (m << 1) if fmt == "e4m3" else (sgn << 7) | ((e + 15) << 2) | m
    return code.to(torch.uint8)


def make_codes(kind: str, seed: int, rows, K: int, fmt: str, salt: int, device="cpu", S: int | None = None):
    """uint8 FP8 codes [len(rows), K] (torch tensor on `device`) for the given row indices of a data kind.
    Deterministic and device independent (integer hashing only), so any row subset can be regenerated cheaply:
      randn40: codes distributed as RNE_fp8(clamp(40 * N(0, 1))), sampled by inverse CDF from a uint32 hash;
      adv:     adversarial for the split-K reduction: per (row, 64-wide k-tile) magnitude 2^e (e in [-6, 5]),
               mantissa in {1, 1.25, 1.5, 1.75}, random sign (exactly representable in both formats); split partials
               of very different magnitudes make the reduction order and tile ownership visible in fp32."""
    import torch
    rows = torch.as_tensor(rows, dtype=torch.int64, device=device)
    cols = torch.arange(K, dtype=torch.int64, device=device)
    if kind == "randn40":
        thr, codes = _randn_table(fmt)
        u = _u32(seed, salt, 0, rows, cols)
        return codes.to(device)[torch.searchsorted(thr.to(device), u, right=True)]
    if kind == "adv":
        T = -(-K // 64)
        e = (_u32(seed, salt, 1, rows, torch.arange(T, dtype=torch.int64, device=device)) % 12) - 6
        e = e.repeat_interleave(64, dim=1)[:, :K]
        u = _u32(seed, salt, 2, rows, cols)
        return _assemble(fmt, (u >> 31) & 1, e, u & 3)
    if kind == "adv_split":
        # split-K structured: with S splits owning the 64-wide tiles t mod S, split 0 and split S-1 get exactly
        # mirrored large tiles (mat1 negated, mat2 equal) so p_{S-1} = -p_0 exactly (the chain truncates toward zero,
        # sign symmetric); the other splits get small values. The fp32 result is then decided by rounding at the
        # large scale, which depends on the reduction order, the tile ownership and S, and survives the rounding to
        # bf16 / fp16 / fp8 outputs. Only the first half of the pairs is mirrored (so S = 2 keeps small residues);
        # a ragged last tile and its partner get small values.
        if not S or S < 2:
            raise ValueError("adv_split needs S >= 2")
        T = -(-K // 64)
        t = cols // 64
        o = cols % 64
        sp = t % S
        full = lambda tt: (tt >= 0) & (tt < T) & ((tt + 1) * 64 <= K)  # noqa: E731
        q = len(range(S - 1, T, S))                 # tiles of split S-1
        nm = max(1, q // 2)                           # mirrored pairs: ordinals j < nm (the rest stay small)
        j = t // S
        mir = (sp == S - 1) & full(t) & full(t - (S - 1)) & (j < nm)
        src = (sp == 0) & full(t) & full(t + (S - 1)) & (j < nm)
        hc = torch.where(mir, (t - (S - 1)) * 64 + o, cols)
        big = (mir | src)[None, :]
        ue = _u32(seed, salt, 3, rows, hc)
        u = _u32(seed, salt, 4, rows, hc)
        e = torch.where(big, 2 + ue % 3, -6 + ue % 4)
        sgn = (u >> 31) & 1
        if salt == 1:
            sgn = torch.where(mir[None, :], 1 - sgn, sgn)
        return _assemble(fmt, sgn, e, u & 3)
    raise ValueError(kind)


def make_inputs(job: dict, rows=None, cols=None, device="cpu"):
    """(a [len(rows), K], b [len(cols), K]) uint8 code tensors of a job (all rows/cols by default), generated in
    blocks of 512 rows."""
    import torch
    d = job["data"]
    rows = range(job["M"]) if rows is None else rows
    cols = range(job["N"]) if cols is None else cols

    def gen(ids, fmt, salt):
        ids = list(ids)
        parts = [make_codes(d["kind"], d["seed"], ids[i:i + 512], job["K"], fmt, salt, device, d.get("S"))
                 for i in range(0, len(ids), 512)]
        return torch.cat(parts) if parts else torch.zeros((0, job["K"]), dtype=torch.uint8, device=device)
    return gen(rows, job["fa"], 1), gen(cols, job["fb"], 2)


def default_scales(job: dict) -> tuple[float, float]:
    """Scales that keep outputs inside the out dtype's range (non-powers of two, so the epilogue rounds)."""
    K = max(job["K"], 1)
    rms = (1600.0 if job["data"]["kind"] == "randn40" else 40.0) * math.sqrt(K)
    target = {"float8_e4m3fn": 60.0, "float8_e5m2": 3000.0, "float16": 3000.0}.get(job["out"], 3000.0)
    alpha = target / rms
    s = math.sqrt(alpha)
    return float(np.float32(s * 1.0123)), float(np.float32(s / 1.0071))


# ---- worker (runs inside the subprocess with CUBLASLT_WORKSPACE_SIZE / CUBLASLT_LOG_LEVEL=2) -----------------------
def worker(jobs_path: str, out_path: str):
    import torch
    dev = torch.device("cuda:0")
    torch.cuda.init()
    cap = common.LtCapture()
    jobs = json.loads(Path(jobs_path).read_text())
    res, arrays = [], {}
    one = torch.ones((), device=dev)
    for j in jobs:
        t0 = time.monotonic()
        a8, b8 = make_inputs(j, device=dev)
        a = a8.view(getattr(torch, FP8[j["fa"]]))
        b = b8.view(getattr(torch, FP8[j["fb"]]))
        sa = torch.tensor(j["scale_a"], dtype=torch.float32, device=dev)
        sb = torch.tensor(j["scale_b"], dtype=torch.float32, device=dev)
        sr = None if j.get("scale_result") is None else torch.tensor(j["scale_result"], dtype=torch.float32,
                                                                     device=dev)
        r = dict(id=j["id"], error=None)
        torch.cuda.synchronize()
        mk = cap.mark()
        try:
            y = torch._scaled_mm(a, b.t(), scale_a=sa, scale_b=sb, scale_result=sr,
                                 out_dtype=getattr(torch, j["out"]), use_fast_accum=bool(j["fast"]))
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001
            y = None
            r["error"] = common.err_text(e)
        calls = cap.matmuls_since(mk)
        r["n_lt_calls"] = len(calls)
        r["lt"] = calls[0] if calls else None
        r["lt_dims_match"] = bool(calls) and tuple(calls[0]["MNK"]) == (j["M"], j["N"], j["K"])
        r["lt_messages"] = cap.other_since(mk)[:5]
        if y is not None:
            full = common.tbits(y)
            r["full_sha256"] = hashlib.sha256(np.ascontiguousarray(full).tobytes()).hexdigest()
            sub = j.get("sub")
            # the stored inputs are regenerated on CPU from the same hash; prove CPU == GPU bytes for the subset
            ra = sub["rows"] if sub else list(range(j["M"]))
            rb = sub["cols"] if sub else list(range(j["N"]))
            ca, cb = make_inputs(j, ra, rb, "cpu")
            r["inputs_cpu_eq_gpu"] = bool(np.array_equal(ca.numpy(), a8[torch.tensor(ra, device=dev)].cpu().numpy())
                                          and np.array_equal(cb.numpy(), b8[torch.tensor(rb, device=dev)].cpu().numpy()))
            if sub:
                rows, cols = np.asarray(sub["rows"]), np.asarray(sub["cols"])
                arrays[f"{j['id']}.out"] = np.ascontiguousarray(full[rows][:, cols])
            else:
                arrays[f"{j['id']}.out"] = full
        # canary: the context still works
        try:
            c = torch._scaled_mm(torch.ones(16, 32, device=dev).to(torch.float8_e4m3fn),
                                 torch.ones(16, 32, device=dev).to(torch.float8_e4m3fn).t(), scale_a=one, scale_b=one,
                                 out_dtype=torch.float32)
            torch.cuda.synchronize()
            r["canary_ok"] = bool((common.tbits(c) == 0x42000000).all())   # 32.0f bit pattern
        except Exception as e:  # noqa: BLE001
            r["canary_ok"] = False
            r["canary_error"] = common.err_text(e)
        r["seconds"] = round(time.monotonic() - t0, 3)
        res.append(r)
        del a, b, a8, b8
    import os
    meta = dict(ws_env=os.environ.get("CUBLASLT_WORKSPACE_SIZE"), log_file=os.environ.get("CUBLASLT_LOG_FILE"),
                torch=torch.__version__, device=torch.cuda.get_device_name(0), results=res)
    np.savez(out_path, **arrays)
    Path(out_path).with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(f"worker: {len(jobs)} jobs, errors={sum(r['error'] is not None for r in res)}, "
          f"no-log={sum(r['n_lt_calls'] == 0 for r in res)}, canary_fail={sum(not r['canary_ok'] for r in res)}",
          flush=True)


def run(jobs_path: str, stage: str):
    jobs = json.loads(Path(jobs_path).read_text())
    st = Path(stage)
    st.mkdir(parents=True, exist_ok=True)
    by_ws = {}
    for j in jobs:
        by_ws.setdefault(int(j["ws"]), []).append(j)
    hw = common.hardware()
    (st / "hardware.json").write_text(json.dumps(hw, indent=1))
    rcs = {}
    for ws, js in sorted(by_ws.items()):
        jp = st / f"jobs_ws{ws}.json"
        jp.write_text(json.dumps(js))
        rcs[ws] = common.run_worker([__file__, "worker", "--jobs", str(jp), "--out", str(st / f"res_ws{ws}.npz")],
                                    ws, st / f"lt_ws{ws}.log")
    print("return codes", rcs)
    if any(rcs.values()):
        sys.exit(1)


def write_recording(rid: str, title: str, jobs_path: str, stage: str, claims: list[str], notes: str,
                    scripts: list[str]) -> dict:
    """Stage (jobs + worker results) -> one gemm recording (tcemu_ada_truth/1) in recordings/rtx4060_2026-10.
    Returns a summary incl. the config check (logged vs table row) of every call."""
    F = common.truth_format()
    jobs = {j["id"]: j for j in json.loads(Path(jobs_path).read_text())}
    st = Path(stage)
    hw = json.loads((st / "hardware.json").read_text())
    group = rid.split(".", 1)[0]
    w = F.RecordingWriter(rid, group, {
        "title": title, "level": "torch._scaled_mm", "kind": "gemm", "hardware": hw,
        "producer": {"agent": "mmaemu WS2 (pre-swap recordings, 2026-10)", "scripts": scripts,
                     "sources": [str(Path(jobs_path).name), "worker logs lt_ws*.log (CUBLASLT_LOG_LEVEL=2)"]},
        "claims": claims, "notes": notes, "params": {}}, root=common.REC_DATA, max_file_bytes=common.MAX_FILE_BYTES)
    summary = dict(n=0, errors=0, config_equal=0, config_differs=[], no_log=0, inputs_cpu_eq_gpu=0,
                   ws_logged_equal=0, canary_fail=0)
    for resf in sorted(st.glob("res_ws*.json")):
        meta = json.loads(resf.read_text())
        arrs = np.load(resf.with_suffix(".npz"), allow_pickle=False)
        for r in meta["results"]:
            j = jobs[r["id"]]
            sub = j.get("sub")
            rows = sub["rows"] if sub else list(range(j["M"]))
            cols = sub["cols"] if sub else list(range(j["N"]))
            a, b = make_inputs(j, rows, cols, "cpu")
            lt = r.get("lt")
            t = j.get("table") or {}
            logged = None if lt is None else [lt["algo"], lt["tile_id"], lt["numSplitsK"], lt["customOption"]]
            table = [t.get("algo"), t.get("tile"), t.get("splitK"), t.get("custom")] if t else None
            equal = logged is not None and table is not None and logged == table
            summary["n"] += 1
            summary["config_equal"] += int(equal)
            summary["no_log"] += int(lt is None)
            summary["inputs_cpu_eq_gpu"] += int(bool(r.get("inputs_cpu_eq_gpu")))
            summary["ws_logged_equal"] += int(lt is not None and lt.get("workspace_bytes") == j["ws"])
            summary["canary_fail"] += int(not r.get("canary_ok", True))
            if table is not None and not equal:
                summary["config_differs"].append(dict(id=r["id"], table=table, logged=logged))
            cp = dict(fmt_a=j["fa"], fmt_b=j["fb"], hw_shape=[j["M"], j["N"], j["K"]],
                      data={k: v for k, v in j["data"].items()}, workspace_bytes=j["ws"])
            if sub:
                cp["sub"] = {"rows": rows, "cols": cols}
            for k in ("combo", "part", "envelope", "discrimination", "tags"):
                if j.get(k) is not None:
                    cp[k] = j[k]
            op = dict(out_dtype=j["out"], fast_accum=bool(j["fast"]), scale_a=common.f32bits(j["scale_a"]),
                      scale_b=common.f32bits(j["scale_b"]),
                      scale_result=None if j.get("scale_result") is None else common.f32bits(j["scale_result"]),
                      splitk=common.splitk_of(lt), cublas=common.cublas_params(lt), table_row=t or None,
                      config_check=dict(table=table, logged=logged, equal=equal,
                                        fields=["algo", "tile", "numSplitsK", "customOption"]))
            if t and not equal:
                op["finding"] = ("logged cuBLASLt config differs from the heuristic table row: kept as recorded "
                                 "(params.splitk is the LOGGED config)")
            out = dict(name="out", params=op)
            if r.get("error"):
                out["hw_error"] = r["error"]
                summary["errors"] += 1
            else:
                out["bits"] = arrs[f"{r['id']}.out"]
                out["full_sha256"] = r.get("full_sha256")
            w.add_case(r["id"], cp, {"a": a.numpy(), "b": b.numpy()}, [out])
    p = w.close()
    print("wrote", p, json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in summary.items()}))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    w = sp.add_parser("worker")
    w.add_argument("--jobs", required=True)
    w.add_argument("--out", required=True)
    r = sp.add_parser("run")
    r.add_argument("--jobs", required=True)
    r.add_argument("--stage", required=True)
    a = ap.parse_args()
    if a.cmd == "worker":
        worker(a.jobs, a.out)
    else:
        run(a.jobs, a.stage)


if __name__ == "__main__":
    main()
