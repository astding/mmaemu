"""Split-K on the H100, offline: hypotheses for how cuBLASLt's sm90 split-K kernels (algo 66) partition K and combine
the S partials, scored bit for bit against the live jobs with numSplitsK > 1 (m4 + m6; pass 1, or pass 2 where pass 1
had an input fault). The instruction model is the measured one (hopper: 32 products per block, 13 fraction bits, c
first); slow accumulation restarts every 128 of a split's own K elements (R = 128, the unsplit restart measured by
M3; R = 256 failed on the first 138 jobs), added serially in fp32 from +0. Slow mode is scored for the 128- and 256-
wide partitions only (the sm90 kernels' k-stage is 128: MATMUL_STAGES_128xAUTO): every 128-tile chain is computed
once and the partitions differ only in which tiles a split sums, in which order.

  partition  il<T>   k-tile t of width T goes to split t mod S (Ada's rule with T = 64)
             ct<T>c  contiguous runs of ceil(T_total / S) k-tiles (the last split takes the rest)
             ct<T>b  contiguous, balanced: the first (T_total mod S) splits get one tile more
             ct<T>l  contiguous, balanced: the last (T_total mod S) splits get one tile more
  combine    serial ((p0 + p1) + p2) ...; rserial ((p_{S-1} + p_{S-2}) + ...); pairwise (Ada, mmaemu.model.plan);
             tree (balanced halves)
  alpha      after (alpha * sum, mmaemu's epilogue) or each (sum of RN32(alpha * p_s), then the epilogue with alpha 1)
usage: splitk_search.py OUT_DIR [--offset 0] [--limit N] [--device cuda] [--only ID,...] --report R.json
(per-job lines are appended to R.jsonl as they finish)"""
from __future__ import annotations

import argparse
import collections
import json
import time
import warnings
from pathlib import Path

import numpy as np

import pc

import torch  # noqa: E402
from mmaemu.model import epilogue  # noqa: E402
from mmaemu.model.chain import chain_gemm  # noqa: E402
from mmaemu.model.plan import reduce_partials  # noqa: E402
from mmaemu.model.presets import Preset  # noqa: E402

warnings.simplefilter("ignore")
HOPPER = Preset("hopper_c_early", 32, 13, 13, -132, False, None)
PARTITIONS = ("il64", "il128", "il256", "ct64c", "ct128c", "ct128b", "ct128l", "ct256c")
COMBINES = ("serial", "rserial", "pairwise", "tree")
ALPHAS = ("after", "each")
SLOW_PARTITIONS = ("il128", "il256", "ct128c", "ct128b", "ct128l", "ct256c")


def partition(K: int, S: int, name: str) -> list:
    """S index vectors into K in chain order (-1 = zero code, padded to a multiple of 32)."""
    T = int(name[2:].rstrip("cbl"))
    nt = -(-K // T)
    if name.startswith("il"):
        tiles = [list(range(s, nt, S)) for s in range(S)]
    else:
        mode = name[-1]
        if mode == "c":
            q = -(-nt // S)
            tiles = [list(range(min(s * q, nt), min((s + 1) * q, nt))) for s in range(S)]
        else:
            q, r = divmod(nt, S)
            sizes = [q + (1 if (s < r if mode == "b" else s >= S - r) else 0) for s in range(S)]
            st = np.cumsum([0] + sizes)
            tiles = [list(range(st[s], st[s + 1])) for s in range(S)]
    out = []
    for ts in tiles:
        k = torch.cat([torch.arange(t * T, (t + 1) * T) for t in ts]) if ts else torch.zeros(0, dtype=torch.int64)
        k = torch.where(k < K, k, torch.full_like(k, -1))
        pad = (-len(k)) % 32
        out.append(torch.cat([k, torch.full((pad,), -1, dtype=torch.int64)]) if pad else k)
    return out


def gather(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    idx = idx.to(x.device)
    g = x[:, idx.clamp_min(0)]
    return torch.where((idx < 0)[None, :], torch.zeros((), dtype=x.dtype, device=x.device), g).contiguous()


def partial(a, b, idx, fast: bool, R: int, fa: str, fb: str) -> torch.Tensor:
    if len(idx) == 0:
        return torch.zeros(a.shape[0], b.shape[0], dtype=torch.float32, device=a.device)
    if fast:
        return chain_gemm(gather(a, idx), gather(b, idx), fmt_a=fa, fmt_b=fb, preset=HOPPER)
    acc = torch.zeros(a.shape[0], b.shape[0], dtype=torch.float32, device=a.device)
    for j in range(0, len(idx), R):
        acc = acc + chain_gemm(gather(a, idx[j:j + R]), gather(b, idx[j:j + R]), fmt_a=fa, fmt_b=fb, preset=HOPPER)
    return acc


def combine(parts: list, name: str) -> torch.Tensor:
    if name in ("serial", "pairwise"):
        return reduce_partials(parts, name)
    if name == "rserial":
        return reduce_partials(parts[::-1], "serial")
    if name == "tree":
        while len(parts) > 1:
            h = -(-len(parts) // 2)
            parts = [parts[i] + parts[i + h] if i + h < len(parts) else parts[i] for i in range(h)]
        return parts[0]
    raise KeyError(name)


def jobs_with_results(o: Path):
    """(job, result, stored subset) of every split call; pass 2 where pass 1 had an input fault."""
    for f in sorted(o.rglob("p1_ws*.json")):
        if f.name.endswith(".jobs.json"):
            continue
        jobs = {j["id"]: j for j in json.loads(f.with_name(f.name.replace(".json", ".jobs.json")).read_text())}
        z = np.load(f.with_suffix(".npz"))
        f2 = Path(str(f.parent).replace("/p1", "/p2")) / f.name.replace("p1_", "p2_", 1)
        p2 = {r["id"]: r for r in json.loads(f2.read_text())["results"]} if f2.exists() else {}
        z2 = None
        for r in json.loads(f.read_text())["results"]:
            if not r.get("lt") or int(r["lt"]["numSplitsK"]) <= 1 or r["error"]:
                continue
            zr = z
            if r.get("inputs_cpu_eq_gpu") is False:
                r = p2.get(r["id"])
                if r is None or r.get("inputs_cpu_eq_gpu") is not True:
                    continue
                z2 = np.load(f2.with_suffix(".npz")) if z2 is None else z2
                zr = z2
            if f"{r['id']}.out" in zr.files:
                yield jobs[r["id"]], r, zr[f"{r['id']}.out"]


def score(j: dict, r: dict, meas: np.ndarray, dev) -> dict:
    import live
    lt = r["lt"]
    S = int(lt["numSplitsK"])
    sub = j.get("sub")
    rows = sub["rows"] if sub else list(range(j["M"]))
    cols = sub["cols"] if sub else list(range(j["N"]))
    a8, b8 = live.make_inputs(j, rows, cols, "cpu")
    a8, b8 = a8.to(dev), b8.to(dev)
    sa = torch.tensor(j["scale_a"], dtype=torch.float32)
    sb = torch.tensor(j["scale_b"], dtype=torch.float32)
    sr = None if j.get("scale_result") is None else torch.tensor(j["scale_result"], dtype=torch.float32)
    od = getattr(torch, j["out"])
    alpha = (sa.double() * sb.double()).float().to(dev)
    one = torch.ones(())
    match, best = [], (None, -1)
    K = j["K"]

    def tile_chains():
        """fp32 chain of every 128-wide k-tile (zero padded), from +0."""
        out = []
        for t in range(-(-K // 128)):
            ix = torch.arange(t * 128, (t + 1) * 128)
            ix = torch.where(ix < K, ix, torch.full_like(ix, -1))
            out.append(chain_gemm(gather(a8, ix), gather(b8, ix), fmt_a=j["fa"], fmt_b=j["fb"], preset=HOPPER))
        return out

    tc = None if j["fast"] else tile_chains()
    for pn in (PARTITIONS if j["fast"] else SLOW_PARTITIONS):
        idx = partition(K, S, pn)
        if j["fast"]:
            parts = [partial(a8, b8, ix, True, None, j["fa"], j["fb"]) for ix in idx]
        else:
            parts = []
            for ix in idx:
                acc = torch.zeros(a8.shape[0], b8.shape[0], dtype=torch.float32, device=dev)
                for t in dict.fromkeys((ix[ix >= 0] // 128).tolist()):    # the split's tiles in chain order
                    acc = acc + tc[t]
                parts.append(acc)
        for cn in COMBINES:
            for al in ALPHAS:
                if al == "after":
                    y = epilogue.apply(combine(parts, cn), sa, sb, out_dtype=od, scale_result=sr)
                else:
                    sc = [(p.double() * alpha.double()).float() for p in parts]
                    y = epilogue.apply(combine(sc, cn), one, one, out_dtype=od, scale_result=sr)
                pred = pc.common.tbits(y)
                ok = int((pred == meas).sum())
                h = f"{pn}/{cn}/{al}" if j["fast"] else f"{pn}/R128/{cn}/{al}"
                if ok == meas.size:
                    match.append(h)
                if ok > best[1]:
                    best = (h, ok)
    return dict(id=j["id"], S=S, fast=bool(j["fast"]), K=j["K"], M=j["M"], N=j["N"], out=j["out"],
                data=j["data"]["kind"], custom=int(lt["customOption"]), tile=lt["tile"], ws=j["ws"],
                values=int(meas.size), match=match, best=best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--report", type=Path, required=True)
    a = ap.parse_args()
    dev = torch.device(a.device)
    only = set(a.only.split(",")) if a.only else None
    items = [t for t in jobs_with_results(a.out) if only is None or t[0]["id"] in only]
    items = items[a.offset:] if a.limit is None else items[a.offset:a.offset + a.limit]
    res, t0 = [], time.monotonic()
    jl = a.report.with_suffix(".jsonl")
    jl.write_text("")
    for n, (j, r, meas) in enumerate(items):
        s = score(j, r, meas, dev)
        res.append(s)
        with jl.open("a") as fh:
            fh.write(json.dumps(s) + "\n")
        print(f"{n} {s['id']} S={s['S']} fast={s['fast']} K={s['K']} custom={s['custom']} {s['data']} "
              f"match={len(s['match'])} {s['match'][:6]} best={s['best']}", flush=True)
    common = None
    for s in res:
        common = set(s["match"]) if common is None else common & set(s["match"])
    by = collections.Counter(h for s in res for h in s["match"])
    rep = dict(n=len(res), seconds=round(time.monotonic() - t0, 1), matching_all=sorted(common or []),
               hypothesis_counts=dict(by.most_common()), no_match=[s["id"] for s in res if not s["match"]],
               jobs=res)
    a.report.write_text(json.dumps(rep, indent=1) + "\n")
    print("matching every job:", rep["matching_all"])
    print("no hypothesis:", len(rep["no_match"]), rep["no_match"][:20])
    print("top:", list(by.most_common(8)))


if __name__ == "__main__":
    main()
