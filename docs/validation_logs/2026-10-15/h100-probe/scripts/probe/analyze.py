"""Offline analysis of a probe run (P3): hypotheses against the recorded bits. Bitwise only.

  m1  c-mode: every recorded MM_cublas64* block (n x n, every shape and mode) against the instruction models
      (ada, hopper = c early, hopper_latec = c added in fp32 after the block sum, b200 = b200_mma_sync), each with
      c in the same chain as the test block ("same") or in an earlier chain ("restart": fp32(c) + chain(a, b)).
  m3  restarts and combination order: every pattern block against {fast: one chain} U {R} x {seq, tree} under the
      instruction models; lists the hypotheses that match every row of the block.
  m2  the 21 Evo 2 projections: full-output sha256 of a hypothesis against the recording.
  m4  the live jobs (m4 + m6, pass 1): stored subsets against a hypothesis (S and scheme from each call's own log).
The m2 / m4 hypothesis: --model (instruction model) and --R (slow-accumulation restart, unsplit calls). Split calls
use mmaemu's plan (Ada k-tile 64, restart every 2 own tiles) only with --model ada; otherwise they are reported as
not modelled.
usage: analyze.py OUT_DIR [--parts m1,m3,m2,m4] [--model hopper] [--R 128] [--device cpu|cuda] --report R.json"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np

import cpu_checks  # noqa: F401  (chains(); sys.path via pc)
import m1_cvec
import m3_patterns as P
import pc

import torch  # noqa: E402
from mmaemu.model import epilogue  # noqa: E402
from mmaemu.model.chain import bfma, chain_gemm, dot  # noqa: E402
from mmaemu.model.plan import execute_plan, make_plan  # noqa: E402
from mmaemu.model.presets import ADA, B200_MMA_SYNC, Preset  # noqa: E402

warnings.simplefilter("ignore")
RS = (32, 64, 128, 256, 512, 1024, 2048)


def model(name: str, R: int | None = None) -> Preset:
    if name == "ada":
        return ADA if R in (None, 128) else Preset("ada_R", 16, 13, 13, -132, False, R)
    if name == "hopper":
        return Preset("hopper_c_early", 32, 13, 13, -132, False, R)
    if name == "hopper_latec":
        return Preset("hopper_c_late", 32, 13, 13, -132, True, R)
    if name == "b200":
        return B200_MMA_SYNC if R is None else Preset("b200_R", 32, 25, 23, -133, True, R)
    raise KeyError(name)


MODELS = ("ada", "hopper", "hopper_latec", "b200")


def canon32(u: np.ndarray) -> np.ndarray:
    return np.where(np.isnan(u.view(np.float32)), np.uint32(0x7FC00000), u)


def tb(t: torch.Tensor) -> np.ndarray:
    return pc.common.tbits(t)


# ---- m1 ------------------------------------------------------------------------------------------------------------
def part_m1(o: Path, dev) -> dict:
    out = {}
    for name in ("m1_torch", "m1_te"):
        f = o / name / f"{name}.json"
        if not f.exists():
            continue
        meta = json.loads(f.read_text())
        z = np.load(o / name / f"{name}.npz")
        cache = {}
        for r in meta["calls"]:
            cid, m = r["id"], r["meta"]
            key = f"{cid}.block"
            if key not in z.files:
                out[cid] = "no data"
                continue
            S = m["set"]
            if S not in cache:
                av, bv, cg, _, _, _ = m1_cvec.load_set(S)
                n = len(av)
                a = torch.from_numpy(av.astype(np.float64)).to(dev)
                b = torch.from_numpy(bv.astype(np.float64)).to(dev)
                cx = torch.from_numpy(cg[:, 0].astype(np.float64)).to(dev)
                cy = torch.from_numpy(cg[:, 1].astype(np.float64)).to(dev)
                A = a[:, None, :].expand(n, n, 32).contiguous()
                B = b[None, :, :].expand(n, n, 32).contiguous()
                C = (cx[:, None] * cy[None, :]).float()
                preds = {}
                for mn in MODELS:
                    p = model(mn)
                    same = dot(A, B, C, preset=p, fmt="e4m3")
                    d0 = dot(A, B, None, preset=p, fmt="e4m3")
                    preds[(mn, "same")] = canon32(tb(same))
                    preds[(mn, "restart")] = canon32(tb(C + d0))
                cache[S] = preds
            meas = canon32(z[key])
            out[cid] = {f"{mn}/{v}": int((pr == meas).sum()) for (mn, v), pr in cache[S].items()}
            out[cid]["n2"] = int(meas.size)
            out[cid]["lt"] = (r["runs"][0]["lt"] or [{}])[0].get("lt_algo_string")
    return out


# ---- m3 ------------------------------------------------------------------------------------------------------------
def sparse_sim(spec: np.ndarray, gvec: np.ndarray, K: int, p: Preset, R: int | None, comb: str, dev) -> torch.Tensor:
    """Exact chain simulation of the <= 3-product pattern rows: only the blocks holding products are evaluated (a block
    of zero products returns its c unchanged for every chain value of these presets: c has <= frac_out + 1 significant
    bits; cpu_checks.check_m1 / selftest_sparse check it). Chains restart every R elements (None: one chain); chain
    results are added in fp32 from +0, in K order ('seq') or as cpu_checks.chains' pairwise tree ('tree')."""
    pos = torch.from_numpy(spec[:, [0, 2, 4]].astype(np.int64)).to(dev)
    av = torch.from_numpy(spec[:, [1, 3, 5]]).to(dev)
    g = torch.from_numpy(gvec).to(dev)
    used = pos >= 0
    pos_s = torch.where(used, pos, torch.full_like(pos, 1 << 40))
    order = pos_s.argsort(1)
    pos_s = pos_s.gather(1, order)
    av = torch.where(used, av, torch.zeros_like(av)).gather(1, order)
    used = used.gather(1, order)
    bv = torch.where(used, g[pos_s.clamp_max(K - 1)], torch.zeros_like(av))
    ch = pos_s // R if R else torch.zeros_like(pos_s)
    bl = pos_s // p.nfma
    n = len(spec)
    new_blk = torch.ones_like(used)
    new_blk[:, 1:] = (bl[:, 1:] != bl[:, :-1]) | (ch[:, 1:] != ch[:, :-1])
    new_blk &= used
    slot = new_blk.long().cumsum(1) - 1                      # block slot of each entry (-1: unused)
    nchain = -(-K // R) if R else 1
    chains = torch.zeros(n, nchain, dtype=torch.float64, device=dev)
    acc = torch.zeros(n, dtype=torch.float64, device=dev)
    cur = torch.full((n,), -1, dtype=torch.int64, device=dev)
    rows = torch.arange(n, device=dev)
    for s in range(3):
        sel = (slot == s) & used
        has = sel.any(1)
        if not bool(has.any()):
            break
        a_s = torch.where(sel, av, torch.zeros_like(av))
        b_s = torch.where(sel, bv, torch.zeros_like(bv))
        ch_s = torch.where(sel, ch, torch.full_like(ch, -1)).amax(1)
        start = has & (ch_s != cur)
        old = start & (cur >= 0)
        chains[rows[old], cur[old]] = acc[old]               # the previous chain is finished
        c_in = torch.where(start, torch.zeros_like(acc), acc)
        pad = torch.zeros(n, p.nfma - 3, dtype=torch.float64, device=dev)
        d = bfma(torch.cat([a_s, pad], 1), torch.cat([b_s, pad], 1), c_in, p, ("e4m3", "e4m3"))
        acc = torch.where(has, d, acc)
        cur = torch.where(has, ch_s, cur)
    last = cur >= 0
    chains[rows[last], cur[last]] = acc[last]
    parts = chains.float()
    if comb == "seq" or nchain == 1:
        out = torch.zeros(n, dtype=torch.float32, device=dev)
        for c in range(nchain):
            out = out + parts[:, c]
        return out
    cols = [parts[:, c] for c in range(nchain)]
    while len(cols) > 1:
        nxt = [cols[i] + cols[i + 1] for i in range(0, len(cols) - 1, 2)]
        if len(cols) % 2:
            nxt.append(cols[-1])
        cols = nxt
    return cols[0]


def selftest_sparse(dev) -> dict:
    """sparse_sim == cpu_checks.chains (dense) on samples of every pattern kind."""
    bad = {}
    for K in (1920, 4096):
        for kind in P.KINDS:
            spec = P.patterns(kind, K)[::7][:600]
            A = torch.from_numpy(P.dense_rows(spec, K)).to(dev)
            for gname, vec in P.groups(kind, K):
                Bm = torch.from_numpy(vec).to(dev)[None, :].expand_as(A).contiguous()
                for mn in MODELS:
                    for R in (None, 64, 128, 512):
                        for comb in ("seq", "tree"):
                            p = model(mn)
                            y1 = cpu_checks.chains(A, Bm, R, comb, p)
                            y2 = sparse_sim(spec, vec, K, p, R, comb, dev)
                            nb = int((tb(y1) != tb(y2)).sum())
                            if nb:
                                bad[f"{kind}.{gname}.K{K}.{mn}.{R}.{comb}"] = nb
    return bad


def part_m3(o: Path, dev, models=MODELS) -> dict:
    out = {}
    for name in ("m3_torch", "m3_te"):
        f = o / name / f"{name}.json"
        if not f.exists():
            continue
        meta = json.loads(f.read_text())
        z = np.load(o / name / f"{name}.npz")
        for key, blk in meta["extra"]["blocks"].items():
            K, M, kind = blk["K"], blk["M"], blk["kind"]
            spec = z[f"{key}.spec"]
            grp = P.groups(kind, K)
            modes = sorted({r["meta"]["mode"] for r in meta["calls"] if r["id"].startswith(key + ".")})
            hyps = [(mn, None, "one") for mn in models] + [(mn, R, c) for mn in models for R in RS if R < K
                                                            for c in ("seq", "tree")]
            for mode in modes:
                vals, var, lts = [], [], set()
                for r0 in blk["gemms"]:
                    cid = f"{key}.r{r0}.{mode}"
                    vals.append(z[f"{cid}.vals"])
                    var.append(z[f"{cid}.colvar"])
                    row = next(r for r in meta["calls"] if r["id"] == cid)
                    for lt in row["runs"][0]["lt"]:
                        lts.add(f"{lt['lt_algo_string']} ws={lt['workspace_bytes']}")
                vals = np.concatenate(vals)
                var = np.concatenate(var)
                res = dict(colvar=int(var.sum()), lt=sorted(lts), groups={})
                for g, (gname, vec) in enumerate(grp):
                    meas = vals[:, g]
                    match, near = [], {}
                    for (mn, R, comb) in hyps:
                        y = sparse_sim(spec, vec, K, model(mn), R, "seq" if comb == "one" else comb, dev)
                        if kind == "tie_bf16":
                            y = y.to(torch.bfloat16)
                        nb = int((tb(y) != meas).sum())
                        h = f"{mn}/{R or 'one-chain'}/{comb}"
                        if nb == 0:
                            match.append(h)
                        else:
                            near[h] = nb
                    best = sorted(near.items(), key=lambda t: t[1])[:3]
                    res["groups"][gname] = dict(rows=len(spec), matching=match, closest_nonmatching=best)
                out[f"{name}:{key}:{mode}"] = res
                print(f"m3 {name}:{key}:{mode} colvar={res['colvar']} " +
                      " ".join(f"{g}={v['matching'][:6]}" for g, v in res["groups"].items()), flush=True)
    return out


# ---- m2 ------------------------------------------------------------------------------------------------------------
def gemm(a_u8, b_u8, fast, p: Preset, dev, fa="e4m3", fb="e4m3", S=1, scheme="none"):
    plan = make_plan(a_u8.shape[1], fast=fast, S=S, scheme=scheme, preset=p)
    return execute_plan(plan, a_u8.to(dev), b_u8.to(dev), fmt_a=fa, fmt_b=fb,
                        chain_fn=lambda x, y: chain_gemm(x, y, fmt_a=fa, fmt_b=fb, preset=p))


def part_m2(o: Path, data: Path, dev, mname: str, R: int | None) -> dict:
    truth = json.loads((data / "truth" / "evo2_1b_w2k.json").read_text())
    rec, split = {}, {}
    for name in ("m2_torch", "m2_te"):
        f = o / name / f"{name}.json"
        if f.exists():
            for r in json.loads(f.read_text())["calls"]:
                rec[r["id"]] = r["runs"][0].get("sha256")
                lt = r["runs"][0].get("lt") or [None]
                sk = pc.common.splitk_of(lt[0])
                split[r["id"]] = (sk["S"], sk.get("scheme", "none"))
    calls = sorted({k.split(".")[0] for k in rec})
    out = {}
    pf, ps = model(mname, None), model(mname, R)
    for c in calls:
        i = int(c[4:])
        d = torch.load(data / "fp8-inputs" / f"{c}.pt", map_location="cpu", weights_only=True)
        p = truth["cases"][i]["params"]
        inv = torch.tensor([p["fp8_scale_inv_in"], p["fp8_scale_inv_w"]], dtype=torch.int32).view(torch.float32)
        one = torch.ones(())
        res, cache = {}, {}
        for fast in (True, False):
            if not fast and R is None:
                continue
            tag = "fast" if fast else "slow"
            for kind, ids in (("acc", (f"{c}.acc_{tag}", f"{c}.te_acc" + ("" if fast else "_split"))),
                              ("y", (f"{c}.y_{tag}", f"{c}.te_y" + ("" if fast else "_split")))):
                for cid in ids:
                    if cid not in rec:
                        continue
                    S, scheme = split[cid]                # each call's own logged split
                    if scheme == "unknown":
                        res[cid.split(".")[1]] = f"not modelled (S={S}, unknown scheme)"
                        continue
                    if (fast, S, scheme) not in cache:
                        acc = gemm(d["xq"], d["wq"], fast, pf if fast else ps, dev, S=S, scheme=scheme)
                        y = epilogue.apply(acc, inv[0], inv[1], out_dtype=torch.bfloat16)
                        acc1 = epilogue.apply(acc, one, one, out_dtype=torch.float32)
                        cache[(fast, S, scheme)] = {"acc": hashlib.sha256(tb(acc1).tobytes()).hexdigest(),
                                                    "y": hashlib.sha256(tb(y).tobytes()).hexdigest()}
                    ok = rec[cid] == cache[(fast, S, scheme)][kind]
                    res[cid.split(".")[1]] = ok if S == 1 else f"{ok} (S={S} {scheme})"
        out[c] = res
        print("m2", c, res, flush=True)
    tot = {}
    for res in out.values():
        for k, v in res.items():
            tot.setdefault(k, [0, 0])
            tot[k][0] += v is True or (isinstance(v, str) and v.startswith("True"))
            tot[k][1] += 1
    return dict(model=mname, R=R, per_call=out, totals={k: f"{a}/{b}" for k, (a, b) in tot.items()})


# ---- m4 ------------------------------------------------------------------------------------------------------------
def part_m4(o: Path, dev, mname: str, R: int | None) -> dict:
    import live
    out = dict(model=mname, R=R, jobs=0, equal=0, different=0, not_modelled=0, errors=0, input_fault=[], by_tag={},
               bad=[], verdicts={})
    for f in sorted(o.rglob("p1_ws*.json")):
        if f.name.endswith(".jobs.json"):
            continue
        jobs = {j["id"]: j for j in json.loads(f.with_name(f.name.replace(".json", ".jobs.json")).read_text())}
        res = json.loads(f.read_text())["results"]
        z = np.load(f.with_suffix(".npz"))
        f2 = Path(str(f.parent).replace("/p1", "/p2")) / f.name.replace("p1_", "p2_", 1)   # the second pass
        p2 = {r["id"]: r for r in json.loads(f2.read_text())["results"]} if f2.exists() else {}
        z2 = None
        for r in res:
            j = jobs[r["id"]]
            tag = j["tags"][0]
            bt = out["by_tag"].setdefault(tag, dict(equal=0, different=0, not_modelled=0))
            out["jobs"] += 1
            zr = z
            if r.get("inputs_cpu_eq_gpu") is False:      # the GEMM saw other inputs than the CPU regeneration
                # pre-registered (plan §5): judge on the pass with clean inputs; both passes faulty = a real problem
                r2 = p2.get(r["id"])
                clean2 = r2 is not None and r2.get("inputs_cpu_eq_gpu") is True
                out["input_fault"].append(dict(id=r["id"], judged_on="p2" if clean2 else "none (both passes faulty)"))
                if not clean2:
                    out["errors"] += 1
                    continue
                z2 = np.load(f2.with_suffix(".npz")) if z2 is None else z2
                r, zr = r2, z2
            if r["error"] or f"{r['id']}.out" not in zr.files:
                out["errors"] += 1
                continue
            sk = pc.common.splitk_of(r["lt"])
            S, scheme = sk["S"], sk.get("scheme", "none")
            if S > 1 and scheme == "unknown":
                out["not_modelled"] += 1
                bt["not_modelled"] += 1
                out["verdicts"][r["id"]] = [tag, S, scheme, bool(j["fast"]), "not_modelled"]
                continue
            sub = j.get("sub")
            rows = sub["rows"] if sub else list(range(j["M"]))
            cols = sub["cols"] if sub else list(range(j["N"]))
            a8, b8 = live.make_inputs(j, rows, cols, "cpu")
            p = model(mname, None if j["fast"] else R)
            try:
                acc = gemm(a8, b8, j["fast"], p, dev, j["fa"], j["fb"], S, scheme)
            except Exception as e:  # noqa: BLE001
                out["not_modelled"] += 1
                bt["not_modelled"] += 1
                out["bad"].append(dict(id=r["id"], error=repr(e)[:200]))
                out["verdicts"][r["id"]] = [tag, S, scheme, bool(j["fast"]), "not_modelled"]
                continue
            sr = None if j.get("scale_result") is None else torch.tensor(j["scale_result"], dtype=torch.float32)
            y = epilogue.apply(acc, torch.tensor(j["scale_a"], dtype=torch.float32),
                               torch.tensor(j["scale_b"], dtype=torch.float32),
                               out_dtype=getattr(torch, j["out"]), scale_result=sr)
            pred = tb(y)
            meas = zr[f"{r['id']}.out"]
            nbad = int((pred != meas).sum())
            out["verdicts"][r["id"]] = [tag, S, scheme, bool(j["fast"]), "equal" if nbad == 0 else f"differ {nbad}/{meas.size}"]
            if nbad == 0:
                out["equal"] += 1
                bt["equal"] += 1
            else:
                out["different"] += 1
                bt["different"] += 1
                if len(out["bad"]) < 60:
                    out["bad"].append(dict(id=r["id"], tags=j["tags"], M=j["M"], N=j["N"], K=j["K"], out=j["out"],
                                           fast=j["fast"], S=S, scheme=scheme, values=int(meas.size), differ=nbad,
                                           inputs_ok=r.get("inputs_cpu_eq_gpu")))
    print("m4", json.dumps({k: v for k, v in out.items() if k not in ("bad", "verdicts")}), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--parts", default="m1,m3,m2,m4")
    ap.add_argument("--model", default="hopper")
    ap.add_argument("--R", type=int, default=None)
    ap.add_argument("--data", type=Path)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--report", type=Path, required=True)
    a = ap.parse_args()
    dev = torch.device(a.device)
    torch.set_num_threads(4)
    rep = dict(out=str(a.out), model=a.model, R=a.R, parts={})
    for part in a.parts.split(","):
        t0 = time.monotonic()
        if part == "m1":
            rep["parts"]["m1"] = part_m1(a.out, dev)
        elif part == "m3":
            rep["parts"]["m3"] = part_m3(a.out, dev)
        elif part == "m2":
            rep["parts"]["m2"] = part_m2(a.out, a.data, dev, a.model, a.R)
        elif part == "m4":
            rep["parts"]["m4"] = part_m4(a.out, dev, a.model, a.R)
        rep.setdefault("seconds", {})[part] = round(time.monotonic() - t0, 1)
        a.report.write_text(json.dumps(rep, indent=1, default=str) + "\n")
    print("wrote", a.report)


if __name__ == "__main__":
    main()
