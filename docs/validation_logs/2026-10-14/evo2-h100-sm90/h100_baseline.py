"""Official-path Evo 2 baseline on the H100: upstream evo2.Evo2('evo2_1b_base') with Transformer Engine FP8.

Mirrors the RTX 4060 goldens (the scripts golden_real.py and brca1_source_script.py that made them), but takes
its inputs from the shipped goldens so they are identical to Ada's:
  windows  token ids of p0..p3, w8k, w2k from data/goldens/evo2_1b_base_real-te-fp8.npz; full bf16 logits, run twice
  brca1    the 500-variant subset from data/brca1/brca1_subset500_deltas.csv (rows in sample order) and GRCh37 chr17,
           8,192-bp windows centred on the variant, upstream score_sequences with default arguments, run twice
usage: python h100_baseline.py {windows,brca1} OUTDIR"""
import csv, gzip, hashlib, importlib.metadata as md, json, os, platform, subprocess, sys, time

import numpy as np
import torch

STAGE, OUT = sys.argv[1], sys.argv[2]
R = os.environ["BASELINE_ROOT"]  # the bundle directory on the rented host
GOLD = f"{R}/data/goldens/evo2_1b_base_real-te-fp8.npz"
WINDOW = 8192
os.makedirs(OUT, exist_ok=True)


def env_meta():
    p = torch.cuda.get_device_properties(0)
    v = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda}
    for k in ("transformer_engine", "transformer_engine_cu12", "transformer_engine_torch", "evo2", "vtx", "flash-attn",
              "nvidia-cublas-cu12", "nvidia-cudnn-cu12", "triton"):
        try:
            v[k] = md.version(k)
        except md.PackageNotFoundError:
            pass
    smi = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,clocks.max.sm,power.limit", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    return dict(gpu=p.name, capability=list(torch.cuda.get_device_capability(0)), sm_count=p.multi_processor_count,
                nvidia_smi=smi, versions=v,
                env={k: os.environ.get(k) for k in ("CUBLASLT_WORKSPACE_SIZE", "CUBLAS_WORKSPACE_CONFIG",
                                                     "CUBLASLT_LOG_LEVEL", "NVTE_DISABLE_NVRTC", "HF_HOME")})


def sha(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


torch.cuda.set_per_process_memory_fraction(0.95)
from evo2 import Evo2
from vortex.model.layers import TELinear

t0 = time.perf_counter()
m = Evo2("evo2_1b_base")
load_s = time.perf_counter() - t0
fp8_mods = [x for x in m.model.modules() if isinstance(x, TELinear) and getattr(x, "use_fp8_input_projections", False)]
# evo2 silently falls back to bf16 projections without TE: refuse to label such a run as the FP8 baseline
assert fp8_mods, "no TELinear module with use_fp8_input_projections: this is not the TE FP8 path"
meta = dict(stage=STAGE, model="evo2_1b_base", fp8_projection_modules=len(fp8_mods), load_s=round(load_s, 1),
            env=env_meta())
print(json.dumps({k: v for k, v in meta.items() if k != "env"}), flush=True)

if STAGE == "windows":
    g = np.load(GOLD)
    names = ["p0", "p1", "p2", "p3", "w8k", "w2k"]
    arrays, per = {}, []
    with torch.inference_mode():
        for rep in (0, 1):
            for name in names:
                ids = torch.tensor(g[f"{name}/ids"].astype(np.int64), device="cuda:0").unsqueeze(0)
                st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                st.record(); lg = m(ids)[0][0][0]; en.record(); torch.cuda.synchronize()
                bits = lg.contiguous().view(torch.int16).cpu().numpy()
                if rep == 0:
                    arrays[f"{name}/ids"] = g[f"{name}/ids"]
                    arrays[f"{name}/logits_bits"] = bits
                    per.append(dict(window=name, bp=int(ids.shape[1]), logits_dtype=str(lg.dtype),
                                    logits_shape=list(lg.shape), logits_sha256=sha(lg),
                                    cuda_event_s=round(st.elapsed_time(en) / 1e3, 3)))
                else:
                    w = next(x for x in per if x["window"] == name)
                    w["rerun_bit_identical"] = bool(np.array_equal(bits, arrays[f"{name}/logits_bits"]))
                    w["rerun_cuda_event_s"] = round(st.elapsed_time(en) / 1e3, 3)
                print(json.dumps(dict(rep=rep, window=name, sha256=sha(lg)[:16])), flush=True)
    np.savez_compressed(os.path.join(OUT, "evo2_1b_base_h100-te-fp8.npz"), **arrays)
    meta["windows"] = per

elif STAGE == "brca1":
    from Bio import SeqIO
    rows = list(csv.DictReader(open(f"{R}/data/brca1/brca1_subset500_deltas.csv")))
    with gzip.open(f"{R}/data/brca1/GRCh37.p13_chr17.fna.gz", "rt") as h:
        chr17 = str(next(SeqIO.parse(h, "fasta")).seq)

    def parse(pos, ref, alt):  # brca1_source_script.parse (upstream notebook logic)
        p = pos - 1
        s, e = max(0, p - WINDOW // 2), min(len(chr17), p + WINDOW // 2)
        r = chr17[s:e]
        k = min(WINDOW // 2, p)
        v = r[:k] + alt + r[k + 1:]
        assert r[k] == ref and v[k] == alt
        return r, v

    ref_seqs, idx, var_seqs, ref_index = [], {}, [], []
    for row in rows:
        r, v = parse(int(row["pos"]), row["ref"], row["alt"])
        if r not in idx:
            idx[r] = len(ref_seqs); ref_seqs.append(r)
        ref_index.append(idx[r]); var_seqs.append(v)
    assert len(var_seqs) == 500 and len(ref_seqs) == 435, (len(var_seqs), len(ref_seqs))
    runs = []
    for rep in (0, 1):
        t = time.perf_counter()
        rs = np.array(m.score_sequences(ref_seqs)); vs = np.array(m.score_sequences(var_seqs))
        runs.append(dict(rs=rs, vs=vs, wall_s=round(time.perf_counter() - t, 1)))
        print(json.dumps(dict(rep=rep, windows=len(ref_seqs) + len(var_seqs), wall_s=runs[-1]["wall_s"])), flush=True)
    np.savez(os.path.join(OUT, "evo2_1b_base_h100-te-fp8_brca1-subset500_scores.npz"),
             rs=runs[0]["rs"], vs=runs[0]["vs"], ref_index=np.array(ref_index))
    np.savez(os.path.join(OUT, "evo2_1b_base_h100-te-fp8_brca1-subset500_scores_rerun.npz"),
             rs=runs[1]["rs"], vs=runs[1]["vs"])
    meta.update(variants=len(var_seqs), unique_ref_windows=len(ref_seqs), score_dtype=str(runs[0]["rs"].dtype),
                wall_s=[x["wall_s"] for x in runs],
                rerun_bit_identical=bool(np.array_equal(runs[0]["rs"], runs[1]["rs"])
                                         and np.array_equal(runs[0]["vs"], runs[1]["vs"])))
else:
    raise SystemExit(f"unknown stage {STAGE}")

json.dump(meta, open(os.path.join(OUT, f"{STAGE}.json"), "w"), indent=1)
print("DONE", STAGE, flush=True)
