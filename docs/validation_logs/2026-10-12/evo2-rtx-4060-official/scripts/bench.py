"""Memory / speed / output benchmark. One configuration per process so peak RSS is clean.

python bench.py --model evo2_7b --device gpu --out results/
python bench.py --model evo2_7b --device cpu --dtype float32 --windows test1 --out results/
Outputs <out>/<tag>.json (memory, timing) and <tag>.npz (per-position log-probs of A,C,G,T and of the true next token).
"""
import argparse, csv, json, os, subprocess, threading, time
from importlib import resources

import numpy as np
import torch

ACGT = None


def rss_gib(field="VmHWM"):
    for line in open("/proc/self/status"):
        if line.startswith(field):
            return int(line.split()[1]) / 2**20
    return None


class RssSampler(threading.Thread):
    """Peak anonymous (non-reclaimable) and file-backed RSS. File-backed pages of memory-mapped weights are page
    cache the kernel can evict; anonymous memory is what the process truly needs."""

    def __init__(self):
        super().__init__(daemon=True)
        self.anon = self.file = 0.0
        self.stop = False

    def run(self):
        while not self.stop:
            for line in open("/proc/self/status"):
                if line.startswith("RssAnon"):
                    self.anon = max(self.anon, int(line.split()[1]) / 2**20)
                elif line.startswith("RssFile"):
                    self.file = max(self.file, int(line.split()[1]) / 2**20)
            time.sleep(0.2)


class SmiSampler(threading.Thread):
    """Peak per-process GPU memory as the driver sees it (includes CUDA context, cuBLAS workspaces, allocator slack)."""

    def __init__(self, pid):
        super().__init__(daemon=True)
        self.pid, self.peak, self.stop = str(pid), 0, False

    def run(self):
        exe = "nvidia-smi" if subprocess.run(["which", "nvidia-smi"], capture_output=True).returncode == 0 else "/usr/lib/wsl/lib/nvidia-smi"
        while not self.stop:
            try:
                out = subprocess.run([exe, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout.strip()
                self.peak = max(self.peak, int(out.splitlines()[0]))
            except Exception:
                pass
            time.sleep(0.5)


def read_prompts():
    with resources.as_file(resources.files("evo2.test.data") / "prompts.csv") as p:
        rows = list(csv.reader(open(p, encoding="utf-8-sig", newline="")))
    return [r[0] for r in rows[1:]]


def windows(kind):
    pr = read_prompts()
    w8k = "".join(pr)[:8192]
    return {"test": [("p%d" % i, s) for i, s in enumerate(pr)] + [("w8k", w8k)],
            "all": [("p%d" % i, s) for i, s in enumerate(pr)] + [("w8k", w8k), ("w2k", w8k[:2048])],
            "test1": [("p1", pr[1])], "w8k": [("w8k", w8k)], "w2k": [("w2k", w8k[:2048])],
            "w4k": [("w4k", w8k[:4096])]}[kind]


def summarize(lg, ids):
    """lg: (L, V) logits; returns log-probs of A,C,G,T at every position, and of the true next token."""
    lp = torch.log_softmax(lg.float(), dim=-1)
    acgt = lp[:, ACGT].cpu().numpy()
    nxt = lp[:-1].gather(-1, ids[0, 1:, None].to(lp.device)).squeeze(-1).cpu().numpy()
    return acgt, nxt


def main():
    global ACGT
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", choices=["gpu", "cpu"], required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--windows", default="test")
    ap.add_argument("--no-lut", action="store_true")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--source", default="checkpoint", choices=["checkpoint", "blocks"])
    ap.add_argument("--evict", action="store_true", help="re-read every block from disk each pass (low-RAM emulation)")
    ap.add_argument("--fp8-mode", default="auto", choices=["auto", "native", "compat", "emulate", "bf16"])
    ap.add_argument("--check-emulate", action="store_true", help="emulate mode: compare every FP8 GEMM against native")
    ap.add_argument("--kernels", action="store_true", help="Triton HC* kernels instead of the dense upstream path")
    ap.add_argument("--dense-chunk", type=int, default=1024, help="channel chunk for the dense path (0 = unchunked)")
    ap.add_argument("--slots", default="auto", help="'auto' or an int (pure streaming with that many slots)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tag = f"{a.model}_{a.device}_{a.dtype}_{a.windows}{'_nolut' if a.no_lut else ''}{'_blocks' if a.source == 'blocks' else ''}{'_evict' if a.evict else ''}{'' if a.fp8_mode == 'auto' else '_' + a.fp8_mode}{'' if a.slots == 'auto' else '_slots' + a.slots}{'_kernels' if a.kernels else ('_chunk%d' % a.dense_chunk if a.dense_chunk else '_dense')}"
    from vortex.model.tokenizer import CharLevelTokenizer

    tok = CharLevelTokenizer(512)
    ACGT = torch.tensor([int(tok.tokenize(c)[0]) for c in "ACGT"], dtype=torch.long)
    rec = dict(tag=tag, model=a.model, device=a.device, dtype=a.dtype, lut=not a.no_lut, pid=os.getpid())
    wins = windows(a.windows)
    rss = RssSampler(); rss.start()
    t0 = time.perf_counter()
    if a.device == "gpu":
        smi = SmiSampler(os.getpid()); smi.start()
        from evo2_anywhere.api import Evo2Anywhere

        m = Evo2Anywhere(a.model, use_lut=not a.no_lut, source=a.source, evict=a.evict, fp8_mode=a.fp8_mode,
                         kernels=a.kernels, dense_chunk=a.dense_chunk or None,
                         slots=int(a.slots) if a.slots != "auto" else "auto")
        rec["fp8_mode"] = m.fp8_mode
        if a.check_emulate:
            from evo2_anywhere import fp8 as _fp8
            _fp8.CHECK["enabled"] = True
            rec["emulate_check"] = _fp8.CHECK
        rec["source"] = a.source
        rec["load_s"] = round(time.perf_counter() - t0, 1)
        rec["rss_after_load_gib"] = round(rss_gib("VmRSS"), 2)
        rec["plan"] = dict(n_resident=m.runner.plan.n_resident, n_slots=m.runner.plan.n_slots, n_blocks=m.runner.plan.n_blocks)
        ids_list = [m._ids(s) for _, s in wins]
        for ids in ids_list:  # warm-up: Triton JIT/autotune per sequence length
            m.runner.run([ids])
        torch.cuda.reset_peak_memory_stats()
        run = lambda ids: m.runner.run([ids])[0][0]
        sync = torch.cuda.synchronize
    else:
        import cpu_patches  # noqa: F401
        from evo2_anywhere import registry
        from evo2_anywhere.build import build_model
        from evo2_anywhere.lut import install_lut, load_lut
        from evo2_anywhere.weights import resolve

        torch.set_num_threads(a.threads)
        spec = registry.get(a.model)
        model, _ = build_model(spec, resolve(spec), kernels=False, flash_attn=False)
        lp = install_lut(model, load_lut(spec)[0]) if (spec.needs_lut and not a.no_lut) else None
        dt = getattr(torch, a.dtype)
        if dt != torch.bfloat16:
            model.to(dt)
            if lp is not None:
                lp.table = lp.table.to(dt)
        for i in model.block_idx_to_device:
            model.block_idx_to_device[i] = "cpu"
        rec["load_s"] = round(time.perf_counter() - t0, 1)
        rec["rss_after_load_gib"] = round(rss_gib("VmRSS"), 2)
        ids_list = [torch.tensor(tok.tokenize(s), dtype=torch.long).unsqueeze(0) for _, s in wins]

        def run(ids):
            if lp is not None:
                lp.ids = ids
            return model(ids)[0][0]

        sync = lambda: None
    per, out = [], {}
    with torch.inference_mode():
        for (name, s), ids in zip(wins, ids_list):
            sync(); t = time.perf_counter()
            lg = run(ids)
            sync(); dt_s = time.perf_counter() - t
            acgt, nxt = summarize(lg, ids)
            out[f"{name}/acgt"], out[f"{name}/next"] = acgt, nxt
            per.append(dict(window=name, bp=len(s), secs=round(dt_s, 2), bp_per_s=round(len(s) / dt_s, 1),
                            mean_next_logprob=float(nxt.mean())))
            print(per[-1], flush=True)
    rec["per_window"] = per
    rec["peak_rss_gib"] = round(rss_gib("VmHWM"), 2)
    rss.stop = True; rss.join(1)
    rec["peak_rss_anon_gib"] = round(rss.anon, 2)
    rec["peak_rss_file_gib"] = round(rss.file, 2)
    if a.device == "gpu":
        smi.stop = True; smi.join(2)
        rec["peak_vram_torch_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        rec["peak_vram_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
        rec["peak_gpu_used_smi_mib_total_device"] = smi.peak
    tot_bp = sum(p["bp"] for p in per); tot_s = sum(p["secs"] for p in per)
    rec["throughput_bp_per_s"] = round(tot_bp / tot_s, 1)
    np.savez_compressed(os.path.join(a.out, tag + ".npz"), **out)
    json.dump(rec, open(os.path.join(a.out, tag + ".json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in rec.items() if k != "per_window"}), flush=True)


if __name__ == "__main__":
    main()
