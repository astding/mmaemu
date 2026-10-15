"""Golden from the REAL upstream model: evo2.Evo2(<model>) (with Transformer Engine when installed), full logits.
usage: python golden_real.py --model evo2_1b_base --windows p0,p1,p2,p3,w8k,w2k"""
import argparse, time
import torch
import common

ap = argparse.ArgumentParser(); ap.add_argument("--model", default="evo2_1b_base")
ap.add_argument("--windows", default="p0,p1,p2,p3,w8k,w2k"); a = ap.parse_args()
torch.cuda.set_per_process_memory_fraction(0.95)
from evo2 import Evo2
m = Evo2(a.model)
tok = m.tokenizer
common.bench.ACGT = torch.tensor([int(tok.tokenize(c)[0]) for c in "ACGT"], dtype=torch.long)
arrays, per = {}, []
with torch.inference_mode():
    for name, s in common.windows(a.windows.split(",")):
        ids = torch.tensor(tok.tokenize(s), dtype=torch.long, device="cuda:0").unsqueeze(0)
        st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        st.record(); lg = m(ids)[0][0][0]; en.record(); torch.cuda.synchronize()
        acgt, nxt = common.bench.summarize(lg, ids)
        arrays[f"{name}/ids"] = ids[0].cpu().numpy().astype("int16")
        arrays[f"{name}/logits_bits"] = common.bits(lg)
        arrays[f"{name}/acgt"], arrays[f"{name}/next"] = acgt, nxt
        per.append(dict(window=name, bp=len(s), logits_dtype=str(lg.dtype), logits_shape=list(lg.shape),
                        logits_sha256=common.sha(lg), cuda_event_s=round(st.elapsed_time(en) / 1e3, 3),
                        mean_next_logprob=float(nxt.mean())))
fp8 = any(getattr(x, "use_fp8_input_projections", False) for x in m.model.modules())
tag = f"{a.model}_real-{'te-fp8' if fp8 else 'bf16'}"
common.save(tag, arrays, dict(tag=tag, kind="real upstream evo2.Evo2", model=a.model, fp8_projections=fp8,
                              windows=per, env=common.env_meta()))
