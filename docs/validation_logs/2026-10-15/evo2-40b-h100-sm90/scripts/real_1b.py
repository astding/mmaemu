"""The real upstream Evo 2 1B on this GPU, for checking official_blockwise.py here: evo2.Evo2("evo2_1b_base") with
Transformer Engine FP8. The test sequences as evo2/test/test_evo2.py runs them (model.model.forward in inference
mode), and the BRCA1 windows of the first N variants with model.score_sequences (batch size 1).
usage: python real_1b.py --inputs inputs.npz --out DIR [--brca1-variants N]"""
import argparse
import hashlib
import json
import os

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--inputs", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--brca1-variants", type=int, default=10)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
torch.cuda.set_per_process_memory_fraction(0.95)
from evo2 import Evo2  # noqa: E402

m = Evo2("evo2_1b_base")
fp8 = sum(bool(getattr(x, "use_fp8_input_projections", False)) for x in m.model.modules())
assert fp8 > 0, "the model has no FP8 TE projections"
inp = np.load(a.inputs)
rows, arrays = [], {}
for name in [str(n) for n in inp["test_names"]]:
    s = bytes(inp[f"test/{name}"].astype(np.uint8)).decode("ascii")
    ids = torch.tensor(m.tokenizer.tokenize(s), dtype=int).to("cuda:0")
    assert np.array_equal(ids.cpu().numpy(), inp[f"test/{name}"])
    with torch.inference_mode():
        logits, _ = m.model.forward(ids.unsqueeze(0))
    lg = logits[0]
    arrays[f"{name}/logits_bits"] = lg.contiguous().view(torch.int16).cpu().numpy()
    rows.append(dict(name=name, logits_sha256=hashlib.sha256(lg.contiguous().view(torch.uint8).cpu().numpy()
                                                              .tobytes()).hexdigest()))
N = a.brca1_variants
ref_index = inp["brca1/ref_index"][:N]
n_ref = int(ref_index.max()) + 1
dec = lambda x: bytes(x.astype(np.uint8)).decode("ascii")  # noqa: E731
rs = np.array(m.score_sequences([dec(inp[f"brca1/ref/{i}"]) for i in range(n_ref)]), dtype=np.float32)
vs = np.array(m.score_sequences([dec(inp[f"brca1/var/{i}"]) for i in range(N)]), dtype=np.float32)
np.savez(os.path.join(a.out, "brca1.npz"), rs=rs, vs=vs)
np.savez_compressed(os.path.join(a.out, "test.npz"), **arrays)
json.dump(dict(model="evo2_1b_base", kind="real upstream evo2.Evo2 with TE FP8", fp8_projection_modules=fp8,
               sequences=rows, brca1_variants=N), open(os.path.join(a.out, "test.json"), "w"), indent=1)
print("real 1B done", flush=True)
