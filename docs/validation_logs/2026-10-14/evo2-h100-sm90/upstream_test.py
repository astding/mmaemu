"""The metric of the upstream Evo 2 test (evo2/test/test_evo2.py) from saved bf16 logits.

usage: python upstream_test.py NAME=LOGITS.npz [NAME=LOGITS.npz ...] > upstream_test.json

Each npz holds <window>/ids (token ids) and <window>/logits_bits (the bf16 logits as int16), the layout of the
golden files. For each of the four test prompts p0..p3: cross entropy of logits[:-1] against ids[1:] and the
fraction of positions whose argmax is the next token; then the mean over the prompts. The upstream test expects
about 0.502 and 0.7956 for 1B.

The upstream test evaluates the cross entropy in bf16 on the GPU, which rounds each prompt's loss to bf16 (steps of
0.001-0.008 at these values) and differs slightly between implementations. Here it is computed in fp32 from the same
bf16 logits, which does not depend on the device; the accuracy is the same either way."""
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

PROMPTS = ["p0", "p1", "p2", "p3"]
out = {}
for arg in sys.argv[1:]:
    name, path = arg.split("=", 1)
    z = np.load(path)
    per = {}
    for w in PROMPTS:
        ids = torch.from_numpy(z[f"{w}/ids"].astype(np.int64))
        logits = torch.from_numpy(z[f"{w}/logits_bits"].copy()).view(torch.bfloat16)
        per[w] = dict(loss=F.cross_entropy(logits[:-1].float(), ids[1:]).item(),
                      accuracy=(logits[:-1].argmax(-1) == ids[1:]).float().mean().item())
    out[name] = dict(loss=float(np.mean([p["loss"] for p in per.values()])),
                     accuracy=float(np.mean([p["accuracy"] for p in per.values()])), prompts=per)
print(json.dumps(out, indent=1))
