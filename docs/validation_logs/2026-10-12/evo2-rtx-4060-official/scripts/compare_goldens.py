"""Bitwise comparison of two golden npz files (logits bit patterns per window)."""
import sys, numpy as np
A, B = np.load(sys.argv[1]), np.load(sys.argv[2])
wa = sorted({k.split("/")[0] for k in A.files}); wb = sorted({k.split("/")[0] for k in B.files})
assert set(wa) == set(wb), (wa, wb)
for w in wa:
    x, y = A[f"{w}/logits_bits"], B[f"{w}/logits_bits"]
    assert x.shape == y.shape, (w, x.shape, y.shape)
    print(w, x.shape, "logits bit-identical:", bool((x == y).all()), "differing elements:", int((x != y).sum()))
