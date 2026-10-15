"""Backend fallback ladder on this device: the selected backend, its source and the BackendFallbackWarnings with
nothing forced and with MMAEMU_TEST_FORCE_FAIL=fatbin, =fatbin,nvrtc and =design_a,window (one child process each,
fresh kernel cache). Run after the first T3 run (run1-76fd4ea); the unused `json` import of the run is removed here.
"""
import os, subprocess, sys, tempfile
CODE = r'''
import json, warnings, torch
from mmaemu.errors import BackendFallbackWarning
from mmaemu.backends import ChainRequest, registry
import mmaemu.api as api
dev = torch.device("cuda", 0)
g = torch.Generator().manual_seed(0)
a = (torch.randn(96, 320, generator=g) * 40).clamp(-448, 448).to(torch.float8_e4m3fn)
b = (torch.randn(80, 320, generator=g) * 40).clamp(-448, 448).to(torch.float8_e4m3fn)
with warnings.catch_warnings(record=True) as rec:
    warnings.simplefilter("always")
    be = registry.select(ChainRequest(device=dev, fmt_a="e4m3", fmt_b="e4m3", preset="ada"))
    o = api.accumulate(a.to(dev), b.to(dev), use_fast_accum=True, splitk=1)
w = [str(x.message) for x in rec if issubclass(x.category, BackendFallbackWarning)]
print(json.dumps(dict(backend=be.name, source=getattr(be, "source", None), warnings=w)))
'''
for force in ("", "fatbin", "fatbin,nvrtc", "design_a,window"):
    env = dict(os.environ, MMAEMU_CACHE_DIR=tempfile.mkdtemp(), PYTHONDONTWRITEBYTECODE="1")
    env.pop("MMAEMU_TEST_FORCE_FAIL", None)
    if force:
        env["MMAEMU_TEST_FORCE_FAIL"] = force
    r = subprocess.run([sys.executable, "-c", CODE], env=env, capture_output=True, text=True, timeout=600)
    out = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-800:]
    print(f"FORCE_FAIL={force or '-'}: {out}")
