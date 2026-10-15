"""Replay the RTX 4060 recordings of recordings/rtx4060_2026-10 with the tcemu audit's FixedModel
(tcemu_ada_truth.models.FixedModel), as an independent tool in its own process. This is the ONLY place where the old
`tcemu` package is imported (FixedModel's exact reference path uses tcemu.reference); it never runs inside the mmaemu
tests.

- the truth package's replay loop is used with loader.list_recordings / loader.load redirected to
  recordings/rtx4060_2026-10 (root=); nothing is read from or written to the truth's own data or replay dirs;
- fastacc: the truth's fastacc.lib() is never called (it may rebuild into $TRUTH/build/lib); a private build of the
  same fastref3.c (tools/record/cpu_model.py, in $MMAEMU_CACHE_DIR) is installed as fastacc._lib;
- run with PYTHONDONTWRITEBYTECODE=1 (no __pycache__ in the tcemu checkout or the truth dir).
Maintainer host only: needs the tcemu prototype checkout ($MMAEMU_TCEMU_DIR) and the hardware ground truth
($MMAEMU_TRUTH_DIR), neither of which is published.

usage: PYTHONDONTWRITEBYTECODE=1 python tools/record/replay_fixed.py [--only REGEX] [--out FILE]
Writes the report JSON (default recordings/rtx4060_2026-10/replay_fixed.json) and prints one line per recording."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

# Checkout of the tcemu prototype (FixedModel's exact reference path imports tcemu.reference).
TCEMU = Path(os.environ["MMAEMU_TCEMU_DIR"]) if os.environ.get("MMAEMU_TCEMU_DIR") else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only")
    ap.add_argument("--skip")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=str(common.REC_ROOT / "replay_fixed.json"))
    a = ap.parse_args()
    if TCEMU is None or not TCEMU.is_dir():
        raise SystemExit("set MMAEMU_TCEMU_DIR to a tcemu prototype checkout (it provides FixedModel's reference)")
    common._require_truth()
    for p in (str(TCEMU), str(common.TRUTH)):
        if p not in sys.path:
            sys.path.append(p)
    import torch
    torch.set_num_threads(a.threads)
    import cpu_model
    from tcemu_ada_truth import fastacc, loader, replay
    from tcemu_ada_truth.models import FixedModel
    fastacc._lib = cpu_model.lib()          # private build; fastacc.lib() then returns it without rebuilding
    root = str(common.REC_ROOT)
    orig_list, orig_load = loader.list_recordings, loader.load
    loader.list_recordings = lambda r=None: orig_list(root)
    loader.load = lambda rid, r=None: orig_load(rid, root=root)
    t0 = time.time()
    rep = replay.replay([FixedModel()], only=a.only, skip=a.skip, hash_only=True)["fixed"]
    rep["root"] = root
    rep["wall_seconds"] = round(time.time() - t0, 1)
    rep["tcemu_imported"] = "tcemu" in sys.modules
    Path(a.out).write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
    print(json.dumps(rep["summary"]))


if __name__ == "__main__":
    main()
