"""Load check: every pre-swap recording loads through the truth package's own loader with sha256 checks of every
input and every stored output (and table payload file), with root=recordings/rtx4060_2026-10.

usage: PYTHONDONTWRITEBYTECODE=1 python tools/record/check_loads.py   (exit 1 on any failure or count mismatch)"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def main():
    L = common.truth_loader()
    F = common.truth_format()
    root = str(common.REC_ROOT)
    man = json.loads((common.REC_ROOT / "manifest.json").read_text())
    on_disk = sorted(p for p in common.REC_DATA.glob("*/*.json"))
    entries = L.list_recordings(root)
    ok = True
    if len(entries) != len(on_disk) or len(entries) != man["totals"]["recordings"]:
        print(f"COUNT MISMATCH: loader {len(entries)}, json files {len(on_disk)}, manifest {man['totals']}")
        ok = False
    n_in = n_out = n_err = n_tab = 0
    for e in entries:
        rec = L.load(e["id"], root=root)
        for f in e["files"]:
            if F.sha256_file(common.REC_ROOT / f["file"]) != f["sha256"]:
                print("FILE SHA256 MISMATCH", f["file"])
                ok = False
        if rec.kind == "table":
            for t in rec.meta.get("table_files", []):
                if F.sha256_file(rec.path.parent / t["file"]) != t["sha256"]:
                    print("TABLE SHA256 MISMATCH", rec.id, t["file"])
                    ok = False
                n_tab += 1
        for c in rec.cases:
            n_in += len(c.inputs())
            for o in c.outputs:
                if o.hw_error:
                    n_err += 1
                elif o.stored:
                    o.bits()
                    n_out += 1
            c.drop_inputs()
    print(f"{len(entries)} recordings load with sha256 OK: {n_in} inputs, {n_out} stored outputs, {n_err} hw_error "
          f"outputs, {n_tab} table payloads")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
