"""Write recordings/rtx4060_2026-10/manifest.json (format tcemu_ada_truth/1) for the pre-swap recordings.

tcemu_ada_truth.loader.list_recordings(root) reads <root>/manifest.json, whose "recordings" entries carry "id" and
"json" (path relative to root). Same layout as the truth dataset: <root>/data/<group>/<name>.json + its .npz files.
Every file's byte size and sha256 is listed; totals at the top. Deterministic (no timestamps, sorted keys)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def build() -> dict:
    F = common.truth_format()
    recs = []
    totals = dict(recordings=0, cases=0, outputs=0, outputs_stored=0, hw_error_outputs=0, files=0, bytes=0)
    for jp in sorted(common.REC_DATA.glob("*/*.json")):
        j = json.loads(jp.read_text())
        if not isinstance(j, dict) or j.get("format") != F.FORMAT:
            continue
        files = []
        names = list(j.get("files", [])) + [t["file"] for t in j.get("table_files", [])]
        for fn in [jp.name] + sorted(set(names)):
            p = jp.parent / fn
            files.append({"file": str(p.relative_to(common.REC_ROOT)), "bytes": p.stat().st_size,
                          "sha256": F.sha256_file(p)})
        n_out = sum(len(c["outputs"]) for c in j.get("cases", []))
        n_err = sum(1 for c in j.get("cases", []) for o in c["outputs"] if o.get("hw_error"))
        recs.append({"id": j["id"], "group": j["group"], "json": str(jp.relative_to(common.REC_ROOT)),
                     "title": j.get("title"), "level": j.get("level"), "kind": j["kind"],
                     "n_cases": len(j.get("cases", [])), "n_outputs": n_out, "n_hw_error": n_err,
                     "stats": j.get("stats"), "files": files})
        totals["recordings"] += 1
        totals["cases"] += len(j.get("cases", []))
        totals["outputs"] += n_out
        totals["outputs_stored"] += (j.get("stats") or {}).get("n_outputs_stored", 0)
        totals["hw_error_outputs"] += n_err
        totals["files"] += len(files)
        totals["bytes"] += sum(f["bytes"] for f in files)
    return {"format": F.FORMAT, "title": "mmaemu WS2 pre-swap recordings on the RTX 4060 (sm_89), 2026-10",
            "layout": "data/<group>/<name>.json + .npz (written with tcemu_ada_truth.format.RecordingWriter, "
                      "root=<this dir>/data); load with tcemu_ada_truth.loader.load(id, root=<this dir>)",
            "totals": totals, "recordings": recs}


def main():
    m = build()
    p = common.REC_ROOT / "manifest.json"
    p.write_text(json.dumps(m, indent=1, sort_keys=True) + "\n")
    print("wrote", p, json.dumps(m["totals"]))


if __name__ == "__main__":
    main()
