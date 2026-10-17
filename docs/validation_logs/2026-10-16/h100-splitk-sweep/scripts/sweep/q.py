"""Heuristic queries of one sweep part (GPU host; no kernels run). Writes SWEEP_OUT/<profile>_a<align>_cNN.npz + .json,
one file per sc.CHUNK queries, each saved as soon as it is done.
  p1, p2, p3: the part's shapes x the 30 settings of the profile's workspace at --align (default: the profile's);
              profile ws0 queries every 4th p1 shape
  p4:         the sealed rows of the profile's workspace (each row carries its setting); nothing about the results
              is printed or summarised (only counts of rows)
usage: q.py PART --profile torch|te [--align none|256]"""
from __future__ import annotations

import argparse
import time

import sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["p1", "p2", "p3", "p4"])
    ap.add_argument("--profile", required=True, choices=sorted(sc.PROFILE_WS))
    ap.add_argument("--align", default=None)
    ap.add_argument("--device", type=int, default=0)
    a = ap.parse_args()
    ws, al = sc.PROFILE_WS[a.profile]
    if a.align is not None:
        al = 0 if a.align == "none" else int(a.align)
    shapes = sc.load_shapes(a.part)
    if a.part == "p4":
        pairs = [((M, N, K), (w, g, o, fa, fb, f)) for (M, N, K, w, g, o, fa, fb, f) in shapes if w == ws]
    else:
        if a.part == "p1" and a.profile == "ws0":
            shapes = shapes[::4]
        pairs = list(sc.cross([tuple(s) for s in shapes], sc.settings(ws, al)))
    hw = sc.hardware(a.device)
    sha = sc.sha256(sc.SHAPES / f"{a.part}.json")
    T0 = time.monotonic()
    chunks = [pairs[i:i + sc.CHUNK] for i in range(0, len(pairs), sc.CHUNK)]
    for q, ch in enumerate(chunks):
        t0 = time.monotonic()
        cols = sc.query(ch, a.device, progress=f"{a.part} {a.profile} align {al} chunk {q}")
        sec = time.monotonic() - t0
        path = sc.OUT / f"{a.profile}_a{al}_c{q:02d}.npz"
        meta = dict(part=a.part, profile=a.profile, ws=ws, align=al, chunk=q, chunks=len(chunks),
                    pairs=[q * sc.CHUNK, q * sc.CHUNK + len(ch)], hardware=hw, seconds=round(sec, 1),
                    shapes_file=f"shapes/{a.part}.json", shapes_sha256=sha)
        if a.part != "p4":
            meta["summary"] = sc.split_summary(cols)
        sc.save(cols, path, meta)
        msg = f"{a.part} {a.profile} align {al} chunk {q + 1}/{len(chunks)}: {len(ch)} queries in {sec:.1f}s " \
              f"({len(ch) / max(sec, 1e-9):.0f}/s), status!=0 {int((cols['status'] != 0).sum())}"
        if a.part != "p4":
            s = meta["summary"]
            msg += f"; split rows {s['split_rows']}, largest split M*N {s['max_split_MN']}, S {s['S_values']}"
        print(msg, flush=True)
    print(f"{a.part} {a.profile} align {al}: {len(pairs)} queries in {time.monotonic() - T0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
