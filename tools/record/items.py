"""Driver for the live-call items R2v, R4, R5: build jobs (CPU), run them (one locked GPU run), write the recording.

usage:
  python tools/record/items.py jobs  {r2v,r4,r5}          CPU: job list (+ coverage list for r4) in the stage dir
  python tools/record/items.py gpu   {r2v,r4,r5}          one locked GPU run (gpu_run.py -> live.py run)
  python tools/record/items.py write {r2v,r4,r5}          CPU: recording + committed side files
Stage dir: $MMAEMU_RECORD_STAGE/<item> (default build/record-stage/<item>)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

HERE = Path(__file__).resolve().parent
SCRIPTS = ["tools/record/items.py", "tools/record/jobs.py", "tools/record/live.py", "tools/record/common.py",
           "tools/record/cpu_model.py", "tools/record/tables.py", "tools/record/coverage.py"]

ITEMS = {
    "r4": dict(rid="r4.unverified_split_combos",
               title="R4: torch._scaled_mm on the smallest torch-reachable table row of every split-K combination "
                     "(fast, S, scheme, out, fmt pair) without a hardware replay (156 of the truth table incl. the "
                     "3 S=29 ones, on randn*40 and split-structured adversarial data; + 153 combinations that only "
                     "the WS2 R2/R2x rows add, adversarial data), bits + logged cuBLASLt config",
               claims=["SPEC v0.1 WS2 R4 / acceptance A2: logged (numSplitsK, customOption) == table row",
                       "closes the coverage gap of SPEC section 5 (156 of 316 combinations, S=29)"]),
    "r5": dict(rid="r5.autosplit_in_envelope",
               title="R5: 300 random in-envelope table shapes at the torch default workspace (1 MiB, no alignment "
                     "preference), mixed out dtype / fast / format pair, bits + logged cuBLASLt config",
               claims=["SPEC v0.1 WS2 R5 / acceptance A2: logged (numSplitsK, customOption) == table row"]),
    "r2v": dict(rid="r2.fp8out_live_verify",
                title="R2-verify: 100 random fp8-output heuristic rows (R2) re-run live through torch._scaled_mm "
                      "(scale_result passed on half of them) + critic gap 17 (e4m3 out at 224x176x80 and "
                      "176x224x80), bits + logged cuBLASLt config",
                claims=["SPEC v0.1 WS2 R2: 100 rows verified live", "critic gap 17"]),
}


def stage(item: str) -> Path:
    return common.STAGE / item


def cmd_jobs(item):
    st = stage(item)
    st.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([sys.executable, str(HERE / "jobs.py"), item, "--out", str(st / "jobs.json")],
                          env=common.worker_env(None, None))
    if item == "r4":
        subprocess.check_call([sys.executable, str(HERE / "coverage.py"), "--out", str(st / "coverage.json")],
                              env=common.worker_env(None, None))


def cmd_gpu(item):
    st = stage(item)
    run = st / "run"
    if run.exists():
        for p in run.iterdir():
            p.unlink()
    rc = subprocess.call([sys.executable, str(HERE / "gpu_run.py"), f"{item.upper()}.live", "--",
                          sys.executable, str(HERE / "live.py"), "run", "--jobs", str(st / "jobs.json"),
                          "--stage", str(run)], env=common.worker_env(None, None))
    sys.exit(rc)


def cmd_write(item):
    import live
    st = stage(item)
    it = ITEMS[item]
    meta = json.loads((st / "jobs.meta.json").read_text())
    notes = (f"Job builder: tools/record/jobs.py ({item}); builder summary: "
             f"{json.dumps({k: v for k, v in meta.items() if k != 'envelopes'}, default=str)[:3000]}. "
             "Inputs: deterministic integer-hash generators (tools/record/live.py make_codes: 'randn40' = codes "
             "distributed as RNE_fp8(clamp(40*N(0,1))); 'adv' = per-tile magnitudes; 'adv_split' = split-structured "
             "mirrored tiles, p_{S-1} = -p_0), generated on the GPU for the call and regenerated on CPU for the "
             "stored subset (byte equality checked per call: inputs_cpu_eq_gpu). Stored operands are a row/column "
             "subset with full K of the GPU call (case params hw_shape + sub); the split partition depends on K "
             "only, so a GEMM of the stored operands with the recorded (S, scheme) reproduces the stored block. "
             "output params.splitk = the LOGGED config of this call; params.table_row = the heuristic-table row "
             "the job was built from; params.config_check compares (algo, tile, numSplitsK, customOption).")
    summ = live.write_recording(it["rid"], it["title"], str(st / "jobs.json"), str(st / "run"), it["claims"],
                                notes, SCRIPTS)
    out = {"item": item, "recording": it["rid"], "summary": summ}
    if item == "r4":
        cov = json.loads((st / "coverage.json").read_text())
        jobs = json.loads((st / "jobs.json").read_text())
        out["combos"] = {"report": cov["report"], "unverified_truth_table": cov["unverified_truth_table"],
                         "unverified_union_only": cov["unverified_union_only"],
                         "jobs": [dict(id=j["id"], combo=j["combo"], part=j["part"], M=j["M"], N=j["N"], K=j["K"],
                                       ws=j["ws"], table_source=j["table"]["source"],
                                       all_distinguished=j["discrimination"]["all_distinguished"])
                                  for j in jobs]}
        (common.REC_ROOT / "r4_unverified_combos.json").write_text(json.dumps(out["combos"], indent=1,
                                                                              sort_keys=True) + "\n")
    (common.REC_ROOT / f"{item}_summary.json").write_text(json.dumps(out["summary"], indent=1, sort_keys=True)
                                                          + "\n")


def main():
    cmd, item = sys.argv[1], sys.argv[2]
    {"jobs": cmd_jobs, "gpu": cmd_gpu, "write": cmd_write}[cmd](item)


if __name__ == "__main__":
    main()
