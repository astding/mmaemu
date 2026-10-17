"""This directory's outputs (2648bba, after the profile rename) against those of the same steps on fa0fd90.

    python compare_to_fa0fd90.py > compare_to_fa0fd90.txt

Before comparing, the old profile names are replaced by the new ones, timings are removed from pytest's lines, and
the keys in DROP (timings, dates, commits, command lines, file paths, Python) are dropped. Prints one line per
output with its number of differences, and ends with DIFFERENCES <n>."""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parent
MAP = [("hopper-h100pcie-torch-ws", "hopper-ws"), ("hopper-h100pcie-torch", "hopper"),
       ("hopper-h100pcie-te", "hopper-te"), ("hopper-h100pcie-ws0", "hopper-ws0"), ("ada-rtx4060-torch", "ada"),
       ("ada-rtx4060-te", "ada-te"), ("ada-rtx4060-ws0", "ada-ws0")]
DROP = {"seconds", "date", "argv", "mmaemu_file", "python", "repo_commit", "mmaemu_commit"}   # keys
TIME = re.compile(r" in [0-9.]+s( \([0-9:]+\))?|\s+[0-9.]+s(?=\s|$)")


def names(s):
    for a, b in MAP:
        s = s.replace(a, b)
    return s


def norm(x):
    if isinstance(x, dict):
        return {k: norm(v) for k, v in x.items() if k not in DROP}
    if isinstance(x, list):
        return [norm(v) for v in x]
    if isinstance(x, str):
        return TIME.sub("", names(x))
    return x


def diff(a, b, p, out):
    if type(a) is not type(b):
        out.append(f"{p}: type {type(a).__name__} vs {type(b).__name__}")
    elif isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k in a and k in b:
                diff(a[k], b[k], f"{p}/{k}", out)
            else:
                out.append(f"{p}/{k}: only in {'fa0fd90' if k in a else '2648bba'}")
    elif isinstance(a, list):
        if len(a) != len(b):
            out.append(f"{p}: length {len(a)} vs {len(b)}")
        for i, (u, v) in enumerate(zip(a, b)):
            diff(u, v, f"{p}[{i}]", out)
    elif a != b:
        out.append(f"{p}: {str(a)[:80]!r} vs {str(b)[:80]!r}")


def lines(path, keep):
    return [TIME.sub("", names(s)).rstrip() for s in path.read_text().splitlines() if keep(s)]


total = 0


def report(label, out):
    global total
    total += len(out)
    print(f"{label}: {len(out)} differences")
    for s in out[:20]:
        print("   ", s)


PAIRS_JSON = [
    ("t2/compare.json", "t2/compare.json"),
    ("t4/mutants-cpu.json", "t4/mutants-cpu.json"),
    ("t4/mutants-gpu.json", "t4/mutants-gpu.json"),
    ("rtx-5070-ti-replays/probe/mmaemu_replay.json", "probe/mmaemu_replay.json"),
    ("rtx-5070-ti-replays/sweep/mmaemu_replay_p5.json", "sweep/mmaemu_replay_p5.json"),
    ("h100-splitk-sweep/results/analysis/resolver_check.json", "resolver_check.json"),
]
for old, new in PAIRS_JSON:
    out = []
    diff(norm(json.loads((OLD / old).read_text())), norm(json.loads((HERE / new).read_text())), "", out)
    report(new, out)

t3old = OLD / "t3-rtx-5070-ti-sm120"
for f in sorted((HERE / "t3").iterdir()):
    o = t3old / f.name
    if not o.exists():
        report(f"t3/{f.name}", ["only in 2648bba"])
        continue
    if f.name == "summary.txt":   # one line per run: name, exit code, seconds, pytest's last line
        keep = lambda s: not s.startswith(("fatbins under test", "logs:"))
        a = [re.sub(r"rc=(\d+)\s+\d+s", r"rc=\1", s) for s in lines(o, keep)]
        b = [re.sub(r"rc=(\d+)\s+\d+s", r"rc=\1", s) for s in lines(f, keep)]
    else:                         # pytest -q: the per-test progress characters, the skip reasons and the counts
        keep = lambda s: (re.match(r"[.sFExX]+\s+\[\s*\d+%\]$", s) or s.startswith(("SKIPPED", "FAILED", "ERROR"))
                          or re.match(r"\d+ (passed|failed|skipped|deselected|error)", s))
        a, b = lines(o, keep), lines(f, keep)
        if not a:
            report(f"t3/{f.name}", ["no pytest output to compare"])
            continue
    report(f"t3/{f.name}", [f"{u!r} vs {v!r}" for u, v in zip(a, b) if u != v]
           + ([f"{len(a)} vs {len(b)} lines"] if len(a) != len(b) else []))

keep = lambda s: (re.match(r"[.sFExX]+\s+\[\s*\d+%\]$", s) or s.startswith(("SKIPPED", "FAILED", "ERROR"))
                  or re.match(r"\d+ (passed|failed|skipped|deselected|error)", s))
a, b = lines(OLD / "rtx-5070-ti-replays/pytest_full.out", keep), lines(HERE / "pytest_full.out", keep)
report("pytest_full.out", [f"{u!r} vs {v!r}" for u, v in zip(a, b) if u != v]
       + ([f"{len(a)} vs {len(b)} lines"] if len(a) != len(b) else []))
print("DIFFERENCES", total)
