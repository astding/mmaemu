"""Command line: ``python -m mmaemu {info,selftest,explain}``.

  info       versions, device, which platform profiles match it, backend status (with the reason when a backend is
             unavailable), fatbins, split-K tables and preset maturity. Always exits 0 when the report was built.
  selftest   runs the kernel gates on a CUDA device (``mmaemu.selftest``). Exit 0 = every kernel required on this
             device passed; 1 = a required kernel failed; 4 = nothing could be tested (no CUDA device, or the
             self-test is unavailable). ``--json PATH`` writes the full report.
  explain    shows what ``mmaemu.scaled_mm`` would do for an M x N x K call (validation, profile, preset maturity,
             split-K decision, backend, fallbacks, warnings) without running the GEMM. Exit 0 = the call is valid,
             1 = the call would raise, 4 = explain is unavailable (also: the device cannot be used).

Exit status 2 is a usage error (including a negative size or a malformed --splitk).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import diagnostics as D

OUT_DTYPES = {"fp32": "float32", "float32": "float32", "bf16": "bfloat16", "bfloat16": "bfloat16",
              "fp16": "float16", "float16": "float16", "e4m3": "float8_e4m3fn", "e5m2": "float8_e5m2"}
IN_DTYPES = {"e4m3": "float8_e4m3fn", "e5m2": "float8_e5m2"}


def _write_json(obj, path: str) -> None:
    text = json.dumps(D.to_jsonable(obj), indent=1, sort_keys=False)
    if path == "-":
        print(text)
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text + "\n")


def cmd_info(args) -> int:
    rep = D.collect_info(args.device)
    if args.json == "-":
        _write_json(rep, "-")
    else:
        print(D.format_info(rep))
        if args.json:
            _write_json(rep, args.json)
    return D.EXIT_OK


def cmd_selftest(args) -> int:
    rep = D.run_selftest(args.device, kernels=args.kernels, force=args.force)
    print(D.format_selftest(rep))
    if args.json:
        _write_json(rep, args.json)
    return D.selftest_exit_code(rep)


def parse_splitk(text: str):
    """--splitk: 'auto', an int, or 'S,scheme'; anything else is a usage error (exit 2)."""
    try:
        if text == "auto":
            return "auto"
        if "," in text:
            s, scheme = text.split(",", 1)
            return (int(s), scheme.strip())
        return int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid split-K {text!r}: use 'auto', '1', or 'S,serial' / "
                                         f"'S,pairwise'") from None


def _size(text: str) -> int:
    try:
        v = int(text)
    except ValueError:
        v = -1
    if v < 0:
        raise argparse.ArgumentTypeError(f"invalid size {text!r}: a non-negative integer")
    return v


def _plan_dict(plan) -> dict:
    d = D.to_jsonable(plan)
    if not isinstance(d, dict):
        d = {"plan": d}
    return d


def format_plan(shape: str, d: dict) -> str:
    L = [f"explain {shape}"]
    for k in ("validation", "profile", "preset", "preset_maturity", "split", "backend", "fallbacks", "warnings"):
        if k in d:
            v = d[k]
            if k == "validation" and isinstance(v, list) and len(v) == 2:
                v = f"{v[0]}: {v[1]}"
            L.append(f"  {k:16s} {v}")
    for k, v in d.items():
        if k not in ("validation", "profile", "preset", "preset_maturity", "split", "backend", "fallbacks",
                     "warnings"):
            L.append(f"  {k:16s} {v}")
    return "\n".join(L)


def cmd_explain(args) -> int:
    import torch

    try:
        api = D._api()
        explain = api.explain
    except Exception as e:  # noqa: BLE001
        print(f"explain unavailable: {D.reason(e)}", file=sys.stderr)
        return D.EXIT_UNAVAILABLE
    a_dt = getattr(torch, IN_DTYPES[args.a])
    b_dt = getattr(torch, IN_DTYPES[args.b])
    out_dt = getattr(torch, OUT_DTYPES[args.out]) if args.out else None
    M, N, K = args.M, args.N, args.K
    try:
        if args.device == "auto":
            use_cuda = not D.cuda_disabled() and torch.cuda.is_available()
            device = torch.device("cuda" if use_cuda else "cpu")
        else:
            device = torch.device(args.device)
        # input [M, K] row-major, mat2 [K, N] column-major (the layout torch._scaled_mm requires); contents are zeros
        a = torch.zeros((M, K), dtype=torch.uint8, device=device).view(a_dt)
        b = torch.zeros((N, K), dtype=torch.uint8, device=device).view(b_dt).t()
        sa = torch.tensor(args.scale_a, dtype=torch.float32, device=device)
        sb = torch.tensor(args.scale_b, dtype=torch.float32, device=device)
        sr = None if args.scale_result is None else torch.tensor(args.scale_result, dtype=torch.float32,
                                                                 device=device)
    except Exception as e:  # noqa: BLE001
        print(f"explain unavailable: device {args.device!r}: {D.reason(e)}", file=sys.stderr)
        return D.EXIT_UNAVAILABLE
    kwargs = {"splitk": args.splitk, "backend": args.backend}
    if args.profile:
        kwargs["profile"] = args.profile
    try:
        plan = explain(a, b, sa, sb, None, sr, out_dt, args.fast, **kwargs)
    except Exception as e:  # noqa: BLE001
        print(f"explain unavailable: {D.reason(e)}", file=sys.stderr)
        return D.EXIT_UNAVAILABLE
    d = _plan_dict(plan)
    shape = (f"M={M} N={N} K={K} {args.a} x {args.b} -> {args.out or 'default (input dtype)'}, "
             f"use_fast_accum={args.fast}, device={device}")
    if args.json == "-":
        _write_json(d, "-")
    else:
        print(format_plan(shape, d))
        if args.json:
            _write_json(d, args.json)
    return D.EXIT_OK if d.get("validation", "ok") == "ok" else D.EXIT_FAIL


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m mmaemu", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("info", help="versions, device, profiles, backends, tables, presets")
    p.add_argument("--device", type=int, default=0, help="CUDA device index (default 0)")
    p.add_argument("--json", metavar="PATH", help="also write the report as JSON ('-' = JSON on stdout only)")
    p.set_defaults(fn=cmd_info)
    p = sub.add_parser("selftest", help="run the kernel gates on a CUDA device")
    p.add_argument("--device", type=int, default=0, help="CUDA device index (default 0)")
    p.add_argument("--kernels", default="all", help="'all' (default) or a comma-separated list such as v1,v4")
    p.add_argument("--force", action="store_true", help="ignore cached verdicts and re-run every gate")
    p.add_argument("--json", metavar="PATH", help="write the full report as JSON")
    p.set_defaults(fn=cmd_selftest)
    p = sub.add_parser("explain", help="show the plan of a scaled_mm call without running it")
    p.add_argument("M", type=_size)
    p.add_argument("N", type=_size)
    p.add_argument("K", type=_size)
    p.add_argument("--a", choices=sorted(IN_DTYPES), default="e4m3", help="input format (default e4m3)")
    p.add_argument("--b", choices=sorted(IN_DTYPES), default="e4m3", help="mat2 format (default e4m3)")
    p.add_argument("--out", choices=sorted(OUT_DTYPES), default=None,
                   help="out_dtype (default: torch's default, the input dtype)")
    p.add_argument("--fast", action="store_true", help="use_fast_accum=True (torch's default is False)")
    p.add_argument("--scale-a", type=float, default=1.0)
    p.add_argument("--scale-b", type=float, default=1.0)
    p.add_argument("--scale-result", type=float, default=None)
    p.add_argument("--profile", default=None, help="platform profile (default: $MMAEMU_PROFILE or the default)")
    p.add_argument("--splitk", type=parse_splitk, default="auto", help="'auto' (default), '1', or 'S,serial' / 'S,pairwise'")
    p.add_argument("--backend", default="auto", help="auto | reference | cuda | cuda:v1 | cuda:v4 | cuda:v9 | cuda:tc3")
    p.add_argument("--device", default="auto", help="auto (CUDA if available) | cpu | cuda | cuda:N")
    p.add_argument("--json", metavar="PATH", help="also write the plan as JSON ('-' = JSON on stdout only)")
    p.set_defaults(fn=cmd_explain)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
