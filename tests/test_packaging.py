"""Packaging tests: version, package resources, licence files, metadata, no CUDA initialisation on import, and
the policy of the CI and release workflows.

Runs from the source tree (T0) and against an installed wheel (CI dist job: MMAEMU_EXPECT_INSTALLED=1, run from a
directory outside the repository). The workflow review also runs without pytest or torch:
    python tests/test_packaging.py --review-workflows
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APACHE_2_SHA256 = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"  # apache.org LICENSE-2.0.txt
LICENCE_FILES = ("LICENSE", "NOTICE", "LICENSES/BSD-2-Clause-MATLAB-tensor-core.txt")
SHA_PIN = re.compile(r"^[^@\s]+@[0-9a-f]{40}$")
CPU_TIERS = "not gpu and not slow and not truth and not profile_device"


# =============================================================================================== workflow review
def _load_yaml(path: Path):
    import yaml  # PyYAML
    return yaml.safe_load(path.read_text())


def _triggers(doc: dict) -> dict:
    # PyYAML (YAML 1.1) reads the key `on` as the boolean True.
    on = doc.get("on", doc.get(True))
    if isinstance(on, str):
        return {on: None}
    if isinstance(on, list):
        return {k: None for k in on}
    return on or {}


def _steps_text(job: dict) -> str:
    parts = []
    for st in job.get("steps", []) or []:
        parts.append(str(st.get("name", "")))
        parts.append(str(st.get("run", "")))
        parts.append(str(st.get("uses", "")))
        parts.append(json.dumps(st.get("with", {}) or {}))
    return "\n".join(parts)


def _all_uses(doc: dict) -> list[str]:
    return [st["uses"] for job in (doc.get("jobs") or {}).values() for st in (job.get("steps") or []) if "uses" in st]


def _perm_has_id_token(perm) -> bool:
    if perm == "write-all":
        return True
    return isinstance(perm, dict) and perm.get("id-token") not in (None, "none")


def review_release_workflow(path: Path) -> list[str]:
    """Problems of .github/workflows/release.yml against docs/release.md (empty list = compliant)."""
    p: list[str] = []
    doc = _load_yaml(path)
    on = _triggers(doc)
    push = on.get("push") or {}
    if push.get("tags") != ["v*"]:
        p.append(f"trigger push.tags must be ['v*'], got {push.get('tags')}")
    if set(push) - {"tags"}:
        p.append(f"push trigger must only filter tags, got {sorted(push)}")
    if "workflow_dispatch" not in on:
        p.append("workflow_dispatch trigger missing")
    if set(on) - {"push", "workflow_dispatch"}:
        p.append(f"unexpected triggers {sorted(set(on) - {'push', 'workflow_dispatch'})}")
    if doc.get("permissions") != {"contents": "read"}:
        p.append(f"top-level permissions must be {{contents: read}}, got {doc.get('permissions')}")
    jobs = doc.get("jobs") or {}
    if set(jobs) != {"build", "publish"}:
        p.append(f"jobs must be build and publish, got {sorted(jobs)}")
    build, pub = jobs.get("build") or {}, jobs.get("publish") or {}
    # id-token: write only in publish
    for name, job in jobs.items():
        if name != "publish" and _perm_has_id_token(job.get("permissions")):
            p.append(f"job {name} must not get id-token")
    if pub.get("permissions") != {"id-token": "write"}:
        p.append(f"publish permissions must be exactly {{id-token: write}}, got {pub.get('permissions')}")
    if _perm_has_id_token(doc.get("permissions")):
        p.append("top-level permissions must not grant id-token")
    env = pub.get("environment")
    if (env.get("name") if isinstance(env, dict) else env) != "pypi":
        p.append(f"publish environment must be pypi, got {env}")
    needs = pub.get("needs")
    if needs != "build" and needs != ["build"]:
        p.append(f"publish must need build, got {needs}")
    cond = str(pub.get("if", ""))
    if "github.event_name == 'push'" not in cond or "refs/tags/v" not in cond:
        p.append(f"publish must run only for pushed v* tags (never for workflow_dispatch), got if: {cond!r}")
    pub_uses = [u for u in _all_uses({"jobs": {"publish": pub}}) if u.startswith("pypa/gh-action-pypi-publish@")]
    if len(pub_uses) != 1 or not SHA_PIN.match(pub_uses[0]):
        p.append(f"publish must use pypa/gh-action-pypi-publish pinned by a 40-hex commit SHA, got {pub_uses}")
    if any(u.startswith("pypa/gh-action-pypi-publish@") for u in _all_uses({"jobs": {"build": build}})):
        p.append("the build job must not publish")
    unpinned = [u for u in _all_uses(doc) if not SHA_PIN.match(u)]
    if unpinned:
        p.append(f"every action must be pinned by commit SHA: {unpinned}")
    text = path.read_text()
    if re.search(r"secrets\.", text):
        p.append("trusted publishing uses no secrets (found 'secrets.')")
    b = _steps_text(build)
    for needle, what in (("tools/build_fatbin.py --check", "build_fatbin --check"),
                         ("tools/check_dist.py --verify-manifest", "manifest sha256 check of the committed fatbins"),
                         ("v${ver}", "tag == version check"),
                         ("python -m build --sdist --wheel", "python -m build (sdist + wheel from the tree)"),
                         ("twine check --strict", "twine check --strict"),
                         ("--require-fatbins", "check_dist --require-fatbins"),
                         ("actions/upload-artifact@", "artifact upload")):
        if needle not in b:
            p.append(f"build job lacks the {what} ({needle!r})")
    runs = "\n".join(str(st.get("run", "")) for job in jobs.values() for st in (job.get("steps") or []))
    if "--release" in runs or re.search(r"\bnvcc\b", runs):
        p.append("release.yml must never rebuild the fatbins (no nvcc, no build_fatbin --release)")
    return p


def review_ci_workflow(path: Path) -> list[str]:
    """Problems of .github/workflows/ci.yml against the CI policy of docs/testing.md (empty list = compliant)."""
    p: list[str] = []
    doc = _load_yaml(path)
    text = path.read_text()
    jobs = doc.get("jobs") or {}
    if doc.get("permissions") != {"contents": "read"}:
        p.append(f"top-level permissions must be {{contents: read}}, got {doc.get('permissions')}")
    for name, job in jobs.items():
        runs_on = json.dumps(job.get("runs-on"))
        if "self-hosted" in runs_on or "gpu" in runs_on.lower():
            p.append(f"job {name} must run on a GitHub-hosted CPU runner, got {runs_on}")
        if _perm_has_id_token(job.get("permissions")):
            p.append(f"job {name} must not get id-token")
        if "environment" in job:
            p.append(f"job {name} must not use a deployment environment")
    if "gh-action-pypi-publish" in text or "twine upload" in text:
        p.append("ci.yml must not publish")
    unpinned = [u for u in _all_uses(doc) if not SHA_PIN.match(u)]
    if unpinned:
        p.append(f"every action must be pinned by commit SHA: {unpinned}")
    lint = _steps_text(jobs.get("lint") or {})
    for needle in ("ruff check", "actionlint", "tools/check_docs.py", "tools/build_fatbin.py --check",
                   "torch[.]equal[(]", "perf_counter", "tests/test_packaging.py --review-workflows"):
        if needle not in lint:
            p.append(f"lint job lacks {needle!r}")
    test = jobs.get("test") or {}
    matrix = ((test.get("strategy") or {}).get("matrix")) or {}
    if [str(x) for x in matrix.get("python", [])] != ["3.10", "3.11", "3.12", "3.13"]:
        p.append(f"test matrix python must be 3.10-3.13, got {matrix.get('python')}")
    if [str(x) for x in matrix.get("torch", [])] != ["2.6.0", "2.7.1", "latest"]:
        p.append(f"test matrix torch must be 2.6.0, 2.7.1, latest, got {matrix.get('torch')}")
    if "download.pytorch.org/whl/cpu" not in text:
        p.append("torch must come from the CPU index")
    if CPU_TIERS not in text:
        p.append(f"the CPU tiers must be selected with -m {CPU_TIERS!r}")
    nv = _steps_text(jobs.get("nvrtc") or {})
    if "nvidia-cuda-nvrtc-cu12==12.8.*" not in nv or "tests/test_nvrtc_compile.py" not in nv:
        p.append("nvrtc job must install nvidia-cuda-nvrtc-cu12==12.8.* and run tests/test_nvrtc_compile.py")
    dist = _steps_text(jobs.get("dist") or {})
    for needle in ("python -m build --sdist --wheel", "twine check --strict", "tools/check_dist.py", "-m venv",
                   "-m mmaemu info", "MMAEMU_EXPECT_INSTALLED"):
        if needle not in dist and needle not in json.dumps(jobs.get("dist")):
            p.append(f"dist job lacks {needle!r}")
    return p


def test_release_workflow_policy():
    _require_yaml()
    path = ROOT / ".github" / "workflows" / "release.yml"
    _skip_unless(path.is_file(), "no .github/workflows in this tree (sdist)")
    assert review_release_workflow(path) == []


def test_ci_workflow_policy():
    _require_yaml()
    path = ROOT / ".github" / "workflows" / "ci.yml"
    _skip_unless(path.is_file(), "no .github/workflows in this tree (sdist)")
    assert review_ci_workflow(path) == []


def test_release_review_detects_violations(tmp_path):
    """The review is not vacuous: each mutation of a compliant workflow is reported."""
    _require_yaml()
    src = ROOT / ".github" / "workflows" / "release.yml"
    _skip_unless(src.is_file(), "no .github/workflows in this tree (sdist)")
    good = src.read_text()
    mutations = {
        "token in build": good.replace("    runs-on: ubuntu-24.04\n    steps:",
                                       "    runs-on: ubuntu-24.04\n    permissions:\n      id-token: write\n    steps:", 1),
        "tag pinned": re.sub(r"gh-action-pypi-publish@[0-9a-f]{40}", "gh-action-pypi-publish@release/v1", good),
        "no env": good.replace("name: pypi\n", "name: testpypi\n"),
        "dispatch publishes": good.replace("if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')",
                                           "if: always()"),
        "no check": good.replace("python tools/build_fatbin.py --check", "true"),
        "no sha": good.replace("python tools/check_dist.py --verify-manifest", "true"),
        "no tag": good.replace('"v${ver}"', '"${ver}"'),
        "branch trigger": good.replace('tags: ["v*"]', 'branches: [master]'),
        "rebuild": good.replace("run: python tools/build_fatbin.py --check",
                                "run: python tools/build_fatbin.py --release && python tools/build_fatbin.py --check"),
    }
    for name, text in mutations.items():
        assert text != good, name
        f = tmp_path / f"{name.replace(' ', '_')}.yml"
        f.write_text(text)
        assert review_release_workflow(f), f"mutation {name!r} not detected"


# =============================================================================================== helpers
def _skip_unless(cond: bool, why: str) -> None:
    if not cond:
        import pytest
        pytest.skip(why)


def _require_yaml() -> None:
    if importlib.util.find_spec("yaml") is None:
        if os.environ.get("MMAEMU_REQUIRE_YAML") == "1":
            raise AssertionError("PyYAML required (MMAEMU_REQUIRE_YAML=1)")
        _skip_unless(False, "PyYAML not installed")


def _mmaemu():
    import mmaemu
    _skip_unless(getattr(mmaemu, "__file__", None) is not None, "mmaemu has no __init__.py in this tree")
    return mmaemu


def _source_tree() -> bool:
    return (ROOT / "pyproject.toml").is_file() and (ROOT / "mmaemu").is_dir()


def _installed_dist():
    """The installed mmaemu distribution if `import mmaemu` resolves to it, else None."""
    import importlib.metadata as md
    mm = _mmaemu()
    try:
        dist = md.distribution("mmaemu")
    except md.PackageNotFoundError:
        return None
    pkg_file = Path(mm.__file__).resolve()
    loc = Path(dist.locate_file("mmaemu/__init__.py")).resolve()
    return dist if loc == pkg_file else None


# =============================================================================================== package
def test_import_location_when_installed():
    """CI dist job: the tests must exercise the installed wheel, not the source tree."""
    mm = _mmaemu()
    if os.environ.get("MMAEMU_EXPECT_INSTALLED") != "1":
        _skip_unless(False, "MMAEMU_EXPECT_INSTALLED not set")
    f = Path(mm.__file__).resolve()
    assert "site-packages" in f.parts or "dist-packages" in f.parts, f
    assert ROOT not in f.parents, f"mmaemu imported from the source tree: {f}"
    assert _installed_dist() is not None


def test_version():
    mm = _mmaemu()
    v = mm.__version__
    assert isinstance(v, str) and re.fullmatch(r"\d+\.\d+\.\d+((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?", v), v
    dist = _installed_dist()
    if dist is not None:
        assert dist.version == v
    if _source_tree():
        text = (ROOT / "mmaemu" / "_version.py").read_text()
        assert re.search(rf"""__version__\s*(?::\s*str\s*)?=\s*["']{re.escape(v)}["']""", text)


def test_py_typed():
    _mmaemu()
    import importlib.resources as ir
    assert ir.files("mmaemu").joinpath("py.typed").is_file()


def _discovered_specs():
    """KernelSpecs via the frozen discovery names; the CUDA backend package must import if it exists."""
    if importlib.util.find_spec("mmaemu.backends") is None or importlib.util.find_spec(
            "mmaemu.backends.cuda.kernels") is None:
        return None, {}
    specs, probes = [], {}
    kernels = importlib.import_module("mmaemu.backends.cuda.kernels")
    specs += list(kernels.SPECS)
    if importlib.util.find_spec("mmaemu.backends.cuda.fast") is not None:
        fast = importlib.import_module("mmaemu.backends.cuda.fast.specs")
        specs += list(fast.SPECS)
        probes.update(getattr(fast, "PROBE_MODULES", {}) or {})
    spec_mod = importlib.import_module("mmaemu.backends.cuda.spec")
    probes.update(getattr(spec_mod, "PROBE_MODULES", {}) or {})
    return specs, probes


def test_kernel_sources_shipped():
    """Every discovered KernelSpec and probe module has its source in the package (NVRTC fallback, sdist)."""
    _mmaemu()
    specs, probes = _discovered_specs()
    _skip_unless(specs is not None, "mmaemu.backends.cuda.kernels not in this tree")
    import importlib.resources as ir
    csrc = ir.files("mmaemu.backends.cuda").joinpath("csrc")
    assert specs, "no KernelSpec discovered"
    for s in specs:
        assert csrc.joinpath(s.source).is_file(), f"{s.name}: csrc/{s.source} missing"
    for stem, src in probes.items():
        assert csrc.joinpath(src).is_file(), f"probe module {stem}: csrc/{src} missing"


def test_fatbins_match_manifest():
    """If the installation carries release fatbins, every manifest entry is present with its sha256."""
    _mmaemu()
    _skip_unless(importlib.util.find_spec("mmaemu.backends") is not None, "mmaemu.backends not in this tree yet")
    import importlib.resources as ir
    bindir = ir.files("mmaemu.backends.cuda").joinpath("bin")
    man = bindir.joinpath("manifest.json")
    _skip_unless(man.is_file(), "no release fatbins in this tree (committed at the release-candidate freeze)")
    m = json.loads(man.read_text())
    assert m["format"] == "mmaemu-fatbin/1" and m["modules"]
    for stem, d in m["modules"].items():
        data = bindir.joinpath(d.get("file", f"{stem}.fatbin")).read_bytes()
        assert hashlib.sha256(data).hexdigest() == d["sha256"], stem


def test_platform_tables_shipped():
    """Every profile's split-K table has its data in mmaemu/platform/data."""
    mm = _mmaemu()
    try:
        names = list(mm.list_profiles())
    except NotImplementedError as e:
        _skip_unless(False, f"profiles not implemented yet ({e})")
    import importlib.resources as ir
    data = ir.files("mmaemu.platform").joinpath("data")
    files = [p.name for p in data.iterdir()] if data.is_dir() else []
    for n in names:
        t = mm.get_profile(n).splitk_table
        if t:
            assert any(f.startswith(t + ".") for f in files), f"profile {n}: table {t} not in platform/data"


def test_no_cuda_init_and_no_tcemu_on_import():
    """`import mmaemu` (and the CLI modules) never initialise CUDA, map no CUDA library beyond those `import torch`
    maps, and never import the old tcemu package."""
    code = ("import re, sys, torch\n"
            "maps = lambda: set(re.findall(r'\\S*libcuda\\.so\\S*', open('/proc/self/maps').read())) "
            "if sys.platform.startswith('linux') else set()\n"
            "before = maps()\n"
            "import mmaemu, mmaemu.diagnostics, mmaemu.__main__\n"
            "print(torch.cuda.is_initialized(), 'tcemu' in sys.modules, sorted(maps() - before))\n")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300,
                       cwd=str(Path.home()) if os.environ.get("MMAEMU_EXPECT_INSTALLED") == "1" else None)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "False False []", r.stdout


# =============================================================================================== licence, metadata
def test_licence_files_installed():
    _mmaemu()
    dist = _installed_dist()
    _skip_unless(dist is not None, "mmaemu is not imported from an installed distribution")
    files = {str(f).replace("\\", "/") for f in (dist.files or [])}
    for lf in LICENCE_FILES:
        assert any(f.endswith("dist-info/licenses/" + lf) for f in files), lf
    meta = dist.metadata
    assert meta["License-Expression"] == "Apache-2.0"
    assert set(LICENCE_FILES) <= set(meta.get_all("License-File") or [])


def test_licence_files_source_tree():
    _skip_unless(_source_tree() and (ROOT / "LICENSE").is_file(), "not a source tree")
    lic = (ROOT / "LICENSE").read_bytes()
    assert hashlib.sha256(lic).hexdigest() == APACHE_2_SHA256, "LICENSE is not the verbatim Apache-2.0 text"
    bsd = (ROOT / "LICENSES" / "BSD-2-Clause-MATLAB-tensor-core.txt").read_text()
    assert "BSD 2-Clause License" in bsd and "Copyright (c) 2025, Faizan A. Khattak and Mantas Mikaitis" in bsd
    notice = (ROOT / "NOTICE").read_text()
    norm = lambda s: " ".join(s.split())  # noqa: E731
    assert norm(bsd) in norm(notice), "NOTICE must reproduce the toolbox's BSD-2 text"
    for needle in ("Apache License, Version 2.0", "Khattak", "Mikaitis", "arXiv:2512.07004", "mmaemu/model/chain.py",
                   "mmaemu/backends/cuda/csrc/", "tests/data/toolbox/"):
        assert needle in notice, needle
    # the toolbox licence also sits next to the test vectors (it ships in the sdist)
    assert any((ROOT / p).is_file() for p in ("tests/data/toolbox/LICENCE.MATLAB-tensor-core",
                                               "tests/data/LICENCE.MATLAB-tensor-core"))


def test_pyproject_metadata():
    _skip_unless(_source_tree(), "not a source tree")
    try:
        import tomllib
    except ModuleNotFoundError:
        _skip_unless(False, "tomllib needs Python >= 3.11")
    d = tomllib.loads((ROOT / "pyproject.toml").read_text())
    pr = d["project"]
    assert pr["name"] == "mmaemu" and pr["dynamic"] == ["version"]
    assert pr["license"] == "Apache-2.0"
    assert pr["license-files"] == ["LICENSE", "NOTICE", "LICENSES/*"]
    assert pr["requires-python"] == ">=3.10"
    deps = {re.split(r"[<>=!~ ]", x, maxsplit=1)[0] for x in pr["dependencies"]}
    assert deps == {"torch", "numpy"}
    assert any(x.startswith("torch>=") for x in pr["dependencies"]), "torch floor missing"
    assert pr["optional-dependencies"]["test"] == ["pytest"]
    assert "Development Status :: 3 - Alpha" in pr["classifiers"]
    assert not [c for c in pr["classifiers"] if c.startswith("License ::")]
    assert all(u.startswith("https://github.com/astding/mmaemu") for u in pr["urls"].values())
    assert {"Source", "Issues"} <= set(pr["urls"])
    hatch = d["tool"]["hatch"]
    assert hatch["version"]["path"] == "mmaemu/_version.py"
    sd = hatch["build"]["targets"]["sdist"]
    for ex in ("/mmaemu/backends/cuda/bin", "/recordings", "/results", "/docs/validation_logs", "/docs/release_logs"):
        assert ex in sd["exclude"], ex
    assert "/tests" in sd["include"] and "/mmaemu" in sd["include"]
    assert hatch["build"]["targets"]["wheel"]["packages"] == ["mmaemu"]
    markers = " ".join(d["tool"]["pytest"]["ini_options"]["markers"])
    for m in ("gpu:", "slow:", "truth:", "profile_device:"):
        assert m in markers, m


def test_citation():
    _skip_unless(_source_tree() and (ROOT / "CITATION.cff").is_file(), "not a source tree")
    text = (ROOT / "CITATION.cff").read_text()
    assert "cff-version: 1.2.0" in text and "arXiv:2512.07004" in text and "license: Apache-2.0" in text
    assert "repository-code: \"https://github.com/astding/mmaemu\"" in text
    m = re.search(r"^version: (\S+)", text, re.M)
    assert m, "CITATION.cff version missing"
    try:
        v = _mmaemu().__version__
    except BaseException:  # noqa: BLE001 - a skip in a stripped tree
        return
    if re.fullmatch(r"\d+\.\d+\.\d+", v):  # final releases: CITATION.cff must name the released version
        assert m.group(1).strip("\"'") == v


def test_gitignore_keeps_fatbins():
    _skip_unless((ROOT / ".gitignore").is_file(), "not a source tree")
    r = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "-q", "mmaemu/backends/cuda/bin/v1.fatbin"],
                       capture_output=True)
    _skip_unless(r.returncode in (0, 1), "not a git checkout")
    assert r.returncode == 1, ".gitignore must not ignore mmaemu/backends/cuda/bin/"


# =============================================================================================== docs
def test_docs_check_passes():
    """tools/check_docs.py (forbidden wording, evidence references, audit scope rows, README statements) is clean."""
    tool = ROOT / "tools" / "check_docs.py"
    _skip_unless(tool.is_file(), "tools/ not in this tree (sdist)")
    r = subprocess.run([sys.executable, str(tool)], capture_output=True, text=True, timeout=300,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    assert r.returncode == 0, r.stdout + r.stderr


def test_docs_check_detects_violations(tmp_path):
    """The docs check is not vacuous: injected violations are reported."""
    tool = ROOT / "tools" / "check_docs.py"
    _skip_unless(tool.is_file() and (ROOT / "docs" / "validation.md").is_file(), "tools/ or docs/ not in this tree")
    import importlib.util as iu
    spec = iu.spec_from_file_location("_check_docs_under_test", tool)
    cd = iu.module_from_spec(spec)
    spec.loader.exec_module(cd)
    fake = tmp_path / "repo"
    (fake / "docs").mkdir(parents=True)
    readme = (ROOT / "README.md").read_text().replace("experimental", "available").replace("not modelled", "planned")
    (fake / "README.md").write_text(readme + "\nIt runs on any GPU.\n")
    val = (ROOT / "docs" / "validation.md").read_text()
    val = val.replace("| Hopper preset |", "| (removed) |") + "\nWe measured 12345 outputs.\nSee `truth:c9.nope`.\n"
    (fake / "docs" / "validation.md").write_text(val)
    cd.ROOT, cd.VALIDATION, cd.README = fake, fake / "docs" / "validation.md", fake / "README.md"
    res = cd.Result()
    files = cd.doc_files()
    cd.check_forbidden(res, files)
    cd.check_refs(res, files, cd.repo_files())
    cd.check_validation_numbers(res)
    cd.check_scope_rows(res)
    cd.check_readme(res)
    text = "\n".join(res.errors)
    for needle in ("[any-gpu]", "`truth:c9.nope`", "12345", "'Hopper preset'", "`hopper` preset experimental",
                   "sm_120 is not modelled"):
        assert needle in text, (needle, text)


# =============================================================================================== script mode
def _main(argv: list[str]) -> int:
    if argv != ["--review-workflows"]:
        print(__doc__)
        return 2
    bad = 0
    for name, fn in (("release.yml", review_release_workflow), ("ci.yml", review_ci_workflow)):
        probs = fn(ROOT / ".github" / "workflows" / name)
        for pr in probs:
            print(f"FAIL {name}: {pr}")
        print(f"{'PASS' if not probs else 'FAIL'} {name}: {len(probs)} problem(s)")
        bad += len(probs)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
