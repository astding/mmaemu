# Releasing mmaemu

The maintainer prepares and publishes a release. No script here pushes or tags; the only upload to PyPI is
release.yml's publish job, which runs on a v* tag pushed by hand and waits for approval of the `pypi` environment.
docs.yml deploys the documentation site to GitHub Pages, and only once the repository is public. Steps marked
**[by hand]** publish something and are done by the maintainer by hand; every other step leaves a committed log.

The environment is the one of [testing.md](testing.md): a venv with torch 2.7.1+cu128, nvcc 12.8.61 for building the
release fatbins, and a GPU lock (`$GPU`, i.e. `$MMAEMU_GPU_LOCK`). T2 and the Evo 2 check of T7 need the unpublished
hardware ground truth (`MMAEMU_TRUTH_DIR`), so a release can only be validated on the maintainer's host.

The wheel is pure Python (`py3-none-any`). It ships the kernel sources and, from the release-candidate freeze on,
GPU-validated fatbins built locally with a pinned nvcc. GitHub runners have no GPU, so the release workflow never
builds kernels: it checks that the committed fatbins are current and intact and packages exactly those bytes.

## 1. Before the freeze

1. On master:
   - `python tools/check_docs.py` exits 0;
   - `tools/ci_local.sh` passes (the CI steps run locally; see section 5).
2. CI (`.github/workflows/ci.yml`) is green on the full matrix: T0 + T1 on Python 3.10-3.13 x torch 2.6.0,
   2.7.1 and the latest torch (CPU). **Torch floor:** set `torch>=` in `pyproject.toml` to the lowest
   torch of that matrix whose T0 + T1 pass, and drop the failing matrix entries.
3. T2, the full replay of the hardware ground truth, is committed under `docs/validation_logs/<date>/` with the
   mmaemu commit, the truth manifest sha256 and the per-recording comparison with the audit's reference replay:
   PASS.
4. T3 (GPU conformance), T3p (live `torch._scaled_mm` parity on the profile device), T4 (mutation matrix) and T6
   (performance) are run on the RTX 4060 and their logs committed. After a GPU swap: T3 on the new GPU, and T3p
   recorded as "profile device unavailable".
5. `docs/validation.md` cites the committed T2/T3/T4 logs (`log:` references) and `docs/performance.md` the T6
   numbers; `python tools/check_docs.py` resolves every reference.

## 2. Release-candidate freeze (maintainer, on the reference host)

    export PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1
    python tools/build_fatbin.py --release --out build/fatbin-release   # nvcc 12.8.61 (--nvcc / $NVCC / $CUDA_HOME / PATH)
    cp build/fatbin-release/*.fatbin build/fatbin-release/manifest.json mmaemu/backends/cuda/bin/
    #   build_report.json and cuobjdump.txt are the build log: keep them under docs/validation_logs/<date>/,
    #   not in bin/ (check_dist accepts only the manifest and its fatbins there)
    python tools/build_fatbin.py --check              # staleness check by source hashes, no nvcc
    python tools/check_dist.py --verify-manifest      # fatbin sha256 == manifest
    tools/run_gpu_tests.sh                            # T3 on the fatbins and on NVRTC (locks per file): bitwise equal
    #   it tests the package's bin/ when that has a manifest ($MMAEMU_TEST_FATBIN_DIR overrides), and a fatbin test
    #   skipped for want of a manifest fails the run

Check the cuobjdump architecture lists against the `sass_archs` of every KernelSpec, 0 spill bytes and the
forbidden-PTX grep, then commit
`mmaemu/backends/cuda/bin/` (the `.gitignore` does not ignore it). Feature branches never commit fatbins.

## 3. Version, changelog, distributions

1. Set the version in `mmaemu/_version.py` (`0.1.0`), `CITATION.cff` (`version:`) and the `CHANGELOG.md` heading.
   The changelog lists the scope, the maturity of every preset and profile and the limitations. If v9 or tc3 did
   not pass acceptance at the freeze (they are not release-blocking), remove them from the CHANGELOG,
   README and docs/performance.md and say they follow in 0.1.x.
2. Build both distributions from the tree and check them:

       python -m pip install build twine              # in a scratch venv
       python -m build --sdist --wheel                # NOT a bare `python -m build`: that builds the wheel from the sdist, which has no fatbins
       twine check --strict dist/*
       python tools/check_dist.py dist/* --require-fatbins --tag v0.1.0

3. **T7**, the release validation of the built wheel on the GPU:

       $GPU timeout 600 python tools/validate_release.py dist/mmaemu-0.1.0-py3-none-any.whl

   It creates a fresh venv with no nvcc on `PATH` and no `CUDA_HOME`, installs torch and the wheel, and runs
   `python -m mmaemu info`, `python -m mmaemu selftest --force`, a T3 conformance subset and one Evo 2 projection
   against its hardware recording. It writes `docs/release_logs/0.1.0_<gpu>.json`; every check must be PASS.
   Commit that file. (A clean-venv *sdist* install on a GPU uses NVRTC; `tools/validate_release.py` with the sdist's
   wheel and `--allow-missing-fatbins` checks that path.)
4. Licence: `LICENSE` is the Apache-2.0 text; `NOTICE` carries the BSD-2 notice of the MATLAB Tensor Core toolbox;
   `check_dist` verifies that both and `LICENSES/` are in the wheel's `dist-info/licenses/` and that the metadata
   says `License-Expression: Apache-2.0`.
5. Mark the validated development commit with `git tag -a rc/v0.1.0 -m "mmaemu 0.1.0 release candidate"`. The
   release tag `v0.1.0` goes on the public snapshot commit (section 4), not on the development history. Nothing
   is pushed before section 4.

## 4. Publishing [by hand]

The public repository carries snapshot trees only, never the development history: each snapshot is one commit
whose tree is exactly a validated candidate tree. The first snapshot has no parent; every later one has the
previous snapshot as its only parent, so the repository never needs a force push.

**Staging.** Before the release, `astding/mmaemu` exists as a *private* repository that receives the pre-release
snapshots in exactly this way (a parentless first snapshot, then snapshots parented on it). Because it never held
any other commit, it can later be made public as it is; its history is then these snapshots and the release. The
maintainer decides when to switch its visibility.

1. **[by hand]** Create the public snapshot commit and its tag in the development clone:

       TREE=$(git rev-parse 'rc/v0.1.0^{tree}')
       export GIT_AUTHOR_NAME=astding GIT_AUTHOR_EMAIL=50975126+astding@users.noreply.github.com
       export GIT_COMMITTER_NAME=astding GIT_COMMITTER_EMAIL=50975126+astding@users.noreply.github.com
       PREV=$(git ls-remote https://github.com/astding/mmaemu.git refs/heads/master | cut -f1)   # empty for the first
       PUB=$(git commit-tree "$TREE" ${PREV:+-p "$PREV"} -m "mmaemu 0.1.0")
       git tag -a v0.1.0 "$PUB" -m "mmaemu 0.1.0"
       test "$(git rev-parse "${PUB}^{tree}")" = "$TREE"                                     # same tree
       git rev-list "${PUB}" | wc -l      # the previous snapshots + 1; `git log` shows no development commit

   (`$PREV` must be fetched into the development clone first: `git fetch https://github.com/astding/mmaemu.git
   master`.) Never force-push to `astding/mmaemu` and never push development commits to it: GitHub keeps
   force-pushed commits reachable by SHA, and a repository that once held other commits must not be made public.
   Push only these explicit refs, never `--all`, `--mirror` or `--tags`:

       git push https://github.com/astding/mmaemu.git "${PUB}:refs/heads/master"

   Pre-release CI runs the same way in the private repository `astding/mmaemu-ci`, which also only ever receives
   snapshots of candidate trees (each parented on the previous one). Never push a `v*` tag to either repository
   before step 5: it starts the release workflow.
2. **[by hand]** Documentation site, once `astding/mmaemu` is public: Settings → Pages → Build and deployment →
   Source: **GitHub Actions**, then run the `docs` workflow (Actions → docs → Run workflow) and check
   <https://astding.github.io/mmaemu/>. Its build job (`mkdocs build --strict`, mkdocs.yml) runs on every push and
   pull request; its deploy job, the only one there with `pages: write` and `id-token: write`, is skipped while the
   repository is private and deploys only master.
3. **[by hand]** On PyPI, add a trusted publisher (pending publisher for a new project): project `mmaemu`, owner
   `astding`, repository `mmaemu`, workflow `release.yml`, environment `pypi`.
4. **[by hand]** On GitHub, create the environment `pypi` with yourself as required reviewer, and protect `v*`
   tags (only maintainers may create them).
5. **[by hand]** Push the tag: `git push https://github.com/astding/mmaemu.git refs/tags/v0.1.0`. The `release`
   workflow then:
   - job **build**: `tools/build_fatbin.py --check`, `tools/check_dist.py --verify-manifest` (committed fatbins vs
     manifest sha256), tag == `mmaemu/_version.py`, `python -m build --sdist --wheel`, `twine check --strict`,
     `tools/check_dist.py --require-fatbins --tag`, and uploads the `dist` artifact (and `SHA256SUMS`);
   - job **publish** waits for the `pypi` environment approval. It is the only job of release.yml and ci.yml with
     `id-token: write` and uses `pypa/gh-action-pypi-publish` pinned by commit SHA (no API token).
6. **[by hand]** Before approving: download the `dist` artifact of that run and validate it on the GPU:

       $GPU timeout 600 python tools/validate_release.py <download>/mmaemu-0.1.0-py3-none-any.whl

   Compare its sha256 with the local build and with `SHA256SUMS` (build locally with the hatchling version named
   in the CI wheel's `WHEEL` Generator line, otherwise the files differ in that line). Then approve the
   deployment.
7. **[by hand]** Post-publish smoke test, in fresh venvs: `pip install mmaemu==0.1.0`, then `python -m mmaemu info`,
   once with CPU torch and once with CUDA torch. Then evo2-anywhere pins `mmaemu>=0.1,<0.2`.
8. **[by hand]** Archive on Zenodo (a DOI for citing): at zenodo.org, log in with GitHub and switch on
   `astding/mmaemu` in the GitHub settings (public repositories only), then on GitHub create a Release from the tag
   `v0.1.0` (Releases → Draft a new release). Zenodo archives the release and takes its metadata from
   `CITATION.cff`. Add the DOI it mints to `CITATION.cff` (`doi:`) and to the README in the next snapshot.

`workflow_dispatch` runs the build job only (a dry run of the release); it never publishes.

## 5. Running the CI steps locally

`tools/ci_local.sh` runs the steps of `ci.yml` on this machine (CPU only; scratch venvs under
`$MMAEMU_CI_DIR`, default a temporary directory):

    tools/ci_local.sh            # lint + T0/T1 (current torch) + NVRTC + dist
    tools/ci_local.sh lint dist  # selected jobs
    MMAEMU_CI_TORCH="2.6.0 2.7.1" tools/ci_local.sh test   # T0/T1 against other CPU torch versions

The full Python x torch matrix runs only on GitHub. The documentation site (the build job of docs.yml):

    python -m pip install mkdocs==1.6.1 mkdocs-material==9.7.7   # in a scratch venv
    mkdocs build --strict --site-dir build/site

## 6. After a release

- 0.1.x: claims for new hardware (e.g. sm_120) are added only with recorded evidence, through the bring-up runbook
  (docs/runbooks/sm120_bringup.md); a new preset or profile starts as experimental.
- Keep `docs/release_logs/<version>_<gpu>.json` for every validated release and GPU.
