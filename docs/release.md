# Releasing mmaemu

The maintainer prepares and publishes a release. No script here pushes or tags; the only upload to PyPI is
release.yml's publish job, which runs on a v* tag pushed by hand and waits for approval of the `pypi` environment.
docs.yml deploys the documentation site to GitHub Pages, and only once the repository is public. Steps marked
**[by hand]** publish something and are done by the maintainer by hand; every other step leaves a committed log.

The environment is the one of [testing.md](testing.md): a venv with torch 2.7.1+cu128, nvcc 12.8.61 for building the
release fatbins, and a GPU lock (`$GPU`, i.e. `$MMAEMU_GPU_LOCK`; the commands are for bash; in zsh, which needs
`setopt interactivecomments` for their trailing comments, `$GPU` is written `${=GPU}`). T2 and the Evo 2 check of T7
need the unpublished hardware ground truth (`MMAEMU_TRUTH_DIR`), so a release can only be validated on the
maintainer's host.

The wheel is pure Python (`py3-none-any`). It ships the kernel sources and GPU-validated fatbins built locally with
a pinned nvcc (section 2). GitHub runners have no GPU, so the release workflow never builds kernels: it checks that
the committed fatbins are current and intact and packages exactly those bytes.

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
    recorded as "profile device unavailable". T3p runs only on the RTX 4060: its tests use `ada`.
    The H100 PCIe profiles have no live tier; their hardware evidence is the committed probe and split-K sweep logs.
    `tests/test_h100_mini.py` replays 124 of the probe's live calls (in CI, T0 + T1 on the CPU, and on the GPU in
    T3), `tests/test_splitk_table.py` checks the H100 table, its envelope and the resolver against both logs, and T7
    compares one Evo 2 projection with the H100's outputs (section 3); the full replays of the probe and the sweep
    (`tools/replay_h100_log.py`) belong to no tier. T3 also runs on an RTX 5070 Ti: v1, v4, h1 and h4 are validated on
    sm_120 as well as on sm_89.

5. `docs/validation.md` cites the committed T2/T3/T4 logs (`log:` references) and `docs/performance.md` the T6
    numbers; `python tools/check_docs.py` resolves every reference.

## 2. Release-candidate freeze (on the maintainer's host)

    export PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1
    python tools/build_fatbin.py --release --out build/fatbin-release   # every module; nvcc 12.8.61 (--nvcc / $NVCC / $CUDA_HOME / PATH)
    for f in build/fatbin-release/*.fatbin; do cmp "$f" "mmaemu/backends/cuda/bin/${f##*/}"; done   # byte for byte
    #   build_report.json, cuobjdump.txt and the comparison are the build log: keep them under
    #   docs/validation_logs/<date>/, not in bin/ (check_dist accepts only the manifest and its fatbins there)
    #   settle bin/ module by module (below), then:
    python tools/build_fatbin.py --check --require-manifest   # staleness check by source hashes, no nvcc
    python tools/check_dist.py --verify-manifest      # fatbin sha256 == manifest
    tools/run_gpu_tests.sh                            # T3 on the fatbins and on NVRTC (locks per file): bitwise equal
    #   it tests the package's bin/ when that has a manifest ($MMAEMU_TEST_FATBIN_DIR overrides), and a fatbin test
    #   skipped for want of a manifest fails the run

Check the cuobjdump architecture lists against the `sass_archs` of every KernelSpec, 0 spill bytes and the
forbidden-PTX grep. Do not copy the build's `manifest.json` into `bin/`: it records the rebuilt bytes. Settle each
module in `mmaemu/backends/cuda/bin/` instead:

- Rebuilt bytes identical to the committed fatbin: the fatbin stays. If its sources changed (a comment, for
  example), its manifest entry takes the new `source_sha256`.
- Rebuilt bytes that differ: compare the SASS of every architecture and the PTX of the two builds before anything
  is replaced. A rebuilt fatbin that replaces the committed one is copied together with its manifest entry, and T3
  runs on it.

At the 0.1.0 freeze h1, h4, v9, tc3 and the probes were rebuilt byte-identical (h1's entry took the `source_sha256`
of a comment change). v1 and v4 differed only in the hash nvcc puts into the names of anonymous-namespace symbols
in the PTX, with identical SASS on every architecture, so their committed bytes, on which T3 had run, were kept
([rc-fatbin](validation_logs/2026-10-16/rc-fatbin/README.md)). Commit `mmaemu/backends/cuda/bin/` (the
`.gitignore` does not ignore it) and the build log.

Between freezes, a branch that adds or changes a kernel commits that module's release build:
`python tools/build_fatbin.py --release --modules <module> --out <dir>` writes a manifest with that module only. Its
fatbin goes to `mmaemu/backends/cuda/bin/` and its entry into `bin/manifest.json`, with the other entries
unchanged. Its `build_report.json` and `cuobjdump.txt` go under `docs/validation_logs/<date>/`. h1
([h1-fatbin](validation_logs/2026-10-15/h1-fatbin/README.md)) and h4
([h4-fatbin](validation_logs/2026-10-16/h4-fatbin/README.md)) were committed this way. The release-candidate
freeze then rebuilds such a module with all the others.

## 3. Version, changelog, distributions

1. Set the version in `mmaemu/_version.py` (`0.1.0`), `CITATION.cff` (`version:`) and the `CHANGELOG.md` heading;
    entries listed under "Unreleased" go into the version's section. Validate `CITATION.cff` with
    `cffconvert --validate -i CITATION.cff` (cffconvert 2.0.0, in a scratch venv): Zenodo reads it at the release
    tag (section 4, step 8). The changelog lists the scope (for 0.1.0 both reference platforms, the H100 PCIe and
    the RTX 4060), the maturity of every preset and profile and the limitations. If v9 or tc3 did not pass
    acceptance at the freeze (they are not release-blocking), remove them from the CHANGELOG, README and
    docs/performance.md and say they follow in 0.1.x.

2. Build both distributions from the tree and check them:

        python -m pip install build twine              # in a scratch venv
        python -m build --sdist --wheel                # NOT a bare `python -m build`: that builds the wheel from the sdist, which has no fatbins
        twine check --strict dist/*
        python tools/check_dist.py dist/* --require-fatbins --tag v0.1.0

3. **T7**, the release validation of the built wheel on the GPU:

        $GPU timeout 600 python tools/validate_release.py dist/mmaemu-0.1.0-py3-none-any.whl

    It creates a fresh venv with no nvcc on `PATH` and no `CUDA_HOME`, installs torch and the wheel, and runs
    `python -m mmaemu info`, `python -m mmaemu selftest --force`, a T3 conformance subset (each kernel against the
    CPU reference of its own preset) and one Evo 2 projection against its hardware recordings: the RTX 4060's under
    `ada` and the H100's (sha256 from the probe log) under `hopper`. It writes
    `docs/release_logs/0.1.0_<gpu>.json`; every check must be PASS.
    Commit that file. (A clean-venv *sdist* install on a GPU uses NVRTC; `tools/validate_release.py` with the sdist's
    wheel and `--allow-missing-fatbins` checks that path.)

4. Licence: `LICENSE` is the Apache-2.0 text; `NOTICE` carries the BSD-2 notice of the MATLAB Tensor Core toolbox;
    `check_dist` verifies that both and `LICENSES/` are in the wheel's `dist-info/licenses/` and that the metadata
    says `License-Expression: Apache-2.0`.

5. Mark the validated development commit with `git tag -f -a rc/v0.1.0 -m "mmaemu 0.1.0 release candidate"`
    (`-f` moves the tag when an earlier candidate holds it). Section 4 takes the snapshot tree from this tag: check
    that `git rev-parse 'rc/v0.1.0^{commit}'` is the validated commit. The release tag `v0.1.0` goes on the public
    snapshot commit (section 4), not on the development history. Nothing is pushed before section 4.

## 4. Publishing [by hand]

The public repository carries snapshot trees only, never the development history: each snapshot is one commit
whose tree is exactly a validated candidate tree. The first snapshot has no parent; every later one has the
previous snapshot as its only parent, so the repository never needs a force push.

**Staging.** Before the release, `astding/mmaemu` exists as a *private* repository that receives the pre-release
snapshots in exactly this way (a parentless first snapshot, then snapshots parented on it). Because it never held
any other commit, it can later be made public as it is; its history is then these snapshots and the release. The
maintainer decides when to switch its visibility. On the GitHub Free plan of `astding/mmaemu`, environments with
required reviewers, rulesets and Pages are available only in public repositories
([environments](https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/manage-environments),
[rulesets](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/about-rulesets)),
so they are set up after the switch to public (step 2) and before the tag push (step 5).

1. **[by hand]** Create the public snapshot commit and its tag in the development clone:

        TREE=$(git rev-parse 'rc/v0.1.0^{tree}')
        export GIT_AUTHOR_NAME=astding GIT_AUTHOR_EMAIL=50975126+astding@users.noreply.github.com
        export GIT_COMMITTER_NAME=astding GIT_COMMITTER_EMAIL=50975126+astding@users.noreply.github.com
        PREV=$(git ls-remote https://github.com/astding/mmaemu.git refs/heads/master | cut -f1)   # empty for the first
        PUB=$(git commit-tree "$TREE" ${PREV:+-p} ${PREV:+"$PREV"} -m "mmaemu 0.1.0")      # bash and zsh
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

2. **[by hand]** Make `astding/mmaemu` public. Making it public also publishes its Actions history, logs and
    unexpired artifacts. Before switching, delete the release runs that `workflow_dispatch` started on pre-release
    snapshots: their `dist` artifacts are pre-release builds of the same version (Actions → release → … next to the
    run → Delete workflow run, or `gh run delete <run-id> -R astding/mmaemu` for each id that
    `gh run list -R astding/mmaemu -w release -e workflow_dispatch` lists). After switching, set Settings → Actions →
    General → "Approval for running fork pull request workflows from contributors" to **Require approval for all
    external contributors** (ci.yml and docs.yml run on pull requests).

    Documentation site: Settings → Pages → Build and deployment → Source: **GitHub Actions**, then run the `docs`
    workflow (Actions → docs → Run workflow) and check <https://astding.github.io/mmaemu/>. Its build job
    (`mkdocs build --strict`, mkdocs.yml) runs on every push and pull request; its deploy job, the only one there
    with `pages: write` and `id-token: write`, is skipped while the repository is private and deploys only master.

3. **[by hand]** On PyPI, add a trusted publisher (pending publisher for a new project): project `mmaemu`, owner
    `astding`, repository `mmaemu`, workflow `release.yml`, environment `pypi`. A pending publisher does not reserve
    the name, and PyPI deletes one that no upload has used within 30 days of its creation
    ([warehouse #19996](https://github.com/pypi/warehouse/pull/19996)). Right before step 5, check that
    <https://pypi.org/manage/account/publishing/> still lists it and that <https://pypi.org/pypi/mmaemu/json>
    returns 404; if it has gone, add it again with the same fields.

4. **[by hand]** On GitHub:

    - Settings → Environments → New environment `pypi`: **Required reviewers**, yourself. Leave **Prevent
        self-review** unticked: the maintainer both pushes the tag and approves the deployment. Optionally, Deployment
        branches and tags → Selected branches and tags → a tag rule `v*`.

    - Protect `v*` tags with a tag ruleset (rulesets replaced GitHub's tag protection rules in 2024): Settings →
        Rulesets → New ruleset → New tag ruleset; Enforcement status **Active** (a new ruleset starts as Disabled); target
        tags matching `v*`; Restrict creations, Restrict updates, Restrict deletions and Block force pushes; bypass list
        Repository admin with **Always allow** (without a bypass the maintainer's own tag push is rejected).

    - Check both (read-only) before step 5:

            gh api repos/astding/mmaemu/environments/pypi --jq '.protection_rules'
            gh api repos/astding/mmaemu/rulesets                 # lists the ruleset id
            gh api repos/astding/mmaemu/rulesets/<id>

        The first must show a `required_reviewers` rule listing the maintainer with `"prevent_self_review": false`; the
        last `"target": "tag"`, `"enforcement": "active"`, `refs/tags/v*` in `conditions.ref_name.include`, the
        rules `creation`, `update`, `deletion` and `non_fast_forward` (Block force pushes) and a `RepositoryRole`
        bypass actor with `"bypass_mode": "always"`. Push the tag only when both do: a job that names an environment
        that does not exist creates it without protection rules, and the publish job would then upload at once,
        without the checks of step 6.

5. **[by hand]** Push the tag: `git push https://github.com/astding/mmaemu.git refs/tags/v0.1.0`. The `release`
    workflow then:

    - job **build**: `tools/build_fatbin.py --check`, `tools/check_dist.py --verify-manifest` (committed fatbins vs
        manifest sha256), tag == `mmaemu/_version.py`, `python -m build --sdist --wheel`, `twine check --strict`,
        `tools/check_dist.py --require-fatbins --tag`, and uploads the artifacts `dist` (wheel and sdist) and
        `release-meta` (`SHA256SUMS`);

    - job **publish** waits for the `pypi` environment approval. It is the only job of release.yml and ci.yml with
        `id-token: write` and uses `pypa/gh-action-pypi-publish` pinned by commit SHA (no API token).

6. **[by hand]** Before approving: download the `dist` and `release-meta` artifacts of that run, one per call so
    that both land in `<download>` (`gh run download <run-id> -R astding/mmaemu -n dist -D <download>`, then the same
    with `-n release-meta`), and validate the wheel on the GPU, with the report outside the tree (the committed
    `docs/release_logs/0.1.0_<gpu>.json` stays the report of section 3):

        $GPU timeout 600 python tools/validate_release.py <download>/mmaemu-0.1.0-py3-none-any.whl --out <dir>/t7.json

    Compare its sha256 with the local build and with `SHA256SUMS` (build locally with the hatchling version named
    in the CI wheel's `WHEEL` Generator line, otherwise the files differ in that line). Then approve the
    deployment (the run's Review deployments → `pypi` → Approve and deploy) within 30 days of the tag push, GitHub's
    limit for a job waiting on approval, and before the pending publisher of step 3 expires.

7. **[by hand]** Post-publish smoke test, in fresh venvs: `pip install mmaemu==0.1.0`, then `python -m mmaemu info`,
    once with CPU torch and once with CUDA torch. Then the maintainer's separate evo2-anywhere project pins
    `mmaemu>=0.1,<0.2`.

8. **[by hand]** Archive on Zenodo (a DOI for citing):

    - At zenodo.org, log in with GitHub (or connect GitHub under Linked accounts), open GitHub from the profile
        menu, click **Sync now** and switch on `astding/mmaemu` (Zenodo lists public repositories only). The
        repository's Settings → Webhooks then shows a zenodo.org hook for release events.

    - On GitHub, Releases → Draft a new release → choose the existing tag `v0.1.0` (never create a tag there) →
        **Publish release**. A draft that is never published does not reach Zenodo. The wheel, sdist and `SHA256SUMS`
        downloaded in step 6 may be attached.

    - Zenodo takes the title, abstract, authors, keywords, licence and message from `CITATION.cff` at the tag
        (validated in section 3) and ignores its other fields: the record's version is the tag name (`v0.1.0`) and its
        date the day Zenodo processes the release (normally the release day); both can be edited on the record.

    - If the webhook delivery failed, redeliver it (Settings → Webhooks → Recent Deliveries, within 3 days). Do not
        enable GitHub's immutable releases before the record exists: the tag name of a deleted immutable release cannot
        be reused, which rules out Zenodo's documented fix for a failed archive (delete the release and create it
        again).

    - A DOI cannot be reserved before the release. Add the concept DOI (the record's all-versions DOI) to
        `CITATION.cff` (`doi:`) and to the README in the next snapshot.

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
