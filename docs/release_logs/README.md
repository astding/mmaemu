# Release validation logs (T7)

One JSON file per released version and GPU, written by

    $MMAEMU_GPU_LOCK timeout 600 python tools/validate_release.py dist/mmaemu-<version>-py3-none-any.whl

and committed as part of the release (docs/release.md, section 3). File name: `<version>_<gpu>.json`, where `<gpu>`
is the device name without vendor words plus its capability (for example `rtx4060-sm89`).

Format `mmaemu-release-validation/1`:

| Key | Content |
|---|---|
| `wheel` | file name, sha256, size, version, the fatbin modules it carries |
| `host`, `source_tree_commit`, `created_utc` | where and from which commit the validation ran |
| `venv` | how the fresh environment was built (uv or venv + pip) and which torch was installed |
| `device` | GPU name, capability, SM count, torch and CUDA versions inside the fresh environment |
| `checks` | `environment`, `venv`, `install`, `info`, `selftest`, `loader`, `conformance`, `evo2`: each with `status` (PASS / FAIL / SKIP), a summary and its evidence |
| `info_before_selftest`, `selftest`, `conformance`, `evo2` | the raw reports of `python -m mmaemu info`, `python -m mmaemu selftest --force` and the two in-venv checks (`conformance.status_after`: backend status, incl. each kernel's loader, after the kernels ran) |
| `verdict` | PASS only if every check is PASS |

Only reports of release candidates and releases are committed here; dry runs go elsewhere (`--out`).
