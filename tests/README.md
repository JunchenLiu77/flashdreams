# flashsim test runners

Three entrypoints for running the flashsim test suite. Pick the one that matches your dev setup.

| Script | Audience | What it does |
| --- | --- | --- |
| [`run_tests_local.sh`](./run_tests_local.sh) | dev already inside a container with deps installed | Just discovers tests and runs `pytest`. No install, no container. |
| [`run_tests_docker.sh`](./run_tests_docker.sh) | local machine with GPU + docker | `docker run` → install deps → run `pytest`. |
| [`run_tests_slurm.sh`](./run_tests_slurm.sh) | login node without GPU | `srun --container-image=…` (Pyxis/enroot) → install deps → run `pytest`. Requires `--partition`, `--account`. |

All three scripts resolve paths relative to their own location and can be invoked from anywhere.

`run_tests_docker` and `run_tests_slurm` dispatch internally (after establishing the environment) to `run_tests_local`.

## Quick examples

```bash
# Already inside a dev container (your venv has flashsim[dev] + integrations)
./tests/run_tests_local.sh
./tests/run_tests_local.sh flashsim/tests/test_attention.py

# Local machine with docker + GPU
./tests/run_tests_docker.sh
./tests/run_tests_docker.sh flashsim/tests/test_attention.py

# Slurm (Pyxis/enroot) from a login node
./tests/run_tests_slurm.sh --partition batch --account nvr_torontoai_videogen --gpus 4
./tests/run_tests_slurm.sh --partition batch --account nvr_torontoai_videogen \
    --qos interactive --gpus 4 --cpus-per-gpu 36 --time 02:00:00 \
    -- flashsim/tests/test_attention.py
```

## What gets run

When no `TEST_TARGET` is given, every script performs global discovery of `**/test_*.py`:

Pytest is invoked with `-m "not manual"` so any test marked `@pytest.mark.manual`
is skipped.

## Shared environment knobs

`run_tests_docker.sh` and `run_tests_slurm.sh` both read these env vars
(`run_tests_local.sh` ignores them — it doesn't manage caches or images):

| Variable | Default | Purpose |
| --- | --- | --- |
| `FLASHSIM_TEST_IMAGE` | `gitlab-master.nvidia.com:5005/sil/flashsim:base-v0.3` | Container image used for the run. |
| `FLASHSIM_UV_CACHE_DIR` | `${HOME}/.cache/uv` | Host dir mounted to `/root/.cache/uv`. |
| `FLASHSIM_HF_CACHE_DIR` | `${HOME}/.cache/huggingface` | Host dir mounted to `/root/.cache/huggingface`. |
| `FLASHSIM_CACHE_DIR` | `${HOME}/.cache/flashsim` | Host dir mounted to `/root/.cache/flashsim`. |
| `FLASHSIM_TRITON_CACHE_DIR` | `${HOME}/.cache/triton` | Host dir mounted to `/root/.cache/triton`; persisted across runs to avoid recompiling Triton kernels (also exported as `TRITON_CACHE_DIR`). |

Each script also has its own `--help` (slurm) or top-of-file usage block
(docker / in-container) for the full set of CLI flags.

## Container image caching (slurm only)

`run_tests_slurm.sh` does a one-time `enroot import` on the login node and
stores the resulting `.sqsh` under `${FLASHSIM_IMAGE_CACHE_DIR}` (defaults to
`${HOME}/.cache/flashsim/containers`). Subsequent runs pass that local file
straight to `--container-image=...`, so they skip the multi-minute
docker→sqsh conversion that pyxis would otherwise repeat inside every job.

| Variable | Default | Purpose |
| --- | --- | --- |
| `FLASHSIM_IMAGE_CACHE_DIR` | `${FLASHSIM_CACHE_DIR}/containers` | Where cached `.sqsh` files live. |

To force a re-import (e.g. after the upstream tag is re-pushed):

```bash
./tests/run_tests_slurm.sh --rebuild-image --partition batch --account <ACCT> --gpus 8
# or just delete the file:
rm "${HOME}/.cache/flashsim/containers/nvcr.io_nvidia_pytorch_26.02-py3.sqsh"
```

If `enroot` isn't on the login node's PATH, the script falls back to letting
pyxis do its own import inside the srun job (and prints a warning).
