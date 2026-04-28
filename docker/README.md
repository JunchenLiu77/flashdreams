# `docker/` — flashdreams container image

This folder contains the recipe and tooling for the flashdreams base container
image. Most users do **not** need anything here — they just reference the
prebuilt image by tag, as documented in the top-level [README](../README.md#instructions-to-run-alpadreams-inference).

Rebuild only when the Dockerfile or pinned dependencies change.

---

## Contents

| File | Purpose |
|---|---|
| `Dockerfile` | Image recipe. Based on `nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04`. |
| `build_with_docker.sh` | Build + push a multi-arch (`linux/arm64` + `linux/amd64`) image to `gitlab-master.nvidia.com:5005/sil/flashdreams`. |
| `docker_farm_setup.sh` | One-time Buildx "farm" setup so arm64 builds run natively on `dgx-spark` instead of under QEMU emulation. |

---

## Canonical image

```
gitlab-master.nvidia.com:5005/sil/flashdreams:base-v0.3-20260424-55bd566
```

Multi-arch (linux/arm64 + linux/amd64) — the container runtime picks the
right variant automatically, so there is no arch-specific tag to choose
between. To pull and run it inside an `srun` session, see the
[top-level README](../README.md#instructions-to-run-alpadreams-inference).

`build_with_docker.sh` additionally publishes dated, SHA-stamped tags of
the form `base-v0.3-YYYYMMDD-<git-short-sha>` for traceability — use those
when you need to pin to an exact build.

---

## For maintainers: building a new image

### 1. One-time — set up the build farm

Multi-arch builds are much faster when each arch runs on a native node.
`docker_farm_setup.sh` wires a local `docker-container` driver plus an SSH
endpoint to `dgx-spark` into a single Buildx builder named `farm`:

```bash
# Prereqs:
#   - `ssh dgx-spark true` succeeds from your workstation
#   - `docker buildx version` works

bash docker/docker_farm_setup.sh
```

Verify:

```bash
docker buildx ls
docker buildx inspect farm
```

You should see two nodes with `linux/amd64` and `linux/arm64` respectively.

Skip this step if you're fine with QEMU emulation for the non-native arch;
`build_with_docker.sh` will still work, just slowly.

#### About `dgx-spark`

`dgx-spark` is the short `~/.ssh/config` alias for a shared NVIDIA DGX
Spark workstation that the project uses as a native **arm64 (Grace)**
build node. `docker_farm_setup.sh` attaches it to the `farm` builder via
an SSH endpoint (`ssh://$USER@dgx-spark`), so any `--platform linux/arm64`
build is scheduled there instead of crawling through QEMU on an amd64
host.

Using it requires:

- Create an account on the machine.
- An SSH key installed on it for your `$USER`.
- A `Host dgx-spark` block in `~/.ssh/config` pointing at the right
  hostname / user / identity file so `ssh dgx-spark true` logs in
  non-interactively.
- Docker installed and runnable by your user on that host.

To request access and get the onboarding steps (SSH Host block, account
provisioning), contact **qiwu@nvidia.com**.

You don't need `dgx-spark` access to build images — dropping it just
means `build_with_docker.sh` will emulate arm64 via QEMU on your amd64
workstation, which is correct but noticeably slower.

### 2. Log in to the registry

```bash
docker login gitlab-master.nvidia.com:5005
# username: your NVIDIA handle
# password: a GitLab personal access token with read_registry + write_registry
```

### 3. Bump the tag (when the Dockerfile changes)

Open `build_with_docker.sh` and bump `TAG` (e.g. `base-v0.3` → `base-v0.4`).
The pushed tag is `$TAG-$(date +%Y%m%d)-$(git rev-parse --short HEAD)`, so
every build is uniquely addressable.

### 4. Build and push

From the repo root:

```bash
bash docker/build_with_docker.sh
```

This builds `linux/arm64` + `linux/amd64`, bundles them into a single
manifest list, and pushes with `--push` (no local image artifact produced).

### 5. Update downstream references

After a successful push, update the `--container-image=` tag in the
top-level `README.md` so users pick up the new build.

---

## Troubleshooting

**`ERROR: failed to solve: ... network ...` during build.**
Inside NVIDIA infra you usually need `--allow network.host --network host`
(already set in `build_with_docker.sh`) so apt/PyPI traffic goes through
the host's configured proxies.

**Buildx can't find an arm64 node.**
Re-run `docker buildx inspect farm --bootstrap`. If the SSH endpoint is
unhealthy, rebuild the farm:

```bash
docker buildx use default
docker buildx rm farm
bash docker/docker_farm_setup.sh
```

**`docker buildx build ... --load` complains about multi-platform.**
`--load` imports a single image into the local Docker daemon and is
incompatible with multi-arch output. Drop one of the `--platform` values
if you need a local-only build for testing.

**Tag already exists / can't overwrite.**
The tag embeds the git short SHA, so make a new commit (even an empty
`git commit --allow-empty`) to get a fresh SHA, or bump `TAG`.
