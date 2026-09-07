# Infrastructure deployment

The Docker image provides the inference service and its dependencies. It runs
as a non-root user, reads checkpoint files from `/ckpt`, uses `/opt/hf-cache`
for backbone assets, and exposes port 8999 with an HTTP healthcheck.

## Launcher

```bash
./run_inference.sh --checkpoint /absolute/path/to/014814
```

Requires Linux x86_64, Python 3, Docker with NVIDIA Container Toolkit and a
supported NVIDIA GPU. The launcher builds `docker/Dockerfile`, selects a GPU
with at least 7000 MiB free and a free port pair, then starts policy and wire
boundary containers. It never stops other processes.

The checkpoint must contain `pretrained_model/` and `norm_stats/` matching the
selected profile. Your checkpoint is mounted read-only. The default
profile is `014814`; `--profile 017799` selects the `dtwin_017799` contract.
For a checkpoint with a different contract, use a
[custom data config](INFERENCE.md#checkpoint-requirements) with the Docker entrypoint.

Useful options: `--gpu`, `--hf-cache`, `--cache-dir`, `--output`,
`--prepare-only` (no GPU processes), and `--dry-run` (no writes or downloads).
See `./run_inference.sh --help`. Simulator tests are not part of this launcher.

Default mode is raw VLA: no adapter, GMM or neck compensation.
`--adapter v1` and `--neck-compensation` are independent opt-ins. Required wire
conversions remain active; see [the inference contract](INFERENCE.md).

The launcher prints connection and container names. Servers remain running;
stop only the named containers when finished. `launch.json`, `policy/` and
`tracking/` are written to the output directory. Simulator settings in the
manifest are external requirements, not settings applied by this launcher.
The manifest records the selected profile, image, paths, ports and runtime options.

## Hugging Face and offline startup

In addition to policy weights, the supplied profiles require Qwen3.5-0.8B
weights, config, tokenizer, chat template and image/video processor files.
Backbone and action expert share the same HF snapshot. A custom checkpoint needs
the backbone assets referenced by its own configuration. The Docker build downloads
the Qwen3.5-0.8B snapshot used by the supplied profiles into `/opt/hf-cache` and
verifies its required files. Backbone assets are cached in a separate image layer
so inference-code changes do not trigger another download.

The image and launcher default to `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.
No HF cache mount, HF token or runtime internet access is needed for the supplied
profiles. For offline deployment, build/pull the image while networking is available,
then run it in the restricted environment.

An optional external cache can override the built-in cache:

```bash
./run_inference.sh --checkpoint /path/to/014814 --hf-cache /path/to/hf-cache
```

The supplied cache is checked before launch; an incomplete cache fails without
attempting a runtime download. Without `--hf-cache`, the launcher uses the image's
cache directly and does not prepare or download another copy on the host.

The cache contains `models--Qwen--Qwen3.5-0.8B/` with `refs/main`, `snapshots/`
and referenced `blobs/`. Preserve symlink targets when copying it. The selected
revision is `2fc06364715b967f1860aea9cf38778875588b17`. File existence and revision
are checked, not checkpoint/HF file checksums. An uncached Docker build needs
access to Hugging Face as well as system and Python package sources.

## Direct Docker launch

```bash
docker build -f docker/Dockerfile -t green-vla:local .
docker run --rm --gpus all -p 8999:8999 green-vla:local
```

The image includes `SberRoboticsCenter/GreenChallengeModel` in `/ckpt` and the
pinned Qwen snapshot in `/opt/hf-cache`. Add
`-v /absolute/path/to/014814:/ckpt:ro` before the image name to select your own
compatible weights. No credentials are included.

Do not mount an empty directory over `/opt/hf-cache`: it hides the files in the
image. For a custom checkpoint that references another backbone, provide its
complete HF cache at that path or include the required assets in your image
during the build. Missing assets fail offline instead of being downloaded.

The entrypoint starts raw policy on loopback 8998, waits for readiness, then
starts the wire boundary on public 8999. If either exits, both stop and the
container exits nonzero. SIGTERM stops both. Connect the simulator to the public
port; bypassing the boundary is not the packaged inference contract.

Settings: `CHECKPOINT_PATH=/ckpt`, `S0S1_DATA_CONFIG=scripted_014814`,
`POLICY_PORT=8999`, `POLICY_RAW_PORT=8998`. Set `POLICY_PORT` and publish that
same port when changing it, so the healthcheck follows. `TRACKING_LOG` defaults
to `/tmp/green-vla/tracking.jsonl`; mount a writable directory to retain it.
Boundary CLI flags are `--adapter {none,v1}` and `--neck-compensation`; other
model flags are forwarded to `serve_policy`. GMM defaults to off.

All Python dependencies are declared in `pyproject.toml`; Docker installs the
runtime and CUDA dependency groups in separate stages. Keep the default build
arguments for the supplied image (Python 3.11, torch 2.7.1+cu128, CUDA 12.8.1,
`TORCH_CUDA_ARCH_LIST="8.9;12.0"`, `MAX_JOBS=8`). PyTorch supplies its matching
NVIDIA libraries; causal-conv1d is compiled against that installation. The pinned
FlashAttention wheel includes Blackwell (`sm_120`) kernels and does not need
source compilation. Other GPU architectures/build configurations
require appropriate build arguments and separate validation.

## External simulator

Configure the simulator itself with `ARM_FF=0 PROPRIO_CMD=all` and connect it to
the printed public endpoint. This repository does not start, install or modify
the simulator. Its `make up` defaults are not overridden by VLA environment
variables. Verify the actual simulator process environment and control tick
before connecting the robot.
