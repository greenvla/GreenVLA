# GreenVLAv1.1

A vision-language-action inference server for the Green humanoid robot.
It accepts camera images, robot state and an instruction, and returns joint commands.

Requires Linux x86_64, Python 3, Git, Docker with NVIDIA Container Toolkit and an NVIDIA GPU.

## Quick start

From the repository root:

```bash
./run_inference.sh --checkpoint "<checkpoint-directory>"
```

Replace the placeholder with your local checkpoint directory containing `pretrained_model/`
and `norm_stats/`. It must match a supported launcher profile; for a non-default profile,
add `--profile "<profile>"`. Available profiles and options are listed by `./run_inference.sh --help`.

The launcher checks the checkpoint, builds the Docker image and starts the
inference endpoint. Raw VLA is the default: adapter, GMM and neck compensation
are disabled. Simulator command conversion is included.
The default embodiment is `scripted_vla_s0s1_step1_state_action_subtasks_shared`.
Your checkpoint is mounted read-only. Building the image requires internet access:
dependencies, the published checkpoint and the pinned Qwen backbone are downloaded
during the build. Model startup and inference require no internet.
The connection address is printed after startup.

The image includes Qwen3.5-0.8B weights, config and tokenizer/processor files.
The launcher uses this built-in cache by default. A complete external cache can
optionally be selected with `--hf-cache /path/to/cache`.
See [offline requirements](docs/INFRA.md#hugging-face-and-offline-startup).

## Your own checkpoint

Use a compatible GreenVLAv1.1 checkpoint with its configuration and normalization
statistics. The data config must match the embodiment, state/action layout and
sampling interval used for training. See [checkpoint requirements](docs/INFERENCE.md#checkpoint-requirements)
for the directory layout and how to select a data config.

## Simulator

Start the simulator separately with `ARM_FF=0 PROPRIO_CMD=all` and connect it to
the public endpoint printed by the launcher. The VLA launcher does not start
tests or change the simulator's `make up` defaults. Servers remain running until
you stop the printed container names.

See [Docker deployment](docs/INFRA.md) and [the inference contract](docs/INFERENCE.md)
for cache requirements, native subtasks and state/action conversion.
