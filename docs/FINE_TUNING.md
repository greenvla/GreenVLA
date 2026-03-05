# Fine-Tuning Green-VLA

> Detailed guide for fine-tuning Green-VLA models on your own data.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Compute Dataset Statistics](#compute-dataset-statistics)
- [Configuration](#configuration)
- [Single-Node Training](#single-node-training)

---

## Prerequisites

Make sure you have completed the [environment setup](../README.md#-getting-started) from the main README.

---

## Compute Dataset Statistics

> **Note:** Statistics can vary depending on the horizon and steps between actions.

```bash
# Create assets directory
mkdir -p <your_assets_dir>

# Compute statistics for your embodiment
for name in <embodiment_config>; do
    accelerate launch lerobot/scripts/compute_dataset_stats.py \
        --data-config $name \
        --assets-dir <your_assets_dir> \
        --max-samples 400000
done
```

<details>
<summary><b>Available Embodiment Configs</b></summary>

| Config | Description |
|--------|-------------|
| `bridge` | Bridge dataset configuration |
| `fractal` | Fractal dataset configuration |
| `...` | More configs available |

Full list: [`lerobot/conf/robotics_dataset/individual/`](../lerobot/conf/robotics_dataset/individual)

</details>

---

## Configuration

Update your fine-tuning config with these parameters:

| Parameter | Description |
|-----------|-------------|
| `assets_dir` | Path to the assets directory with dataset stats |
| `robotics_dataset` | Dataset config: `individual/<name>` or `mixture/<name>` |

> **Tip:** We recommend using `mixture/<dataset_name>` for fine-tuning as it splits the dataset between processes for better parallelization.

Example configs are in [`lerobot/conf/`](../lerobot/conf/).

---

## Single-Node Training

### Bridge Dataset

```bash
TOKENIZERS_PARALLELISM=false \
HYDRA_FULL_ERROR=1 \
accelerate launch \
    --num_processes=8 \
    --config_file <path_to_accelerate_config> \
    lerobot/scripts/train_with_validation.py \
    --config-name=finetune_greenvla_bridge
```

### Fractal Dataset

```bash
TOKENIZERS_PARALLELISM=false \
HYDRA_FULL_ERROR=1 \
accelerate launch \
    --num_processes=8 \
    --config_file <path_to_accelerate_config> \
    lerobot/scripts/train_with_validation.py \
    --config-name=finetune_greenvla_fractal
```

---

## Distributed Training

```bash
TOKENIZERS_PARALLELISM=false \
HYDRA_FULL_ERROR=1 \
accelerate launch \
    --config_file <path_to_accelerate_config> \
    --main_process_ip <MASTER_IP> \
    --main_process_port 8887 \
    --num_processes <TOTAL_GPUS> \
    --num_machines <NUM_NODES> \
    --machine_rank <THIS_NODE_RANK> \
    lerobot/scripts/train_with_validation.py \
    --config-name=finetune_greenvla_bridge
```
