import os

os.environ["OPENPI_DATA_HOME"] = (
    "/mnt/virtual_ai0001071-01239_SR006-nfs2/apanasevich/openpi/assets"
)
os.environ["HF_HOME"] = "/mnt/virtual_ai0001071-01239_SR006-nfs2/.cache/huggingface"
os.environ["XDG_CACHE_HOME"] = "/mnt/virtual_ai0001071-01239_SR006-nfs2/.cache"

import argparse
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from hydra.utils import instantiate
from omegaconf import OmegaConf
from rich_argparse import RichHelpFormatter
from torch.utils.data import DataLoader
from tqdm import tqdm

from lerobot.common.datasets.create_dataloader import create_lerobot_dataset_by_config
from lerobot.common.policies.greenvla_policy.configuration_greenvla_policy import GreenVLAPolicyConfig
from lerobot.common.utils.normalize import NormStats, RunningStats
from lerobot.common.utils.normalize import save as save_stats


class MaskedRunningStats:
    """
    Compute running statistics per dimension, supporting masked updates.
    Only valid (non-masked) values contribute to statistics for each dimension.
    """

    def __init__(self):
        self._per_dim_count = None
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None

    def update(self, batch: np.ndarray, mask: np.ndarray) -> None:
        """
        Update running statistics with a batch of vectors, using mask to filter valid values.

        Args:
            batch: 2D array of shape [num_samples, num_dims]
            mask: 2D boolean array of same shape. True = valid, False = masked/padded.
        """
        if batch.ndim == 1:
            batch = batch.reshape(-1, 1)
            mask = mask.reshape(-1, 1)

        num_samples, num_dims = batch.shape

        if mask.shape != batch.shape:
            raise ValueError(f"Mask shape {mask.shape} doesn't match batch shape {batch.shape}")

        if self._per_dim_count is None:
            self._per_dim_count = np.zeros(num_dims, dtype=np.int64)
            self._mean = np.zeros(num_dims)
            self._mean_of_squares = np.zeros(num_dims)
            self._min = np.full(num_dims, np.inf)
            self._max = np.full(num_dims, -np.inf)

        # Update statistics per dimension
        for i in range(num_dims):
            valid_mask = mask[:, i]
            valid_values = batch[:, i][valid_mask]

            if len(valid_values) == 0:
                continue

            num_valid = len(valid_values)
            old_count = self._per_dim_count[i]
            new_count = old_count + num_valid

            batch_mean = np.mean(valid_values)
            batch_mean_of_squares = np.mean(valid_values ** 2)

            if old_count == 0:
                self._mean[i] = batch_mean
                self._mean_of_squares[i] = batch_mean_of_squares
                self._min[i] = np.min(valid_values)
                self._max[i] = np.max(valid_values)
            else:
                # Update running mean and mean of squares using Welford's method
                self._mean[i] += (batch_mean - self._mean[i]) * (num_valid / new_count)
                self._mean_of_squares[i] += (batch_mean_of_squares - self._mean_of_squares[i]) * (num_valid / new_count)
                self._min[i] = min(self._min[i], np.min(valid_values))
                self._max[i] = max(self._max[i], np.max(valid_values))

            self._per_dim_count[i] = new_count

    def get_statistics(self) -> NormStats:
        """Compute and return the statistics."""
        if self._per_dim_count is None:
            raise ValueError("No data has been processed yet.")

        num_dims = len(self._per_dim_count)
        mean = np.zeros(num_dims)
        stddev = np.ones(num_dims)  # Default to 1 for dimensions with no data
        q01 = np.zeros(num_dims)
        q99 = np.zeros(num_dims)

        # Check which dimensions have enough samples
        insufficient_dims = []
        for i in range(num_dims):
            if self._per_dim_count[i] < 2:
                insufficient_dims.append(i)
                # Use default values: mean=0, std=1
                mean[i] = 0.0
                stddev[i] = 0.0
                q01[i] = 0.0
                q99[i] = 0.0
            else:
                variance = self._mean_of_squares[i] - self._mean[i] ** 2
                mean[i] = self._mean[i]
                stddev[i] = np.sqrt(max(0, variance))
                q01[i] = self._min[i]
                q99[i] = self._max[i]

        if insufficient_dims:
            print(f"⚠️  Warning: Dimensions {insufficient_dims} have fewer than 2 valid samples. "
                  f"Using default values (mean=0, std=1) for these dimensions.")

        return NormStats(mean=mean, std=stddev, q01=q01, q99=q99)

    def merge(self, other: "MaskedRunningStats") -> None:
        """Merge another MaskedRunningStats instance into this one."""
        if other._per_dim_count is None:
            return
        if self._per_dim_count is None:
            self._per_dim_count = other._per_dim_count.copy()
            self._mean = other._mean.copy()
            self._mean_of_squares = other._mean_of_squares.copy()
            self._min = other._min.copy()
            self._max = other._max.copy()
            return

        num_dims = len(self._per_dim_count)
        for i in range(num_dims):
            n1 = self._per_dim_count[i]
            n2 = other._per_dim_count[i]

            if n2 == 0:
                continue
            if n1 == 0:
                self._per_dim_count[i] = n2
                self._mean[i] = other._mean[i]
                self._mean_of_squares[i] = other._mean_of_squares[i]
                self._min[i] = other._min[i]
                self._max[i] = other._max[i]
            else:
                total_n = n1 + n2
                self._mean[i] = (n1 * self._mean[i] + n2 * other._mean[i]) / total_n
                self._mean_of_squares[i] = (n1 * self._mean_of_squares[i] + n2 * other._mean_of_squares[i]) / total_n
                self._min[i] = min(self._min[i], other._min[i])
                self._max[i] = max(self._max[i], other._max[i])
                self._per_dim_count[i] = total_n


def compute_dataset_stats(data_config: str, max_samples: int, assets_dir: str):
    
    config_dir = (Path(__file__).parent / ".." / "conf").resolve()
    
    yaml_path = config_dir / "robotics_dataset" / "individual" / f"{data_config}.yaml"
    cfg = OmegaConf.load(yaml_path)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    data_config = instantiate(cfg)

    model_cfg = GreenVLAPolicyConfig(max_state_dim=48, max_action_dim=48, model_mode="flow_matching")
    episodes = None
    # with open(data_config.episodes_list_file, 'r') as f:
    #     episodes = json.load(f)
    
    print("Creating LeRobot dataset")
    lerobot_dataset = create_lerobot_dataset_by_config(
        data_config_factory=data_config,
        model_config=model_cfg,
        assets_dirs=assets_dir,
        normalization_mode="mean_std", #it does not matter
        skip_norm_stats=True,
        skip_model_transforms=True,
        return_norm_stats=False,
        episodes=episodes,
        )

    lerobot_pytorch_dataloader = DataLoader(lerobot_dataset, batch_size=16, shuffle=True, num_workers=16)

    lerobot_pytorch_dataloader.dataset._dataset._dataset.return_fake_images = True

    init_pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=10800))
    accelerator = Accelerator(kwargs_handlers=[init_pg_kwargs])
    if accelerator.is_local_main_process:
        print(f"✅ Instantiated config: {type(data_config).__name__}")
        print(f"   - Dataset name: {data_config.name}")
        print(f"   - State dim: {data_config.state_dim}")
        print(f"   - Control mode: {data_config.control_mode}")
        print(f"   - Action horizon: {data_config.action_horizon}")

    max_steps = None
    if max_samples is not None:
        max_steps = (
            max_samples
            // lerobot_pytorch_dataloader.batch_size
            // accelerator.num_processes
        )

    keys = ["state", "actions"]
    
    dataloader = accelerator.prepare(lerobot_pytorch_dataloader)

    if max_steps is not None:
        max_steps = min(max_steps, len(dataloader))
    else:
        max_steps = len(dataloader)

    # Check first batch to see if we have action_loss_mask
    first_batch = next(iter(dataloader))
    has_mask = "action_loss_mask" in first_batch
    
    if has_mask:
        if accelerator.is_local_main_process:
            print("📊 Detected action_loss_mask - using masked statistics (padded values excluded)")
        stats = {key: MaskedRunningStats() for key in keys}
    else:
        if accelerator.is_local_main_process:
            print("📊 No action_loss_mask detected - using standard statistics")
        stats = {key: RunningStats() for key in keys}

    # Re-create dataloader iterator since we consumed one batch
    dataloader = accelerator.prepare(lerobot_pytorch_dataloader)

    total_steps = 0
    for batch in tqdm(
        dataloader,
        total=max_steps,
        disable=not accelerator.is_local_main_process,
        desc="Computing dataset stats",
    ):
        if has_mask:
            # Get action_loss_mask: shape [batch_size, action_dim]
            action_mask = batch["action_loss_mask"].detach().cpu().numpy()
            
            for key in keys:
                values = batch[key].detach().cpu().numpy()
                
                if key == "actions":
                    # values shape: [batch, horizon, action_dim]
                    # Flatten to [batch * horizon, action_dim] and tile mask accordingly
                    batch_size, horizon, action_dim = values.shape
                    values_flat = values.reshape(-1, action_dim)
                    # Tile the mask for each horizon step: [batch, action_dim] -> [batch * horizon, action_dim]
                    mask_flat = np.tile(action_mask, (horizon, 1)).reshape(-1, action_dim)
                    stats[key].update(values_flat, mask=mask_flat)
                else:  # state
                    # values shape: [batch, state_dim]
                    stats[key].update(values, mask=action_mask)
        else:
            # Original behavior without mask
            for key in keys:
                values = batch[key].detach().cpu().numpy()
                stats[key].update(values.reshape(-1, values.shape[-1]))

        total_steps += 1
        if max_steps is not None and total_steps >= max_steps:
            break

    accelerator.wait_for_everyone()
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        gathered_stats = [None] * world_size
        dist.all_gather_object(gathered_stats, stats)
    else:
        gathered_stats = [stats]

    if accelerator.is_local_main_process:
        for i in range(1, len(gathered_stats)):
            for key in keys:
                gathered_stats[0][key].merge(gathered_stats[i][key])

        save_path = Path(assets_dir) / data_config.asset_id

        norm_stats = {
            key: stats.get_statistics() for key, stats in gathered_stats[0].items()
        }
        save_stats(save_path, norm_stats)


def main():
    parser = argparse.ArgumentParser(
        description="Computes norm stats for specified data config",
        formatter_class=RichHelpFormatter,
    )
    parser.add_argument(
        "--data-config",
        type=str,
        required=True,
        help="Data config to compute stats for",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to compute stats for",
    )
    parser.add_argument(
        "--assets-dir",
        type=str,
        required=True,
        help="Assets directory where norm stats will be saved",
    )
    args = parser.parse_args()

    compute_dataset_stats(args.data_config, args.max_samples, args.assets_dir)


if __name__ == "__main__":
    main()
