#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from pathlib import Path
from typing import Optional, Any
import subprocess
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
import os
import threading
import queue
import time
from dataclasses import dataclass
from contextlib import nullcontext

from termcolor import colored
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
import json
import numpy as np
import random
import hydra
from omegaconf import DictConfig

from lerobot.common.datasets.create_dataloader import create_lerobot_dataset_by_config
from lerobot.common.utils.inference_transforms import get_torch_input_transforms, get_torch_output_transforms
from lerobot.common.datasets.torch_transforms import compose
from lerobot.common.utils.torch_observation import torch_preprocess_dict
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.utils import has_method

# Configure matplotlib for non-interactive use
try:
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    plt.ioff()  # Turn off interactive mode
except ImportError:
    matplotlib = None
    plt = None

try:
    from clearml import Task
    CLEARML_AVAILABLE = True
except ImportError:
    logging.warning("ClearML is not available")
    CLEARML_AVAILABLE = False
    Task = None


from lerobot.common.constants import (
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    PRETRAINED_MODEL_DIR,
    TRAINING_STATE_DIR,
    TRAINING_STEP,
)
import numpy as np
from lerobot.common.datasets.utils import load_json, write_json
from lerobot.common.optim.optimizers import load_optimizer_state, save_optimizer_state
from lerobot.common.optim.schedulers import load_scheduler_state, save_scheduler_state
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.utils.random_utils import load_rng_state, save_rng_state
from lerobot.configs.train import TrainPipelineConfig




# Global upload queue instance


def _create_checkpoint_preview_dict_from_path(checkpoint_dir: Path) -> dict:
    """Create a preview dictionary for ClearML UI like in upload_checkpoint_as_artifact"""
    preview_dict = {}
    
    # Try to load config files from the checkpoint directory if they exist
    pretrained_dir = checkpoint_dir / "pretrained_model"
    
    # Load config.json if it exists
    config_json_path = pretrained_dir / "config.json"
    if config_json_path.exists():
        try:
            with open(config_json_path, 'r') as f:
                config_data = json.load(f)
            preview_dict["config.json"] = config_data
        except Exception as e:
            logging.warning(f"Failed to load config.json for preview: {e}")
    
    # Load train_config.json if it exists  
    train_config_json_path = pretrained_dir / "train_config.json"
    if train_config_json_path.exists():
        try:
            with open(train_config_json_path, 'r') as f:
                train_config_data = json.load(f)
            preview_dict["train_config.json"] = train_config_data
        except Exception as e:
            logging.warning(f"Failed to load train_config.json for preview: {e}")
    
    return preview_dict


def load_safetensors_weights(pretrained_path: str) -> dict[str, torch.Tensor]:
    """Load safetensors weights from a local file, local directory, or HuggingFace Hub model ID.

    Resolution order:
        1. Local ``.safetensors`` file  – loaded directly.
        2. Local directory              – all ``*.safetensors`` inside are loaded and merged.
        3. HuggingFace Hub repo ID      – only safetensors (+ index) files are downloaded,
           then loaded and merged as in (2).

    Returns:
        Merged state dict ready for ``model.load_state_dict(..., strict=False)``.
    """
    from safetensors.torch import load_file

    path = Path(pretrained_path)

    if path.suffix == ".safetensors" and path.exists():
        logging.info(f"Loading weights from local file: {path}")
        return load_file(str(path))

    if path.exists() and path.is_dir():
        logging.info(f"Loading weights from local directory: {path}")
        return _merge_safetensors_dir(path)

    # Treat as a HuggingFace Hub repo ID (e.g. "org/model-name").
    from huggingface_hub import snapshot_download

    logging.info(f"Downloading safetensors from HuggingFace Hub: {pretrained_path}")
    local_dir = snapshot_download(
        pretrained_path,
        allow_patterns=["*.safetensors", "*.safetensors.index.json"],
    )
    return _merge_safetensors_dir(Path(local_dir))


def _merge_safetensors_dir(directory: Path) -> dict[str, torch.Tensor]:
    """Load and merge all ``*.safetensors`` shards in *directory* (searched recursively)."""
    from safetensors.torch import load_file

    files = sorted(directory.rglob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors files found in {directory}")

    state_dict: dict[str, torch.Tensor] = {}
    for f in files:
        state_dict.update(load_file(str(f)))

    logging.info(f"Loaded {len(state_dict)} tensors from {len(files)} shard(s)")
    return state_dict


def log_output_dir(out_dir):
    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {out_dir}")


def get_step_identifier(step: int, total_steps: int) -> str:
    num_digits = max(6, len(str(total_steps)))
    return f"{step:0{num_digits}d}"


def get_step_checkpoint_dir(output_dir: Path, total_steps: int, step: int) -> Path:
    """Returns the checkpoint sub-directory corresponding to the step number."""
    step_identifier = get_step_identifier(step, total_steps)
    return output_dir / CHECKPOINTS_DIR / step_identifier


def save_training_step(step: int, save_dir: Path) -> None:
    write_json({"step": step}, save_dir / TRAINING_STEP)


def load_training_step(save_dir: Path) -> int:
    training_step = load_json(save_dir / TRAINING_STEP)
    return training_step["step"]


def update_last_checkpoint(checkpoint_dir: Path) -> Path:
    last_checkpoint_dir = checkpoint_dir.parent / LAST_CHECKPOINT_LINK
    if last_checkpoint_dir.is_symlink():
        last_checkpoint_dir.unlink()
    relative_target = checkpoint_dir.relative_to(checkpoint_dir.parent)
    last_checkpoint_dir.symlink_to(relative_target)


def save_norm_stats(checkpoint_dir: Path, norm_stats: dict) -> None:
    save_dir = checkpoint_dir / "norm_stats"
    print(f"Saving norm stats to {save_dir}")
    print(f"Norm stats keys:\n{norm_stats.keys()}")
    for embodiment_key, norm_stats in norm_stats.items():
        save_stats_dir = save_dir / embodiment_key
        save_stats_dir.mkdir(parents=True, exist_ok=True)
        with open(save_stats_dir / "norm_stats.json", "w") as f:
            json.dump({"norm_stats": norm_stats}, f, indent=2)
            
            
def save_embodiments_configs(checkpoint_dir: Path, embodiments_configs: dict) -> None:
    import yaml
    save_dir = checkpoint_dir / "embodiments_configs"
    save_dir.mkdir(parents=True, exist_ok=True)
    for config_name, config in embodiments_configs.items():
    #saving configs as yaml
        with open(save_dir / f"{config_name}.yaml", "w") as f:
            yaml.dump(config, f, indent=2)




def save_checkpoint(
    checkpoint_dir: Path,
    step: int,
    cfg: TrainPipelineConfig,
    policy: PreTrainedPolicy,
    optimizer: Optimizer,
    scheduler: LRScheduler | None = None,
    norm_stats: dict | None = None,
    embodiments_configs: dict | None = None,
    clearml_task: Optional["Task"] = None,
    train_loss: Optional[float] = None,
    val_loss: Optional[float] = None,
    dataset_config_name: Optional[str] = None,
    model_total_params: Optional[int] = None,
    model_learnable_params: Optional[int] = None,
    upload_method: str = "none",
) -> None:
    """This function creates the following directory structure:

    005000/  # training step at checkpoint
    ├── pretrained_model/
    │   ├── config.json          # policy config
    │   ├── model.safetensors    # policy weights
    │   └── train_config.json    # train config
    ├── training_state/
    │   ├── optimizer_param_groups.json  # optimizer param groups
    │   ├── optimizer_state.safetensors  # optimizer state
    │   ├── rng_state.safetensors        # RNG states
    │   ├── scheduler_state.json         # scheduler state
    │   └── training_step.json           # training step
    ├── norm_stats/                       # per-embodiment normalization stats
    │   ├── embodiment_name_1/
    │   │   └── norm_stats.json
    │   ├── embodiment_name_2/
    │   │   └── norm_stats.json
    │   └── ...
    └── embodiments_configs/             # per-embodiment config files
        ├── embodiment1.yaml
        ├── embodiment2.yaml
        └── ...

    Notes:
    - The `norm_stats` directory (created when `norm_stats` is provided) stores
      normalization statistics per embodiment. Each subfolder corresponds to an
      embodiment key (e.g., `embodiment_name_i`) and contains a `norm_stats.json`.
    - The `embodiments_configs` directory (if present) holds YAML config files for
      each embodiment (e.g., `embodiment1.yaml`, `embodiment2.yaml`, ...).

    Args:
        cfg (TrainPipelineConfig): The training config used for this run.
        step (int): The training step at that checkpoint.
        policy (PreTrainedPolicy): The policy to save.
        optimizer (Optimizer | None, optional): The optimizer to save the state from. Defaults to None.
        scheduler (LRScheduler | None, optional): The scheduler to save the state from. Defaults to None.
        norm_stats (dict | None, optional): Normalization statistics to save. Defaults to None.
        clearml_task (Task | None, optional): ClearML task instance. If None, uses current task. Defaults to None.
        train_loss (float | None, optional): Training loss at this checkpoint. Defaults to None.
        val_loss (float | None, optional): Validation loss at this checkpoint. Defaults to None.
        dataset_config_name (str | None, optional): Name of the dataset config used for training. Defaults to None.
        model_total_params (int | None, optional): Total number of model parameters. Defaults to None.
        model_learnable_params (int | None, optional): Number of learnable model parameters. Defaults to None.
        upload_method (str, optional): Upload method - "none", "rclone", or "artifact". Defaults to "rclone".
    """
    pretrained_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
    policy.save_pretrained(pretrained_dir)
    cfg.save_pretrained(pretrained_dir)
    save_training_state(checkpoint_dir, step, None, scheduler)
    if norm_stats is not None:
        save_norm_stats(checkpoint_dir, norm_stats)
    if embodiments_configs is not None:
        save_embodiments_configs(checkpoint_dir, embodiments_configs)
        
    # Upload checkpoint using specified method
    if upload_method == "none":
        logging.info("Upload method set to 'none', skipping checkpoint upload")
    else:
        logging.warning(f"Unknown upload_method: {upload_method}. Skipping upload.")


def save_training_state(
    checkpoint_dir: Path,
    train_step: int,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
) -> None:
    """
    Saves the training step, optimizer state, scheduler state, and rng state.

    Args:
        save_dir (Path): The directory to save artifacts to.
        train_step (int): Current training step.
        optimizer (Optimizer | None, optional): The optimizer from which to save the state_dict.
            Defaults to None.
        scheduler (LRScheduler | None, optional): The scheduler from which to save the state_dict.
            Defaults to None.
    """
    save_dir = checkpoint_dir / TRAINING_STATE_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    save_training_step(train_step, save_dir)
    save_rng_state(save_dir)
    if optimizer is not None:
        save_optimizer_state(optimizer, save_dir)
    if scheduler is not None:
        save_scheduler_state(scheduler, save_dir)


def load_training_state(
    checkpoint_dir: Path, optimizer: Optimizer, scheduler: LRScheduler | None
) -> tuple[int, Optimizer, LRScheduler | None]:
    """
    Loads the training step, optimizer state, scheduler state, and rng state.
    This is used to resume a training run.

    Args:
        checkpoint_dir (Path): The checkpoint directory. Should contain a 'training_state' dir.
        optimizer (Optimizer): The optimizer to load the state_dict to.
        scheduler (LRScheduler | None): The scheduler to load the state_dict to (can be None).

    Raises:
        NotADirectoryError: If 'checkpoint_dir' doesn't contain a 'training_state' dir

    Returns:
        tuple[int, Optimizer, LRScheduler | None]: training step, optimizer and scheduler with their
            state_dict loaded.
    """
    training_state_dir = checkpoint_dir / TRAINING_STATE_DIR
    if not training_state_dir.is_dir():
        raise NotADirectoryError(training_state_dir)

    load_rng_state(training_state_dir)
    step = load_training_step(training_state_dir)
    optimizer = load_optimizer_state(optimizer, training_state_dir)
    if scheduler is not None:
        scheduler = load_scheduler_state(scheduler, training_state_dir)

    return step, optimizer, scheduler

def visualize_joint_actions(
    predictions: dict[str, np.ndarray],  # dataset_name -> predictions
    ground_truth: dict[str, np.ndarray], # dataset_name -> ground_truth
    iteration: int,
    logger,
    viz_config: dict,
    should_visualize: bool = True,
    sample_info: str = None
) -> dict:
    """Compute metrics and optionally log aggregated metrics visualization.
    
    Args:
        predictions (dict[str, np.ndarray]): Dict mapping dataset names to prediction arrays
        ground_truth (dict[str, np.ndarray]): Dict mapping dataset names to ground truth arrays
        iteration (int): Current step/iteration for logging
        logger: ClearML logger instance
        viz_config (dict): Visualization configuration
        should_visualize (bool): Whether to create plots this validation cycle
    
    Returns:
        dict: Dictionary containing aggregated metrics per dataset
    """
    # Always compute metrics
    all_metrics = compute_metrics_only(predictions, ground_truth)
    
    # Early return if no visualization needed or matplotlib not available
    if not should_visualize or plt is None or logger is None:
        return all_metrics
        
    try:
        # Get horizon information for logging
        horizon_info = get_dataset_horizon_info(predictions)
        logging.info(f"Dataset horizons: {horizon_info}")
        
        # Log metrics to ClearML
        for ds_name, metrics in all_metrics.items():
            for metric_name, value in metrics.items():
                logger.report_scalar(
                    title=f"Validation Metrics/{ds_name}",
                    series=metric_name,
                    value=value,
                    iteration=iteration
                )
            
            # Create debug visualization for first few samples
            if ds_name in predictions and ds_name in ground_truth:
                pred = predictions[ds_name]
                gt = ground_truth[ds_name]
                
                # Get number of plots to generate
                plots_per_validation = viz_config.get('plots_per_validation', 5)
                if plots_per_validation <= 0:
                    continue
                    
                # Handle batched data
                if len(pred.shape) > 2:  # [batch, horizon, joints] or [batch, joints]
                    num_available = pred.shape[0]
                    num_plots = min(plots_per_validation, num_available)
                    logging.info(f"Creating {num_plots} debug plots from {num_available} available samples for {ds_name}")
                    
                    for plot_idx in range(num_plots):
                        # Take single batch
                        batch_pred = pred[plot_idx:plot_idx+1]  # Keep batch dim of 1
                        batch_gt = gt[plot_idx:plot_idx+1]
                        
                        n_joints = batch_pred.shape[-1]
                        if n_joints > 0:
                            # Create figure with subplots for each joint
                            n_cols = min(4, n_joints)  # Max 4 columns
                            n_rows = (n_joints + n_cols - 1) // n_cols
                            fig = plt.figure(figsize=(4 * n_cols, 3 * n_rows))
                            
                            for joint_idx in range(n_joints):
                                # Extract joint data
                                pred_joint = batch_pred[..., joint_idx].flatten()
                                gt_joint = batch_gt[..., joint_idx].flatten()
                                
                                # Create subplot
                                ax = plt.subplot(n_rows, n_cols, joint_idx + 1)
                                ax.plot(pred_joint, label='Pred', alpha=0.7)
                                ax.plot(gt_joint, label='GT', alpha=0.7)
                                ax.set_title(f'Joint {joint_idx}')
                                if joint_idx == 0:
                                    ax.legend()
                                ax.grid(True)
                            
                            plt.tight_layout()
                            
                            # Log to ClearML
                            title = f"Debug Samples/{ds_name}"
                            series = f"sample{plot_idx}" if sample_info is None else f"{sample_info}_sample{plot_idx}"
                            logger.report_matplotlib_figure(
                                title=title,
                                series=series,
                                figure=fig,
                                iteration=iteration,
                                report_image=True,
                                report_interactive=False
                            )
                            plt.close(fig)
                else:
                    # Non-batched data, just plot once
                    n_joints = pred.shape[-1]
                    if n_joints > 0:
                        # Create figure with subplots for each joint
                        n_cols = min(4, n_joints)  # Max 4 columns
                        n_rows = (n_joints + n_cols - 1) // n_cols
                        fig = plt.figure(figsize=(4 * n_cols, 3 * n_rows))
                        
                        for joint_idx in range(n_joints):
                            # Extract joint data
                            pred_joint = pred[..., joint_idx].flatten()
                            gt_joint = gt[..., joint_idx].flatten()
                            
                            # Create subplot
                            ax = plt.subplot(n_rows, n_cols, joint_idx + 1)
                            ax.plot(pred_joint, label='Pred', alpha=0.7)
                            ax.plot(gt_joint, label='GT', alpha=0.7)
                            ax.set_title(f'Joint {joint_idx}')
                            if joint_idx == 0:
                                ax.legend()
                            ax.grid(True)
                        
                        plt.tight_layout()
                        
                        # Log to ClearML
                        title = f"Debug Samples/{ds_name}"
                        series = "sample0" if sample_info is None else sample_info
                        logger.report_matplotlib_figure(
                            title=title,
                            series=series,
                            figure=fig,
                            iteration=iteration,
                            report_image=True,
                            report_interactive=False
                        )
                        plt.close(fig)
                    
    except Exception as e:
        logging.error(f"Error in visualization: {str(e)}")
    
    return all_metrics

def compute_metrics_only(predictions: dict[str, np.ndarray], ground_truth: dict[str, np.ndarray]) -> dict:
    """Compute aggregated metrics without creating visualizations."""
    all_metrics = {}
    for ds_name, pred in predictions.items():
        if ds_name not in ground_truth:
            logging.warning(f"No ground truth found for dataset {ds_name}, skipping metrics")
            continue
            
        gt = ground_truth[ds_name]
        metrics = {}
        
        # Handle different shapes gracefully
        if pred.shape != gt.shape:
            logging.warning(f"Shape mismatch for {ds_name}: pred {pred.shape} != gt {gt.shape}")
            # Try to align shapes by taking minimum dimensions
            min_shape = tuple(min(p, g) for p, g in zip(pred.shape, gt.shape))
            pred = pred[tuple(slice(0, s) for s in min_shape)]
            gt = gt[tuple(slice(0, s) for s in min_shape)]
            logging.info(f"Aligned shapes to {pred.shape} for {ds_name}")
        
        # Compute aggregated metrics across all joints
        mse = float(np.mean((pred - gt) ** 2))
        mae = float(np.mean(np.abs(pred - gt)))
        metrics["val_action_mse"] = mse
        metrics["val_action_mae"] = mae
        
        # Compute per-joint average metrics
        n_joints = pred.shape[-1]
        joint_mse = np.mean([(np.mean((pred[..., i].flatten() - gt[..., i].flatten()) ** 2)) for i in range(n_joints)])
        joint_mae = np.mean([np.mean(np.abs(pred[..., i].flatten() - gt[..., i].flatten())) for i in range(n_joints)])
        metrics["val_action_joint_mse_avg"] = float(joint_mse)
        metrics["val_action_joint_mae_avg"] = float(joint_mae)
        
        all_metrics[ds_name] = metrics
    return all_metrics

def get_dataset_horizon_info(predictions: dict[str, np.ndarray]) -> dict[str, int]:
    """Get horizon information for each dataset.
    
    Args:
        predictions: Dict mapping dataset names to prediction arrays
        
    Returns:
        Dict mapping dataset names to their horizon lengths
    """
    horizon_info = {}
    for ds_name, pred in predictions.items():
        if len(pred.shape) > 2:
            horizon = pred.shape[-2]  # Second to last dimension is typically horizon
        else:
            horizon = 1  # Single timestep
        horizon_info[ds_name] = horizon
    return horizon_info

def get_distinct_colors(n_colors: int) -> list:
    """Generate visually distinct colors for plotting.
    Uses a combination of qualitative colormaps for better distinction than a single colormap.
    """
    if n_colors <= 10:
        # Use tab10 for up to 10 colors - most distinct
        return plt.cm.tab10(np.linspace(0, 1, n_colors))
    elif n_colors <= 20:
        # Combine tab10 and Set3 for up to 20 colors
        colors1 = plt.cm.tab10(np.linspace(0, 1, 10))
        colors2 = plt.cm.Set3(np.linspace(0, 1, n_colors - 10))
        return np.vstack([colors1, colors2])
    else:
        # For more colors, use hsv colormap which can generate arbitrary number of distinct colors
        return plt.cm.hsv(np.linspace(0, 1, n_colors))

def plot_single_dataset(
    pred_actions: np.ndarray,
    gt_actions: np.ndarray,
    ds_name: str,
    iteration: int,
    logger,
    subplot_height: int,
    subplot_width: int,
    save_dir: Path = None,
    checkpoint_name: str = None,
    sample_info: str = None
) -> dict:
    """Create subplot figure for a single dataset."""
    metrics = {}
    try:
        # Handle shape mismatches gracefully
        if pred_actions.shape != gt_actions.shape:
            logging.warning(f"Shape mismatch for {ds_name}: pred_actions {pred_actions.shape} != gt_actions {gt_actions.shape}")
            # Try to align shapes by taking minimum dimensions
            min_shape = tuple(min(p, g) for p, g in zip(pred_actions.shape, gt_actions.shape))
            pred_actions = pred_actions[tuple(slice(0, s) for s in min_shape)]
            gt_actions = gt_actions[tuple(slice(0, s) for s in min_shape)]
            logging.info(f"Aligned shapes to {pred_actions.shape} for {ds_name}")
        
        n_joints = pred_actions.shape[-1]
        if n_joints == 0:
            raise ValueError(f"No joints found in data: shape {pred_actions.shape}")
            
        # Log dataset-specific horizon information
        horizon = pred_actions.shape[-2] if len(pred_actions.shape) > 2 else 1
        logging.info(f"Plotting {n_joints} joints for dataset {ds_name} with processed horizon {horizon}")
        logging.info(f"Note: This is the horizon after output pipeline processing. Raw model horizon may differ.")
    except Exception as e:
        logging.error(f"Error initializing plot for dataset {ds_name}: {str(e)}")
        return metrics
    
    # Create subplot grid
    n_cols = 2
    n_rows = (n_joints + 1) // 2
    fig = plt.figure(figsize=(subplot_width, subplot_height * n_rows))
    
    for joint_idx in range(min(n_joints, pred_actions.shape[-1])):
        try:
            # Extract data
            pred_joint = pred_actions[..., joint_idx].flatten()
            gt_joint = gt_actions[..., joint_idx].flatten()
                
            logging.debug(f"Processing joint {joint_idx}: pred shape {pred_joint.shape}, gt shape {gt_joint.shape}")
        except Exception as e:
            logging.error(f"Error processing joint {joint_idx}: {str(e)}")
            continue
        
        # Compute metrics
        joint_mse = float(np.mean((pred_joint - gt_joint) ** 2))
        joint_mae = float(np.mean(np.abs(pred_joint - gt_joint)))
        metrics[f"val_action_mse_joint_{joint_idx}"] = joint_mse
        metrics[f"val_action_mae_joint_{joint_idx}"] = joint_mae
        
        # Create subplot
        ax = plt.subplot(n_rows, n_cols, joint_idx + 1)
        ax.plot(pred_joint, label='Predicted', alpha=0.7)
        ax.plot(gt_joint, label='Ground Truth', alpha=0.7)
        
        # Add error bands
        error = np.abs(pred_joint - gt_joint)
        error_std = np.std(error)
        ax.fill_between(
            range(len(pred_joint)),
            pred_joint - error_std,
            pred_joint + error_std,
            alpha=0.2,
            label=f'Error Std (±{error_std:.3f})'
        )
        
        ax.set_title(f'Joint {joint_idx} (MSE: {joint_mse:.3f})')
        ax.grid(True)
        # if joint_idx == 0:
        ax.legend()
    
    plt.tight_layout()
    
    # Save plot to file and ClearML
    try:
        # Save to file if directory is provided
        if save_dir is not None and checkpoint_name is not None:
            # Create filename with checkpoint name, dataset, sample, and step info
            sample_suffix = f"_{sample_info}" if sample_info else ""
            plot_filename = f"{checkpoint_name}_{ds_name}{sample_suffix}_step{iteration}.png"
            plot_path = save_dir / plot_filename
            
            # Ensure directory exists
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save figure
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            logging.info(f"Saved plot to {plot_path}")

        # Save to ClearML if logger is provided
        if logger is not None:
            # Include sample information in the title and series for better identification
            horizon = pred_actions.shape[-2] if len(pred_actions.shape) > 2 else 1
            title = f"{ds_name} - actions"
            series = sample_info if sample_info else ds_name
            
            logger.report_matplotlib_figure(
                title=title,
                series=series,
                figure=fig,
                iteration=iteration,
                report_image=True,
                report_interactive=False
            )
    except Exception as e:
        logging.error(f"Error saving plot for dataset {ds_name}: {str(e)}")
    finally:
        plt.close(fig)
    
    return metrics


def log_batch_dataset_proportions(batch, iteration, logger, grad_accum_steps, batch_size, num_gpus):
    """Log the proportions of samples from each dataset in the current batch.
    
    Args:
        batch: The batch containing dataset_ids
        iteration: Current training step
        logger: ClearML logger instance
        grad_accum_steps: Number of gradient accumulation steps
        batch_size: Per-device batch size
        num_gpus: Number of GPUs being used
    """
    try:
        if "dataset_id" not in batch:
            logging.warning("Batch does not contain dataset_id, skipping proportion logging")
            return
            
        # Get dataset IDs from batch and convert to numpy array
        dataset_ids = batch["dataset_id"]
        if torch.is_tensor(dataset_ids):
            dataset_ids = dataset_ids.cpu().numpy()
        elif isinstance(dataset_ids, list):
            dataset_ids = np.array(dataset_ids)
        
        # Count occurrences of each dataset ID
        unique_ids, counts = np.unique(dataset_ids, return_counts=True)
        total_samples = len(dataset_ids)
        
        # Calculate proportions
        proportions = counts / total_samples
        
        # Add global batch size info
        global_batch_size = batch_size * grad_accum_steps * num_gpus
        # Log proportions as scalar metrics to ClearML
        for dataset_id, proportion in zip(unique_ids, proportions):
            logger.report_scalar(
                title="Dataset Proportions",
                series=f"Dataset {dataset_id}",
                value=float(proportion),
                iteration=iteration
            ) 
    except Exception as e:
        logging.error(f"Error in log_batch_dataset_proportions: {str(e)}")

# _seed_worker function moved to seed_utils.py to avoid circular imports


def instantiate_data_config(cfg: DictConfig, add_kwargs: dict = None):
    """Instantiate robotics dataset config.

    - For mixture configs: mutate each nested data_config only for keys that already exist
    (e.g., map_to_unified_space, map_to_humanoid), then instantiate with _recursive_=False.
    - For individual configs: set keys on the config if they already exist, then instantiate with _recursive_=True.
    """
    try:
        is_mixture = cfg._target_.split(".")[-1] == "MixtureDataConfigFactory"
    except Exception:
        logging.warning(f"Could not determine if config is a mixture config")
        is_mixture = False

    if is_mixture and hasattr(cfg, "datasets_with_weights") and cfg.datasets_with_weights is not None:
        assert cfg.data_configs is None, "both datasets_with_weights and data_configs are set"
        assert cfg.weights is None, "both datasets_with_weights and weights are set"
        datasets_list = [ds_cfg.path for ds_cfg in cfg.datasets_with_weights]
        weights_list = [ds_cfg.weight for ds_cfg in cfg.datasets_with_weights]
        cfg.data_configs = datasets_list
        cfg.weights = weights_list
        del cfg.datasets_with_weights

    if add_kwargs:
        if is_mixture and hasattr(cfg, "data_configs") and cfg.data_configs is not None:
            # Update only existing keys in each nested dataset cfg to avoid unknown-arg errors
            for idx in range(len(cfg.data_configs)):
                dc = cfg.data_configs[idx]
                try:
                    # DictConfig supports 'in' and item assignment
                    for k, v in add_kwargs.items():
                        if isinstance(dc, DictConfig):
                            if k in dc:
                                dc[k] = v
                        elif isinstance(dc, dict):
                            if k in dc:
                                dc[k] = v
                except Exception as e:
                    logging.warning(f"Error updating dataset config: {str(e)}")
                    # Best-effort; skip problematic entries
                    pass
        else:
            # Individual dataset config: only set keys that exist in cfg
            for k, v in add_kwargs.items():
                try:
                    if k in cfg:
                        cfg[k] = v
                except Exception as e:
                    logging.warning(f"Error updating dataset config: {str(e)}")
                    pass

    if is_mixture:
        return hydra.utils.instantiate(cfg, _recursive_=False)
    else:
        return hydra.utils.instantiate(cfg, _recursive_=True)


def create_train_val_datasets_distributed(
    data_config_factory: Any,
    model_config: Any,
    assets_dirs: str,
    accelerator: Accelerator,  # Add accelerator parameter
    val_split: float = 0.1,
    seed: int = 42,
    map_to_unified_space: bool = False,
    use_validation_list: bool = False,
    recompute_norm_stats: bool = False,
    **dataset_kwargs
):
    """
    Create train and validation datasets with distributed-aware episode splitting.
    Only main process loads metadata, then broadcasts splits to all processes.
    
    Logic flow:
    1. First, check if validation episodes are provided in data config via 'validation_episodes' key
    2. If validation episodes files are found, load them and create val_episodes_dict from them
    3. If no validation episodes files are found, do random shuffling and splitting based on val_split ratio
    4. Train episodes are always created as the complement of validation episodes
    
    The validation_episodes JSON file can contain:
    - Direct list: [0, 1, 2, ...] (applies to all datasets)
    - Multi-dataset dict: {dataset_name: [0, 1, 2, ...]}
    """
    class DummyFactory:
        def __init__(self, cfg_item):
            self.cfg_item = cfg_item
        
        def create(self, *args, **kwargs):
            return self.cfg_item
    
    # Get data config to determine dataset type (needed for output pipeline creation)
    data_cfg = data_config_factory.create(assets_dirs, model_config)
    
    # Check if we have mixture configs
    if hasattr(data_cfg, 'mixture_configs') and data_cfg.mixture_configs:
        # Multi-dataset scenario
        dataset_configs = data_cfg.mixture_configs
        dataset_names = [cfg_item.repo_id for cfg_item in dataset_configs] 
        for i, cfg_item in enumerate(dataset_configs):
            cfg_name = dataset_names[i]
    else:
        # Single dataset scenario
        dataset_configs = [data_cfg]
        dataset_names = ["main"]
    
    # ONLY MAIN PROCESS loads metadata and creates splits
    if accelerator.is_main_process:
        train_episodes_dict = {}
        val_episodes_dict = {}
        # First, try to load validation episodes from data config
        validation_episodes_loaded = False
        
        for i, cfg_item in enumerate(dataset_configs):
            cfg_name = dataset_names[i]
            
            # Check if validation episodes are provided for this dataset
            if use_validation_list and hasattr(cfg_item, 'validation_episodes') and cfg_item.validation_episodes:
                validation_episodes_path = cfg_item.validation_episodes
                if os.path.exists(validation_episodes_path):
                    try:
                        with open(validation_episodes_path, 'r') as f:
                            validation_episodes_data = json.load(f)
                        
                        # Handle different JSON structures for this specific dataset
                        if isinstance(validation_episodes_data, list):
                            # Direct list format: [0, 1, 2, ...]
                            val_episodes_dict[cfg_name] = validation_episodes_data
                        elif isinstance(validation_episodes_data, dict):
                            # Multi-dataset format: {dataset_name: [episodes]}
                            if cfg_name in validation_episodes_data:
                                val_episodes_dict[cfg_name] = validation_episodes_data[cfg_name]
                            else:
                                # Fallback to random splitting for this dataset
                                val_episodes_dict[cfg_name] = None
                        else:
                            # Fallback to random splitting for this dataset
                            val_episodes_dict[cfg_name] = None
                        
                        if val_episodes_dict[cfg_name] is not None:
                            validation_episodes_loaded = True
                            logging.info(f"Loaded validation episodes for {cfg_name} from data config {validation_episodes_path}: {val_episodes_dict[cfg_name]}")
                        else:
                            logging.warning(f"Could not extract validation episodes for {cfg_name} from {validation_episodes_path}")
                    except Exception as e:
                        logging.warning(f"Failed to load validation episodes for {cfg_name} from {validation_episodes_path}: {e}")
                        val_episodes_dict[cfg_name] = None
                else:
                    logging.warning(f"Validation episodes file {validation_episodes_path} not found for {cfg_name}")
                    val_episodes_dict[cfg_name] = None
            else:
                val_episodes_dict[cfg_name] = None
        
        # If no validation episodes were loaded, do random splitting
        if not validation_episodes_loaded:
            logging.warning("No validation episodes files found or use_validation_list is False, using random splitting")
            for i, cfg_item in enumerate(dataset_configs):
                cfg_name = dataset_names[i]
                
                # Load metadata for this dataset
                info_path = Path(cfg_item.root_dir) / "meta" / "info.json"
                with open(info_path, "r") as f:
                    total_episodes = json.load(f)["total_episodes"]
                
                # Create random episode splits
                all_episodes = list(range(total_episodes))
                random.seed(seed + hash(cfg_name))
                random.shuffle(all_episodes)
                
                n_val = max(1, int(total_episodes * val_split))
                val_episodes = all_episodes[:n_val]
                train_episodes = all_episodes[n_val:]
                
                val_episodes_dict[cfg_name] = val_episodes
                train_episodes_dict[cfg_name] = train_episodes
        else:
            # Create train episodes as complement of validation episodes
            for i, cfg_item in enumerate(dataset_configs):
                cfg_name = dataset_names[i]
                
                # Load metadata for this dataset
                info_path = Path(cfg_item.root_dir) / "meta" / "info.json"
                with open(info_path, "r") as f:
                    total_episodes = json.load(f)["total_episodes"]
                
                if val_episodes_dict[cfg_name] is not None:
                    # Create train episodes as complement of validation episodes
                    all_episodes = set(range(total_episodes))
                    val_episodes_set = set(val_episodes_dict[cfg_name])
                    train_episodes = sorted(list(all_episodes - val_episodes_set))
                    train_episodes_dict[cfg_name] = train_episodes
                else:
                    # Fallback to random splitting for this dataset
                    all_episodes = list(range(total_episodes))
                    random.seed(seed + hash(cfg_name))
                    random.shuffle(all_episodes)
                    
                    n_val = max(1, int(total_episodes * val_split))
                    val_episodes = all_episodes[:n_val]
                    train_episodes = all_episodes[n_val:]
                    
                    val_episodes_dict[cfg_name] = val_episodes
                    train_episodes_dict[cfg_name] = train_episodes
    else:
        # Other processes wait for broadcast
        train_episodes_dict = None
        val_episodes_dict = None
    
    # BROADCAST episode splits from main process to all processes
    train_episodes_dict = broadcast_object_list([train_episodes_dict])[0]
    val_episodes_dict = broadcast_object_list([val_episodes_dict])[0]
    
    # Check if we have mixture configs (data_cfg already created above)
    is_mixture_cfg = hasattr(data_cfg, 'mixture_configs') and data_cfg.mixture_configs

    # If a dataset provides an episodes allowlist file, intersect BEFORE sharding to avoid empty subsets
    cfg_map = {}
    if is_mixture_cfg:
        for item in data_cfg.mixture_configs:
            cfg_map[item.repo_id] = item
    else:
        cfg_map["main"] = data_cfg

    allowed_by_cfg = {}
    for name, cfg_item in cfg_map.items():
        allowlist_path = getattr(cfg_item, 'episodes_list_file', None)
        allowed_set = None
        try:
            if allowlist_path and os.path.exists(allowlist_path):
                with open(allowlist_path, 'r') as f:
                    allowed_set = set(json.load(f))
        except Exception:
            allowed_set = None
        allowed_by_cfg[name] = allowed_set

    def _apply_allowlist(eps_dict: dict[str, list[int]]):
        out = {}
        for name, eps in eps_dict.items():
            allowed = allowed_by_cfg.get(name)
            if isinstance(eps, list) and allowed is not None:
                filtered = [e for e in eps if e in allowed]
                if len(filtered) == 0 and len(allowed) > 0:
                    # Fallback to all allowed episodes to avoid empty datasets
                    filtered = sorted(list(allowed))
                elif len(allowed) == 0:
                    raise Exception(f'The validation episodes are empty for the dataset: {name}. Please check the episodes list file: {allowlist_path} or the validation episodes list: meta/validation_episodes.json')
                out[name] = filtered
            else:
                out[name] = eps
        return out

    train_episodes_dict = _apply_allowlist(train_episodes_dict)
    val_episodes_dict = _apply_allowlist(val_episodes_dict)

    # Partition validation episodes across processes so each worker evaluates on a different subset
    # Keep keys identical across ranks to avoid collective mismatches during metric aggregation
    if hasattr(data_cfg, 'mixture_configs') and data_cfg.mixture_configs:
        world_size = accelerator.num_processes
        rank = accelerator.process_index
        val_episodes_dict_local = {}
        for cfg_name, eps in val_episodes_dict.items():
            if isinstance(eps, list):
                if len(eps) >= world_size:
                    # Round-robin sharding
                    shard = eps[rank::world_size]
                elif len(eps) > 0:
                    # Too few episodes for strict sharding; assign at least one per rank via wrap-around
                    shard = [eps[rank % len(eps)]]
                else:
                    shard = []
                val_episodes_dict_local[cfg_name] = shard
            else:
                # Fallback: if episodes are not a list, keep as-is
                val_episodes_dict_local[cfg_name] = eps
        
        
        # Shard training episodes across processes with safe fallback for tiny splits
        train_episodes_dict_local = {}
        for cfg_name, eps in train_episodes_dict.items():
            if isinstance(eps, list):
                if len(eps) >= world_size:
                    shard = eps[rank::world_size]
                elif len(eps) > 0:
                    shard = [eps[rank % len(eps)]]
                else:
                    shard = []
                train_episodes_dict_local[cfg_name] = shard
            else:
                train_episodes_dict_local[cfg_name] = eps
    else:
        train_episodes_dict_local = train_episodes_dict
        val_episodes_dict_local = val_episodes_dict


    # ALL PROCESSES create datasets using their local episode splits
    train_dataset, norm_stats = create_lerobot_dataset_by_config(
        data_config_factory=data_config_factory,
        model_config=model_config,
        assets_dirs=assets_dirs,
        episodes=train_episodes_dict_local,
        normalization_mode=model_config.normalization_mode,
        return_norm_stats=True,
        map_to_unified_space=map_to_unified_space,
        recompute_norm_stats=recompute_norm_stats,
        **dataset_kwargs
    )
    # If mixture dataset, set per-rank RNG to reduce cross-rank duplication
    if hasattr(train_dataset, "set_rng"):
        try:
            import numpy as _np
            train_dataset.set_rng(_np.random.RandomState(seed + rank))
        except Exception:
            pass
    
    val_datasets_dict = {}
    # data_cfg already created above, reuse it
    is_mixture = hasattr(data_cfg, 'mixture_configs') and data_cfg.mixture_configs
    
    # Initialize output_pipeline_dict for all processes
    output_pipeline_dict = {}
    
    for cfg_name, eps in val_episodes_dict_local.items():
        if is_mixture:
            cfg_item = next(item for item in data_cfg.mixture_configs if item.repo_id == cfg_name)
        else:
            cfg_item = data_cfg
        norm_stats_item = norm_stats[cfg_item.repo_id]
        factory = DummyFactory(cfg_item)
        val_dataset = create_lerobot_dataset_by_config(
            data_config_factory=factory,
            model_config=model_config,
            assets_dirs=assets_dirs,
            episodes=eps,
            normalization_mode=model_config.normalization_mode,
            return_norm_stats=False,
            recompute_norm_stats=False,
            precomputed_norm_stats=norm_stats,
            map_to_unified_space=map_to_unified_space,
            **dataset_kwargs
        )
        val_datasets_dict[cfg_name] = val_dataset
        output_pipeline_dict[cfg_name] = compose(
            get_torch_output_transforms(
                norm_stats=norm_stats_item,
                policy_config=model_config,
                data_config_factory=DummyFactory(cfg_item),
                assets_dirs=assets_dirs,
                normalization_mode=model_config.normalization_mode,
                map_to_unified_space=map_to_unified_space
        ))

    return train_dataset, val_datasets_dict, norm_stats, output_pipeline_dict

def evaluate_policy(
    policy: PreTrainedPolicy,
    val_dataloaders: dict[str, DataLoader],
    accelerator: Accelerator,
    max_eval_steps: int = None,
    output_pipeline_dict: dict=None,
    viz_config: dict = None,
    iteration: int = 0,
    metrics_to_log: list[str] = None,
) -> dict:
    """Evaluate the policy on validation data using all distributed processes."""
    policy.eval()
    # unwrapped_policy = accelerator.unwrap_model(policy)
    unwrapped_policy = policy
    all_metrics = {}  # e.g., {"dataset1": {"val_loss": ...}, "overall": {...}}
    
    # Track raw model output shapes for logging
    raw_model_horizons = {}
    
    # Initialize predictions and ground truth for each dataset
    predictions = {}
    ground_truth = {}
    
    for ds_name, dl in val_dataloaders.items():
        metrics = {name: AverageMeter(name, ":.6f") for name in metrics_to_log}
        # metrics = {"val_loss": AverageMeter("val_loss", ":.6f")}
        # metrics["val_action_mse"] = AverageMeter("val_action_mse", ":.6f")
        # metrics["val_accuracy"] = AverageMeter("val_accuracy", ":.6f")

        eval_steps = 0
        
        # Initialize lists to collect predictions and ground truth
        ds_predictions = []
        ds_ground_truth = []
        
        with torch.no_grad():
            for batch in dl:
                if max_eval_steps is not None and eval_steps >= max_eval_steps:
                    break
                batch = torch_preprocess_dict(batch, augmentations_pipeline=None)
                loss, output_dict = policy.forward(batch)
                if "val_loss" in metrics_to_log:
                    metrics["val_loss"].update(loss.item())
                if "val_accuracy" in metrics_to_log:
                    metrics["val_accuracy"].update(output_dict["accuracy"])    

                
                
                # MSE in original action space using output pipeline from remote_inference
                if output_pipeline_dict is not None:
                    pred_norm = unwrapped_policy.select_action(batch)
                    if "action" not in batch:
                        logging.error(f"The MSE metric is not available for FAST model")
                        output_pipeline_dict = None
                        continue
                    
                    # Log the raw model output shape (this is the actual local horizon)
                    raw_pred_shape = pred_norm.shape if isinstance(pred_norm, torch.Tensor) else np.array(pred_norm).shape
                    logging.debug(f"Raw model output shape for {ds_name}: {raw_pred_shape}")
                    
                    # Store raw model horizon for this dataset
                    if len(raw_pred_shape) > 2:
                        raw_horizon = raw_pred_shape[-2]
                        if ds_name not in raw_model_horizons:
                            raw_model_horizons[ds_name] = raw_horizon
                    
                    output_gt = output_pipeline_dict[ds_name]({
                        "actions": batch["action"].detach().cpu().numpy() if isinstance(batch["action"], torch.Tensor) else batch["action"],
                        "state": batch["state"].detach().cpu().numpy() if isinstance(batch["state"], torch.Tensor) else batch["state"],
                    })['actions']
                    output_pred = output_pipeline_dict[ds_name]({
                        "actions": pred_norm.detach().cpu().numpy() if isinstance(pred_norm, torch.Tensor) else pred_norm,
                        "state": batch["state"].detach().cpu().numpy() if isinstance(batch["state"], torch.Tensor) else batch["state"],
                    })['actions']
                    
                    # Log the processed output shape (this might be different due to output pipeline)
                    processed_pred_shape = output_pred.shape
                    logging.debug(f"Processed output shape for {ds_name}: {processed_pred_shape}")
                    
                    mse = float(np.mean((output_pred - output_gt) ** 2))
                    if "val_action_mse" in metrics_to_log:
                        metrics["val_action_mse"].update(mse)
                    
                    # Collect predictions and ground truth for later metric computation
                    ds_predictions.append(output_pred)
                    ds_ground_truth.append(output_gt)
                
                eval_steps += 1
        
        # Aggregate across processes with proper weighting (sum/count)
        for key, meter in metrics.items():
            sum_tensor = torch.tensor(meter.sum, device=accelerator.device, dtype=torch.float32)
            cnt_tensor = torch.tensor(meter.count, device=accelerator.device, dtype=torch.float32)
            global_sum = accelerator.gather(sum_tensor).sum()
            global_cnt = accelerator.gather(cnt_tensor).sum()
            meter.avg = (global_sum / torch.clamp(global_cnt, min=1.0)).item()

        
        all_metrics[ds_name] = {key: meter.avg for key, meter in metrics.items()}
        
        # Store concatenated predictions and ground truth for this dataset
        if ds_predictions and ds_ground_truth:
            predictions[ds_name] = np.concatenate(ds_predictions, axis=0)
            ground_truth[ds_name] = np.concatenate(ds_ground_truth, axis=0)
    
    # Log raw model horizons for debugging
    if raw_model_horizons:
        logging.info(f"Raw model local horizons detected: {raw_model_horizons}")
        logging.info("Note: These are the actual model output horizons before output pipeline processing")
    
    # Compute overall: Weighted average just for val_loss (by dataset size)
    if len(val_dataloaders) > 1:
        num_datasets = len(val_dataloaders)
        overall_val_loss = sum(all_metrics[ds].get('val_loss', 0) for ds in all_metrics if ds != 'overall') / num_datasets
        overall_val_action_mse = sum(all_metrics[ds].get('val_action_mse', 0) for ds in all_metrics if ds != 'overall') / num_datasets
        all_metrics['overall'] = {
            'val_loss': overall_val_loss,
            'val_action_mse': overall_val_action_mse,
        }
    elif len(val_dataloaders) == 1:
        single_ds = next(iter(all_metrics))
        all_metrics['overall'] = {
            'val_loss': all_metrics[single_ds].get('val_loss', float('inf')),
            'val_action_mse': all_metrics[single_ds].get('val_action_mse', float('inf')),
        }
    
    # Compute additional metrics if we have predictions and ground truth
    if predictions and ground_truth and accelerator.is_main_process:
        try:
            from lerobot.common.utils.train_utils import visualize_joint_actions
            logger = Task.current_task().get_logger()
            if logger is not None:
                # Compute and log aggregated metrics
                joint_metrics = visualize_joint_actions(
                    predictions, ground_truth,
                    iteration, logger, viz_config,
                    should_visualize=True, 
                    sample_info=None
                )
                
                # Update metrics with joint metrics
                for ds_name, metrics_dict in joint_metrics.items():
                    if ds_name not in all_metrics:
                        all_metrics[ds_name] = {}
                    all_metrics[ds_name].update(metrics_dict)
            else:
                logging.warning("ClearML logger not available for metrics logging")
        except Exception as e:
            logging.error(f"Failed to compute joint metrics: {str(e)}")
    
    policy.train()
    return all_metrics

def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    lr_scheduler=None,
    lock=nullcontext(),
    accelerator: Accelerator = None,
    grad_accum_steps: int = 1,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    # policy.train()
    with lock, accelerator.accumulate(policy):
        # device is handled by accelerator.prepare on the dataloader
        # autocast is handled by accelerator
        loss, output_dict = policy(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)

        # scale loss for gradient accumulation to keep effective loss magnitude
        scaled_loss = loss #/ max(1, grad_accum_steps)
        accelerator.backward(scaled_loss)

        if grad_clip_norm is not None and accelerator.sync_gradients:
            grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
        else:
            grad_norm_sq = torch.tensor(0.0, device=scaled_loss.device)
            if accelerator.sync_gradients:
                for p in policy.parameters():
                    if p.grad is not None:
                        grad_norm_sq += p.grad.data.norm(2) ** 2
            grad_norm = torch.sqrt(grad_norm_sq).item()

        # Only step optimizer and scheduler when gradients are synced (end of accumulation)
        if accelerator.sync_gradients:
            optimizer.step()
            optimizer.zero_grad()
            if lr_scheduler is not None:
                lr_scheduler.step()

        if has_method(policy, "update"):
            # To possibly update an internal buffer (for instance an Exponential Moving Average like in TDMPC).
            policy.update()

    # log the unscaled loss value for readability
    train_metrics.loss = loss.item()
    # grad_norm can be either a tensor or float, handle both cases
    if hasattr(grad_norm, 'item'):
        train_metrics.grad_norm = grad_norm.item()
    else:
        train_metrics.grad_norm = grad_norm
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


# Distributed dataset summary aggregation across all workers
def _collect_local_rows(ds):
    rows = []
    # CoTrainDataset: recursively collect from both datasets
    if hasattr(ds, "robotics_dataset") and hasattr(ds, "vlm_dataset"):
        rows.extend(_collect_local_rows(ds.robotics_dataset))
        rows.extend(_collect_local_rows(ds.vlm_dataset))
        return rows
    # LeRobotMixtureDataset: iterate sub-datasets
    if hasattr(ds, "_datasets") and isinstance(ds._datasets, (list, tuple)):
        for sub in ds._datasets:
            if hasattr(sub, "get_dataset_summary"):
                meta = sub.get_dataset_summary()
                rows.append(
                    {
                        "dataset_id": meta.get("dataset_id", "unknown"),
                        "num_samples": int(meta.get("num_samples", 0)),
                        "num_episodes": int(meta.get("num_episodes", 0)),
                        "duration_s": float(meta.get("dataset_length", 0.0)),
                    }
                )
        return rows
    # TorchTransformedDataset or base dataset with single summary
    if hasattr(ds, "get_dataset_summary"):
        meta = ds.get_dataset_summary()
        rows.append(
            {
                "dataset_id": meta.get("dataset_id", "unknown"),
                "num_samples": int(meta.get("num_samples", 0)),
                "num_episodes": int(meta.get("num_episodes", 0)),
                "duration_s": float(meta.get("dataset_length", 0.0)),
            }
        )
    return rows