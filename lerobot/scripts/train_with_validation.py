import logging
import time
from pathlib import Path

import humanfriendly as _hf
import hydra
import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, TorchDynamoPlugin
from accelerate.utils import set_seed as accelerate_set_seed
from clearml import Task
from omegaconf import DictConfig, OmegaConf
from termcolor import colored
from torch.utils.data import DataLoader
from tqdm import tqdm

from lerobot.common.datasets.create_dataloader import create_lerobot_dataset_by_config
from lerobot.common.datasets.transforms import JaxLikeAugmentations
from lerobot.common.datasets.utils import cycle
from lerobot.common.datasets.cotrain_dataset import CoTrainDataset
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.seed_utils import _seed_worker
from lerobot.common.utils.torch_observation import torch_preprocess_dict
from lerobot.common.utils.train_utils import (_collect_local_rows, create_train_val_datasets_distributed,
                                              evaluate_policy, get_step_checkpoint_dir, instantiate_data_config,
                                              load_safetensors_weights, log_batch_dataset_proportions,
                                              save_checkpoint, update_last_checkpoint, update_policy)
from lerobot.common.utils.utils import format_big_number, init_logging

OmegaConf.register_new_resolver(
    "_load_config", lambda rel_path: OmegaConf.load(Path.cwd() / rel_path)
)

@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def hydra_train(cfg: DictConfig):
    try:
        return train(cfg)
    except Exception as e:
        if e is KeyboardInterrupt:
            logging.info("Training interrupted by user")
        raise e

def train(cfg: DictConfig):
    output_dir = Path(cfg.training.output_dir)
    grad_accum_steps = OmegaConf.select(cfg, "training.grad_accum_steps", default=1)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=cfg.training.find_unused_parameters)

    dynamo = None
    if cfg.training.get("compile", False):
        import torch._dynamo

        torch._dynamo.reset()
        torch._dynamo.config.verbose = True

        logging.info("Policy will be compiled, It may take a while.")
        dynamo = TorchDynamoPlugin(
            backend="inductor",
            mode="default",
            fullgraph=False,
            dynamic=True,
        )

    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs],
        log_with="clearml",
        gradient_accumulation_steps=grad_accum_steps,
        dynamo_plugin=dynamo,
    )

    init_logging(accelerator.is_main_process)

    accelerate_set_seed(cfg.training.seed, device_specific=True)
    import torch  # Sometimes it is necessary

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    if accelerator.is_main_process:
        logging.info(OmegaConf.to_yaml(cfg, resolve=True))

    accelerator.init_trackers(
        project_name=cfg.logging.clearml.task.project_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        init_kwargs={
            "clearml": {
                "task_name": cfg.logging.clearml.task.task_name,
                "task_type": cfg.logging.clearml.task.task_type,
            }
        },
    )

    # find the ClearML tracker that Accelerate created
    # ─── re‑init the Task so current_task() is not None ───

    # Create dataset factory and policy config first
    output_dir = Path(cfg.training.output_dir)

    map_to_unified_space = cfg.get("train_on_unified_space", False)
    map_to_humanoid = cfg.get("map_to_humanoid", False)
    add_kwargs = {
        "map_to_unified_space": map_to_unified_space,
        "map_to_humanoid": map_to_humanoid,
    }
    # time.sleep(accelerator.process_index * 1)
    robotics_dataset_factory = instantiate_data_config(cfg.robotics_dataset, add_kwargs)
    policy_config = hydra.utils.instantiate(cfg.policy.policy_config)

    # Check if validation is enabled
    enable_validation = cfg.training.get("enable_validation", False)
    use_validation_list = cfg.training.get("use_validation_list", False)
    val_split = cfg.training.get("val_split", 0.03)
    val_freq = cfg.training.get("val_freq", 5000)
    max_val_steps = cfg.training.get("max_val_steps", None)


    # Extract dataset config name
    dataset_config_name = cfg.robotics_dataset.get("name", "unknown")

    if accelerator.is_main_process:
        task = Task.init(
            project_name=cfg.logging.clearml.task.project_name,
            task_name=cfg.logging.clearml.task.task_name,
            task_type=cfg.logging.clearml.task.task_type,
            reuse_last_task_id=True,
        )
        logger = task.get_logger()

    if enable_validation:
        # Create train/validation split
        robotics_dataset, val_datasets_dict, norm_stats, output_pipeline_dict = (
            create_train_val_datasets_distributed(
                data_config_factory=robotics_dataset_factory,
                model_config=policy_config,
                assets_dirs=cfg.assets_dir,
                accelerator=accelerator,
                val_split=val_split,
                seed=cfg.training.get("seed", 42),
                map_to_unified_space=map_to_unified_space,
                use_validation_list=use_validation_list,
                recompute_norm_stats=cfg.training.recompute_norm_stats,
            )
        )
        model_mode = getattr(policy_config, "model_mode", "flow_matching")
        if model_mode == "token_prediction":
            output_pipeline_dict = None

        # Create validation dataloaders
        val_dataloaders = {}
        for name, ds in val_datasets_dict.items():
            dl = DataLoader(
                ds,
                batch_size=cfg.training.batch_size,
                shuffle=False,
                num_workers=cfg.training.num_workers_validation,
                pin_memory=False,
                persistent_workers=False,
                worker_init_fn=_seed_worker,
            )
            val_dataloaders[name] = accelerator.prepare(dl)

        if accelerator.is_main_process:
            logging.info(f"Validation enabled:")
            logging.info(f"  Split: {val_split:.1%}")
            logging.info(f"  Frequency: every {val_freq} steps")
            logging.info(f"  Max steps per validation: {max_val_steps}")
    else:
        # Use existing single dataset creation
        robotics_dataset, norm_stats = create_lerobot_dataset_by_config(
            data_config_factory=robotics_dataset_factory,
            model_config=policy_config,
            assets_dirs=cfg.assets_dir,
            normalization_mode=policy_config.normalization_mode,
            return_norm_stats=True,
            map_to_unified_space=map_to_unified_space,
            recompute_norm_stats=cfg.training.recompute_norm_stats,
        )
        val_dataloaders = None
        if accelerator.is_main_process:
            logging.info("Validation disabled")
    if (
        ("qwen_vlm_dataset" in cfg and cfg.qwen_vlm_dataset is not None)
    ):
        if accelerator.is_main_process:
            logging.info("Using VLM dataset")
        vlm_dataset_config = cfg.get(
            "qwen_vlm_dataset",
        )
        vlm_dataset = hydra.utils.instantiate(vlm_dataset_config.vlm_dataset.dataset)
        dataset = CoTrainDataset(
            robotics_dataset,
            vlm_dataset,
            robotics_prob=cfg.robotics_prob,
            state_dim=policy_config.max_state_dim,
            action_horizon=policy_config.n_action_steps,
        )
    else:
        dataset = robotics_dataset
        vlm_dataset = None

    local_rows = _collect_local_rows(dataset)

    # Gather rows from all ranks
    all_rows = None
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            gathered = [None for _ in range(accelerator.num_processes)]
            dist.all_gather_object(gathered, local_rows)
            all_rows = []
            for part in gathered:
                if part:
                    all_rows.extend(part)
        else:
            all_rows = local_rows
    except Exception:
        logging.warning(
            "Failed to gather dataset summary across all workers, using local rows"
        )
        # Fallback: best-effort without distributed
        all_rows = local_rows

    # Distributed dataset summary aggregation across all workers
    if accelerator.is_main_process:
        # Aggregate by dataset_id
        agg = {}
        for r in all_rows:
            dsid = r["dataset_id"]
            a = agg.setdefault(
                dsid,
                {
                    "dataset_id": dsid,
                    "num_samples": 0,
                    "num_episodes": 0,
                    "duration_s": 0.0,
                },
            )
            a["num_samples"] += int(r.get("num_samples", 0))
            a["num_episodes"] += int(r.get("num_episodes", 0))
            a["duration_s"] += float(r.get("duration_s", 0.0))

        # Optional: derive weights similar to default rule (samples**0.43)
        total_weight = 0.0
        for a in agg.values():
            a["weight"] = (
                float(a["num_samples"] ** 0.43) if a["num_samples"] > 0 else 0.0
            )
            total_weight += a["weight"]
        if total_weight > 0:
            for a in agg.values():
                a["weight"] = a["weight"] / total_weight

        df_rows = list(agg.values())
        # Total row
        total_row = {
            "dataset_id": (
                "Mixture Total"
                if len(df_rows) > 1
                else df_rows[0]["dataset_id"] if df_rows else "Total"
            ),
            "num_samples": int(sum(r["num_samples"] for r in df_rows)),
            "num_episodes": int(sum(r["num_episodes"] for r in df_rows)),
            "duration_s": float(sum(r["duration_s"] for r in df_rows)),
            "weight": "",
        }
        
        def _fmt_duration(seconds: float) -> str:
            try:
                return _hf.format_timespan(seconds)
            except Exception:
                logging.warning("Failed to format duration, using default format")
                return f"{seconds:.1f}s"

        # Build pretty DataFrame
        summary_df = pd.DataFrame(
            [
                {
                    "dataset_id": r["dataset_id"],
                    "num_samples": r["num_samples"],
                    "num_episodes": r["num_episodes"],
                    "duration": _fmt_duration(r["duration_s"]),
                    "weight": (
                        ""
                        if isinstance(r.get("weight", ""), str)
                        else float(r.get("weight", 0.0))
                    ),
                }
                for r in df_rows
            ]
            + [
                {
                    "dataset_id": total_row["dataset_id"],
                    "num_samples": total_row["num_samples"],
                    "num_episodes": total_row["num_episodes"],
                    "duration": _fmt_duration(total_row["duration_s"]),
                    "weight": "",
                }
            ]
        )

        logger.report_table(
            title="Training Dataset Summary (Global)",
            series="Robot Dataset Summary",
            iteration=0,
            table_plot=summary_df,
        )

        # Log training setup summary
        num_gpus = accelerator.num_processes
        per_device_bs = cfg.training.batch_size
        grad_accum = max(1, grad_accum_steps)
        global_bs = per_device_bs * grad_accum * num_gpus
        setup_df = pd.DataFrame(
            {
                "Num GPUs": [num_gpus],
                "Per-Device Batch Size": [per_device_bs],
                "Grad Accum Steps": [grad_accum],
                "Global Batch Size": [global_bs],
            }
        )
        logger.report_table(
            title="Training Setup",
            series="Training Setup",
            iteration=0,
            table_plot=setup_df,
        )

        if vlm_dataset is not None:  # logging vlm dataset summary
            vlm_dataset.log_dataset_summary(logger)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=cfg.training.shuffle,
        num_workers=cfg.training.num_workers,
        prefetch_factor=cfg.training.prefetch_factor,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=_seed_worker,
    )

    # Instantiate the policy with the config
    policy = hydra.utils.instantiate(cfg.policy.policy)
    if cfg.training.get("gradient_checkpointing_enabled", False):
        if hasattr(policy.model, "gradient_checkpointing_enable"):
            policy.model.gradient_checkpointing_enable()
        else:
            logging.info(
                "Policy does not support gradient checkpointing but gradient_checkpointing_enabled is set to true"
            )

    # Loading pretrained model weights
    if cfg.pretrained_path is not None:
        logging.info(f"Loading pretrained model weights from {cfg.pretrained_path}")
        state_dict = load_safetensors_weights(cfg.pretrained_path)

        # Filter state_dict to only keep keys that start with weight_prefix_to_keep
        prefixes = OmegaConf.select(cfg, "training.weight_prefix_to_keep", default=None)
        if prefixes:
            filtered_state_dict = {}
            for key, value in state_dict.items():
                if any(key.startswith(prefix) for prefix in prefixes):
                    filtered_state_dict[key] = value
                else:
                    logging.debug(f"Skipping key (not in weight_prefix_to_keep): {key}")

            logging.info(f"Filtered state_dict: {len(filtered_state_dict)}/{len(state_dict)} keys kept")
            state_dict = filtered_state_dict

        missing_keys, unexpected_keys = policy.load_state_dict(
            state_dict, strict=False
        )
        logging.info(f"Missing keys: {missing_keys}")
        logging.info(f"Unexpected keys: {unexpected_keys}")

    params = policy.get_optim_params()
    optimizer_preset = hydra.utils.instantiate(cfg.optimizer)
    optimizer = optimizer_preset.build(params)
    lr_scheduler_preset = hydra.utils.instantiate(cfg.scheduler)

    lr_scheduler = lr_scheduler_preset.build(
        optimizer=optimizer,
        num_training_steps=cfg.training.steps,
        num_processes=accelerator.num_processes,
    )

    num_learnable_params = sum(
        p.numel() for p in policy.parameters() if p.requires_grad
    )
    num_total_params = sum(p.numel() for p in policy.parameters())

    if accelerator.is_main_process:
        logging.info(
            colored("Output dir:", "yellow", attrs=["bold"]) + f" {output_dir}"
        )
        logging.info(
            f"steps={cfg.training.steps} ({format_big_number(cfg.training.steps)})"
        )
        logging.info(
            f"{num_learnable_params=} ({format_big_number(num_learnable_params)})"
        )
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # For any operation use `original_policy` object, `policy` is only for training
    original_policy = policy
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )
    dl_iter = cycle(dataloader)

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    # For tracking dataset proportions across gradient accumulation steps
    accumulated_dataset_ids = []
    num_frames = robotics_dataset._num_samples
    num_episodes = robotics_dataset._num_episodes

    step = 0
    best_val_loss = float("inf")

    if policy_config.model_mode == "token_prediction":
        metrics_to_log = ["val_loss", "val_accuracy"]
    elif policy_config.model_mode == "flow_matching":
        metrics_to_log = ["val_loss", "val_action_mse"]
    elif policy_config.model_mode == "mixed":
        metrics_to_log = ["val_loss", "val_action_mse"]
    else:
        raise ValueError(
            f"Invalid model mode: {policy_config.model_mode}, only token_prediction and flow_matching are supported"
        )

    # Use global effective batch size to reflect true throughput across GPUs and accumulation
    effective_batch_size_for_tracker = (
        cfg.training.batch_size * max(1, grad_accum_steps) * accelerator.num_processes
    )
    train_tracker = MetricsTracker(
        effective_batch_size_for_tracker,
        num_frames,
        num_episodes,
        train_metrics,
        initial_step=step,
    )
    logging.info("Start offline training on a fixed dataset")

    # Load augmentation config from Hydra

    image_augmentations_config = cfg.get("image_augmentations", None)
    if image_augmentations_config is not None:
        jax_augmenter = JaxLikeAugmentations(config=image_augmentations_config)
    else:
        jax_augmenter = None
        logging.info("No image augmentations configured")
    optimizer_step = 0
    total_micro_steps = cfg.training.steps * max(1, cfg.training.grad_accum_steps)
    # Only set first_val_step if validation is enabled
    first_val_step = enable_validation
    accelerator.wait_for_everyone()

    policy.train()
    for _ in tqdm(
        range(step, total_micro_steps), disable=not accelerator.is_main_process
    ):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        # Accumulate dataset IDs if on main process
        if accelerator.is_main_process and "dataset_id" in batch:
            # Handle both tensor and list cases for dataset_id
            dataset_ids = batch["dataset_id"]
            if torch.is_tensor(dataset_ids):
                dataset_ids = dataset_ids.cpu().numpy()
            accumulated_dataset_ids.extend(dataset_ids)

        batch = torch_preprocess_dict(batch, augmentations_pipeline=jax_augmenter)

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            lr_scheduler=lr_scheduler,
            accelerator=accelerator,
            grad_accum_steps=grad_accum_steps,
        )
        if accelerator.sync_gradients:
            optimizer_step += 1
            train_tracker.step()

            is_val_step = (
                enable_validation and val_freq > 0 and optimizer_step % val_freq == 0
            )

            if enable_validation and (is_val_step or first_val_step):
                first_val_step = False
                if accelerator.is_main_process:
                    logging.info(f"Running validation at step {optimizer_step}")

                val_metrics = evaluate_policy(
                    policy=original_policy,
                    val_dataloaders=val_dataloaders,
                    accelerator=accelerator,
                    max_eval_steps=max_val_steps,
                    output_pipeline_dict=output_pipeline_dict,
                    metrics_to_log=metrics_to_log,
                    viz_config=cfg.training.get("visualization", {}),
                    iteration=optimizer_step,
                )

                overall_metrics = val_metrics.get("overall", {})
                val_loss = overall_metrics.get("val_loss", float("inf"))
                # val_mse = overall_metrics.get('val_action_mse', float('inf'))

                if accelerator.is_main_process:
                    logging.info(f"Validation Loss: {val_loss:.6f}")
                    # logging.info(f"Validation Action MSE: {val_mse:.6f}")
                    for ds, metrics_dict in val_metrics.items():
                        if ds != "overall":
                            for metric_name, metric_val in metrics_dict.items():
                                logging.info(f"  {ds} {metric_name}: {metric_val:.6f}")

                    log_dict = {}
                    for ds, metrics_dict in val_metrics.items():
                        for metric_name, metric_val in metrics_dict.items():
                            log_dict[f"val/{ds}/{metric_name}"] = metric_val

                    accelerator.log(log_dict, step=optimizer_step)

                    # Log all metrics to ClearML
                    for ds, metrics_dict in val_metrics.items():
                        # Log standard validation metrics
                        if "val_loss" in metrics_dict:
                            logger.report_scalar(
                                title="Validation Loss",
                                series=ds,
                                value=metrics_dict["val_loss"],
                                iteration=optimizer_step,
                            )
                        if "val_action_mse" in metrics_dict:
                            logger.report_scalar(
                                title="Validation Action MSE",
                                series=ds,
                                value=metrics_dict["val_action_mse"],
                                iteration=optimizer_step,
                            )

                        # Log aggregated joint metrics
                        if "val_action_joint_mse_avg" in metrics_dict:
                            logger.report_scalar(
                                title="Joint MSE (Avg)",
                                series=ds,
                                value=metrics_dict["val_action_joint_mse_avg"],
                                iteration=optimizer_step,
                            )
                        if "val_action_joint_mae_avg" in metrics_dict:
                            logger.report_scalar(
                                title="Joint MAE (Avg)",
                                series=ds,
                                value=metrics_dict["val_action_joint_mae_avg"],
                                iteration=optimizer_step,
                            )

            # Handle logging and checkpointing outside validation block
            if accelerator.is_main_process:
                is_log_step = (
                    cfg.training.log_freq > 0
                    and optimizer_step % cfg.training.log_freq == 0
                )
                is_saving_step = (
                    optimizer_step % cfg.training.save_freq == 0
                    or optimizer_step == cfg.training.steps
                )

                if is_log_step:
                    logging.info(train_tracker)
                    log_dict = train_tracker.to_dict()
                    if output_dict:
                        log_dict.update(output_dict)
                    accelerator.log(
                        {f"train/{k}": v for k, v in log_dict.items()},
                        step=optimizer_step,
                    )

                    # Log accumulated dataset proportions during training metrics logging
                    if logger is not None and accumulated_dataset_ids:
                        # Create a dummy batch with accumulated IDs
                        accumulated_batch = {"dataset_id": accumulated_dataset_ids}
                        log_batch_dataset_proportions(
                            batch=accumulated_batch,
                            iteration=optimizer_step,
                            logger=logger,
                            grad_accum_steps=cfg.training.grad_accum_steps,
                            batch_size=cfg.training.batch_size,
                            num_gpus=accelerator.num_processes,
                        )
                        # Reset accumulation
                        accumulated_dataset_ids.clear()

                    train_tracker.reset_averages()

                if is_saving_step:
                    logging.info(f"Checkpoint policy after step {optimizer_step}")
                    # unwrapped_policy = accelerator.unwrap_model(policy)
                    checkpoint_dir = get_step_checkpoint_dir(
                        output_dir, cfg.training.steps, optimizer_step
                    )

                    # Get current loss values
                    current_train_loss = (
                        train_tracker.loss if hasattr(train_tracker, "loss") else None
                    )
                    current_val_loss = val_loss if "val_loss" in locals() else None

                    save_checkpoint(
                        checkpoint_dir=checkpoint_dir,
                        step=optimizer_step,
                        cfg=policy_config,
                        policy=original_policy,
                        optimizer=optimizer,
                        scheduler=lr_scheduler,
                        norm_stats=norm_stats,
                        clearml_task=task,
                        train_loss=current_train_loss,
                        val_loss=current_val_loss,
                        dataset_config_name=dataset_config_name,
                        model_total_params=num_total_params,
                        model_learnable_params=num_learnable_params,
                    )

                    update_last_checkpoint(checkpoint_dir)
                    logging.info(f"Checkpoint policy saved after step {optimizer_step}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.end_training()


if __name__ == "__main__":
    init_logging()
    hydra_train()
