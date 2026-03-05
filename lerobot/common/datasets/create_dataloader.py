import logging
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader as TorchDataLoader
import numpy as np
import json
from typing import Sequence, Any, Callable, Dict, Tuple, Optional, Set
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import os
from lerobot.common.datasets.torch_transforms import (
    TorchTransformedDataset,
    PromptFromLeRobotTaskTorch,
    NormalizeTorch,
    PyTorchModelTransformFactory,
    ResizeImagesTorch,
    RemoveStrings,
    MapToUnifiedSpaceTorch,
    Group as TorchGroup,
)
from lerobot.common.datasets.fake_dataset import FakeDataset
from lerobot.common.datasets.data_config import DataConfig
from lerobot.common.utils.normalize import compute_dataset_stats_from_dataset, NormStats

from lerobot.common.datasets.utils import print_dataset_summary

from tqdm import tqdm
from rich.table import Table
from rich.console import Console
from rich import print
from typing import Union


from lerobot.common.datasets.qwen_vlm_datasets.qwen_vlm_dataset import (
    MixtureQwenVLMDataset,
    QwenVLMDataset,
    QwenVLMDatasetConfig,
    MixtureQwenVLMDatasetConfig,
)

from lerobot.common.utils.normalize import norm_stats_to_dict
import humanfriendly
import pandas as pd


def create_lerobot_dataset(
    data_config: DataConfig,
    model_config: Any,
    return_meta: bool = False,
    dataset_root: str = None,
    episodes: list[int] | None = None,
    **kwargs,
) -> TorchDataset | Tuple[TorchDataset, lerobot_dataset.LeRobotDatasetMetadata]:
    """Create a LeRobot PyTorch dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None and dataset_root is None:
        raise ValueError(
            "Repo ID is not set and dataset_root is not provided. Cannot create dataset."
        )
    if repo_id == "fake":
        fake_ds = FakeDataset(model_config, num_samples=1024)
        if return_meta:
            return fake_ds, None
        return fake_ds

    logging.info(f"Creating LeRobot dataset for repo_id: {repo_id}")
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=dataset_root)
    return_subtasks = False


    if hasattr(data_config, 'episodes_list_file') and data_config.episodes_list_file is not None and episodes is not None:
        if not os.path.exists(data_config.episodes_list_file):
            logging.warning(f"Episodes list file {data_config.episodes_list_file} does not exist, using all episodes")
            logging.info(f"[bold red]Warning:[/bold red] Episodes list file {data_config.episodes_list_file} does not exist, using all episodes")
        with open(data_config.episodes_list_file, 'r') as f:
            filtered_episodes = json.load(f)
        filtered_episodes = [x for x in episodes if x in set(filtered_episodes)]
        logging.info(f'filtering {data_config.repo_id}: {len(episodes)} -> {len(filtered_episodes)} episodes')
        episodes = filtered_episodes
    return_subtasks = getattr(data_config, 'return_subtasks', False)
    subtask_transition_percent = getattr(data_config, 'subtask_transition_percent', 0.9)
    return_subtasks_mode = "optional" if return_subtasks else "disabled"
        
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        root=dataset_root,
        episodes=episodes,
        delta_timestamps={
            
            key: [
                (state_delta_idx * data_config.action_sample_step + data_config.action_offset - 1) / dataset_meta.fps
                for state_delta_idx in range(1, data_config.action_horizon + 1)]
                # 1. For most datasets actions[:-1] == states[1:] and offset is 0. For mode ground truth we sample
                # states at [step, 2 * step, ...] == actions at [step - 1, 2 * step - 1, ...]
                # 2. For datasets with actions == states, the offset is 1, and
                # States at [step, 2 * step, ...] == actions at [step, 2 * step, ...]
                # 3. For datasets with real control actions, we would like to train on
                # actions at [step, 2 * step, ...], so we set offset=1 in the config as in case 2.
                #
                # For better performance convert dataset as in case 2 rather than case 1. This is not critical,
                # the only difference is that padded initial actions will be used in case 1
                # at the first chunk of each episode if action_sample_step < 1.
            for key in data_config.action_sequence_keys
        },
        return_subtasks_mode=return_subtasks_mode,
        subtask_transition_percent=subtask_transition_percent,
        **kwargs,
    )

    if data_config.prompt_from_task:
        prompt_transform = PromptFromLeRobotTaskTorch(dataset_meta.tasks)
        dataset = TorchTransformedDataset(dataset, prompt_transform)

    if return_meta:
        return dataset, dataset_meta
    return dataset


def transform_le_robot_dataset(
    dataset: TorchDataset,
    data_config: DataConfig,
    model_config: Any,
    *,
    norm_stats: dict[str, Any] | None = None,
    skip_norm_stats: bool = False,
    skip_model_transforms: bool = False,
    normalization_mode: str | None = None,
    return_norm_stats: bool = False,
    map_to_unified_space: bool = False,
) -> TorchDataset:
    """Transform the LeRobot PyTorch dataset by applying the data transforms."""
    if (norm_stats is None) and (data_config.repo_id != "fake") and (not skip_norm_stats):
        if data_config.norm_stats is None:
            raise ValueError(
                "Warning: Normalization stats not found in DataConfig. "
                "LeRobot might handle normalization differently or expect it to be pre-computed/loaded."
            )
        else:
            norm_stats = data_config.norm_stats

    all_transforms = []
    if data_config.repack_transforms and data_config.repack_transforms.inputs:
        all_transforms.extend(data_config.repack_transforms.inputs)

    if data_config.data_transforms and data_config.data_transforms.inputs:
        all_transforms.extend(data_config.data_transforms.inputs)

    if norm_stats is not None:
        norm_stats = {k: NormStats(**v) if isinstance(v, dict) else v
                        for k, v in norm_stats.items()}
        all_transforms.append(
            NormalizeTorch(norm_stats, normalization_mode=normalization_mode)
        )
    logging.info(f"map_to_unified_space: {map_to_unified_space}")
    if map_to_unified_space:
        all_transforms.append(MapToUnifiedSpaceTorch(target_dim=model_config.unified_space_dim, axis=-1, pad_value=0.0))

    if not skip_model_transforms:
        model_transform_group_or_callable = data_config.model_transforms
        if isinstance(data_config.model_transforms, PyTorchModelTransformFactory):
            model_transform_group_or_callable = data_config.model_transforms(
                model_config
            )

        if model_transform_group_or_callable:
            if isinstance(model_transform_group_or_callable, TorchGroup):
                if model_transform_group_or_callable.inputs:
                    all_transforms.extend(model_transform_group_or_callable.inputs)
            elif callable(model_transform_group_or_callable):
                all_transforms.append(model_transform_group_or_callable)
            else:
                logging.info(
                    "Warning: model_transforms is not a callable or TorchGroup, skipping."
                )
    else:
        # all_transforms.append(RemoveStrings())
        all_transforms.append(ResizeImagesTorch(224, 224))

    def composed_transform(item):
        for t in all_transforms:
            item = t(item)
        return item

    if return_norm_stats:
        return (
            TorchTransformedDataset(dataset, composed_transform),
            {data_config.repo_id: norm_stats_to_dict(norm_stats)},
        )
    else:
        return TorchTransformedDataset(dataset, composed_transform)


class LeRobotMixtureDataset(TorchDataset):
    """PyTorch Dataset that samples from multiple LeRobot datasets with weights."""

    def __init__(
        self,
        datasets: Sequence[TorchDataset],
        weights: Sequence[float] | None = None,
        *,
        seed: int | None = None,
    ):
        if not datasets:
            raise ValueError("At least one dataset must be provided.")

        self._datasets = datasets

        if weights is None or len(weights) == 0:
            # making here warning that weights are not provided and using the default weights
            logging.info(
                "[bold red]Warning:[/bold red] Weights are not provided. Using default weights."
            )
            weights = []
            for i in range(len(datasets)):
                weights.append(
                    datasets[i]._num_samples ** 0.43
                )  # same as in the Pi0 paper

        if len(weights) != len(datasets):
            raise ValueError(
                f"Number of weights ({len(weights)}) must match number of datasets ({len(datasets)})."
            )

        self._weights = np.array(weights, dtype=np.float64)
        self._weights = self._weights / np.sum(self._weights)

        self._num_datasets = len(datasets)
        self._rng = np.random.RandomState(seed)
        # obtaining info from datasets
        num_samples = 0
        num_episodes = 0
        for dataset in datasets:
            num_samples += dataset._num_samples
            num_episodes += dataset._num_episodes
        self._num_samples = num_samples
        self._num_episodes = num_episodes

        self._length = sum(len(dataset) for dataset in datasets)

    def __len__(self) -> int:
        return 100000000

    def __getitem__(self, index: int) -> Any:
        # if not 0 <= index < self._length:
        #     raise IndexError(
        #         f"Index {index} out of range for LeRobotMixtureDataset of length {self._length}"
        #     )
        
        

        choosed_dataset_idx = self._rng.choice(self._num_datasets, p=self._weights)
        choosed_dataset = self._datasets[choosed_dataset_idx]
        choosed_element_idx = self._rng.randint(0, len(choosed_dataset))
        
        # Get the batch from the chosen dataset
        try:      
            batch = choosed_dataset[choosed_element_idx]
        except Exception as e:
            print(f"[bold red]Error getting item {index} from dataset {choosed_dataset._dataset_id} at index {choosed_element_idx}: {e}[/bold red]")
            return self[index]
        
        # Add dataset_id to the batch
        if isinstance(batch, dict):
            batch["dataset_id"] = choosed_dataset._dataset_id
        else:
            # If batch is not a dict, wrap it in one
            batch = {"data": batch, "dataset_id": choosed_dataset._dataset_id}
            
        return batch
    
    def set_rng(self, rng):
        self._rng = rng

    def print_dataset_summary(self) -> dict:
        summary = {}
        summary["dataset_id"] = "Mixture of datasets"
        num_samples = 0
        num_episodes = 0
        total_duration = 0
        for dataset in self._datasets:
            dataset_meta = dataset.get_dataset_summary()
            print_dataset_summary(dataset_meta)
            num_samples += dataset_meta["num_samples"]
            num_episodes += dataset_meta["num_episodes"]
            total_duration += dataset_meta["dataset_length"]
        summary["num_samples"] = num_samples
        summary["num_episodes"] = num_episodes
        summary["dataset_length"] = total_duration
        print_dataset_summary(summary)

        # printing the weights
        table = Table(title="Weights", expand=False)
        table.add_column("Dataset ID", style="bold cyan", no_wrap=True)
        table.add_column("Weight", style="bold magenta")
        for i, dataset in enumerate(self._datasets):
            table.add_row(dataset._dataset_id, f"{self._weights[i]:.4f}")
        console = Console()
        console.print(table)

    def _build_summary_df(self) -> pd.DataFrame:
        """
        Build a DataFrame summarizing each sub‑dataset and the overall mixture,
        with human-friendly duration strings.
        """
        rows = []
        # per‑dataset rows
        for ds, w in zip(self._datasets, self._weights):
            meta = ds.get_dataset_summary()
            dur_s = meta["dataset_length"]
            dur_str = humanfriendly.format_timespan(dur_s)
            rows.append(
                {
                    "dataset_id": meta["dataset_id"],
                    "num_samples": meta["num_samples"],
                    "num_episodes": meta["num_episodes"],
                    "duration": dur_str,
                    "weight": float(w),
                }
            )

        # total row
        total_s = sum(
            ds.get_dataset_summary()["dataset_length"] for ds in self._datasets
        )
        rows.append(
            {
                "dataset_id": "Mixture Total",
                "num_samples": self._num_samples,
                "num_episodes": self._num_episodes,
                "duration": humanfriendly.format_timespan(total_s),
                "weight": "",  # leave blank for total
            }
        )

        return pd.DataFrame(rows)

    def log_dataset_summary(self, logger) -> None:
        """
        Log the mixture summary to ClearML (once, at iteration=0).
        """
        df = self._build_summary_df()
        logger.report_table(
            title="Training Dataset Summary",
            series="Robot Dataset Summary",
            iteration=0,
            table_plot=df,
        )


def create_lerobot_dataloader(
    data_config_factory: Any,
    model_config: Any,
    assets_dirs: str,
    batch_size: int,
    *,
    skip_norm_stats: bool = False,
    skip_model_transforms: bool = False,
    shuffle: bool = False,
    num_workers: int = 0,
    seed: int = 42,
    pin_memory: bool = True,
    drop_last: bool = True,
    return_dataset: bool = False,
    return_dataloader: bool = True,
    normalization_mode: str | None = None,
    map_to_unified_space: bool = False,
    **kwargs,
) -> TorchDataLoader | TorchDataset:
    """Create a LeRobot PyTorch DataLoader."""
    data_cfg = data_config_factory.create(assets_dirs, model_config)

    if data_cfg.mixture_configs:
        datasets = []
        for i, cfg_item in enumerate(data_cfg.mixture_configs):
            raw_dataset = create_lerobot_dataset(
                cfg_item, model_config, dataset_root=cfg_item.root_dir, **kwargs
            )
            transformed_ds = transform_le_robot_dataset(
                raw_dataset,
                cfg_item,
                model_config,
                skip_norm_stats=skip_norm_stats,
                skip_model_transforms=skip_model_transforms,
                normalization_mode=normalization_mode,
                map_to_unified_space=map_to_unified_space,
            )
            datasets.append(transformed_ds)

        final_dataset = LeRobotMixtureDataset(
            datasets,
            weights=data_cfg.mixture_weights,
            seed=seed,
        )
    else:
        raw_dataset = create_lerobot_dataset(
            data_cfg, model_config, dataset_root=data_cfg.root_dir, **kwargs
        )
        final_dataset = transform_le_robot_dataset(
            raw_dataset,
            data_cfg,
            model_config,
            skip_norm_stats=skip_norm_stats,
            skip_model_transforms=skip_model_transforms,
            normalization_mode=normalization_mode,
            map_to_unified_space=map_to_unified_space,
        )

    dataloader_generator = None
    if shuffle:
        dataloader_generator = torch.Generator()
        dataloader_generator.manual_seed(seed)

    dataloader = TorchDataLoader(
        final_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        generator=dataloader_generator,
    )

    if return_dataset:
        return dataloader, raw_dataset, final_dataset

    return dataloader


def create_lerobot_dataset_by_config(
    data_config_factory: Any,
    model_config: Any,
    assets_dirs: str,
    skip_norm_stats: bool = False,
    recompute_norm_stats: bool = False,
    precomputed_norm_stats: dict[str, NormStats] | None = None,
    skip_model_transforms: bool = False,
    normalization_mode: str | None = None,
    return_norm_stats: bool = False,
    episodes: list[int] | dict[str, list[int]] | None = None,
    map_to_unified_space: bool = False,
    **kwargs,
) -> TorchDataset:
    """Create a LeRobot PyTorch dataset by config."""
    data_cfg = data_config_factory.create(assets_dirs, model_config)
    norm_stats = {}

    if data_cfg.mixture_configs:
        datasets = []
        for i, cfg_item in enumerate(data_cfg.mixture_configs):
            # Extract episodes for this specific config item
            cfg_episodes = None
            if episodes is not None:
                if isinstance(episodes, dict):
                    # For mixture datasets, episodes should be a dict with config names as keys
                    cfg_name = cfg_item.repo_id if hasattr(cfg_item, 'repo_id') else f"config_{i}"
                    cfg_episodes = episodes.get(cfg_name, None)
                else:
                    # If episodes is a list, use it for all configs
                    cfg_episodes = episodes
            
            raw_dataset = create_lerobot_dataset(
                cfg_item, model_config, dataset_root=cfg_item.root_dir, episodes=cfg_episodes, **kwargs
            )
            dataset_norm_stats = None

            if recompute_norm_stats:
                dataset_without_norm = transform_le_robot_dataset(
                    raw_dataset,
                    cfg_item,
                    model_config,
                    skip_norm_stats=True,
                    skip_model_transforms=True,
                    normalization_mode=normalization_mode,
                )
                dataset_norm_stats = compute_dataset_stats_from_dataset(dataset_without_norm)
                skip_norm_stats = False
            elif precomputed_norm_stats:
                dataset_norm_stats = precomputed_norm_stats.get(cfg_item.repo_id, None)
                skip_norm_stats = False

            if return_norm_stats:
                transformed_ds, dataset_norm_stats = transform_le_robot_dataset(
                    raw_dataset,
                    cfg_item,
                    model_config,
                    norm_stats=dataset_norm_stats,
                    skip_norm_stats=skip_norm_stats,
                    skip_model_transforms=skip_model_transforms,
                    normalization_mode=normalization_mode,
                    return_norm_stats=return_norm_stats,
                    map_to_unified_space=map_to_unified_space,
                )
                norm_stats.update(dataset_norm_stats)
            else:
                transformed_ds = transform_le_robot_dataset(
                    raw_dataset,
                    cfg_item,
                    model_config,
                    norm_stats=dataset_norm_stats,
                    skip_norm_stats=skip_norm_stats,
                    skip_model_transforms=skip_model_transforms,
                    normalization_mode=normalization_mode,
                    return_norm_stats=return_norm_stats,
                    map_to_unified_space=map_to_unified_space,
                )

            datasets.append(transformed_ds)
        final_dataset = LeRobotMixtureDataset(
            datasets,
            weights=data_cfg.mixture_weights,
        )
    else:
        # For single dataset, extract episodes from the dict if provided
        dataset_episodes = None
        if episodes is not None:
            if isinstance(episodes, dict):
                dataset_episodes = episodes.get("main", None)
            else:
                dataset_episodes = episodes
                
        raw_dataset = create_lerobot_dataset(
            data_cfg, model_config, dataset_root=data_cfg.root_dir, episodes=dataset_episodes, **kwargs
        )

        dataset_norm_stats = None
        if recompute_norm_stats:
            dataset_without_norm = transform_le_robot_dataset(
                raw_dataset,
                data_cfg,
                model_config,
                skip_norm_stats=True,
                skip_model_transforms=True,
                normalization_mode=normalization_mode,
            )
            dataset_norm_stats = compute_dataset_stats_from_dataset(dataset_without_norm)
            skip_norm_stats = False
        elif precomputed_norm_stats:
            dataset_norm_stats = precomputed_norm_stats.get(data_cfg.repo_id, None)
            skip_norm_stats = False
        if return_norm_stats:
            final_dataset, dataset_norm_stats = transform_le_robot_dataset(
                raw_dataset,
                data_cfg,
                model_config,
                norm_stats=dataset_norm_stats,
                skip_norm_stats=skip_norm_stats,
                skip_model_transforms=skip_model_transforms,
                normalization_mode=normalization_mode,
                return_norm_stats=return_norm_stats,
                map_to_unified_space=map_to_unified_space,
            )
            norm_stats.update(dataset_norm_stats)
        else:
            final_dataset = transform_le_robot_dataset(
                raw_dataset,
                data_cfg,
                model_config,
                norm_stats=dataset_norm_stats,
                skip_norm_stats=skip_norm_stats,
                skip_model_transforms=skip_model_transforms,
                normalization_mode=normalization_mode,
                return_norm_stats=return_norm_stats,
                map_to_unified_space=map_to_unified_space,
            )
    if return_norm_stats:
        return final_dataset, norm_stats
    else:
        return final_dataset


def create_vlm_dataset_by_config(
    data_config: Any,
    **kwargs,
) -> TorchDataset:
    """Create a VLM PyTorch dataset by config."""
    if isinstance(data_config, QwenVLMDatasetConfig):
        return QwenVLMDataset(data_config, **kwargs)
    elif isinstance(data_config, MixtureQwenVLMDatasetConfig):
        return MixtureQwenVLMDataset(data_config, **kwargs)
    else:
        raise ValueError(f"Invalid data config type: {type(data_config)}")


def create_cotrain_dataset(
    robotics_data_config: DataConfig,
    vlm_data_config: Union[
        QwenVLMDatasetConfig,
        MixtureQwenVLMDatasetConfig,
    ],
    model_config: Any,
    assets_dir,
    use_quantiles: bool = False,
    robotics_prob: float = 0.5,
    state_dim: int = 32,
    action_horizon: int = 50,
    **kwargs,
) -> TorchDataset:
    """Create a Co-training dataset by config."""
    robotics_dataset = create_lerobot_dataset_by_config(
        robotics_data_config,
        model_config,
        assets_dir,
        use_quantiles=use_quantiles,
    )
    vlm_dataset = create_vlm_dataset_by_config(
        vlm_data_config,
        **kwargs,
    )

    return CoTrainDataset(robotics_dataset, vlm_dataset, robotics_prob)
