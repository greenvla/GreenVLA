from typing import Any, List, Callable, Dict, Union
from pathlib import Path
import numpy as np

from lerobot.configs.policies import PreTrainedConfig
from lerobot.common.datasets.data_config import DataConfigFactory, DataConfig
from lerobot.common.datasets.torch_transforms import (
    UnnormalizeTorch,
    NormalizeTorch,
    InjectDefaultPromptTorch,
    MapToUnifiedSpaceTorch,
    MapToSingleSpaceTorch,
)
import json
from lerobot.common.utils.normalize import NormStats, deserialize_json



def get_torch_input_transforms(
    policy_config: PreTrainedConfig,
    data_config_factory: DataConfigFactory,
    assets_dirs: str,
    repack_transforms_output: List[Callable] = None,
    default_prompt: str | None = None,
    normalization_mode: str | None = None,
    map_to_unified_space: bool = False,
) -> List[Callable]:
    """
    Creates a list of PyTorch output transforms similar to OpenPI's create_pretrained_transforms
    for the LeRobot ecosystem.

    Args:
        policy_config: The pre-trained policy configuration (e.g., an instance of PolicyConfig).
        data_config_factory: A factory to create the DataConfig (e.g., LeRobotRDTDataConfig()).
        assets_dirs: Path string to the root directory for assets (e.g. norm_stats).
        repack_transforms_output: Optional list of custom output repackaging transforms.

    Returns:
        A list of callable PyTorch transforms to be applied to the policy output.
    """
    resolved_assets_dirs = Path(assets_dirs)
    data_cfg: DataConfig = data_config_factory.create(
        resolved_assets_dirs, policy_config
    )
    
    assert normalization_mode in ["mean_std", "quantile", "min_max"], f"Invalid normalization mode: {normalization_mode}"

    input_transforms: List[Callable] = []

    if repack_transforms_output:
        input_transforms.extend(repack_transforms_output)
    if default_prompt:
        input_transforms.append(InjectDefaultPromptTorch(default_prompt))
    if data_cfg.data_transforms.inputs:
        input_transforms.extend(data_cfg.data_transforms.inputs)
    if data_cfg.norm_stats:
        input_transforms.append(
            NormalizeTorch(data_cfg.norm_stats, normalization_mode=normalization_mode)
        )
    if data_cfg.model_transforms.inputs:
        input_transforms.extend(data_cfg.model_transforms.inputs)
    if map_to_unified_space:
        input_transforms.append(MapToUnifiedSpaceTorch(target_dim=policy_config.unified_space_dim, axis=-1, pad_value=0.0))
    return input_transforms


def get_torch_output_transforms(
    policy_config: PreTrainedConfig,
    data_config_factory: DataConfigFactory,
    assets_dirs: str,
    norm_stats: dict | None = None,
    repack_transforms_output: List[Callable] = None,
    normalization_mode: str | None = None,
    map_to_unified_space: bool = False,
) -> Union[List[Callable]]:
    """
    Creates output transforms for single or mixture datasets.

    Args:
        policy_config: The pre-trained policy configuration (e.g., an instance of PolicyConfig).
        data_config_factory: A factory to create the DataConfig (e.g., LeRobotRDTDataConfig()).
        assets_dirs: Path string to the root directory for assets (e.g. norm_stats).
        norm_stats: Optional precomputed norm stats to use.
        repack_transforms_output: Optional list of custom output repackaging transforms.
        normalization_mode: The normalization mode to use (mean_std, quantile, or min_max).
        map_to_unified_space: Whether to map outputs to unified space.

    Returns:
        A list of callable transforms.
    """
    assert normalization_mode in ["mean_std", "quantile", "min_max"], f"Invalid normalization mode: {normalization_mode}"
    resolved_assets_dirs = Path(assets_dirs)
    data_cfg: DataConfig = data_config_factory.create(
        resolved_assets_dirs, policy_config
    )

    output_transforms: List[Callable] = []
    if data_cfg.model_transforms.outputs:
        output_transforms.extend(data_cfg.model_transforms.outputs)
    if map_to_unified_space:
        if getattr(data_cfg.data_transforms.inputs[0], "mapping_for_unified_space", None) is not None:
            mapping_actions = getattr(data_cfg.data_transforms.inputs[0], "mapping_for_unified_space")
            mapping_state = getattr(data_cfg.data_transforms.inputs[0], "mapping_for_unified_space")
        else:
            mapping_actions = mapping_state = getattr(data_cfg.data_transforms.inputs[0], "mapping_for_unified_space_actions")
            mapping_state = getattr(data_cfg.data_transforms.inputs[0], "mapping_for_unified_space_state")
        output_transforms.append(MapToSingleSpaceTorch(target_dim=policy_config.max_action_dim, 
                                                       axis=-1, 
                                                       pad_value=0.0,
                                                       mapping_actions=mapping_actions,
                                                       mapping_state=mapping_state))

    if norm_stats is None:
        norm_stats = data_cfg.norm_stats
    if norm_stats:
        norm_stats_object = {}
        for k, v in norm_stats.items():
            if isinstance(v, dict):
                norm_stats_object[k] = NormStats(**v)
            else:
                norm_stats_object[k] = v
        output_transforms.append(
            UnnormalizeTorch(norm_stats_object, normalization_mode=normalization_mode)
        )
    if data_cfg.data_transforms.outputs:
        output_transforms.extend(data_cfg.data_transforms.outputs)
    if repack_transforms_output:
        output_transforms.extend(repack_transforms_output)
    return output_transforms
