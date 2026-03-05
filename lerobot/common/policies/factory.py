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
import os
from pathlib import Path
from typing import Callable

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError
from omegaconf import OmegaConf
from hydra.utils import instantiate
from torch import nn
from lerobot.common.datasets.torch_transforms import compose

from lerobot.common.constants import PRETRAINED_MODEL_DIR
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import get_config_root
from lerobot.common.utils.inference_transforms import (
    get_torch_input_transforms,
    get_torch_output_transforms,
)
from lerobot.configs.policies import PreTrainedConfig
from lerobot.common.policies.greenvla_policy.configuration_greenvla_policy import GreenVLAPolicyConfig
from lerobot.common.policies.pretrained import PreTrainedPolicy

def get_policy_class(name: str) -> PreTrainedPolicy:
    """Get the policy's class and config class given a name (matching the policy class' `name` attribute)."""
    
    
    if name == "greenvlapolicy":
        from lerobot.common.policies.greenvla_policy.modeling_greenvla_policy import GreenVLAPolicy
        return GreenVLAPolicy
    else:
        raise NotImplementedError(f"Policy with name {name} is not implemented.")


def make_policy_config(policy_type: str, **kwargs) -> PreTrainedConfig:
    if policy_type == "greenvlapolicy":
        return GreenVLAPolicyConfig(**kwargs)
    else:
        raise ValueError(f"Policy type '{policy_type}' is not available.")


def make_policy(
    cfg: PreTrainedConfig,
    ds_meta: LeRobotDatasetMetadata | None = None,
    env_cfg: None = None,
) -> PreTrainedPolicy:
    """Make an instance of a policy class.

    This function exists because (for now) we need to parse features from either a dataset or an environment
    in order to properly dimension and instantiate a policy for that dataset or environment.

    Args:
        cfg (PreTrainedConfig): The config of the policy to make. If `pretrained_path` is set, the policy will
            be loaded with the weights from that path.
        ds_meta (LeRobotDatasetMetadata | None, optional): Dataset metadata to take input/output shapes and
            statistics to use for (un)normalization of inputs/outputs in the policy. Defaults to None.
        env_cfg (EnvConfig | None, optional): The config of a gym environment to parse features from. Must be
            provided if ds_meta is not. Defaults to None.

    Raises:
        ValueError: Either ds_meta or env and env_cfg must be provided.
        NotImplementedError: if the policy.type is 'vqbet' and the policy device 'mps' (due to an incompatibility)

    Returns:
        PreTrainedPolicy: _description_
    """

    policy_cls = get_policy_class(cfg.type)

    kwargs = {}

    assert cfg.pretrained_path, "You are instantiating a policy from scratch. Set cfg.pretrained_path to the model config"


    kwargs["config"] = cfg

    if cfg.pretrained_path:
        # Load a pretrained policy and override the config if needed (for example, if there are inference-time
        # hyperparameters that we want to vary).
        kwargs["pretrained_name_or_path"] = cfg.pretrained_path
        policy = policy_cls.from_pretrained(**kwargs)
    else:
        # Make a fresh policy.
        policy = policy_cls(**kwargs)

    policy.to(cfg.device)
    assert isinstance(policy, nn.Module)

    # policy = torch.compile(policy, mode="reduce-overhead")

    return policy


def _resolve_norm_stats_dir(
    pretrained_name_or_path: str | Path,
    data_config_name: str,
) -> str:
    """Resolve the norm_stats directory from a local checkpoint path or HF Hub repo.

    For local paths, looks for norm_stats/ in the checkpoint directory.
    For HF Hub repos, downloads norm_stats/{data_config_name}/norm_stats.json
    and returns the parent directory in the local cache.

    Returns:
        Local path to the directory containing {data_config_name}/norm_stats.json
    """
    model_id = str(pretrained_name_or_path)
    norm_stats_subpath = f"norm_stats/{data_config_name}/norm_stats.json"

    if Path(model_id).is_dir():
        # Local path: check norm_stats/ directory
        local_norm_stats = Path(model_id) / norm_stats_subpath
        if local_norm_stats.is_file():
            return str(Path(model_id) / "norm_stats")
        raise FileNotFoundError(
            f"Norm stats not found at {local_norm_stats}. "
            f"Expected {norm_stats_subpath} inside {model_id}"
        )
    else:
        # HF Hub: download the norm_stats file and resolve cache path
        try:
            downloaded = hf_hub_download(
                repo_id=model_id,
                filename=norm_stats_subpath,
            )
        except HfHubHTTPError as e:
            raise FileNotFoundError(
                f"Norm stats file '{norm_stats_subpath}' not found on the "
                f"HuggingFace Hub in {model_id}"
            ) from e
        # downloaded points to .../norm_stats/{data_config_name}/norm_stats.json
        # We need the .../norm_stats/ directory
        return str(Path(downloaded).parent.parent)


def load_pretrained_policy(
    pretrained_name_or_path: str | Path,
    data_config_name: str | None = "bridge",
    *,
    config_overrides: dict | None = None,
) -> tuple[PreTrainedPolicy, list[Callable] | None, list[Callable] | None]:
    """Load a pretrained policy with input/output transforms from a local path or HF Hub.

    This is the main entry point for loading a checkpoint for inference. It handles:
    - Loading the config (from pretrained_model/config.json)
    - Loading the model weights (from pretrained_model/model.safetensors)
    - Downloading and setting up norm stats
    - Building input and output transform pipelines

    Works with both local checkpoint directories and HuggingFace Hub repo IDs.

    Args:
        pretrained_name_or_path: Either a local directory path (e.g.
            "/path/to/checkpoints/300000") or a HuggingFace Hub repo ID
            (e.g. "SberRoboticsCenter/GreenVLA-5b-corrupted").
        data_config_name: Name of the robot/data config to use for transforms
            (e.g. "bridge"). Must match a yaml file in lerobot/conf/robotics_dataset/individual/.
            If ``None``, transforms are skipped and the function returns
            ``(policy, None, None)``.
        config_overrides: Optional dict of config fields to override after loading
            (e.g. {"device": "cuda:0", "use_amp": True, "compile_sample_actions": True}).

    Returns:
        A tuple of (policy, input_transforms, output_transforms) where:
        - policy: The loaded PreTrainedPolicy ready for inference.
        - input_transforms: Callable to apply to raw observations before
            passing them to the policy, or ``None`` when *data_config_name* is ``None``.
        - output_transforms: Callable to apply to the policy's raw output
            to get denormalized actions, or ``None`` when *data_config_name* is ``None``.

    Example::

        from lerobot.common.policies.factory import load_pretrained_policy

        # From HuggingFace Hub
        policy, input_transforms, output_transforms = load_pretrained_policy(
            "SberRoboticsCenter/GreenVLA-5b-corrupted",
            data_config_name="bridge",
        )

        # From a local checkpoint
        policy, input_transforms, output_transforms = load_pretrained_policy(
            "/path/to/checkpoints/300000",
            data_config_name="bridge",
        )
    """
    model_id = str(pretrained_name_or_path)

    # 1. Load config
    cfg = PreTrainedConfig.from_pretrained(model_id)

    # 2. Apply user-provided overrides
    if config_overrides:
        for key, value in config_overrides.items():
            if not hasattr(cfg, key):
                logging.warning(f"Config has no attribute '{key}', skipping override.")
                continue
            setattr(cfg, key, value)

    # 3. Set pretrained path and load the policy
    cfg.pretrained_path = model_id
    policy = make_policy(cfg)

    # 4. If no data config is requested, return the bare policy without transforms.
    if data_config_name is None:
        return policy, None, None

    # 5. Resolve norm stats (downloads from Hub if needed)
    assets_path = _resolve_norm_stats_dir(model_id, data_config_name)

    # 6. Build data config factory
    config_dir = get_config_root()
    data_factory_config_yaml_path = (
        config_dir / "robotics_dataset" / "individual" / f"{data_config_name}.yaml"
    )
    if not data_factory_config_yaml_path.is_file():
        raise FileNotFoundError(
            f"Data config not found at {data_factory_config_yaml_path}. "
            f"Available configs are in {config_dir / 'robotics_dataset' / 'individual'}/"
        )
    data_factory_config = OmegaConf.load(data_factory_config_yaml_path)
    data_factory = instantiate(data_factory_config)

    # 7. Build input/output transforms
    input_transforms = compose(get_torch_input_transforms(
        policy_config=cfg,
        data_config_factory=data_factory,
        assets_dirs=assets_path,
        normalization_mode=cfg.normalization_mode,
    ))
    output_transforms = compose(get_torch_output_transforms(
        policy_config=cfg,
        data_config_factory=data_factory,
        assets_dirs=assets_path,
        normalization_mode=cfg.normalization_mode,
    ))

    return policy, input_transforms, output_transforms
