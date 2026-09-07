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

"""Instantiate the GreenVLAv1.1 whole-body policy from a checkpoint.

This fork serves one policy family, so the registry other LeRobot forks keep here is a
single implementation: GreenVLAv1.1. Legacy checkpoint identifiers remain accepted.
"""

from torch import nn

from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.configs.policies import PreTrainedConfig

from lerobot.common.policies.greenvla_v1_1.identity import POLICY_TYPE, POLICY_ALIASES


def get_policy_class(name: str) -> type[PreTrainedPolicy]:
    """Get the policy's class given a name (matching the policy class' `name` attribute)."""
    if name not in POLICY_ALIASES:
        raise NotImplementedError(f"Policy with name {name} is not implemented.")
    from lerobot.common.policies.greenvla_v1_1.modeling_greenvla_v1_1 import GreenVLAv11Policy

    return GreenVLAv11Policy




def make_policy(cfg: PreTrainedConfig, ds_meta=None, env_cfg=None) -> PreTrainedPolicy:
    """Make an instance of a policy class.

    Args:
        cfg: The policy config read from the checkpoint. `pretrained_path` must be set --
            this fork loads trained checkpoints and never builds a policy from scratch.
        ds_meta: Unused; kept so callers written against the upstream signature still work.
        env_cfg: Unused, same reason.

    Returns:
        The policy, moved to `cfg.device`.
    """
    policy_cls = get_policy_class(cfg.type)

    assert cfg.pretrained_path, "You are instantiating a policy from scratch. Set cfg.pretrained_path to the model config"

    policy = policy_cls.from_pretrained(config=cfg, pretrained_name_or_path=cfg.pretrained_path)

    policy.to(cfg.device)
    assert isinstance(policy, nn.Module)

    return policy
