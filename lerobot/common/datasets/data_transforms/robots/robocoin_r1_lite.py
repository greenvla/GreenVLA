import dataclasses
import numpy as np
import torch
from typing import List, Union
from lerobot.common.datasets.torch_transforms import (
    DataTransformFn,
    pad_to_dim,
    parse_image_helper,
    BaseModelConfigPlaceholder as ModelConfig,  # Use the placeholder
)


@dataclasses.dataclass(frozen=True)
class RobocoinR1LiteInputsTransform(DataTransformFn):
    max_action_dim: int
    action_space: str = "joint"

    map_to_unified_space: bool = False
    map_to_humanoid: bool = False
    # Masks as lists
    left_arm_joints_mask = [0, 1, 2, 3, 4, 5]  # 6 joints
    right_arm_joints_mask = [7, 8, 9, 10, 11, 12]  # 6 joints
    left_gripper_mask = [6]  # 1 gripper
    right_gripper_mask = [13]  # 1 gripper

    def __post_init__(self):
        state_mask = [
            *self.left_arm_joints_mask,
            *self.left_gripper_mask,
            *self.right_arm_joints_mask,
            *self.right_gripper_mask,
        ]
        object.__setattr__(self, "state_mask", state_mask)
        object.__setattr__(self, "action_dim", len(state_mask))

    def __call__(self, data: dict) -> dict:
        state = data["observation/state"][self.state_mask]
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()

        state = pad_to_dim(
            state, self.max_action_dim, value=0.0
        )  # pad with float if state is float
        
        head_image = parse_image_helper(data["observation/head_image"])
        left_wrist_image = parse_image_helper(data["observation/left_wrist_image"])
        right_wrist_image = parse_image_helper(data["observation/right_wrist_image"])

        # Ensure images are torch tensors if not already
        if isinstance(head_image, np.ndarray):
            head_image = torch.from_numpy(head_image).permute(
                2, 0, 1
            )  # HWC to CHW for torch
            left_wrist_image = torch.from_numpy(left_wrist_image).permute(2, 0, 1)
            right_wrist_image = torch.from_numpy(right_wrist_image).permute(2, 0, 1)
        elif (
            head_image.ndim == 3 and head_image.shape[2] == 3
        ):  # HWC torch tensor to CHW
            head_image = head_image.permute(2, 0, 1)
            left_wrist_image = left_wrist_image.permute(2, 0, 1)
            right_wrist_image = right_wrist_image.permute(2, 0, 1)

        inputs = {
            "state": state,
            "image": {
                # Key names adapted from OpenPI's CobotMagicInputs
                # LeRobot might use different keys internally or expect a flatter structure.
                "base_0_rgb": head_image,  # CHW
                "left_wrist_0_rgb": left_wrist_image,  # CHW
                "right_wrist_0_rgb": right_wrist_image,  # CHW
            },
            "image_mask": {
                "base_0_rgb": torch.tensor(True),
                "left_wrist_0_rgb": torch.tensor(True),
                "right_wrist_0_rgb": torch.tensor(True),
            },
        }

        if "actions" in data:
            actions = data["actions"][:, self.state_mask]
            if isinstance(actions, np.ndarray):
                actions = torch.from_numpy(actions).float()
            # Assuming actions are [horizon, dim], pad the dim part
            inputs["actions"] = pad_to_dim(
                actions, self.max_action_dim, axis=-1, value=0.0
            )

        # Handle both "prompt" and "task" keys for backward compatibility
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        elif "task" in data:
            inputs["prompt"] = data["task"]
        
        if "subtask" in data:
            inputs["subtask"] = data["subtask"]
        if "next_subtask" in data:
            inputs["next_subtask"] = data["next_subtask"]
        if "is_subtask_transition" in data:
            inputs["is_subtask_transition"] = data["is_subtask_transition"]
        if self.map_to_unified_space:
            inputs["mapping_for_unified_space"] = self.mapping_for_unified_space

        return inputs


@dataclasses.dataclass(frozen=True)
class RobocoinR1LiteOutputsTransform(DataTransformFn):
    action_dim: int
    
    def __call__(self, data: dict) -> dict:
        actions = data["actions"]
        if isinstance(actions, np.ndarray):
            if actions.ndim == 2:
                # If it's already numpy, just slice
                return {"actions": actions[:, : self.action_dim]}
            elif actions.ndim == 3:
                return {"actions": actions[:, :, : self.action_dim]}
            else:
                raise ValueError(f"Unsupported action shape: {actions.shape}")
        elif isinstance(actions, torch.Tensor):
            # If it's a torch tensor, slice and then decide if to convert to numpy or keep as tensor
            # Based on LeRobot conventions, policy output might be expected as numpy
            if actions.ndim == 2:
                return {"actions": actions[:, : self.action_dim].cpu().numpy()}
            elif actions.ndim == 3:
                return {"actions": actions[:, :, : self.action_dim].cpu().numpy()}
            else:
                raise ValueError(f"Unsupported action shape: {actions.shape}")
        else:
            raise TypeError(f"Unsupported action type: {type(actions)}")
