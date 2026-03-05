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
class RobocoinAirbotInputsTransform(DataTransformFn):
    max_action_dim: int
    action_space: str = "joint"
    head_image_keys: Union[None, List[str]] = None

    map_to_unified_space: bool = False
    map_to_humanoid: bool = False
    # Masks as lists
    left_arm_joints_mask = [0, 1, 2, 3, 4, 5]  # 6 joints
    right_arm_joints_mask = [6, 7, 8, 9, 10, 11]  # 6 joints
    left_gripper_joints_mask = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]  # 12 joints
    right_gripper_joints_mask = [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]  # 12 joints

    def __post_init__(self):
        state_mask = [
                *self.left_arm_joints_mask,
                *self.left_gripper_joints_mask,
                *self.right_arm_joints_mask,
                *self.right_gripper_joints_mask,
            ]
        # Use object.__setattr__ to bypass frozen restriction
        object.__setattr__(self, "state_mask", state_mask)
        object.__setattr__(self, "action_dim", len(state_mask))

    def __call__(self, data: dict) -> dict:
        state = data["observation/state"][self.state_mask]
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()

        state = pad_to_dim(
            state, self.max_action_dim, value=0.0
        )  # pad with float if state is float

        if self.head_image_keys is not None:
            choosen_head_image_key = np.random.choice(self.head_image_keys)
            head_image = parse_image_helper(data[choosen_head_image_key])
        else:
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
class RobocoinAirbotOutputsTransform(DataTransformFn):
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
