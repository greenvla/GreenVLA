import dataclasses
import numpy as np
import torch
from lerobot.common.datasets.torch_transforms import (
    DataTransformFn,
    pad_to_dim,
    parse_image_helper,
    BaseModelConfigPlaceholder as ModelConfig,  # Use the placeholder
)


@dataclasses.dataclass(frozen=True)
class CentaurDexHandInputsTransform(DataTransformFn):
    action_dim: int

    # specifying slices for arms
    left_arm_joints_state_indices   = [*range(0, 7)] 
    right_arm_joints_state_indices  = [*range(7, 14)]
    left_arm_gripper_state_indices  = [*range(16,22)]
    right_arm_gripper_state_indices = [*range(22,28)]
    

    state_mask = [
        *left_arm_joints_state_indices,
        *left_arm_gripper_state_indices,
        *right_arm_joints_state_indices,
        *right_arm_gripper_state_indices,
    ]

    left_arm_joints_action_indices   = [*range(0, 7)] 
    right_arm_joints_action_indices  = [*range(7, 14)]
    left_arm_gripper_action_indices  = [*range(16,22)]
    right_arm_gripper_action_indices = [*range(22,28)]

    action_mask = [
        *left_arm_joints_action_indices,
        *left_arm_gripper_action_indices,
        *right_arm_joints_action_indices,
        *right_arm_gripper_action_indices,
    ]

    def __call__(self, data: dict) -> dict:
        state = data["observation/state"][self.state_mask]
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()

        state = pad_to_dim(
            state, self.action_dim, value=0.0
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
            actions = data["actions"][:, self.action_mask]
            if isinstance(actions, np.ndarray):
                actions = torch.from_numpy(actions).float()
            # Assuming actions are [horizon, dim], pad the dim part
            inputs["actions"] = pad_to_dim(actions, self.action_dim, axis=-1, value=0.0)

        # Handle both "prompt" and "task" keys for backward compatibility
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        elif "task" in data:
            inputs["prompt"] = data["task"]

        return inputs


@dataclasses.dataclass(frozen=True)
class CentaurDexHandOutputsTransform(DataTransformFn):
    def __call__(self, data: dict) -> dict:
        actions = data["actions"]
        if isinstance(actions, np.ndarray):
            # If it's already numpy, just slice
            if actions.ndim==2:
                return {"actions": actions[:, :26]}
            elif actions.ndim==3:
                return {"actions": actions[:, :, :26]}
            else:
                raise ValueError(f"Unsupported action shape: {actions.shape}")
        elif isinstance(actions, torch.Tensor):
            # If it's a torch tensor, slice and then decide if to convert to numpy or keep as tensor
            # Based on LeRobot conventions, policy output might be expected as numpy
            if actions.ndim==2:
                return {"actions": actions[:, :26].cpu().numpy()}
            elif actions.ndim==3:
                return {"actions": actions[:, :, :26].cpu().numpy()}
            else:
                raise ValueError(f"Unsupported action shape: {actions.shape}")
        else:
            raise TypeError(f"Unsupported action type: {type(actions)}")
