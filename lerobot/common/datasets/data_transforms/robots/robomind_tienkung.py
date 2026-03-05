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
class RobomindTienkungInputsTransform(DataTransformFn):
    action_dim: int

    # specifying slices for arms
    left_arm_joints_state_indices   = [*range(0, 7)] 
    left_arm_gripper_state_indices  = [*range(7, 13)]
    right_arm_joints_state_indices  = [*range(13, 20)]
    right_arm_gripper_state_indices = [*range(20, 26)]
    
    map_to_unified_space: bool = True
    map_to_humanoid: bool = True
    
    mapping_for_unified_space = [
        (list(range(0, 7)), list(range(0, 7))),        # left arm joints
        (list(range(7, 13)), list(range(7, 13))),      # left gripper joints
        (list(range(13, 20)), list(range(14, 21))),    # right arm joints
        (list(range(20, 26)), list(range(21, 27))),    # right gripper joints
    ]
    
    state_mask = [
        *left_arm_joints_state_indices,
        *left_arm_gripper_state_indices,
        *right_arm_joints_state_indices,
        *right_arm_gripper_state_indices,
    ]

    left_arm_joints_action_indices   = [*range(0, 7)] 
    left_arm_gripper_action_indices  = [*range(7, 13)]
    right_arm_joints_action_indices  = [*range(13, 20)]
    right_arm_gripper_action_indices = [*range(20, 26)]

    action_mask = [
        *left_arm_joints_action_indices,
        *left_arm_gripper_action_indices,
        *right_arm_joints_action_indices,
        *right_arm_gripper_action_indices,
    ]

    def transform_to_humanoid(self, x: torch.Tensor | np.ndarray):
        need_to_squeeze = False
        if x.ndim == 1:
            need_to_squeeze = True
            x = x.unsqueeze(0)
        x[:, 1] = (-torch.pi/2) - x[:, 1]
        x[:, 4] = (-torch.pi/2) - x[:, 4]
        x[:, 6] = -x[:, 6]
        
        x[:, 7] = (1-x[:, 7]) * 1.4 - 0.1
        x[:, 8] = (1-x[:, 8]) * 0.5
        x[:, 9] = (1-x[:, 9]) * 1.7
        x[:, 10] = (1-x[:, 10]) * 1.7
        x[:, 11] = (1-x[:, 11]) * 1.7
        x[:, 12] = (1-x[:, 12]) * 1.7

        x[:, 13] = -x[:, 13]
        x[:, 15] = (-torch.pi/2) - x[:, 15]
        x[:, 16] = -x[:, 16]
        x[:, 17] = (-torch.pi/2) - x[:, 17]
        
        
        x[:, 20] = (1-x[:, 20]) * 1.4 - 0.1
        x[:, 21] = (1-x[:, 21]) * 0.5
        x[:, 22] = (1-x[:, 22]) * 1.7
        x[:, 23] = (1-x[:, 23]) * 1.7
        x[:, 24] = (1-x[:, 24]) * 1.7
        x[:, 25] = (1-x[:, 25]) * 1.7
        if need_to_squeeze:
            x = x.squeeze(0)
        return x
    
    def __call__(self, data: dict) -> dict:
        state = data["observation/state"][self.state_mask]
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        if self.map_to_humanoid:
            state = self.transform_to_humanoid(state)
        state = pad_to_dim(
            state, self.action_dim, value=0.0
        )  # pad with float if state is float

        head_image = parse_image_helper(data["observation/head_image"])
        left_wrist_image = torch.zeros_like(head_image).to(head_image.dtype).permute(2, 0, 1)
        right_wrist_image = torch.zeros_like(head_image).to(head_image.dtype).permute(2, 0, 1)

        # Ensure images are torch tensors if not already
        if isinstance(head_image, np.ndarray):
            head_image = torch.from_numpy(head_image).permute(2, 0, 1)  # HWC to CHW for torch
        elif (
            head_image.ndim == 3 and head_image.shape[2] == 3
        ):  # HWC torch tensor to CHW
            head_image = head_image.permute(2, 0, 1)

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
                "left_wrist_0_rgb": torch.tensor(False),
                "right_wrist_0_rgb": torch.tensor(False),
            },
        }

        if "actions" in data:
            actions = data["actions"][:, self.action_mask]
            if isinstance(actions, np.ndarray):
                actions = torch.from_numpy(actions).float()
            if self.map_to_humanoid:
                actions = self.transform_to_humanoid(actions)
            # Assuming actions are [horizon, dim], pad the dim part
            inputs["actions"] = pad_to_dim(actions, self.action_dim, axis=-1, value=0.0)

        # Handle both "prompt" and "task" keys for backward compatibility
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        elif "task" in data:
            inputs["prompt"] = data["task"]
            
        if self.map_to_unified_space:
            inputs["mapping_for_unified_space"] = self.mapping_for_unified_space

        return inputs


@dataclasses.dataclass(frozen=True)
class RobomindTienkungOutputsTransform(DataTransformFn):
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
