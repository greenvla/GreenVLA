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
class FourierGR1InputsTransform(DataTransformFn):
    action_dim: int

    map_to_unified_space: bool = True
    map_to_humanoid: bool = True
    
    mapping_for_unified_space = [
        (list(range(0, 7)), list(range(0, 7))),        # left arm joints
        (list(range(7, 13)), list(range(7, 13))),      # left gripper joints
        (list(range(13, 20)), list(range(14, 21))),    # right arm joints
        (list(range(20, 26)), list(range(21, 27))),    # right gripper joints
    ]
    
    # specifying slices for arms
    left_arm_joints_state_indices  = [*range(6, 13)]
    right_arm_joints_state_indices = [*range(13, 20)]
    left_arm_gripper_state_indices = [*range(20,26)]
    right_arm_gripper_state_indices = [*range(26,32)]
    torso_state_indices = [*range(0,3)] # torso and head not used because they do not change and can cause problems in normalization
    head_state_indices = [*range(3,6)]

    state_mask = [
        *left_arm_joints_state_indices,
        *left_arm_gripper_state_indices,
        *right_arm_joints_state_indices,
        *right_arm_gripper_state_indices,
    ]

    def transform_to_humanoid(self, x: torch.Tensor | np.ndarray):
        need_to_squeeze = False
        if x.ndim == 1:
            need_to_squeeze = True
            x = x.unsqueeze(0)
        x[:,5] = -x[:,5]
        
        x[:, 7] = x[:, 7] * 0.14 - 0.1
        x[:, 8] = (x[:, 8]- 2.0) * 0.04
        x[:, 9] = x[:, 9] * 0.17
        x[:, 10] = x[:, 10] * 0.17
        x[:, 11] = x[:, 11] * 0.17
        x[:, 12] = x[:, 12] * 0.17

        x[:, 18] = -x[:, 18]
        
        x[:, 20] = x[:, 20] * 0.14 - 0.1
        x[:, 21] = (x[:, 21]- 2.0) * 0.04
        x[:, 22] = x[:, 22] * 0.17
        x[:, 23] = x[:, 23] * 0.17
        x[:, 24] = x[:, 24] * 0.17
        x[:, 25] = x[:, 25] * 0.17
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

        # Ensure images are torch tensors if not already
        if isinstance(head_image, np.ndarray):
            head_image = torch.from_numpy(head_image).permute(
                2, 0, 1
            )  # HWC to CHW for torch
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
                "left_wrist_0_rgb":  torch.zeros_like(head_image).to(head_image.dtype),  # CHW
                "right_wrist_0_rgb": torch.zeros_like(head_image).to(head_image.dtype),  # CHW
            },
            "image_mask": {
                "base_0_rgb": torch.tensor(True),
                "left_wrist_0_rgb": torch.tensor(False),
                "right_wrist_0_rgb": torch.tensor(False),
            },
        }

        if "actions" in data:
            actions = data["actions"][:, self.state_mask]
            
                
            if isinstance(actions, np.ndarray):
                actions = torch.from_numpy(actions).float()
                
            if self.map_to_humanoid:
                actions = self.transform_to_humanoid(actions)
            # Assuming actions are [horizon, dim], pad the dim part
            inputs["actions"] = pad_to_dim(actions, self.action_dim, axis=-1, value=0.0)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        if self.map_to_unified_space:
            inputs["mapping_for_unified_space"] = self.mapping_for_unified_space
            
        return inputs


@dataclasses.dataclass(frozen=True)
class FourierGR1OutputsTransform(DataTransformFn):
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
