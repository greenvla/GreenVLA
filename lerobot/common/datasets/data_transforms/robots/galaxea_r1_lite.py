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
class GalaxeaR1LiteInputsTransform(DataTransformFn):
    action_dim: int
    
    map_to_unified_space: bool = True
    
    mapping_for_unified_space = [
        (list(range(0, 6)), list(range(1, 7))),        # left arm joints
        (list(range(6, 7)), list(range(13, 14))),      # left gripper joints
        (list(range(7, 13)), list(range(15, 21))),    # right arm joints
        (list(range(13, 14)), list(range(27, 28))),    # right gripper joints
        (list(range(14, 17)), list(range(28, 31))),    # torso
        (list(range(17, 20)), list(range(32, 35))),    # head
        (list(range(20, 23)), list(range(45, 48))),    # head
    ]
    
    left_arm_joints_state_indices = [*range(0, 6)]
    left_gripper_state_indices = [12]
    right_arm_joints_state_indices = [*range(13, 19)]
    right_gripper_state_indices = [25]
    torso_state_indices = [*range(26,29)] #the last one torso state is not changed in dataset

    state_mask = [
        *left_arm_joints_state_indices,
        *left_gripper_state_indices,
        *right_arm_joints_state_indices,
        *right_gripper_state_indices,
        *torso_state_indices
    ]

    def __call__(self, data: dict) -> dict:
        if data['observation/state'].shape[0] == 33:
            state = data["observation/state"][self.state_mask]
        else:
            raise ValueError(f"State shape is {data['observation/state'].shape}, expected 33")
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()

        state = pad_to_dim(state, self.action_dim, value=0.0)
        
        base_image = parse_image_helper(data["observation/head_image"])
        left_wrist_image = parse_image_helper(data["observation/left_wrist_image"])
        right_wrist_image = parse_image_helper(data["observation/right_wrist_image"])
        
                # Ensure images are torch tensors if not already
        if isinstance(base_image, np.ndarray):
            base_image = torch.from_numpy(base_image).permute(
                2, 0, 1
            )  # HWC to CHW for torch
            left_wrist_image = torch.from_numpy(left_wrist_image).permute(2, 0, 1)
            right_wrist_image = torch.from_numpy(right_wrist_image).permute(2, 0, 1)
        elif (
            base_image.ndim == 3 and base_image.shape[2] == 3
        ):  # HWC torch tensor to CHW
            base_image = base_image.permute(2, 0, 1)
            left_wrist_image = left_wrist_image.permute(2, 0, 1)
            right_wrist_image = right_wrist_image.permute(2, 0, 1)

        inputs = {
            "state": state,
            "image": {
                # Key names adapted from OpenPI's CobotMagicInputs
                # LeRobot might use different keys internally or expect a flatter structure.
                "base_0_rgb": base_image,  # CHW
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
            if data['actions'].shape[1] == 33:
                actions = data["actions"][:, self.state_mask]
            else:
                actions = data["actions"]
            if isinstance(actions, np.ndarray):
                actions = torch.from_numpy(actions).float()
            inputs["actions"] = pad_to_dim(actions, self.action_dim, axis=-1, value=0.0)
            
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
            
        if self.map_to_unified_space:
            inputs["mapping_for_unified_space"] = self.mapping_for_unified_space
            
        return inputs
    
    


@dataclasses.dataclass(frozen=True)
class GalaxeaR1LiteOutputsTransform(DataTransformFn):
    def __call__(self, data: dict) -> dict:
        actions = data["actions"]
        if isinstance(actions, np.ndarray):
            # If it's already numpy, just slice
            if actions.ndim==2:
                return {"actions": actions[:, :20]}
            elif actions.ndim==3:
                return {"actions": actions[:, :, :20]}
            else:
                raise ValueError(f"Unsupported action shape: {actions.shape}")
        elif isinstance(actions, torch.Tensor):
            # If it's a t   orch tensor, slice and then decide if to convert to numpy or keep as tensor
            # Based on LeRobot conventions, policy output might be expected as numpy
            if actions.ndim==2:
                return {"actions": actions[:, :20].cpu().numpy()}
            elif actions.ndim==3:
                return {"actions": actions[:, :, :20].cpu().numpy()}
            else:
                raise ValueError(f"Unsupported action shape: {actions.shape}")
        else:
            raise TypeError(f"Unsupported action type: {type(actions)}")