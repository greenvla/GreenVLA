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
class AgibotTwoFingerInputsTransform(DataTransformFn):
    action_dim: int
    map_to_unified_space: bool = True
    map_to_humanoid: bool = True
    mapping_for_unified_space = [
        (list(range(0, 7)), list(range(0, 7))),        # left arm joints
        (list(range(7, 8)), list(range(13, 14))),      # left gripper joints
        (list(range(8, 15)), list(range(14, 21))),    # right arm joints
        (list(range(15, 16)), list(range(27, 28))),    # right gripper joints
        (list(range(16, 18)), [31, 29]),    # torso
        (list(range(18, 20)), list(range(32, 34))),    # head
    ]
    
    # specifying slices for arms
    left_arm_joints_state_indices = [*range(32, 39)]
    right_arm_joints_state_indices = [*range(39, 46)]
    left_arm_gripper_state_indices = [0]
    right_arm_gripper_state_indices = [1]
    torso_state_indices = [*range(53, 55)]
    head_state_indices = [*range(16, 18)]

    state_mask = [
        *left_arm_joints_state_indices,
        *left_arm_gripper_state_indices,
        *right_arm_joints_state_indices,
        *right_arm_gripper_state_indices,
        *torso_state_indices,
        *head_state_indices,
    ]

    left_arm_joints_action_indices = [*range(18, 25)]
    right_arm_joints_action_indices = [*range(25, 32)]
    left_arm_gripper_action_indices = [0]
    right_arm_gripper_action_indices = [1]
    torso_action_indices = [*range(34, 36)]
    head_action_indices = [*range(16, 18)]

    action_mask = [
        *left_arm_joints_action_indices,
        *left_arm_gripper_action_indices,
        *right_arm_joints_action_indices,
        *right_arm_gripper_action_indices,
        *torso_action_indices,
        *head_action_indices,
    ]

    def transform_to_humanoid(self, x: torch.Tensor | np.ndarray):
        need_to_squeeze = False
        if x.ndim == 1:
            need_to_squeeze = True
            x = x.unsqueeze(0)
        
        x[:,1] = (torch.pi/2)- x[:,1]
        x[:,2] = -x[:,2]
        x[:, 4] = -x[:, 4]
        
        x[:, 8] = -x[:, 8]
        x[:, 9] =  (torch.pi/2) - x[:, 9]
        x[:, 10] = -x[:, 10]
        x[:, 11] = -x[:, 11]
        x[:, 12] = -x[:, 12]
                
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
            if self.map_to_humanoid:
                actions = self.transform_to_humanoid(actions)
            # Assuming actions are [horizon, dim], pad the dim part
            inputs["actions"] = pad_to_dim(actions, self.action_dim, axis=-1, value=0.0)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
            
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
class AgibotTwoFingerOutputsTransform(DataTransformFn):
    def __call__(self, data: dict) -> dict:
        actions = data["actions"]
        if isinstance(actions, np.ndarray):
            if actions.ndim==2:
            # If it's already numpy, just slice
                return {"actions": actions[:, :20]}
            elif actions.ndim==3:
                return {"actions": actions[:, :, :20]}
            else:
                raise ValueError(f"Unsupported action shape: {actions.shape}")
        elif isinstance(actions, torch.Tensor):
            # If it's a torch tensor, slice and then decide if to convert to numpy or keep as tensor
            # Based on LeRobot conventions, policy output might be expected as numpy
            if actions.ndim==2:
                return {"actions": actions[:, :20].cpu().numpy()}
            elif actions.ndim==3:
                return {"actions": actions[:, :, :20].cpu().numpy()}
            else:
                raise ValueError(f"Unsupported action shape: {actions.shape}")
        else:
            raise TypeError(f"Unsupported action type: {type(actions)}")
