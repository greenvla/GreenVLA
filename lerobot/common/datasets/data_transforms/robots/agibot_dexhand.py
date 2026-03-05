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
class AgibotDexHandInputsTransform(DataTransformFn):
    action_dim: int
    map_to_unified_space: bool = False
    map_to_humanoid: bool = True
    
    mapping_for_unified_space = [
        (list(range(0, 7)), list(range(0, 7))),        # left arm joints
        (list(range(7, 13)), list(range(7, 13))),      # left gripper joints
        (list(range(13, 20)), list(range(14, 21))),    # right arm joints
        (list(range(20, 26)), list(range(21, 27))),    # right gripper joints
        (list(range(26, 28)), [31, 29]),    # torso
        (list(range(28, 30)), list(range(32, 34))),    # head
    ]
    # specifying slices for arms
    left_arm_joints_state_indices = [*range(42, 49)]
    right_arm_joints_state_indices = [*range(49, 56)]
    left_arm_gripper_state_indices = [*range(0, 6)]
    right_arm_gripper_state_indices = [*range(6, 12)]
    torso_state_indices = [*range(63, 65)]
    head_state_indices = [*range(26, 28)]

    state_mask = [
        *left_arm_joints_state_indices,
        *left_arm_gripper_state_indices,
        *right_arm_joints_state_indices,
        *right_arm_gripper_state_indices,
        *torso_state_indices,
        *head_state_indices,
    ]

    left_arm_joints_action_indices = [*range(28, 35)]
    right_arm_joints_action_indices = [*range(35, 42)]
    left_arm_gripper_action_indices = [*range(0, 6)]
    right_arm_gripper_action_indices = [*range(6, 12)]
    torso_action_indices = [*range(44, 46)]
    head_action_indices = [*range(26, 28)]

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
        
        x[:, 7] = -x[:, 7] + 0.3
        x[:, 8] = x[:, 8] + 0.3
        x[:, 9] = x[:, 9] + 0.3
        x[:, 10] = x[:, 10] + 0.3
        x[:, 11] = x[:, 11] + 0.3
        x[:, 12] = -x[:, 12]
        
        x[:, 7:13] = x[:, 7:13][:, [4, 3, 2, 1, 0, 5]]
        
        
        
        x[:, 13] = -x[:, 13]
        x[:, 14] =  (torch.pi/2) - x[:, 14]
        x[:, 15] = -x[:, 15]
        x[:, 16] = -x[:, 16]
        x[:, 17] = -x[:, 17]
        
        x[:, 20] = -x[:, 20] + 0.3
        x[:, 21] = x[:, 21] + 0.3
        x[:, 22] = x[:, 22] + 0.3
        x[:, 23] = x[:, 23] + 0.3
        x[:, 24] = x[:, 24] + 0.3
        x[:, 25] = x[:, 25] + 0.3
        x[:, 26] = -x[:, 26]
        
        x[:, 20:26] = x[:, 20:26][:, [4, 3, 2, 1, 0, 5]]
        
        #x[:, 29], x[:, 31] = x[:, 31], x[:, 29]
        
        
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
            # Assuming actions are [horizon, dim], pad the dim part
            
            if self.map_to_humanoid:
                actions = self.transform_to_humanoid(actions)
                
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
class AgibotDexHandOutputsTransform(DataTransformFn):
    def __call__(self, data: dict) -> dict:
        actions = data["actions"]
        if isinstance(actions, np.ndarray):
            if actions.ndim==2:
            # If it's already numpy, just slice
                return {"actions": actions[:, :30]}
            elif actions.ndim==3:
                return {"actions": actions[:, :, :30]}
            else:
                raise ValueError(f"Unsupported action shape: {actions.shape}")
        elif isinstance(actions, torch.Tensor):
            if actions.ndim==2:
                return {"actions": actions[:, :30].cpu().numpy()}
            elif actions.ndim==3:
                return {"actions": actions[:, :, :30].cpu().numpy()}
            else:
                raise ValueError(f"Unsupported action shape: {actions.shape}")
            # If it's a torch tensor, slice and then decide if to convert to numpy or keep as tensor
            # Based on LeRobot conventions, policy output might be expected as numpy
            
        else:
            raise TypeError(f"Unsupported action type: {type(actions)}")
