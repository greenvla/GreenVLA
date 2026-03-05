import dataclasses
import numpy as np
import torch
from lerobot.common.datasets.torch_transforms import (
    DataTransformFn,
    pad_to_dim,
    parse_image_helper,
    BaseModelConfigPlaceholder as ModelConfig,  # Use the placeholder
)

# TRANSFORMS FOR COBOT MAGIC ALIKE - PyTorch Version


@dataclasses.dataclass(frozen=True)
class CobotMagicInputsTransform(DataTransformFn):
    action_dim: int
    map_to_unified_space: bool = True
    
    mapping_for_unified_space = [
        (list(range(0, 6)), list(range(1, 7))),        # left arm joints
        (list(range(6, 7)), list(range(13, 14))),      # left gripper joints
        (list(range(7, 13)), list(range(15, 21))),    # right arm joints
        (list(range(13, 14)), list(range(27, 28))),    # right gripper joints
    ]
    
    def __call__(self, data: dict) -> dict:
        state = data["observation/state"]
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        state_padded = pad_to_dim(
            state, self.action_dim, value=0.0
        )  # pad with float if state is float

        base_image = parse_image_helper(data["observation/base_image"])
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
            "state": state_padded,
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
            actions = data["actions"]
            if isinstance(actions, np.ndarray):
                actions = torch.from_numpy(actions).float()
            # Assuming actions are [horizon, dim], pad the dim part
            inputs["actions"] = pad_to_dim(
                actions[:, :14], self.action_dim, axis=-1, value=0.0
            )
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
class CobotMagicOutputsTransform(DataTransformFn):
    def __call__(self, data: dict) -> dict:
        actions = data.pop("actions")
        if isinstance(actions, np.ndarray):
            if actions.ndim == 2:
                return data | {"actions": actions[:, :14]}
            elif actions.ndim == 3:
                return data | {"actions": actions[:, :, :14]}
            else:
                raise ValueError(f"Unsupported action shape: {actions.shape}")
        elif isinstance(actions, torch.Tensor):
            if actions.ndim == 2:
                return data | {"actions": actions[:, :14].cpu().numpy()}
            elif actions.ndim == 3:
                return data | {"actions": actions[:, :, :14].cpu().numpy()}
            else:
                raise ValueError(f"Unsupported action shape: {actions.shape}")
        else:
            raise TypeError(f"Unsupported action type: {type(actions)}")
