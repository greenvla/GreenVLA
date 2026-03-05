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
class FractalInputsTransform(DataTransformFn):
    action_dim: int
    
    map_to_unified_space: bool = True
    mapping_for_unified_space_actions = [
        (list(range(0, 3)), list(range(35, 38))),        
        (list(range(3, 6)), list(range(42, 45))),     
        (list(range(6, 7)), list(range(13, 14))),
    ]
    
    mapping_for_unified_space_state = [
        (list(range(0, 3)), list(range(35, 38))),        
        (list(range(3, 7)), list(range(38, 42))),     
        (list(range(7, 8)), list(range(13, 14))),
    ]
    
    def __call__(self, data: dict) -> dict:
        state = data["observation/state"]
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        state = pad_to_dim(state, self.action_dim, value=0.0)
        
        base_image = parse_image_helper(data["observation/image"])
        
        if isinstance(base_image, np.ndarray):
            base_image = torch.from_numpy(base_image).permute(
                2, 0, 1
            )  # HWC to CHW for torch
        elif (
            base_image.ndim == 3 and base_image.shape[2] == 3
        ):  # HWC torch tensor to CHW
            base_image = base_image.permute(2, 0, 1)
            
        
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": torch.zeros_like(base_image).to(base_image.dtype),
                "right_wrist_0_rgb": torch.zeros_like(base_image).to(base_image.dtype),
            },
            "image_mask": {
                "base_0_rgb": torch.tensor(True),
                "left_wrist_0_rgb": torch.tensor(False),
                "right_wrist_0_rgb": torch.tensor(False),
            },
        }
        
        if "actions" in data:
            actions = data["actions"]
            if isinstance(actions, np.ndarray):
                actions = torch.from_numpy(actions).float()
            inputs["actions"] = pad_to_dim(actions, self.action_dim, axis=-1, value=0.0)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
            
        if self.map_to_unified_space:
            inputs["mapping_for_unified_space_actions"] = self.mapping_for_unified_space_actions
            inputs["mapping_for_unified_space_state"] = self.mapping_for_unified_space_state
        return inputs
    

@dataclasses.dataclass(frozen=True)
class FractalOutputsTransform(DataTransformFn):
    def __call__(self, data: dict) -> dict:
        actions = data.pop("actions")
        if isinstance(actions, np.ndarray):
            if actions.ndim == 2:
                return data | {"actions": actions[:, :7]}
            elif actions.ndim == 3:
                return data | {"actions": actions[:, :, :7]}
            else:
                raise ValueError(f"Unsupported action shape: {actions.shape}")
        elif isinstance(actions, torch.Tensor):
            if actions.ndim == 2:
                return data | {"actions": actions[:, :7].cpu().numpy()}
            elif actions.ndim == 3:
                return data | {"actions": actions[:, :, :7].cpu().numpy()}
            else:
                raise ValueError(f"Unsupported action shape: {actions.shape}")
        else:
            raise TypeError(f"Unsupported action type: {type(actions)}")