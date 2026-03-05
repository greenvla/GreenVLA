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
class RobosetInputsTransform(DataTransformFn):
    action_dim: int
    state_mask = [i for i in range(8)] # 7 joints + gripper
    possible_base_img_keys = ["observation/cam_left", "observation/cam_right", "observation/cam_top"]
    map_to_unified_space: bool = True
    
    mapping_for_unified_space = [
        (list(range(0, 7)), list(range(0, 7))),        
        (list(range(7, 8)), list(range(13, 14))),     
    ]
    
    def __call__(self, data: dict) -> dict:
        state = data["observation/state"]
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        state = pad_to_dim(state[self.state_mask], self.action_dim, value=0.0)
        
        base_image_key = np.random.choice(self.possible_base_img_keys)
        base_image = parse_image_helper(data[base_image_key])
        left_wrist_image = parse_image_helper(data["observation/cam_wrist"])
        if isinstance(base_image, np.ndarray):
            base_image = torch.from_numpy(base_image).permute(
                2, 0, 1
            )  # HWC to CHW for torch
            left_wrist_image = torch.from_numpy(left_wrist_image).permute(2, 0, 1)
        elif (
            base_image.ndim == 3 and base_image.shape[2] == 3
        ):  # HWC torch tensor to CHW
            base_image = base_image.permute(2, 0, 1)
            left_wrist_image = left_wrist_image.permute(2, 0, 1)
        
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": torch.zeros_like(base_image).to(base_image.dtype),
            },
            "image_mask": {
                "base_0_rgb": torch.tensor(True),
                "left_wrist_0_rgb": torch.tensor(True),
                "right_wrist_0_rgb": torch.tensor(False),
            },
        }
        if "actions" in data:
            actions = data["actions"]
            if isinstance(actions, np.ndarray):
                actions = torch.from_numpy(actions).float()
            actions[:,-1] = 1 - actions[:,-1] # gripper is inverted
            inputs["actions"] = pad_to_dim(actions, self.action_dim, axis=-1, value=0.0)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        
        if self.map_to_unified_space:
            inputs["mapping_for_unified_space"] = self.mapping_for_unified_space
            
        return inputs

@dataclasses.dataclass(frozen=True)
class RobosetOutputsTransform(DataTransformFn):
    def __call__(self, data: dict) -> dict:
        actions = data["actions"]
        actions[:,-1] = 1 - actions[:,-1] # gripper is inverted
        if isinstance(actions, np.ndarray):
            if actions.ndim==2:
                data["actions"] = actions[:,:8]
            elif actions.ndim==3:
                data["actions"] = actions[:,:8]
            else:
                raise ValueError(f"Unsupported action shape: {actions.shape}")
            data["actions"] = actions[:,:8]
        elif isinstance(actions, torch.Tensor):
            if actions.ndim==2:
                data["actions"] = actions[:,:8].cpu().numpy()
            elif actions.ndim==3:
                data["actions"] = actions[:,:8].cpu().numpy()
            else:
                raise ValueError(f"Unsupported action shape: {actions.shape}")
            data["actions"] = actions[:,:8].cpu().numpy()
        else:
            raise TypeError(f"Unsupported action type: {type(actions)}")
        return data
    

