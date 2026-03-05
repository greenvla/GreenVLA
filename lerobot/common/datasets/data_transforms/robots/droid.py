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
class DroidInputsTransformOld(DataTransformFn):
    action_dim: int
    state_mask = [i for i in range(8) if i != 6] # skipping pad; IDC what does it means
    possible_base_img_keys = ["observation/exterior_image_1", "observation/exterior_image_2"]

    map_to_unified_space: bool = True
    # mapping_for_unified_space = [
    #     (list(range(0, 3)), list(range(35, 38))),        
    #     (list(range(3, 6)), list(range(42, 45))),     
    #     (list(range(6, 7)), list(range(13, 14))),
    # ]
    mapping_for_unified_space = [
        (list(range(0, 6)), list(range(1, 7))),        
        (list(range(6, 7)), list(range(13, 14))),
    ]
    
    # def __init__(self, *args, **kwargs):
    #     raise DeprecationWarning("Use DroidInputsTransform instead")
    
    def __call__(self, data: dict) -> dict:
        state = data["observation/state"]
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        state = pad_to_dim(state[self.state_mask], self.action_dim, value=0.0)
        
        base_image_key = np.random.choice(self.possible_base_img_keys)
        base_image = parse_image_helper(data[base_image_key])
        left_wrist_image = parse_image_helper(data["observation/wrist_image"])
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
class DroidOutputsTransformOld(DataTransformFn):
    # def __init__(self, *args, **kwargs):
    #     raise DeprecationWarning("Use DroidOutputsTransform instead")
    
    def __call__(self, data: dict[str, np.ndarray | torch.Tensor]) -> dict:
        actions = data["actions"]
        
        assert actions.ndim in (2,3), f"Unsupported action shape: {actions.shape}"
        assert isinstance(actions, (np.ndarray, torch.Tensor)), f"Unsupported action type: {type(actions)}"
        
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        
        actions = actions[..., :7]
        # actions[..., -1] = 1 - actions[..., -1] # gripper is inverted TODO: Do we need to invert here?
        
        data["actions"] = actions
        
        return data


@dataclasses.dataclass(frozen=True)
class DroidInputsTransform(DataTransformFn):
    action_dim: int
    
    control_type: str = "cartesian" # "cartesian" or "joint"
    
    cartesian_state_indices = [*range(0, 6)]
    joint_state_indices     = [*range(6, 13)]
    gripper_state_indices   = [*range(13, 14)]
    
    cartesian_action_indices = [*range(0, 6)]
    joint_action_indices     = [*range(6, 13)]
    gripper_action_indices   = [*range(13, 14)]
    cartesian_action_velocity_indices = [*range(14, 20)]
    joint_action_velocity_indices     = [*range(20, 27)]
    gripper_action_velocity_indices   = [*range(27, 28)]
    
    possible_base_img_keys = ["observation/exterior_image_1", "observation/exterior_image_2"]
    
    def remap_cartesian_state_and_action(self, data: dict) -> dict:
        current_euler_phi = data["observation/state"][self.cartesian_state_indices[3]]
        neg_inds = current_euler_phi < 0
        current_euler_phi[neg_inds] = current_euler_phi[neg_inds] + 2 * np.pi
        
        data["observation/state"][self.cartesian_state_indices[3]] = current_euler_phi
        
        current_euler_phi = data["actions"][self.cartesian_action_indices[3]]
        neg_inds = current_euler_phi < 0
        current_euler_phi[neg_inds] = current_euler_phi[neg_inds] + 2 * np.pi
        
        data["actions"][self.cartesian_action_indices[3]] = current_euler_phi
        
        return data
        

    def __call__(self, data: dict) -> dict:
        
        # remapping carteesian_position_euler_phi
        data = self.remap_cartesian_state_and_action(data)
        
        if self.control_type == "cartesian":
            state_mask = self.cartesian_state_indices + self.gripper_state_indices
            action_mask = self.cartesian_action_indices + self.gripper_action_indices
        elif self.control_type == "joint":
            state_mask = self.joint_state_indices + self.gripper_state_indices
            action_mask = self.joint_action_indices + self.gripper_action_indices
        else:
            raise ValueError(f"Invalid control type: {self.control_type}")
        
        state = data["observation/state"]
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        state = pad_to_dim(data["observation/state"][state_mask], self.action_dim, value=0.0)
        
        base_image_key = np.random.choice(self.possible_base_img_keys)
        base_image = parse_image_helper(data[base_image_key])
        left_wrist_image = parse_image_helper(data["observation/wrist_image"])
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
            actions = data["actions"][:, action_mask]
            if isinstance(actions, np.ndarray):
                actions = torch.from_numpy(actions).float()
            actions[:,-1] = 1 - actions[:,-1] # gripper is inverted
            inputs["actions"] = pad_to_dim(actions, self.action_dim, axis=-1, value=0.0)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs

@dataclasses.dataclass(frozen=True)
class DroidOutputsTransform(DataTransformFn):
    
    control_type: str = 'cartesian'
    
    def remap_cartesian_action(self, actions: np.ndarray) -> np.ndarray:
        current_euler_phi = actions[3]
        neg_inds = current_euler_phi > np.pi
        
        current_euler_phi[neg_inds] = current_euler_phi[neg_inds] - 2 * np.pi
        actions[3] = current_euler_phi
        return actions
    
    def __call__(self, data: dict) -> dict:
        actions = data["actions"]
        actions[:,-1] = 1 - actions[:,-1] # gripper is inverted
        
        action_dims = 7 if self.control_type == 'cartesian' else 8
        
        if isinstance(actions, np.ndarray):
            data["actions"] = actions[:,:action_dims]
        elif isinstance(actions, torch.Tensor):
            data["actions"] = actions[:,:action_dims].cpu().numpy()
        else:
            raise TypeError(f"Unsupported action type: {type(actions)}")
        
        if self.control_type == 'cartesian':
            data["actions"] = self.remap_cartesian_action(data["actions"])
        
        return data
