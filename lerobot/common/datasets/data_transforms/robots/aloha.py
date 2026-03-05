import dataclasses
import numpy as np
import torch
from lerobot.common.datasets.torch_transforms import (
    DataTransformFn,
    pad_to_dim,
    parse_image_helper,
    BaseModelConfigPlaceholder as ModelConfig,  # Use the placeholder
)


def _joint_flip_mask() -> np.ndarray:
    """Used to convert between aloha and pi joint angles."""
    return torch.tensor([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])


def _normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def _unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def _gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with model which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = _unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (
            2 * horn_radius * linear_position
        )
        return torch.arcsin(torch.clip(value, -1.0, 1.0))

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return _normalize(value, min_val=0.4, max_val=1.5)


def _gripper_from_angular(value):
    # Convert from the gripper position used by model to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = _unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return _normalize(value, min_val=-0.6213, max_val=1.4910)


def _gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = _unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return _normalize(value, min_val=0.4, max_val=1.5)


def _decode_state(state: torch.Tensor, *, adapt_to_pi: bool = False) -> torch.Tensor:
    if adapt_to_pi:
        # Flip the joints.
        state = _joint_flip_mask() * state
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        state[[6, 13]] = _gripper_to_angular(state[[6, 13]])
    return state


def _encode_actions(
    actions, *, adapt_to_pi: bool = False
):
    """
    Convert actions from model angular space to Aloha control space.

    Supports actions of shape (..., 14) where ... can be empty, (T,) or (B, T).
    Accepts torch.Tensor or np.ndarray and returns the same type as input.
    """
    is_numpy = isinstance(actions, np.ndarray)
    a = torch.as_tensor(actions) if is_numpy else actions

    if adapt_to_pi:
        # Flip joints on the last dimension
        mask = _joint_flip_mask().to(device=a.device, dtype=a.dtype)
        a = a * mask  # broadcast along the last dim
        # Grippers mapping on the last dimension (indices 6 and 13)
        a[..., [6, 13]] = _gripper_from_angular(a[..., [6, 13]])

    return a.cpu().numpy() if is_numpy else a


def _encode_actions_inv(
    actions, *, adapt_to_pi: bool = False
):
    """
    Inverse of _encode_actions. Converts Aloha control space back to model angular space.

    Supports actions of shape (..., 14) where ... can be empty, (T,) or (B, T).
    Accepts torch.Tensor or np.ndarray and returns the same type as input.
    """
    is_numpy = isinstance(actions, np.ndarray)
    a = torch.as_tensor(actions) if is_numpy else actions

    if adapt_to_pi:
        mask = _joint_flip_mask().to(device=a.device, dtype=a.dtype)
        a = a * mask
        a[..., [6, 13]] = _gripper_from_angular_inv(a[..., [6, 13]])

    return a.cpu().numpy() if is_numpy else a


@dataclasses.dataclass(frozen=True)
class AlohaInputsTransform(DataTransformFn):
    action_dim: int

    map_to_unified_space: bool = True
    
    mapping_for_unified_space = [
        (list(range(0, 6)), list(range(1, 7))),        # left arm joints
        (list(range(6, 7)), list(range(13, 14))),      # left gripper joints
        (list(range(7, 13)), list(range(15, 21))),    # right arm joints
        (list(range(13, 14)), list(range(27, 28))),    # right gripper joints
    ]
    
    adapt_to_pi: bool = True
    
    
    left_arm_indices = [*range(7,14)]
    right_arm_indices = [*range(0,7)]
    state_mask = [*left_arm_indices, *right_arm_indices]

    def __call__(self, data: dict) -> dict:
        state = data["observation/state"][self.state_mask]
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()

        state = _decode_state(state, adapt_to_pi=self.adapt_to_pi)

        state = pad_to_dim(
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
            actions = data["actions"][..., self.state_mask]
            if isinstance(actions, np.ndarray):
                actions = torch.from_numpy(actions).float()
            actions = _encode_actions(actions, adapt_to_pi=self.adapt_to_pi)
            inputs["actions"] = pad_to_dim(actions, self.action_dim, axis=-1, value=0.0)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        
        if self.map_to_unified_space:
            inputs["mapping_for_unified_space"] = self.mapping_for_unified_space
        return inputs


@dataclasses.dataclass(frozen=True)
class AlohaOutputsTransform(DataTransformFn):
    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        actions = data["actions"]
        a = _encode_actions_inv(actions[..., :14], adapt_to_pi=self.adapt_to_pi)
        if isinstance(a, torch.Tensor):
            a = a.cpu().numpy()
        data["actions"] = a
        return data
