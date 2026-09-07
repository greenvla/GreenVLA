from typing import TypeVar, Dict, Any
import logging

import torch
import numpy as np

logger = logging.getLogger("torch_preprocess")

ArrayT = TypeVar("ArrayT", torch.Tensor, np.ndarray)

# The model always expects these images
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)

IMAGE_RESOLUTION = (224, 224)


def move_batch_to_device(batch, target_device):
    if isinstance(batch, torch.Tensor):
        return batch.to(target_device)
    elif isinstance(batch, dict):
        return {k: move_batch_to_device(v, target_device) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return type(batch)(move_batch_to_device(x, target_device) for x in batch)
    return batch




def torch_preprocess_dict_inference(
    data: Dict[str, Any], dtype: torch.dtype = torch.float32
):
    for key in data["image"]:
        assert data["image"][key].dtype == torch.uint8, f"Image {key} is not uint8"
        data["image"][key] = data["image"][key].float() / 255.0
        if data["image"][key].shape[-1] == 1 or data["image"][key].shape[-1] == 3:
            data["image"][key] = data["image"][key].permute(0, 3, 1, 2)

    if "state" in data:
        data["state"] = data["state"].to(dtype)

    if "action" in data:
        data["actions"] = data["action"].to(dtype)
    return data


def move_dict_to_batch_for_inference(data_tree, device="cpu"):
    if isinstance(data_tree, dict):
        return {
            k: move_dict_to_batch_for_inference(v, device) for k, v in data_tree.items()
        }
    elif isinstance(data_tree, (list, tuple)):
        return type(data_tree)(
            move_dict_to_batch_for_inference(item, device) for item in data_tree
        )
    elif isinstance(data_tree, np.ndarray):
        # Convert numpy array to tensor, add batch dimension, move to device
        return torch.from_numpy(data_tree).unsqueeze(0).to(device)
    elif isinstance(data_tree, torch.Tensor):
        return data_tree.unsqueeze(0).to(device)
    else:
        # For other types (e.g., strings, scalars not meant to be tensors), keep them as is
        return data_tree










