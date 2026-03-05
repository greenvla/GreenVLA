import abc
from collections.abc import Sequence
import dataclasses
from typing import Generic, TypeVar, Dict, Optional, Tuple, Any, Callable
import logging

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import ColorJitter
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


def torch_preprocess_dict(
    data: Dict[str, Any],
    dtype: torch.dtype = torch.float32,
    augmentations_pipeline: Optional[Callable] = None,
):
    for key in data["image"]:
        assert data["image"][key].dtype == torch.uint8, f"Image {key} is not uint8"
        data["image"][key] = data["image"][key].float() / 255.0
        if augmentations_pipeline is not None:
            augmentations_mask = None
            if "data_source" in data:
                augmentations_mask = data["data_source"] == 0 # 0 is robotics, 1 is vlm
            data["image"][key] = augmentations_pipeline(
                data["image"][key].to(dtype), key, augmentations_mask
            )

    if "state" in data:
        data["state"] = data["state"].to(dtype)
    if "action" in data:
        data["action"] = data["action"].to(dtype)
    if "actions" in data:
        data["action"] = data["actions"].to(dtype)
    return data


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


@dataclasses.dataclass
class TorchObservation(Generic[ArrayT]):
    """PyTorch equivalent of the JAX Observation class."""

    # Images, in [-1, 1] float32
    images: Dict[str, torch.Tensor]
    # Image masks, with same keys as images
    image_masks: Dict[str, torch.Tensor]
    # Low-dimensional robot state
    state: torch.Tensor

    # Tokenized prompt
    tokenized_prompt: Optional[torch.Tensor] = None
    # Tokenized prompt mask
    tokenized_prompt_mask: Optional[torch.Tensor] = None

    # fast model specific fields
    token_ar_mask: Optional[torch.Tensor] = None
    token_loss_mask: Optional[torch.Tensor] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any], device="cuda") -> "TorchObservation":
        """Convert dictionary data to TorchObservation format."""
        # Ensure tokenized_prompt and tokenized_prompt_mask are provided together
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError(
                "tokenized_prompt and tokenized_prompt_mask must be provided together."
            )

        # Convert images to [-1, 1] float32 if they are uint8
        images = {}
        for key in data["image"]:
            img = data["image"][key]
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)
            if img.dtype == torch.uint8:
                img = img.float() / 255.0
            elif img.dtype == torch.float64:
                img = img.float()  # Convert to float32
            images[key] = img.to(device)

        # Convert state to tensor with consistent dtype
        state = data["state"]
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
        if state.dtype == torch.float64:
            state = state.float()  # Convert to float32
        state = state.to(device)

        # Convert image masks to tensors
        image_masks = {}
        for key in data["image_mask"]:
            mask = data["image_mask"][key]
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask)
            image_masks[key] = mask.to(device)

        # Convert optional fields with dtype consistency
        tokenized_prompt = data.get("tokenized_prompt")
        if tokenized_prompt is not None:
            if isinstance(tokenized_prompt, np.ndarray):
                tokenized_prompt = torch.from_numpy(tokenized_prompt)
            tokenized_prompt = tokenized_prompt.to(device)

        tokenized_prompt_mask = data.get("tokenized_prompt_mask")
        if tokenized_prompt_mask is not None:
            if isinstance(tokenized_prompt_mask, np.ndarray):
                tokenized_prompt_mask = torch.from_numpy(tokenized_prompt_mask)
            tokenized_prompt_mask = tokenized_prompt_mask.to(device)

        token_ar_mask = data.get("token_ar_mask")
        if token_ar_mask is not None:
            if isinstance(token_ar_mask, np.ndarray):
                token_ar_mask = torch.from_numpy(token_ar_mask)
            token_ar_mask = token_ar_mask.to(device)

        token_loss_mask = data.get("token_loss_mask")
        if token_loss_mask is not None:
            if isinstance(token_loss_mask, np.ndarray):
                token_loss_mask = torch.from_numpy(token_loss_mask)
            token_loss_mask = token_loss_mask.to(device)

        return cls(
            images=images,
            image_masks=image_masks,
            state=state,
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
            token_ar_mask=token_ar_mask,
            token_loss_mask=token_loss_mask,
        )

    def to_dict(self, device: str = None) -> Dict[str, Any]:
        """Convert TorchObservation to dictionary format.

        Args:
            device: Device to move tensors to. If None, keeps current device.
        """

        def move_tensor(tensor):
            if device is not None and hasattr(tensor, "to"):
                return tensor.to(device)
            return tensor

        result = {
            "image": {k: move_tensor(v) for k, v in self.images.items()},
            "image_mask": {k: move_tensor(v) for k, v in self.image_masks.items()},
            "state": move_tensor(self.state),
        }
        if self.tokenized_prompt is not None:
            result["tokenized_prompt"] = move_tensor(self.tokenized_prompt)
        if self.tokenized_prompt_mask is not None:
            result["tokenized_prompt_mask"] = move_tensor(self.tokenized_prompt_mask)
        if self.token_ar_mask is not None:
            result["token_ar_mask"] = move_tensor(self.token_ar_mask)
        if self.token_loss_mask is not None:
            result["token_loss_mask"] = move_tensor(self.token_loss_mask)
        return result


def resize_with_pad_torch(
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """PyTorch equivalent of JAX resize_with_pad function.

    Args:
        images: Tensor of shape (*batch, H, W, C) for uint8 or float32
        height: Target height
        width: Target width
        mode: Interpolation mode for F.interpolate

    Returns:
        Resized and padded images of shape (*batch, height, width, C)
    """
    has_batch_dim = images.ndim == 4
    if not has_batch_dim:
        images = images.unsqueeze(0)

    # Get current dimensions
    cur_height, cur_width = images.shape[-3:-1]

    # Calculate resize ratio to maintain aspect ratio
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    # Convert NHWC to NCHW for F.interpolate
    images_nchw = images.permute(0, 3, 1, 2)

    # For uint8 images, convert to float for interpolation to avoid precision issues
    original_dtype = images.dtype
    if original_dtype == torch.uint8:
        images_nchw = images_nchw.float()

    # Resize images
    resized_images_nchw = F.interpolate(
        images_nchw,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
        antialias=True,  # Add antialiasing for better quality
    )

    # Convert back to NHWC
    resized_images = resized_images_nchw.permute(0, 2, 3, 1)

    # Handle dtype conversion back to original
    if original_dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(-1.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    # Calculate padding
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    # Pad images - F.pad expects (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
    # For NHWC format, we need (0, 0, pad_w0, pad_w1, pad_h0, pad_h1, 0, 0)
    pad_value = 0 if original_dtype == torch.uint8 else -1.0
    padded_images = F.pad(
        resized_images,
        (0, 0, pad_w0, pad_w1, pad_h0, pad_h1, 0, 0),
        mode="constant",
        value=pad_value,
    )

    if not has_batch_dim:
        padded_images = padded_images.squeeze(0)

    return padded_images


def random_crop_torch(
    image: torch.Tensor, crop_height: int, crop_width: int, generator: torch.Generator
) -> torch.Tensor:
    """Random crop implementation for PyTorch."""
    _, h, w, _ = image.shape
    if crop_height >= h and crop_width >= w:
        return image

    # Generate random top-left corner
    max_y = h - crop_height
    max_x = w - crop_width

    y = torch.randint(0, max_y + 1, (1,), generator=generator).item()
    x = torch.randint(0, max_x + 1, (1,), generator=generator).item()

    return image[:, y : y + crop_height, x : x + crop_width, :]


def rotate_torch(image: torch.Tensor, angle: float) -> torch.Tensor:
    """Rotate image using torchvision."""
    # Convert NHWC to NCHW
    image_nchw = image.permute(0, 3, 1, 2)

    # Apply rotation
    rotated_nchw = TF.rotate(image_nchw, angle, fill=0.5)  # Fill with middle gray value

    # Convert back to NHWC
    return rotated_nchw.permute(0, 2, 3, 1)


def preprocess_observation_torch(
    generator: Optional[torch.Generator] = None,
    observation: TorchObservation = None,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: Tuple[int, int] = IMAGE_RESOLUTION,
) -> TorchObservation:
    """PyTorch equivalent of JAX preprocess_observation function.

    Args:
        generator: PyTorch random generator (equivalent of JAX rng)
        observation: TorchObservation object
        train: Whether in training mode
        image_keys: Sequence of image keys to process
        image_resolution: Target image resolution (height, width)

    Returns:
        Preprocessed TorchObservation
    """
    if not set(image_keys).issubset(observation.images.keys()):
        raise ValueError(
            f"images dict missing keys: expected {image_keys}, got {list(observation.images.keys())}"
        )

    batch_shape = observation.state.shape[:-1]

    out_images = {}
    for key in image_keys:
        image = observation.images[key]

        # Resize if necessary
        if image.shape[-3:-1] != image_resolution:
            logger.info(
                f"Resizing image {key} from {image.shape[-3:-1]} to {image_resolution}"
            )
            image = resize_with_pad_torch(image, *image_resolution)

        if train and generator is not None:

            # Apply different transforms based on camera type
            if "wrist" not in key:
                # For non-wrist cameras: random crop, resize, rotate
                height, width = image.shape[-3:-1]
                crop_height = int(height * 0.95)
                crop_width = int(width * 0.95)

                # Random crop
                image = random_crop_torch(image, crop_height, crop_width, generator)

                # Resize back to original size
                image = resize_with_pad_torch(image, height, width)

                # Random rotation (-5 to 5 degrees)
                angle_range = 5.0
                angle = (
                    (torch.rand(1, generator=generator).item() - 0.5) * 2 * angle_range
                )
                image = rotate_torch(image, angle)

            # Color jitter for all cameras
            color_jitter = ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)

            # Apply color jitter per batch item
            batch_size = image.shape[0]
            jittered_images = []
            for i in range(batch_size):
                # Convert NHWC to CHW for torchvision
                img_chw = image[i].permute(2, 0, 1)
                jittered_chw = color_jitter(img_chw)
                # Convert back to HWC
                jittered_hwc = jittered_chw.permute(1, 2, 0)
                jittered_images.append(jittered_hwc)

            image = torch.stack(jittered_images, dim=0)

        out_images[key] = image

    # Create output masks
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # Default to no masking
            out_masks[key] = torch.ones(batch_shape, dtype=torch.bool)
        else:
            out_masks[key] = observation.image_masks[key]

    return TorchObservation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
    )
