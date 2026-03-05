import logging
from typing import Protocol, Any, Sequence, TypeAlias, Callable, TypeVar, Dict
import dataclasses
import numpy as np
from scipy.interpolate import pchip_interpolate
import torch
import einops
from torch.utils.data import Dataset as TorchDataset
from lerobot.common.utils.image_tools import (
    resize_without_pad,
    resize_image_tensor,
)

try:
    import humanfriendly

    _HAS_HUMANFRIENDLY = True
except ImportError:
    _HAS_HUMANFRIENDLY = False


from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import print_dataset_summary
from rich import print
import pandas as pd

DataDict: TypeAlias = dict[str, Any]


class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested
                  dictionary that contains unbatched data elements.
                  Each leaf is expected to be a numpy array or torch tensor.

        Returns:
            The transformed data. Could be the input `data` that was modified
            in place, or a new data structure.
        """
        ...


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


@dataclasses.dataclass(frozen=True)
class Group:
    """A group of transforms."""

    inputs: Sequence[DataTransformFn] = ()
    outputs: Sequence[DataTransformFn] = ()

    def push(
        self,
        *,
        inputs: Sequence[DataTransformFn] = (),
        outputs: Sequence[DataTransformFn] = (),
    ) -> "Group":
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


@dataclasses.dataclass(frozen=True)
class RemoveStrings(DataTransformFn):
    def __call__(self, data: DataDict) -> DataDict:
        return {k: v for k, v in data.items() if not isinstance(v, str)}


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """
    Repacks an input dictionary into a new dictionary based on a defined structure.
    Keys in the structure are new keys, values are paths to old keys (e.g., "observation/images/top").
    This implementation handles simple, direct remapping.
    A more general version would require tree traversal like JAX's flatten_dict/unflatten_dict.
    """

    structure: dict[str, Any]  # Simplified: new_key -> old_key_path as string

    def _get_nested(self, data: DataDict, path: str) -> Any:
        keys = path.split("/")
        value = data
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    raise KeyError(f"Path {path} not found in data structure.")
            else:
                raise KeyError(f"Cannot traverse path {path}, {key} not in dict.")
        return value

    def _set_nested(self, data: DataDict, path: str, value: Any):
        keys = path.split("/")
        current = data
        for key in keys[:-1]:
            current = current.setdefault(key, {})
            if not isinstance(current, dict):
                raise ValueError(f"Conflict in path {path} at {key}")
        current[keys[-1]] = value

    def __call__(self, data: DataDict) -> DataDict:
        output_data: DataDict = {}

        # This is a simplified interpretation of JAX's RepackTransform.
        # It maps flat paths in the original data to potentially nested paths in the new structure.
        # For simplicity, let's assume structure is new_key -> old_key_path for now.
        # A full JAX-like repack would involve tree_map and flatten/unflatten.
        flat_item: DataDict = {}

        def _flatten_dict_simple(d, parent_key="", sep="/"):
            items = {}
            for k, v in d.items():
                new_key = parent_key + sep + k if parent_key else k
                if isinstance(v, dict):  # Assuming dicts are mutable for this example
                    items.update(_flatten_dict_simple(v, new_key, sep=sep))
                else:
                    items[new_key] = v
            return items

        flat_item = _flatten_dict_simple(data)

        def _map_structure(struct_node, current_output_dict):
            if isinstance(struct_node, dict):
                for k, v_path_or_struct in struct_node.items():
                    if isinstance(v_path_or_struct, str):  # it's a path to the old key
                        current_output_dict[k] = flat_item[v_path_or_struct]
                    elif isinstance(v_path_or_struct, dict):
                        current_output_dict[k] = {}
                        _map_structure(v_path_or_struct, current_output_dict[k])
                    else:  # Should not happen with the example structure
                        current_output_dict[k] = v_path_or_struct

        _map_structure(self.structure, output_data)
        return output_data


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


@dataclasses.dataclass(frozen=True)
class TokenizeGreenVLAInputsTransform:
    tokenizer: None

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")
        if not isinstance(prompt, str):
            prompt = prompt.item()
        state = data["state"]
        image_mask = data["image_mask"]
        actions = data.get("actions", None)
        subtask = data.pop("subtask", None)
        next_subtask = data.pop("next_subtask", None)
        is_subtask_transition = data.pop("is_subtask_transition", None)
        
        tokenized_data = self.tokenizer.tokenize(prompt, state, image_mask, actions, subtask, next_subtask, is_subtask_transition)

        action_loss_mask = data["action_loss_mask"] if "action_loss_mask" in data else tokenized_data["action_loss_mask"]
        return {
            **data,
            "input_ids": tokenized_data["input_ids"],
            "padded_mask": tokenized_data["padded_mask"],
            "attention_mask": tokenized_data["attention_mask"],
            "loss_mask": tokenized_data["loss_mask"],
            "action_loss_mask": action_loss_mask,
            "token_type_ids": tokenized_data["token_type_ids"],
        }
        
@dataclasses.dataclass(frozen=True)
class ExtractGreenVLAActionsTorch:
    tokenizer: None
    action_horizon: int
    action_dim: int
    model_mode: str
    inference_mode: str

    def __call__(self, data: DataDict) -> DataDict:
        if self.model_mode == "flow_matching" or (self.model_mode == "mixed" and self.inference_mode == "flow_matching"): #TODO: how to handle that mixed model can generate actions in both ways?
            return data
        elif self.model_mode == "token_prediction" or (self.model_mode == "mixed" and self.inference_mode == "token_prediction"):
            if "actions" not in data:
                return data
            if data['actions'].ndim == 1:
                tokens = data.pop("actions").astype(np.int32)
            else:
                tokens = data.pop("actions").astype(np.int32)[0]
            actions = self.tokenizer.extract_actions(
                tokens, self.action_horizon, self.action_dim
            )
            if isinstance(actions, torch.Tensor):
                actions = actions.cpu().numpy()
            data["actions"] = actions[None, ...]
            return data
        else:
            raise ValueError(f"Invalid model mode: {self.model_mode}")


@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]

        is_torch = isinstance(state, torch.Tensor)

        if is_torch:
            mask_tensor = torch.tensor(
                self.mask, dtype=torch.bool, device=actions.device
            )
            state_masked_part = state[..., : mask_tensor.shape[-1]]
            actions_masked_part = actions[..., : mask_tensor.shape[-1]]

            delta = torch.where(
                mask_tensor, state_masked_part, torch.zeros_like(state_masked_part)
            )
            actions_masked_part -= delta.unsqueeze(
                -2
            )  # assuming actions has horizon dim

            # Update the slice of actions
            if actions.shape[-1] == mask_tensor.shape[-1]:
                data["actions"] = actions_masked_part
            else:  # mask is shorter than action dim
                data["actions"][..., : mask_tensor.shape[-1]] = actions_masked_part
        else:  # numpy
            mask_arr = np.asarray(self.mask, dtype=bool)
            dims = mask_arr.shape[-1]

            delta_values = np.where(mask_arr, state[..., :dims], 0)
            # Ensure delta_values can be broadcast if actions has an extra dimension (horizon)
            if actions.ndim > delta_values.ndim:
                delta_values_expanded = np.expand_dims(delta_values, axis=-2)
            else:
                delta_values_expanded = delta_values

            actions[..., :dims] -= delta_values_expanded
            data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class InterpolateActions(DataTransformFn):
    sample_step: int = 1
    actions_type: str = "absolute"
    def __call__(self, data: DataDict) -> DataDict:
        assert self.actions_type in ["absolute", "relative", "delta"]
        if "actions" not in data:
            return data
        state, actions = data["state"], data["actions"]
        if not isinstance(actions, np.ndarray):
            actions = actions.cpu().numpy()
        if not isinstance(state, np.ndarray):
            state = state.cpu().numpy()
        
        if self.actions_type == "absolute":
            zero_action = np.expand_dims(state, axis=-2)
        elif self.actions_type == "relative":
            zero_action = np.zeros_like(zero_action)
        elif self.actions_type == "delta":
            zero_action = actions[..., :1, :]
        actions = np.concatenate([zero_action, actions], axis=-2)
        batch_mode = actions.ndim == 3
        if not batch_mode:
            actions = actions[None, ...]
        query_indices = np.arange(0, actions.shape[1] - 1 + 1e-6, self.sample_step)
        interpolated_actions = np.zeros((actions.shape[0], query_indices.shape[0] - 1, actions.shape[2]),
                                        dtype=actions.dtype)
        for batch_idx in range(actions.shape[0]):
            for joint_idx in range(actions.shape[2]):
                interpolated_actions[batch_idx, :, joint_idx] = pchip_interpolate(
                    np.arange(actions.shape[1]), actions[batch_idx, :, joint_idx], query_indices
                )[1:]
        if not batch_mode:
            interpolated_actions = interpolated_actions[0]
        data["actions"] = interpolated_actions
        return data

@dataclasses.dataclass(frozen=True)
class AbsoluteActions(DataTransformFn):
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        if not isinstance(actions, np.ndarray):
            actions = actions.cpu().numpy()
        if not isinstance(state, np.ndarray):
            state = state.cpu().numpy()
        is_torch = isinstance(state, torch.Tensor)

        if is_torch:
            mask_tensor = torch.tensor(
                self.mask, dtype=torch.bool, device=actions.device
            )
            state_masked_part = state[..., : mask_tensor.shape[-1]]
            actions_masked_part = actions[..., : mask_tensor.shape[-1]]

            delta = torch.where(
                mask_tensor, state_masked_part, torch.zeros_like(state_masked_part)
            )
            actions_masked_part += delta.unsqueeze(-2)

            if actions.shape[-1] == mask_tensor.shape[-1]:
                data["actions"] = actions_masked_part
            else:
                data["actions"][..., : mask_tensor.shape[-1]] = actions_masked_part

        else:  # numpy
            mask_arr = np.asarray(self.mask, dtype=bool)
            dims = mask_arr.shape[-1]

            delta_values = np.where(mask_arr, state[..., :dims], 0)
            if actions.ndim > delta_values.ndim:
                delta_values_expanded = np.expand_dims(delta_values, axis=-2)
            else:
                delta_values_expanded = delta_values

            actions[..., :dims] += delta_values_expanded
            data["actions"] = actions

        return data

class SmoothActions(DataTransformFn):
    @staticmethod
    def smooth_array(array: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
        assert kernel_size % 2 == 1
        assert array.ndim == 2
        half_kernel_size = kernel_size // 2
        gaussian_kernel = torch.exp(-torch.arange(-half_kernel_size, half_kernel_size + 1, device=array.device, dtype=array.dtype)**2)
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        # array shape is (horizon, act_dim)
        # kernel shape should be (act_dim, act_dim, kernel_size)
        gaussian_kernel = gaussian_kernel[None, None] * torch.eye(array.shape[1], device=array.device, dtype=array.dtype)[:, :, None]
        # using reflect padding to mimic kernel crop on the array sides (mode='symmetric' would be better though)
        padded_array = torch.nn.functional.pad(array[None], pad=(0, 0, half_kernel_size, half_kernel_size), mode='reflect')
        smoothed_array = torch.nn.functional.conv1d(padded_array.permute(0, 2, 1), gaussian_kernel).squeeze(0).T
        return smoothed_array

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data
        actions = data["actions"]
        smoothed_actions = self.smooth_array(actions)
        data["actions"] = smoothed_actions
        return data

def pad_to_dim(
    x: np.ndarray | torch.Tensor, target_dim: int, axis: int = -1, value=0
) -> np.ndarray | torch.Tensor:
    current_dim = x.shape[axis]
    if current_dim >= target_dim:
        return x

    pad_width_shape = [(0, 0)] * x.ndim
    pad_width_shape[axis] = (0, target_dim - current_dim)

    if isinstance(x, torch.Tensor):
        # PyTorch pad format is (pad_left, pad_right, pad_top, pad_bottom, etc.)
        # for the last N dimensions.
        # We need to create a flat list of pad values for torch.nn.functional.pad
        # It pads from the last dim to the first dim specified in pad.
        # So if axis is -1, pad is (0, target_dim - current_dim)
        # If axis is -2, pad is (0, 0, 0, target_dim - current_dim)

        torch_pad = []
        for i in range(x.ndim):
            # Iterate from last dim towards first
            actual_axis_idx = x.ndim - 1 - i
            if actual_axis_idx == axis or (
                axis < 0 and x.ndim + axis == actual_axis_idx
            ):
                torch_pad.extend(
                    [
                        pad_width_shape[actual_axis_idx][0],
                        pad_width_shape[actual_axis_idx][1],
                    ]
                )
            else:
                torch_pad.extend([0, 0])
        return torch.nn.functional.pad(
            x, tuple(torch_pad), mode="constant", value=value
        )
    else:  # numpy
        return np.pad(x, pad_width_shape, mode="constant", constant_values=value)


def parse_image_helper(image: Any) -> np.ndarray | torch.Tensor:
    """Converts image to numpy HWC uint8 format if it's float or CHW."""
    is_torch = isinstance(image, torch.Tensor)

    if is_torch:
        # Handle torch tensor
        if image.is_floating_point():
            image = (255 * image).byte()  # to uint8 equivalent for torch
        if image.ndim == 3 and image.shape[0] == 3:  # CHW to HWC
            image = image.permute(1, 2, 0)
        return image
    else:  # Assume numpy
        image_np = np.asarray(image)
        if np.issubdtype(image_np.dtype, np.floating):
            image_np = (255 * image_np).astype(np.uint8)
        if image_np.ndim == 3 and image_np.shape[0] == 3:  # CHW to HWC
            image_np = einops.rearrange(image_np, "c h w -> h w c")
        return image_np


# Placeholder for ModelType to avoid direct openpi.models dependency here
@dataclasses.dataclass
class BaseModelConfigPlaceholder:
    action_dim: int
    model_type: str = "greenvlapolicy"  # Example, adjust as needed by LeRobot


@dataclasses.dataclass(frozen=True)
class PyTorchModelTransformFactory:
    """Creates model transforms for PyTorch models."""

    default_prompt: str | None = None

    # These would typically come from a PyTorch/LeRobot specific model config
    max_token_len: int = 64
    action_horizon: int = 10

    def __call__(self, model_config: BaseModelConfigPlaceholder) -> Group:

        logging.info(
            f"PyTorchModelTransformFactory called with model_config: {model_config}, default_prompt: {self.default_prompt}"
        )
        logging.info(
            f"Max token len: {self.max_token_len}, Action horizon: {model_config.action_horizon if hasattr(model_config, 'action_horizon') else self.action_horizon}"
        )

        return Group(inputs=[], outputs=[])


SampleType = TypeVar("SampleType")


class TorchTransformedDataset(TorchDataset[SampleType]):
    """A PyTorch Dataset that applies a sequence of transforms."""

    def __init__(
        self,
        dataset: TorchDataset[SampleType],
        transform: Callable[[SampleType], SampleType] | None = None,
    ):
        self._dataset = dataset
        self._transform = transform if transform is not None else lambda x: x
        # check that _dataset._dataset exist
        if hasattr(self._dataset, "_dataset") and isinstance(
            self._dataset._dataset, LeRobotDataset
        ):
            self._dataset_id = self._dataset._dataset.repo_id
            self._num_samples = self._dataset._dataset.num_frames
            self._num_episodes = self._dataset._dataset.num_episodes
            self._fps = self._dataset._dataset.fps

    def __getitem__(self, index: int) -> SampleType:
        try:
            item = self._dataset[index]
        except Exception as e:
            print(f"[bold red]Error getting item {index} from dataset : {e}[/bold red]")
            raise e
        return self._transform(item)

    def __len__(self) -> int:
        return len(self._dataset)

    def get_dataset_summary(self) -> dict[str, Any]:
        summary_info = {}
        try:
            summary_info["dataset_id"] = self._dataset_id
            summary_info["num_samples"] = self._num_samples
            summary_info["num_episodes"] = self._num_episodes
            summary_info["dataset_length"] = self._num_samples / self._fps
        except AttributeError as e:
            raise
        return summary_info

    def print_dataset_summary(self):
        summary = self.get_dataset_summary()
        print_dataset_summary(summary)

    def log_dataset_summary(self, logger) -> pd.DataFrame:
        """
        Build a pandas DataFrame from get_dataset_summary().
        """
        summary = self.get_dataset_summary()
        if _HAS_HUMANFRIENDLY:
            duration_str = humanfriendly.format_timespan(summary["dataset_length"])
        else:
            duration_str = f"{summary['dataset_length']:.2f} seconds"

        # rename or reorder columns as you like
        df = pd.DataFrame(
            {
                "Dataset ID": [summary["dataset_id"]],
                "Amount of Samples": [summary["num_samples"]],
                "Amount of Episodes": [summary["num_episodes"]],
                "Total Duration": [duration_str],
            }
        )
        logger.report_table(
            title="Training Dataset Summary",
            series="Robot Dataset Summary",
            iteration=0,
            table_plot=df,
        )


# Placeholder for PyTorch-native PromptFromLeRobotTask
class PromptFromLeRobotTaskTorch:
    def __init__(self, tasks: Sequence[str] | None):
        self.tasks = tasks
        # In a real implementation, this might involve a tokenizer
        # or specific logic to select/format prompts.
        if tasks is not None:
            self.task_prompts = {i: task for i, task in enumerate(tasks)}
        else:
            self.task_prompts = {}
        print(
            f"Initialized PromptFromLeRobotTaskTorch with {len(self.task_prompts)} tasks."
        )

    def __call__(self, item: Dict[str, Any]) -> Dict[str, Any]:
        task_index = int(item["task_index"])
        if (prompt := self.tasks.get(task_index)) is None:
            raise ValueError(f"{task_index=} not found in task mapping: {self.tasks}")

        return {**item, "prompt": prompt}


class MapToUnifiedSpaceTorch(DataTransformFn):
    def __init__(self, target_dim: int = 44, axis: int = -1, pad_value: float = 0.0):
        self.target_dim = target_dim
        self.axis = axis
        self.pad_value = pad_value

    def __call__(self, x: Dict[str, Any]):
        """
        Map a vector (or last-dim of a tensor) into a unified 43-dim vector
        Returns (padded, pad_mask) where pad_mask is True for padded positions.
        """
        mapping_actions = x.get("mapping_for_unified_space_actions", None)
        mapping_state = x.get("mapping_for_unified_space_state", None)
        if mapping_actions is None and mapping_state is None:
            both = x.get("mapping_for_unified_space", None)
            mapping_state = both
            mapping_actions = both
        if mapping_actions is None and mapping_state is None:
            raise ValueError("mapping_for_unified_space not found in x")
        if "state" in x and mapping_state is not None:
            x["state"], x["state_pad_mask"] = self.map_to_unified_space(x["state"], mapping_state)
        if "actions" in x and mapping_actions is not None:
            x["actions"], x["actions_pad_mask"] = self.map_to_unified_space(x["actions"], mapping_actions)
        x.pop("mapping_for_unified_space_actions", None)
        x.pop("mapping_for_unified_space_state", None)
        x.pop("mapping_for_unified_space", None)
        
        return x
        
    
    def map_to_unified_space(self, x: torch.Tensor | np.ndarray, mapping: list[tuple[list[int], list[int]]]):
        if not isinstance(x, (torch.Tensor, np.ndarray)):
            raise TypeError(f"Unsupported type for pad_to_dim_unified_space: {type(x)}")

        # Only support mapping along the last axis for now (minimal change footprint)
        if self.axis not in (-1, x.ndim - 1):
            raise ValueError("pad_to_dim_unified_space currently supports mapping along the last dimension only")

        out_shape = list(x.shape)
        out_shape[-1] = self.target_dim

        if isinstance(x, torch.Tensor):
            padded = torch.full(out_shape, self.pad_value, dtype=x.dtype, device=x.device)
            pad_mask = torch.ones(out_shape, dtype=torch.bool, device=x.device)
            for src_idx, dst_idx in mapping:
                # Copy only overlapping indices to handle mismatched slice lengths safely
                k = min(len(src_idx), len(dst_idx))
                if k > 0:
                    padded[..., dst_idx[:k]] = x[..., src_idx[:k]]
                    pad_mask[..., dst_idx[:k]] = False
            return padded, pad_mask
        else:
            padded = np.full(out_shape, self.pad_value, dtype=x.dtype)
            pad_mask = np.ones(out_shape, dtype=bool)
            for src_idx, dst_idx in mapping:
                k = min(len(src_idx), len(dst_idx))
                if k > 0:
                    padded[..., dst_idx[:k]] = x[..., src_idx[:k]]
                    pad_mask[..., dst_idx[:k]] = False
            return padded, pad_mask
    
class MapToSingleSpaceTorch(DataTransformFn):
        
    def __init__(self, 
                 target_dim: int = 32, 
                 axis: int = -1, 
                 pad_value: float = 0.0,
                 mapping_actions: list[tuple[list[int], list[int]]] = None,
                 mapping_state: list[tuple[list[int], list[int]]] = None):
        self.target_dim = target_dim
        self.axis = axis
        self.pad_value = pad_value
        self.mapping_actions = mapping_actions
        self.mapping_state = mapping_state
    
    def __call__(self, x: Dict[str, Any]):
        """
        Inverse mapping: from unified space back to compact/single space.
        Uses the same mapping definition as MapToUnifiedSpaceTorch where
        each tuple is (src_compact_indices, dst_unified_indices).
        """
        mapping_actions = None
        mapping_state = None
        if "mapping_for_unified_space_actions" in x:
            mapping_actions = x.get("mapping_for_unified_space_actions", None)
        if "mapping_for_unified_space_state" in x:
            mapping_state = x.get("mapping_for_unified_space_state", None)
        if mapping_actions is None and mapping_state is None and "mapping_for_unified_space" in x:
            mapping_actions = mapping_state = x.get("mapping_for_unified_space", None)
        if mapping_actions is None and mapping_state is None:
            mapping_actions = self.mapping_actions
            mapping_state = self.mapping_state
            
        if mapping_actions is None and mapping_state is None:
            raise ValueError("mapping_for_unified_space_actions or mapping_for_unified_space_state not found in x")

        if "state" in x and mapping_state is not None:
            x["state"] = self.map_to_single_space(x["state"], mapping_state)
            x["state"] = pad_to_dim(x["state"], self.target_dim, axis=self.axis, value=self.pad_value)
        
        #print('before', x["actions"][:, :2])
        if "actions" in x and mapping_actions is not None:
            x["actions"] = self.map_to_single_space(x["actions"], mapping_actions)
            x["actions"] = pad_to_dim(x["actions"], self.target_dim, axis=self.axis, value=self.pad_value)
        #print('after', x["actions"][:, :2])
        return x
    
    def map_to_single_space(self, x: torch.Tensor | np.ndarray, mapping: list[tuple[list[int], list[int]]]):
        """
        Given an input in unified space (last dim is unified), build a compact vector of size target_dim by
        copying values back from unified indices (dst) into compact indices (src).
        """
        if not isinstance(x, (torch.Tensor, np.ndarray)):
            raise TypeError(f"Unsupported type for map_to_single_space: {type(x)}")

        if self.axis not in (-1, x.ndim - 1):
            raise ValueError("MapToSingleSpaceTorch supports mapping along the last dimension only")

        out_shape = list(x.shape)
        out_shape[self.axis if self.axis >= 0 else (x.ndim + self.axis)] = self.target_dim

        if isinstance(x, torch.Tensor):
            compact = torch.full(out_shape, self.pad_value, dtype=x.dtype, device=x.device)
            for src_idx, dst_idx in mapping:
                compact[..., src_idx] = x[..., dst_idx]
            return compact
        else:
            compact = np.full(out_shape, self.pad_value, dtype=x.dtype)
            for src_idx, dst_idx in mapping:
                compact[..., src_idx] = x[..., dst_idx]
            return compact

class InjectDefaultPromptTorch:
    def __init__(self, prompt: str | None):
        self.prompt = prompt

    def __call__(self, data: Dict[str, Any]):
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = self.prompt
        return data


class ResizeImagesTorch:
    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width

    def __call__(self, data: DataDict) -> DataDict:
        # Assuming images are in (..., C, H, W) or (..., H, W, C) format
        # parse_image_helper is used to ensure HWC uint8 before PIL,
        # but here we want to keep it as tensor and determine format.
        transformed_images = {}
        for k, v_img_tensor in data["image"].items():
            if not isinstance(v_img_tensor, torch.Tensor):
                # Fallback or error if not a tensor, though ideally it should be
                # Forcing to numpy HWC for the old path if it's not a tensor
                img_np_hwc = parse_image_helper(
                    v_img_tensor
                )  # Ensure HWC numpy for resize_with_pad
                resized_img = resize_without_pad(img_np_hwc, self.height, self.width)
                transformed_images[k] = resized_img  # This will be a numpy array
            else:
                # Determine if channels_last (..., H, W, C)
                # Heuristic: if last dim is 3 or 1, assume channels_last. Otherwise CHW.
                # This might need to be more robust depending on actual data shapes.
                is_channels_last = (
                    v_img_tensor.shape[-1] == 3 or v_img_tensor.shape[-1] == 1
                ) and v_img_tensor.ndim >= 3

                # Ensure uint8 if it's a float tensor, common for image data
                # The new resizer handles float inputs too, but this aligns with parse_image_helper's conversion
                processed_tensor = v_img_tensor
                if v_img_tensor.is_floating_point():
                    processed_tensor = (v_img_tensor * 255).round().to(torch.uint8)

                resized_tensor = resize_image_tensor(
                    processed_tensor,
                    self.height,
                    self.width,
                    channels_last=is_channels_last,
                )
                transformed_images[k] = resized_tensor
        data["image"] = transformed_images
        return data


# Placeholder for PyTorch-native Normalize transform
class NormalizeTorch:
    def __init__(
        self,
        norm_stats: Dict[str, Any],
        normalization_mode: str | None = None,
        strict: bool = False,
    ):
        self.norm_stats = norm_stats
        assert normalization_mode in [
            "mean_std",
            "quantile",
            "min_max",
        ], f"Invalid normalization mode: {normalization_mode}"
        self.normalization_mode = normalization_mode
        self.strict = strict
        # Actual normalization logic would be more complex, handling different keys etc.

    def __call__(self, item: Dict[str, Any]) -> Dict[str, Any]:
        for key, stats in self.norm_stats.items():
            if key in item:
                data_value = item[key]

                if isinstance(data_value, np.ndarray):
                    data_tensor = torch.from_numpy(data_value).contiguous()
                elif isinstance(data_value, torch.Tensor):
                    data_tensor = data_value
                else:
                    raise TypeError(
                        f"Data for key '{key}' must be a torch.Tensor or np.ndarray, "
                        f"got {type(data_value)}."
                    )

                if self.normalization_mode == "mean_std":
                    _mean = torch.as_tensor(
                        stats.mean, dtype=data_tensor.dtype, device=data_tensor.device
                    )
                    _std = torch.as_tensor(
                        stats.std, dtype=data_tensor.dtype, device=data_tensor.device
                    )
                    normalized_tensor = (data_tensor - _mean) / (_std + 1e-6)

                elif self.normalization_mode == "quantile":
                    if stats.q01 is None or stats.q99 is None:
                        # This state should ideally be prevented by __post_init__ / _validate_quantile_stats.
                        # Raising an error here as a safeguard.
                        raise ValueError(
                            f"Quantile stats (q01, q99) for key '{key}' are None, "
                            "but use_quantiles is True. This indicates an issue not caught during initialization."
                        )
                    _q01 = torch.as_tensor(
                        stats.q01, dtype=data_tensor.dtype, device=data_tensor.device
                    )
                    _q99 = torch.as_tensor(
                        stats.q99, dtype=data_tensor.dtype, device=data_tensor.device
                    )

                    denominator_quantile = _q99 - _q01
                    normalized_tensor = (data_tensor - _q01) / (
                        denominator_quantile + 1e-6
                    ) * 2.0 - 1.0
                    # # clipping in range (-1, 1)
                    # normalized_tensor = torch.clip(
                    #     input=normalized_tensor, min=-1, max=1
                    # )

                item[key] = normalized_tensor

            # elif self.strict:
            #     raise KeyError(f"Key '{key}' from norm_stats not found in item, and 'strict' is True.")

        return item


@dataclasses.dataclass(frozen=True)
class UnnormalizeTorch:
    norm_stats: Dict[str, Any]  # Expects NormStats-like objects
    normalization_mode: str | None = None
    strict: bool = False

    def __post_init__(self):
        assert self.normalization_mode in [
            "mean_std",
            "quantile",
            "min_max",
        ], f"Invalid normalization mode: {self.normalization_mode}"
        if self.normalization_mode == "quantile":
            for key, stats in self.norm_stats.items():
                if (
                    not hasattr(stats, "q01")
                    or not hasattr(stats, "q99")
                    or stats.q01 is None
                    or stats.q99 is None
                ):
                    raise ValueError(
                        f"Quantile stats (q01, q99) for key '{key}' must be provided and not None "
                        "when normalization_mode is 'quantile'."
                    )

    def _unnormalize(self, data_tensor: torch.Tensor, stats: Any) -> torch.Tensor:
        _mean = torch.as_tensor(
            stats.mean, dtype=data_tensor.dtype, device=data_tensor.device
        )
        _std = torch.as_tensor(
            stats.std, dtype=data_tensor.dtype, device=data_tensor.device
        )
        return data_tensor * (_std + 1e-6) + _mean

    def _unnormalize_quantile(
        self, data_tensor: torch.Tensor, stats: Any
    ) -> torch.Tensor:
        _q01 = torch.as_tensor(
            stats.q01, dtype=data_tensor.dtype, device=data_tensor.device
        )
        _q99 = torch.as_tensor(
            stats.q99, dtype=data_tensor.dtype, device=data_tensor.device
        )
        denominator_quantile = _q99 - _q01
        return (data_tensor + 1.0) / 2.0 * (denominator_quantile + 1e-6) + _q01

    def __call__(self, item: Dict[str, Any]) -> Dict[str, Any]:
        if not self.norm_stats:
            return item

        for key, stats in self.norm_stats.items():
            if key in item:
                data_value = item[key]

                if isinstance(data_value, np.ndarray):
                    # Ensure contiguous array before converting to tensor
                    data_tensor = torch.from_numpy(np.ascontiguousarray(data_value))
                elif isinstance(data_value, torch.Tensor):
                    data_tensor = data_value
                else:
                    raise TypeError(
                        f"Data for key '{key}' must be a torch.Tensor or np.ndarray, "
                        f"got {type(data_value)}."
                    )

                if self.normalization_mode == "quantile":
                    item[key] = self._unnormalize_quantile(data_tensor, stats)
                elif self.normalization_mode == "mean_std":
                    item[key] = self._unnormalize(data_tensor, stats)

            elif self.strict:
                raise KeyError(
                    f"Key '{key}' from norm_stats not found in item, and 'strict' is True."
                )

        return item


# Placeholder for PyTorchModelTransformFactory
# This factory should create a callable (like a TorchGroup instance)
# that applies PyTorch-native model-specific transformations.
class PyTorchModelTransformFactory:
    def __init__(self, default_prompt: str | None = None):
        self.default_prompt = default_prompt

    def __call__(self, model_config: Any) -> Callable[[Dict[str, Any]], Dict[str, Any]]:

        return lambda x: x  # Identity function as a basic placeholder
