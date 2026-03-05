#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import contextlib
import importlib.resources
import json
import logging
from collections.abc import Iterator
from itertools import accumulate
from pathlib import Path
from pprint import pformat
from types import SimpleNamespace
from typing import Any, Union

import datasets
import jsonlines
import numpy as np
import packaging.version
import torch
from datasets.table import embed_table_storage
from huggingface_hub import DatasetCard, DatasetCardData, HfApi
from huggingface_hub.errors import RevisionNotFoundError
from PIL import Image as PILImage
from torchvision import transforms
from rich.console import Console
from rich.table import Table
from omegaconf import OmegaConf

import fsspec

from lerobot.common.datasets.backward_compatibility import (
    V21_MESSAGE,
    BackwardCompatibilityError,
    ForwardCompatibilityError,
)
from lerobot.common.utils.utils import is_valid_numpy_dtype_string
from lerobot.configs.types import DictLike

DEFAULT_CHUNK_SIZE = 1000  # Max number of episodes per chunk

INFO_PATH = "meta/info.json"
EPISODES_PATH = "meta/episodes.jsonl"
STATS_PATH = "meta/stats.json"
EPISODES_STATS_PATH = "meta/episodes_stats.jsonl"
TASKS_PATH = "meta/tasks.jsonl"
VARIATIONS_PATH = "meta/variations.jsonl"
SUBTASKS_PATH = "meta/subtasks.jsonl"

DEFAULT_VIDEO_PATH = (
    "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
)
DEFAULT_PARQUET_PATH = (
    "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
)
DEFAULT_IMAGE_PATH = (
    "images/{image_key}/episode_{episode_index:06d}/frame_{frame_index:06d}.png"
)

DATASET_CARD_TEMPLATE = """
---
# Metadata will go there
---
This dataset was created using [LeRobot](https://github.com/huggingface/lerobot).

## {}

"""

DEFAULT_FEATURES = {
    "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
    "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
    "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
    "index": {"dtype": "int64", "shape": (1,), "names": None},
    "task_index": {"dtype": "int64", "shape": (1,), "names": None},
    "variation_index": {"dtype": "int64", "shape": (1,), "names": None},
}


def open_file(path: Union[str, Path], mode: str = "r"):
    """
    Generic opener that delegates to fsspec if path is an S3 URI. Otherwise, built‐in open().
    """
    if isinstance(path, str) and path.startswith("s3://"):
        return fsspec.open(path, mode=mode)
    else:
        return open(path, mode)


def flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict:
    """Flatten a nested dictionary structure by collapsing nested keys into one key with a separator.

    For example:
    ```
    >>> dct = {"a": {"b": 1, "c": {"d": 2}}, "e": 3}`
    >>> print(flatten_dict(dct))
    {"a/b": 1, "a/c/d": 2, "e": 3}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: dict, sep: str = "/") -> dict:
    outdict = {}
    for key, value in d.items():
        parts = key.split(sep)
        d = outdict
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return outdict


def get_nested_item(obj: DictLike, flattened_key: str, sep: str = "/") -> Any:
    split_keys = flattened_key.split(sep)
    getter = obj[split_keys[0]]
    if len(split_keys) == 1:
        return getter

    for key in split_keys[1:]:
        getter = getter[key]

    return getter


def serialize_dict(stats: dict[str, torch.Tensor | np.ndarray | dict]) -> dict:
    serialized_dict = {}
    for key, value in flatten_dict(stats).items():
        if isinstance(value, (torch.Tensor, np.ndarray)):
            serialized_dict[key] = value.tolist()
        elif isinstance(value, np.generic):
            serialized_dict[key] = value.item()
        elif isinstance(value, (int, float)):
            serialized_dict[key] = value
        else:
            raise NotImplementedError(
                f"The value '{value}' of type '{type(value)}' is not supported."
            )
    return unflatten_dict(serialized_dict)


def embed_images(dataset: datasets.Dataset) -> datasets.Dataset:
    # Embed image bytes into the table before saving to parquet
    format = dataset.format
    dataset = dataset.with_format("arrow")
    dataset = dataset.map(embed_table_storage, batched=False)
    dataset = dataset.with_format(**format)
    return dataset


def load_json(fpath: Union[str, Path]) -> Any:
    """
    Exactly like json.load(open(path)), but if path starts with "s3://", uses fsspec.open.
    """
    with open_file(fpath, "r") as f:
        return json.load(f)


def convert_config_to_dict(data):
    # If the data is a ListConfig or DictConfig, convert it to a standard Python object
    if OmegaConf.is_list(data) or OmegaConf.is_dict(data):
        return OmegaConf.to_container(data, resolve=True)
    elif isinstance(data, list):
        # Recursively convert each item in the list
        return [convert_config_to_dict(item) for item in data]
    elif isinstance(data, dict):
        # Recursively convert each value in the dictionary
        return {key: convert_config_to_dict(value) for key, value in data.items()}
    else:
        # Return the data as is if it's not a ListConfig, DictConfig, list, or dict
        return data


def convert_config_to_dict(data):
    # If the data is a ListConfig or DictConfig, convert it to a standard Python object
    if OmegaConf.is_list(data) or OmegaConf.is_dict(data):
        return OmegaConf.to_container(data, resolve=True)
    elif isinstance(data, list):
        # Recursively convert each item in the list
        return [convert_config_to_dict(item) for item in data]
    elif isinstance(data, dict):
        # Recursively convert each value in the dictionary
        return {key: convert_config_to_dict(value) for key, value in data.items()}
    else:
        # Return the data as is if it's not a ListConfig, DictConfig, list, or dict
        return data


def write_json(data: dict, fpath: Path) -> None:
    fpath.parent.mkdir(exist_ok=True, parents=True)

    data = convert_config_to_dict(data)
    with open(fpath, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_jsonlines(fpath: Union[str, Path]) -> list[Any]:
    """
    Read a .jsonl from disk or from S3. If S3, open via fsspec.
    """
    if isinstance(fpath, str) and fpath.startswith("s3://"):
        with open_file(fpath, "r") as f:
            reader = jsonlines.Reader(f)
            return list(reader)
    else:
        with jsonlines.open(fpath, "r") as reader:
            return list(reader)


def write_jsonlines(data: dict, fpath: Path) -> None:
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with jsonlines.open(fpath, "w") as writer:
        writer.write_all(data)


def append_jsonlines(data: dict, fpath: Path) -> None:
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with jsonlines.open(fpath, "a") as writer:
        writer.write(data)


def write_info(info: dict, local_dir: Path):
    write_json(info, local_dir / INFO_PATH)


def load_info(local_dir: Union[str, Path]) -> dict:
    """
    Load meta/info.json from either a local Path or an S3 URI (string).
    After loading, convert 'shape' lists back to tuples.
    """
    if isinstance(local_dir, str) and local_dir.startswith("s3://"):
        fpath = f"{local_dir}/{INFO_PATH}"
    else:
        fpath = Path(local_dir) / INFO_PATH

    info = load_json(fpath)
    for ft in info["features"].values():
        ft["shape"] = tuple(ft["shape"])
    return info


def write_stats(stats: dict, local_dir: Path):
    serialized_stats = serialize_dict(stats)
    write_json(serialized_stats, local_dir / STATS_PATH)


def cast_stats_to_numpy(stats) -> dict[str, dict[str, np.ndarray]]:
    stats = {key: np.array(value) for key, value in flatten_dict(stats).items()}
    return unflatten_dict(stats)


def load_stats(local_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    if not (local_dir / STATS_PATH).exists():
        return None
    stats = load_json(local_dir / STATS_PATH)
    return cast_stats_to_numpy(stats)


def write_task(task_index: int, task: dict, local_dir: Path):
    task_dict = {
        "task_index": task_index,
        "task": task,
    }
    append_jsonlines(task_dict, local_dir / TASKS_PATH)


def load_tasks(local_dir: Union[str, Path]) -> tuple[dict, dict]:
    """
    Load meta/tasks.jsonl from local disk or S3.
    """
    if isinstance(local_dir, str) and local_dir.startswith("s3://"):
        fpath = f"{local_dir}/{TASKS_PATH}"
    else:
        fpath = Path(local_dir) / TASKS_PATH

    tasks = load_jsonlines(fpath)
    tasks = {
        item["task_index"]: item["task"]
        for item in sorted(tasks, key=lambda x: x["task_index"])
    }
    task_to_task_index = {task: task_index for task_index, task in tasks.items()}
    return tasks, task_to_task_index

def load_subtasks(local_dir: Union[str, Path]) -> tuple[dict, dict]:
    """
    Load meta/tasks.jsonl from local disk or S3.
    """
    if isinstance(local_dir, str) and local_dir.startswith("s3://"):
        fpath = f"{local_dir}/{SUBTASKS_PATH}"
    else:
        fpath = Path(local_dir) / SUBTASKS_PATH

    if not fpath.exists():
        return None

    subtasks = load_jsonlines(fpath)
    subtasks = {
        item["episode_index"]: item["subtasks"]
        for item in sorted(subtasks, key=lambda x: x["episode_index"])
    }
    return subtasks


def write_episode(episode: dict, local_dir: Path):
    append_jsonlines(episode, local_dir / EPISODES_PATH)


def load_episodes(local_dir: Union[str, Path]) -> dict:
    if isinstance(local_dir, str) and local_dir.startswith("s3://"):
        fpath = f"{local_dir}/{EPISODES_PATH}"
    else:
        fpath = Path(local_dir) / EPISODES_PATH
    episodes = load_jsonlines(fpath)
    return {
        item["episode_index"]: item
        for item in sorted(episodes, key=lambda x: x["episode_index"])
    }


def write_episode_stats(episode_index: int, episode_stats: dict, local_dir: Path):
    # We wrap episode_stats in a dictionary since `episode_stats["episode_index"]`
    # is a dictionary of stats and not an integer.
    episode_stats = {
        "episode_index": episode_index,
        "stats": serialize_dict(episode_stats),
    }
    append_jsonlines(episode_stats, local_dir / EPISODES_STATS_PATH)


def load_episodes_stats(local_dir: Union[str, Path]) -> dict:
    if isinstance(local_dir, str) and local_dir.startswith("s3://"):
        fpath = f"{local_dir}/{EPISODES_STATS_PATH}"
    else:
        fpath = Path(local_dir) / EPISODES_STATS_PATH
    episodes_stats = load_jsonlines(fpath)
    return {
        item["episode_index"]: cast_stats_to_numpy(item["stats"])
        for item in sorted(episodes_stats, key=lambda x: x["episode_index"])
    }


def backward_compatible_episodes_stats(
    stats: dict[str, dict[str, np.ndarray]], episodes: list[int]
) -> dict[str, dict[str, np.ndarray]]:
    return dict.fromkeys(episodes, stats)


def load_image_as_numpy(
    fpath: str | Path, dtype: np.dtype = np.float32, channel_first: bool = True
) -> np.ndarray:
    img = PILImage.open(fpath).convert("RGB")
    img_array = np.array(img, dtype=dtype)
    if channel_first:  # (H, W, C) -> (C, H, W)
        img_array = np.transpose(img_array, (2, 0, 1))
    if np.issubdtype(dtype, np.floating):
        img_array /= 255.0
    return img_array


def hf_transform_to_torch(items_dict: dict[torch.Tensor | None]):
    """Get a transform function that convert items from Hugging Face dataset (pyarrow)
    to torch tensors. Importantly, images are converted from PIL, which corresponds to
    a channel last representation (h w c) of uint8 type, to a torch image representation
    with channel first (c h w) of float32 type in range [0,1].
    """
    for key in items_dict:
        first_item = items_dict[key][0]
        if isinstance(first_item, PILImage.Image):
            to_tensor = transforms.ToTensor()
            items_dict[key] = [to_tensor(img) for img in items_dict[key]]
        elif first_item is None:
            pass
        else:
            items_dict[key] = [
                x if isinstance(x, str) else torch.tensor(x) for x in items_dict[key]
            ]
    return items_dict


def is_valid_version(version: str) -> bool:
    try:
        packaging.version.parse(version)
        return True
    except packaging.version.InvalidVersion:
        return False


def check_version_compatibility(
    repo_id: str,
    version_to_check: str | packaging.version.Version,
    current_version: str | packaging.version.Version,
    enforce_breaking_major: bool = True,
) -> None:
    v_check = (
        packaging.version.parse(version_to_check)
        if not isinstance(version_to_check, packaging.version.Version)
        else version_to_check
    )
    v_current = (
        packaging.version.parse(current_version)
        if not isinstance(current_version, packaging.version.Version)
        else current_version
    )
    if v_check.major < v_current.major and enforce_breaking_major:
        raise BackwardCompatibilityError(repo_id, v_check)
    elif v_check.minor < v_current.minor:
        logging.warning(V21_MESSAGE.format(repo_id=repo_id, version=v_check))


def get_repo_versions(repo_id: str) -> list[packaging.version.Version]:
    """Returns available valid versions (branches and tags) on given repo."""
    api = HfApi()
    repo_refs = api.list_repo_refs(repo_id, repo_type="dataset")
    repo_refs = [b.name for b in repo_refs.branches + repo_refs.tags]
    repo_versions = []
    for ref in repo_refs:
        with contextlib.suppress(packaging.version.InvalidVersion):
            repo_versions.append(packaging.version.parse(ref))

    return repo_versions


def get_safe_version(repo_id: str, version: str | packaging.version.Version) -> str:
    """
    Returns the version if available on repo or the latest compatible one.
    Otherwise, will throw a `CompatibilityError`.
    """
    target_version = (
        packaging.version.parse(version)
        if not isinstance(version, packaging.version.Version)
        else version
    )
    hub_versions = get_repo_versions(repo_id)

    if not hub_versions:
        raise RevisionNotFoundError(
            f"""Your dataset must be tagged with a codebase version.
            Assuming _version_ is the codebase_version value in the info.json, you can run this:
            ```python
            from huggingface_hub import HfApi

            hub_api = HfApi()
            hub_api.create_tag("{repo_id}", tag="_version_", repo_type="dataset")
            ```
            """
        )

    if target_version in hub_versions:
        return f"v{target_version}"

    compatibles = [
        v
        for v in hub_versions
        if v.major == target_version.major and v.minor <= target_version.minor
    ]
    if compatibles:
        return_version = max(compatibles)
        if return_version < target_version:
            logging.warning(
                f"Revision {version} for {repo_id} not found, using version v{return_version}"
            )
        return f"v{return_version}"

    lower_major = [v for v in hub_versions if v.major < target_version.major]
    if lower_major:
        raise BackwardCompatibilityError(repo_id, max(lower_major))

    upper_versions = [v for v in hub_versions if v > target_version]
    assert len(upper_versions) > 0
    raise ForwardCompatibilityError(repo_id, min(upper_versions))


def get_hf_features_from_features(features: dict) -> datasets.Features:
    hf_features = {}
    for key, ft in features.items():
        if ft["dtype"] == "video":
            continue
        elif ft["dtype"] == "image":
            hf_features[key] = datasets.Image()
        elif ft["shape"] == (1,):
            hf_features[key] = datasets.Value(dtype=ft["dtype"])
        elif len(ft["shape"]) == 1:
            hf_features[key] = datasets.Sequence(
                length=ft["shape"][0], feature=datasets.Value(dtype=ft["dtype"])
            )
        elif len(ft["shape"]) == 2:
            hf_features[key] = datasets.Array2D(shape=ft["shape"], dtype=ft["dtype"])
        elif len(ft["shape"]) == 3:
            hf_features[key] = datasets.Array3D(shape=ft["shape"], dtype=ft["dtype"])
        elif len(ft["shape"]) == 4:
            hf_features[key] = datasets.Array4D(shape=ft["shape"], dtype=ft["dtype"])
        elif len(ft["shape"]) == 5:
            hf_features[key] = datasets.Array5D(shape=ft["shape"], dtype=ft["dtype"])
        else:
            raise ValueError(f"Corresponding feature is not valid: {ft}")

    return datasets.Features(hf_features)


def create_empty_dataset_info(
    codebase_version: str,
    fps: int,
    robot_type: str | None,
    features: dict,
    use_videos: bool,
) -> dict:
    return {
        "codebase_version": codebase_version,
        "robot_type": robot_type,
        "total_episodes": 0,
        "total_frames": 0,
        "total_tasks": 0,
        "total_variations": 0,
        "total_videos": 0,
        "total_chunks": 0,
        "chunks_size": DEFAULT_CHUNK_SIZE,
        "fps": fps,
        "splits": {},
        "data_path": DEFAULT_PARQUET_PATH,
        "video_path": DEFAULT_VIDEO_PATH if use_videos else None,
        "features": features,
    }


def get_episode_data_index(
    episode_dicts: dict[dict], episodes: list[int] | None = None
) -> dict[str, torch.Tensor]:
    episode_lengths = {
        ep_idx: ep_dict["length"] for ep_idx, ep_dict in episode_dicts.items()
    }
    if episodes is not None:
        episode_lengths = {ep_idx: episode_lengths[ep_idx] for ep_idx in episodes}

    cumulative_lengths = list(accumulate(episode_lengths.values()))
    return {
        "from": torch.LongTensor([0] + cumulative_lengths[:-1]),
        "to": torch.LongTensor(cumulative_lengths),
    }


def check_timestamps_sync(
    timestamps: np.ndarray,
    episode_indices: np.ndarray,
    episode_data_index: dict[str, np.ndarray],
    fps: int,
    tolerance_s: float,
    raise_value_error: bool = True,
) -> bool:
    """
    This check is to make sure that each timestamp is separated from the next by (1/fps) +/- tolerance
    to account for possible numerical error.

    Args:
        timestamps (np.ndarray): Array of timestamps in seconds.
        episode_indices (np.ndarray): Array indicating the episode index for each timestamp.
        episode_data_index (dict[str, np.ndarray]): A dictionary that includes 'to',
            which identifies indices for the end of each episode.
        fps (int): Frames per second. Used to check the expected difference between consecutive timestamps.
        tolerance_s (float): Allowed deviation from the expected (1/fps) difference.
        raise_value_error (bool): Whether to raise a ValueError if the check fails.

    Returns:
        bool: True if all checked timestamp differences lie within tolerance, False otherwise.

    Raises:
        ValueError: If the check fails and `raise_value_error` is True.
    """
    if timestamps.shape != episode_indices.shape:
        raise ValueError(
            "timestamps and episode_indices should have the same shape. "
            f"Found {timestamps.shape=} and {episode_indices.shape=}."
        )

    # Consecutive differences
    diffs = np.diff(timestamps)
    within_tolerance = np.abs(diffs - (1.0 / fps)) <= tolerance_s

    # Mask to ignore differences at the boundaries between episodes
    mask = np.ones(len(diffs), dtype=bool)
    ignored_diffs = (
        episode_data_index["to"][:-1] - 1
    )  # indices at the end of each episode
    mask[ignored_diffs] = False
    filtered_within_tolerance = within_tolerance[mask]

    # Check if all remaining diffs are within tolerance
    if not np.all(filtered_within_tolerance):
        # Track original indices before masking
        original_indices = np.arange(len(diffs))
        filtered_indices = original_indices[mask]
        outside_tolerance_filtered_indices = np.nonzero(~filtered_within_tolerance)[0]
        outside_tolerance_indices = filtered_indices[outside_tolerance_filtered_indices]

        outside_tolerances = []
        for idx in outside_tolerance_indices:
            entry = {
                "timestamps": [timestamps[idx], timestamps[idx + 1]],
                "diff": diffs[idx],
                "episode_index": (
                    episode_indices[idx].item()
                    if hasattr(episode_indices[idx], "item")
                    else episode_indices[idx]
                ),
            }
            outside_tolerances.append(entry)

        if raise_value_error:
            raise ValueError(
                f"""One or several timestamps unexpectedly violate the tolerance inside episode range.
                This might be due to synchronization issues during data collection.
                \n{pformat(outside_tolerances)}"""
            )
        return False

    return True


def check_delta_timestamps(
    delta_timestamps: dict[str, list[float]],
    fps: int,
    tolerance_s: float,
    raise_value_error: bool = True,
) -> bool:
    """This will check if all the values in delta_timestamps are multiples of 1/fps +/- tolerance.
    This is to ensure that these delta_timestamps added to any timestamp from a dataset will themselves be
    actual timestamps from the dataset.
    """
    outside_tolerance = {}
    for key, delta_ts in delta_timestamps.items():
        within_tolerance = [
            abs(ts * fps - round(ts * fps)) / fps <= tolerance_s for ts in delta_ts
        ]
        if not all(within_tolerance):
            outside_tolerance[key] = [
                ts
                for ts, is_within in zip(delta_ts, within_tolerance, strict=True)
                if not is_within
            ]

    if len(outside_tolerance) > 0:
        if raise_value_error:
            raise ValueError(
                f"""
                The following delta_timestamps are found outside of tolerance range.
                Please make sure they are multiples of 1/{fps} +/- tolerance and adjust
                their values accordingly.
                \n{pformat(outside_tolerance)}
                """
            )
        return False

    return True


def get_delta_indices(
    delta_timestamps: dict[str, list[float]], fps: int, return_int: bool = True
) -> dict[str, list[int | float]]:
    delta_indices = {}
    for key, delta_ts in delta_timestamps.items():
        delta_indices[key] = [int(round(d * fps)) if return_int else d * fps for d in delta_ts]

    return delta_indices


def cycle(iterable):
    """The equivalent of itertools.cycle, but safe for Pytorch dataloaders.

    See https://github.com/pytorch/pytorch/issues/23900 for information on why itertools.cycle is not safe.
    """
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


def create_branch(repo_id, *, branch: str, repo_type: str | None = None) -> None:
    """Create a branch on a existing Hugging Face repo. Delete the branch if it already
    exists before creating it.
    """
    api = HfApi()

    branches = api.list_repo_refs(repo_id, repo_type=repo_type).branches
    refs = [branch.ref for branch in branches]
    ref = f"refs/heads/{branch}"
    if ref in refs:
        api.delete_branch(repo_id, repo_type=repo_type, branch=branch)

    api.create_branch(repo_id, repo_type=repo_type, branch=branch)


def create_lerobot_dataset_card(
    tags: list | None = None,
    dataset_info: dict | None = None,
    **kwargs,
) -> DatasetCard:
    """
    Keyword arguments will be used to replace values in ./lerobot/common/datasets/card_template.md.
    Note: If specified, license must be one of https://huggingface.co/docs/hub/repositories-licenses.
    """
    card_tags = ["LeRobot"]

    if tags:
        card_tags += tags
    if dataset_info:
        dataset_structure = "[meta/info.json](meta/info.json):\n"
        dataset_structure += f"```json\n{json.dumps(dataset_info, indent=4)}\n```\n"
        kwargs = {**kwargs, "dataset_structure": dataset_structure}
    card_data = DatasetCardData(
        license=kwargs.get("license"),
        tags=card_tags,
        task_categories=["robotics"],
        configs=[
            {
                "config_name": "default",
                "data_files": "data/*/*.parquet",
            }
        ],
    )

    card_template = (
        importlib.resources.files("lerobot.common.datasets") / "card_template.md"
    ).read_text()

    return DatasetCard.from_template(
        card_data=card_data,
        template_str=card_template,
        **kwargs,
    )


class IterableNamespace(SimpleNamespace):
    """
    A namespace object that supports both dictionary-like iteration and dot notation access.
    Automatically converts nested dictionaries into IterableNamespaces.

    This class extends SimpleNamespace to provide:
    - Dictionary-style iteration over keys
    - Access to items via both dot notation (obj.key) and brackets (obj["key"])
    - Dictionary-like methods: items(), keys(), values()
    - Recursive conversion of nested dictionaries

    Args:
        dictionary: Optional dictionary to initialize the namespace
        **kwargs: Additional keyword arguments passed to SimpleNamespace

    Examples:
        >>> data = {"name": "Alice", "details": {"age": 25}}
        >>> ns = IterableNamespace(data)
        >>> ns.name
        'Alice'
        >>> ns.details.age
        25
        >>> list(ns.keys())
        ['name', 'details']
        >>> for key, value in ns.items():
        ...     print(f"{key}: {value}")
        name: Alice
        details: IterableNamespace(age=25)
    """

    def __init__(self, dictionary: dict[str, Any] = None, **kwargs):
        super().__init__(**kwargs)
        if dictionary is not None:
            for key, value in dictionary.items():
                if isinstance(value, dict):
                    setattr(self, key, IterableNamespace(value))
                else:
                    setattr(self, key, value)

    def __iter__(self) -> Iterator[str]:
        return iter(vars(self))

    def __getitem__(self, key: str) -> Any:
        return vars(self)[key]

    def items(self):
        return vars(self).items()

    def values(self):
        return vars(self).values()

    def keys(self):
        return vars(self).keys()


def validate_frame(frame: dict, features: dict):
    optional_features = {"timestamp", "variation"}
    expected_features = (set(features) - set(DEFAULT_FEATURES.keys())) | {"task"}
    actual_features = set(frame.keys())

    error_message = validate_features_presence(
        actual_features, expected_features, optional_features
    )

    if "task" in frame:
        error_message += validate_feature_string("task", frame["task"])

    common_features = actual_features & (expected_features | optional_features)
    for name in common_features - {"task", "variation"}:
        error_message += validate_feature_dtype_and_shape(
            name, features[name], frame[name]
        )

    if error_message:
        raise ValueError(error_message)


def validate_features_presence(
    actual_features: set[str], expected_features: set[str], optional_features: set[str]
):
    error_message = ""
    missing_features = expected_features - actual_features
    extra_features = actual_features - (expected_features | optional_features)

    if missing_features or extra_features:
        error_message += "Feature mismatch in `frame` dictionary:\n"
        if missing_features:
            error_message += f"Missing features: {missing_features}\n"
        if extra_features:
            error_message += f"Extra features: {extra_features}\n"

    return error_message


def validate_feature_dtype_and_shape(
    name: str, feature: dict, value: np.ndarray | PILImage.Image | str
):
    expected_dtype = feature["dtype"]
    expected_shape = feature["shape"]
    if is_valid_numpy_dtype_string(expected_dtype):
        return validate_feature_numpy_array(name, expected_dtype, expected_shape, value)
    elif expected_dtype in ["image", "video"]:
        return validate_feature_image_or_video(name, expected_shape, value)
    elif expected_dtype == "string":
        return validate_feature_string(name, value)
    else:
        raise NotImplementedError(
            f"The feature dtype '{expected_dtype}' is not implemented yet."
        )


def validate_feature_numpy_array(
    name: str, expected_dtype: str, expected_shape: list[int], value: np.ndarray
):
    error_message = ""
    if isinstance(value, np.ndarray):
        actual_dtype = value.dtype
        actual_shape = value.shape

        if actual_dtype != np.dtype(expected_dtype):
            error_message += f"The feature '{name}' of dtype '{actual_dtype}' is not of the expected dtype '{expected_dtype}'.\n"

        if actual_shape != tuple(expected_shape):
            error_message += f"The feature '{name}' of shape '{actual_shape}' does not have the expected shape '{expected_shape}'.\n"
    else:
        error_message += f"The feature '{name}' is not a 'np.ndarray'. Expected type is '{expected_dtype}', but type '{type(value)}' provided instead.\n"

    return error_message


def validate_feature_image_or_video(
    name: str, expected_shape: list[str], value: np.ndarray | PILImage.Image
):
    # Note: The check of pixels range ([0,1] for float and [0,255] for uint8) is done by the image writer threads.
    error_message = ""
    if isinstance(value, np.ndarray):
        actual_shape = value.shape
        c, h, w = expected_shape
        if len(actual_shape) != 3 or (
            actual_shape != (c, h, w) and actual_shape != (h, w, c)
        ):
            error_message += f"The feature '{name}' of shape '{actual_shape}' does not have the expected shape '{(c, h, w)}' or '{(h, w, c)}'.\n"
    elif isinstance(value, PILImage.Image):
        pass
    else:
        error_message += f"The feature '{name}' is expected to be of type 'PIL.Image' or 'np.ndarray' channel first or channel last, but type '{type(value)}' provided instead.\n"

    return error_message


def validate_feature_string(name: str, value: str):
    if not isinstance(value, str):
        return f"The feature '{name}' is expected to be of type 'str', but type '{type(value)}' provided instead.\n"
    return ""


def validate_episode_buffer(episode_buffer: dict, total_episodes: int, features: dict):
    if "size" not in episode_buffer:
        raise ValueError("size key not found in episode_buffer")

    if "task" not in episode_buffer:
        raise ValueError("task key not found in episode_buffer")

    if episode_buffer["episode_index"] != total_episodes:
        # TODO(aliberts): Add option to use existing episode_index
        raise NotImplementedError(
            "You might have manually provided the episode_buffer with an episode_index that doesn't "
            "match the total number of episodes already in the dataset. This is not supported for now."
        )

    if episode_buffer["size"] == 0:
        raise ValueError(
            "You must add one or several frames with `add_frame` before calling `add_episode`."
        )

    buffer_keys = set(episode_buffer.keys()) - {"task", "size", "variation"}
    if not buffer_keys == set(features):
        raise ValueError(
            f"Features from `episode_buffer` don't match the ones in `features`."
            f"In episode_buffer not in features: {buffer_keys - set(features)}"
            f"In features not in episode_buffer: {set(features) - buffer_keys}"
        )


def load_variations(root: Path) -> tuple[dict[int, str], dict[str, int]]:
    """Load variations from variations.jsonl file.

    Args:
        root (Path): Root directory of the dataset.

    Returns:
        tuple[dict[int, str], dict[str, int]]: A tuple containing:
            - A dictionary mapping variation indices to variation names
            - A dictionary mapping variation names to variation indices
    """
    variations = {}
    variation_to_variation_index = {}
    variations_path = root / VARIATIONS_PATH
    if variations_path.is_file():
        with open(variations_path, "r") as f:
            for line in f:
                variation_dict = json.loads(line)
                variation_index = variation_dict["variation_index"]
                variation = variation_dict["variation"]
                variations[variation_index] = variation
                variation_to_variation_index[variation] = variation_index
    return variations, variation_to_variation_index


try:
    import humanfriendly

    _HAS_HUMANFRIENDLY = True
except ImportError:
    _HAS_HUMANFRIENDLY = False


def print_dataset_summary(summary: dict):
    total_seconds = summary["dataset_length"]
    if _HAS_HUMANFRIENDLY:
        duration_str = humanfriendly.format_timespan(total_seconds)
    else:
        duration_str = f"{total_seconds:.2f} seconds"

    # Build a Rich table
    table = Table(
        title=f"Dataset Summary for [bold green]{summary['dataset_id']}[/bold green]",
        expand=False,
    )
    table.add_column("Info", style="bold cyan", no_wrap=True)
    table.add_column("Value", style="bold magenta")

    table.add_row("Amount of Samples", str(format(summary["num_samples"], ",")))
    table.add_row("Amount of Episodes", str(format(summary["num_episodes"], ",")))
    table.add_row("Total Duration", duration_str)

    console = Console()
    console.print(table)

def get_config_root() -> Path:
    """
    Returns the absolute path to the lerobot/conf directory.
    
    This function finds the project root by locating the lerobot directory
    and returns the path to the conf subdirectory within it.
    
    Returns:
        Path: Absolute path to the lerobot/conf directory
        
    Raises:
        FileNotFoundError: If the lerobot/conf directory cannot be found
    """
    # Start from the current file's directory and go up to find the project root
    current_file = Path(__file__).resolve()
    
    # Go up from lerobot/common/utils/utils.py to the project root
    project_root = current_file.parent.parent.parent.parent
    
    # Construct path to lerobot/conf
    config_root = project_root / "lerobot" / "conf"
    
    if not config_root.exists():
        raise FileNotFoundError(f"Config directory not found at: {config_root}")
    
    return config_root


def map_subtask_id_to_episode_id(global_subtask_timestamp: int, data: dict, episodes_indices: np.ndarray, subtasks_indices: np.ndarray):
    """
    Map an index from compressed M-space (only inside allowed intervals) to original N-space index.
    Conventions:
    - episodes_indices[e] is episode e end in N-space (exclusive)
    - subtasks_indices[e] is episode e end in M-space (exclusive)
    - local intervals are [start, end_excl)
    """
    if global_subtask_timestamp < 0:
        raise IndexError("global_subtask_timestamp must be non-negative")
    total_M = int(subtasks_indices[-1]) if len(subtasks_indices) > 0 else 0
    if global_subtask_timestamp >= total_M:
        raise IndexError(f"global_subtask_timestamp out of range: {global_subtask_timestamp} >= {total_M}")

    # 1) Episode in M-space
    episode_id = int(np.searchsorted(subtasks_indices, global_subtask_timestamp, side="right"))
    prev_M_end = int(subtasks_indices[episode_id - 1]) if episode_id > 0 else 0
    offset_in_episode_allowed = int(global_subtask_timestamp - prev_M_end)

    ep = data[episode_id]
    prefix = ep["subtask_prefix"]

    # 2) Interval inside episode by allowed offset
    interval_id = int(np.searchsorted(prefix, offset_in_episode_allowed, side="right"))
    prev_prefix = int(prefix[interval_id - 1]) if interval_id > 0 else 0
    shift_in_interval = int(offset_in_episode_allowed - prev_prefix)

    start = int(ep["start"][interval_id])
    end_excl = int(ep["end_excl"][interval_id])
    interval_len = end_excl - start

    # Safety
    if not (0 <= shift_in_interval < interval_len):
        raise AssertionError(
            f"Shift out of bounds: {shift_in_interval=}, {interval_len=}, {episode_id=}, {interval_id=}"
        )

    local_episode_timestamp = start + shift_in_interval

    # 3) Convert episode-local to global N-space
    prev_N_end = int(episodes_indices[episode_id - 1]) if episode_id > 0 else 0
    global_episode_timestamp = prev_N_end + local_episode_timestamp

    # Final safety: must be strictly inside this episode [prev_N_end, episodes_indices[episode_id])
    if not (global_episode_timestamp < int(episodes_indices[episode_id])):
        raise AssertionError(
            f"Mapped to episode boundary or beyond: "
            f"{global_episode_timestamp=} >= {int(episodes_indices[episode_id])=}. "
            f"Likely interval end convention mismatch."
        )

    return global_episode_timestamp, episode_id, interval_id


def parse_subtask_info(episodes, subtasks):
    """
    Build per-episode interval metadata in a single consistent convention:
    - Episode timeline is [0, episode_length)
    - Each interval is [start, end_excl)
    """

    data = {}

    # IMPORTANT: rely on insertion order of episodes/subtasks keys (Py3.7+ keeps dict order)
    # If you have non-aligned ordering, sort keys explicitly and use that order everywhere.
    for eid, episode in episodes.items():
        ep_len = int(episode["length"])

        starts = np.asarray([s["start_frame"] for s in subtasks[eid]], dtype=np.int64)
        ends_raw = np.asarray([s["end_frame"] for s in subtasks[eid]], dtype=np.int64)

        # --- Normalize "end" to end-exclusive ---
        # Heuristic:
        # 1) If any end equals episode_length, it's almost certainly end-exclusive already.
        # 2) Else, if all ends <= episode_length-1 and any end equals episode_length-1, it is likely inclusive.
        # 3) Otherwise, fallback: if all ends < episode_length -> assume inclusive; if any end > episode_length -> error.
        if np.any(ends_raw == ep_len):
            ends_excl = ends_raw
        else:
            if np.any(ends_raw == ep_len - 1):
                # Likely inclusive [start, end_incl], convert to exclusive
                ends_excl = ends_raw + 1
            else:
                if np.all(ends_raw < ep_len):
                    # Ambiguous; choose inclusive -> exclusive (matches your previous formula)
                    ends_excl = ends_raw + 1
                else:
                    raise ValueError(f"Episode {eid}: ends exceed episode length: {ends_raw.max()} > {ep_len}")

        # --- Validate interval bounds in end-exclusive form ---
        if not np.all(starts >= 0):
            raise ValueError(f"Episode {eid}: start < 0 encountered")
        if not np.all(ends_excl <= ep_len):
            raise ValueError(f"Episode {eid}: end_excl > episode_length encountered: {ends_excl.max()} > {ep_len}")
        if not np.all(starts < ends_excl):
            raise ValueError(f"Episode {eid}: start >= end_excl encountered")

        # Lengths and prefix in end-exclusive convention
        lengths = ends_excl - starts
        prefix = np.cumsum(lengths, dtype=np.int64)
        total_allowed = int(prefix[-1]) if len(prefix) > 0 else 0

        data[eid] = {
            "episode_length": ep_len,
            "start": starts,
            "end_excl": ends_excl,
            "subtask_lengths": lengths,
            "subtask_prefix": prefix,
            "subtask_length": total_allowed,
        }

    # Global episode end-exclusive boundaries in N-space
    episode_lengths = np.asarray([v["episode_length"] for v in data.values()], dtype=np.int64)
    episodes_indices = np.cumsum(episode_lengths, dtype=np.int64)

    # Global allowed-length end-exclusive boundaries in M-space
    allowed_lengths = np.asarray([v["subtask_length"] for v in data.values()], dtype=np.int64)
    subtasks_indices = np.cumsum(allowed_lengths, dtype=np.int64)

    episode_length_total = int(episodes_indices[-1]) if len(episodes_indices) > 0 else 0
    subtask_length_total = int(subtasks_indices[-1]) if len(subtasks_indices) > 0 else 0

    return data, episodes_indices, subtasks_indices, episode_length_total, subtask_length_total