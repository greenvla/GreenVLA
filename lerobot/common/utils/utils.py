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
import logging
import os
from copy import copy
from pathlib import Path
import torch






def auto_select_torch_device() -> torch.device:
    """Tries to select automatically a torch device."""
    if torch.cuda.is_available():
        logging.info("Cuda backend detected, using cuda.")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        logging.info("Metal backend detected, using cuda.")
        return torch.device("mps")
    else:
        logging.warning(
            "No accelerated backend detected. Using default cpu, this will be slow."
        )
        return torch.device("cpu")


# TODO(Steven): Remove log. log shouldn't be an argument, this should be handled by the logger level


def get_safe_dtype(dtype: torch.dtype, device: str | torch.device):
    """
    mps is currently not compatible with float64
    """
    if isinstance(device, torch.device):
        device = device.type
    if device == "mps" and dtype == torch.float64:
        return torch.float32
    else:
        return dtype


def is_torch_device_available(try_device: str) -> bool:
    try_device = str(try_device)  # Ensure try_device is a string
    if try_device == "cuda":
        return torch.cuda.is_available()
    elif try_device == "mps":
        return torch.backends.mps.is_available()
    elif try_device == "cpu":
        return True
    else:
        raise ValueError(
            f"Unknown device {try_device}. Supported devices are: cuda, mps or cpu."
        )


def is_amp_available(device: str):
    if device in ["cuda", "cpu"]:
        return True
    elif device == "mps":
        return False
    else:
        raise ValueError(f"Unknown device '{device}.")



















def move_batch_to_device(batch, target_device):
    if isinstance(batch, torch.Tensor):
        return batch.to(target_device)
    elif isinstance(batch, dict):
        return {k: move_batch_to_device(v, target_device) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return type(batch)(move_batch_to_device(x, target_device) for x in batch)
    return batch




def get_local_hf_snapshot_or_repo_id(repo_id: str, revision: str = "main") -> str:
    """
    If HF_HOME is set and a cached snapshot exists, return its path.
    If HF_HOME is NOT set, just return the original repo_id (so normal HF logic is used).

    Example:
        model_source = get_local_hf_snapshot_or_repo_id("Qwen/Qwen3-VL-4B-Instruct")
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_source, local_files_only=True)
        processor = AutoProcessor.from_pretrained(model_source, local_files_only=True)
    """
    hf_home_env = os.environ.get("HF_HOME")
    if not hf_home_env:
        # No custom cache configured → fall back to standard HF behavior
        return repo_id

    hf_home = Path(hf_home_env)
    repo_cache_name = "models--" + repo_id.replace("/", "--")
    repo_dir = hf_home / "hub" / repo_cache_name

    if not repo_dir.exists():
        # Cache for this repo not found under HF_HOME → fall back to repo_id
        return repo_id

    # Try to resolve the requested revision via refs/<revision>
    if revision is not None:
        ref_file = repo_dir / "refs" / revision
        if ref_file.exists():
            commit_hash = ref_file.read_text().strip()
            snap_dir = repo_dir / "snapshots" / commit_hash
            if snap_dir.exists():
                return str(snap_dir)

    # Fallback: latest snapshot directory, if any
    snapshots_root = repo_dir / "snapshots"
    if not snapshots_root.exists():
        return repo_id

    snapshot_dirs = sorted([p for p in snapshots_root.iterdir() if p.is_dir()])
    if not snapshot_dirs:
        return repo_id

    return str(snapshot_dirs[-1])