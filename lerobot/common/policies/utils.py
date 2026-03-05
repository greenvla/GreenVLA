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

import torch
from torch import nn
from typing import List, TypeVar, Union

T = TypeVar("T")


def populate_queues(queues, batch):
    for key in batch:
        # Ignore keys not in the queues already (leaving the responsibility to the caller to make sure the
        # queues have the keys they want).
        if key not in queues:
            continue
        if len(queues[key]) != queues[key].maxlen:
            # initialize by copying the first observation several times until the queue is full
            while len(queues[key]) != queues[key].maxlen:
                queues[key].append(batch[key])
        else:
            # add latest observation to the queue
            queues[key].append(batch[key])
    return queues


def get_device_from_parameters(module: nn.Module) -> torch.device:
    """Get a module's device by checking one of its parameters.

    Note: assumes that all parameters have the same device
    """
    return next(iter(module.parameters())).device


def get_dtype_from_parameters(module: nn.Module) -> torch.dtype:
    """Get a module's parameter dtype by checking one of its parameters.

    Note: assumes that all parameters have the same dtype.
    """
    return next(iter(module.parameters())).dtype


def get_output_shape(module: nn.Module, input_shape: tuple) -> tuple:
    """
    Calculates the output shape of a PyTorch module given an input shape.

    Args:
        module (nn.Module): a PyTorch module
        input_shape (tuple): A tuple representing the input shape, e.g., (batch_size, channels, height, width)

    Returns:
        tuple: The output shape of the module.
    """
    dummy_input = torch.zeros(size=input_shape)
    with torch.inference_mode():
        output = module(dummy_input)
    return tuple(output.shape)


def compute_lps(pattern: List[T]) -> List[int]:
    """
    Compute the longest proper prefix which is also suffix (lps) array for KMP.

    Args:
        pattern: The pattern list for which to compute the LPS array.

    Returns:
        A list lps where lps[i] is the length of the longest proper prefix of
        pattern[0:i+1] which is also a suffix of pattern[0:i+1].
    """
    length = 0
    lps = [0] * len(pattern)
    # the loop calculates lps[i] for i from 1 to len(pattern)-1
    i = 1
    while i < len(pattern):
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
                # note: we do not increment i here
            else:
                lps[i] = 0
                i += 1
    return lps


def find_sublist_indices(text: List[T], pattern: List[T]) -> List[int]:
    """
    Find all starting indices in 'text' where 'pattern' occurs as a contiguous sublist.

    Args:
        text: The list to search within.
        pattern: The sublist pattern to search for.

    Returns:
        A list of integers representing all starting indices in 'text' where
        'pattern' is found. Returns an empty list if 'pattern' is not found.

    Raises:
        ValueError: If the pattern is empty.
    """
    if not pattern:
        raise ValueError("Empty pattern is not allowed")

    # Preprocess pattern to get lps array
    lps = compute_lps(pattern)
    indices: List[int] = []

    i = 0  # index for text
    j = 0  # index for pattern
    while i < len(text):
        if text[i] == pattern[j]:
            i += 1
            j += 1
            if j == len(pattern):
                # Match found; append starting index
                indices.append(i - j)
                # Continue searching for next possible match
                j = lps[j - 1]
        else:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1

    return indices


def find_sublist_index(text: List[T], pattern: List[T]) -> Union[int, None]:
    """
    Find the first occurrence of 'pattern' in 'text'.

    Args:
        text: The list to search within.
        pattern: The sublist pattern to search for.

    Returns:
        The starting index of the first occurrence of 'pattern' in 'text',
        or -1 if 'pattern' is not found.
    """
    indices = find_sublist_indices(text, pattern)
    return indices[0] if indices else None
