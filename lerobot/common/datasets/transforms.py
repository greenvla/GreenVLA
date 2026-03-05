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
import collections
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence, Union

import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import Transform
from torchvision.transforms.v2 import functional as F  # noqa: N812
import torch.nn.functional as F_nn
import math
from dataclasses import dataclass
from typing import Optional, Tuple


class RandomSubsetApply(Transform):
    """Apply a random subset of N transformations from a list of transformations.

    Args:
        transforms: list of transformations.
        p: represents the multinomial probabilities (with no replacement) used for sampling the transform.
            If the sum of the weights is not 1, they will be normalized. If ``None`` (default), all transforms
            have the same probability.
        n_subset: number of transformations to apply. If ``None``, all transforms are applied.
            Must be in [1, len(transforms)].
        random_order: apply transformations in a random order.
    """

    def __init__(
        self,
        transforms: Sequence[Callable],
        p: list[float] | None = None,
        n_subset: int | None = None,
        random_order: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(transforms, Sequence):
            raise TypeError("Argument transforms should be a sequence of callables")
        if p is None:
            p = [1] * len(transforms)
        elif len(p) != len(transforms):
            raise ValueError(
                f"Length of p doesn't match the number of transforms: {len(p)} != {len(transforms)}"
            )

        if n_subset is None:
            n_subset = len(transforms)
        elif not isinstance(n_subset, int):
            raise TypeError("n_subset should be an int or None")
        elif not (1 <= n_subset <= len(transforms)):
            raise ValueError(
                f"n_subset should be in the interval [1, {len(transforms)}]"
            )

        self.transforms = transforms
        total = sum(p)
        self.p = [prob / total for prob in p]
        self.n_subset = n_subset
        self.random_order = random_order

        self.selected_transforms = None

    def forward(self, *inputs: Any) -> Any:
        needs_unpacking = len(inputs) > 1

        selected_indices = torch.multinomial(torch.tensor(self.p), self.n_subset)
        if not self.random_order:
            selected_indices = selected_indices.sort().values

        self.selected_transforms = [self.transforms[i] for i in selected_indices]

        for transform in self.selected_transforms:
            outputs = transform(*inputs)
            inputs = outputs if needs_unpacking else (outputs,)

        return outputs

    def extra_repr(self) -> str:
        return (
            f"transforms={self.transforms}, "
            f"p={self.p}, "
            f"n_subset={self.n_subset}, "
            f"random_order={self.random_order}"
        )


class RandomElasticDeform(Transform):
    """Grid-like elastic deformation via smooth random displacement + grid_sample.

    If Kornia is available, uses its RandomElasticTransform; otherwise a lightweight
    implementation that samples a low-res displacement field and upsamples it.
    """

    def __init__(
        self,
        p: float = 0.3,
        grid_xy: Tuple[int, int] = (10, 10),
        magnitude: float = 2.0,
        mode: str = "bilinear",
        padding_mode: str = "border",
        align_corners: bool = True,
    ):
        super().__init__()
        self.p = float(p)
        self.grid_xy = grid_xy
        self.magnitude = float(magnitude)
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners 

    @staticmethod
    def _make_normalized_grid(B: int, H: int, W: int, device, dtype):
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device, dtype=dtype),
            torch.linspace(-1, 1, W, device=device, dtype=dtype),
            indexing="ij",
        )
        grid = torch.stack((xx, yy), dim=-1)  # (H, W, 2)
        return grid.expand(B, H, W, 2)  # (B, H, W, 2)

    def _elastic_displacement(self, B, H, W, device, dtype):
        # sample a coarse displacement field and upsample to HxW
        gx, gy = self.grid_xy
        # +2 to reduce edge falloff after interpolation
        disp = torch.randn(B, 2, gy + 2, gx + 2, device=device, dtype=dtype)
        disp = F_nn.interpolate(disp, size=(H, W), mode="bicubic", align_corners=True)
        # normalize and scale to pixels -> convert to normalized grid units later
        disp = disp / (disp.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6))
        # scale by magnitude in pixels -> normalized coords scale is 2 / size
        scale_x = (2.0 / max(W - 1, 1)) * self.magnitude
        scale_y = (2.0 / max(H - 1, 1)) * self.magnitude
        disp[:, 0, :, :] *= scale_x
        disp[:, 1, :, :] *= scale_y
        # (B, H, W, 2)
        disp = disp.permute(0, 2, 3, 1).contiguous()
        return disp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if self.p < 1.0:
            apply = (torch.rand((B,), device=x.device) < self.p)
        else:
            apply = torch.ones((B,), device=x.device, dtype=torch.bool)

        if not torch.any(apply):
            return x


        # Torch-only path
        base_grid = self._make_normalized_grid(B, H, W, x.device, x.dtype)
        disp = self._elastic_displacement(B, H, W, x.device, x.dtype)
        grid = base_grid + disp  # (B, H, W, 2)
        x_warp = F_nn.grid_sample(
            x, grid, mode=self.mode, padding_mode=self.padding_mode, align_corners=self.align_corners
        )
        return torch.where(apply.view(B, 1, 1, 1), x_warp, x)


class RandomMultiErasing(Transform):
    """Apply RandomErasing multiple times to approximate multiple coarse holes."""

    def __init__(
        self,
        p: float,
        passes: int,
        value: str | float | Tuple[float, ...] = "random",
        scale: Tuple[float, float] = (0.02, 0.2),
        ratio: Tuple[float, float] = (0.3, 3.3),
    ):
        super().__init__()
        self.p = float(p)
        self.passes = int(passes)
        self.erase = v2.RandomErasing(p=1.0, value=value, scale=scale, ratio=ratio, inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        if self.p < 1.0:
            apply = (torch.rand((B,), device=x.device) < self.p)
        else:
            apply = torch.ones((B,), device=x.device, dtype=torch.bool)
        if not torch.any(apply):
            return x

        out = x
        # do 'passes' erasures; erasing op is fast and differentiable
        for _ in range(self.passes):
            out_candidate = self.erase(out)
            out = torch.where(apply.view(B, 1, 1, 1), out_candidate, out)
        return out
    
    

class RandomPixelDropout(Transform):
    """Drop random pixels. Expects (B,C,H,W) in [0,1]."""

    def __init__(
        self,
        dropout_prob: float,
        p: float = 1.0,
        per_channel: bool = False,
        fill_random: bool = False,
    ):
        super().__init__()
        self.p = float(p)
        self.dropout_prob = float(dropout_prob)
        self.per_channel = bool(per_channel)
        self.fill_random = bool(fill_random)

    def _get_mask(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if self.per_channel:
            shape = (B, C, H, W)
        else:
            shape = (B, 1, H, W)
        return torch.rand(shape, device=x.device, dtype=x.dtype) < self.dropout_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p < 1.0:
            # per-sample application probability
            apply = (torch.rand((x.shape[0],), device=x.device) < self.p).view(-1, 1, 1, 1)
            apply = apply.to(dtype=torch.bool)
        else:
            apply = torch.ones((x.shape[0], 1, 1, 1), device=x.device, dtype=torch.bool)

        mask = self._get_mask(x) & apply  # broadcast apply over spatial
        if self.fill_random:
            fill = torch.rand_like(x)
        else:
            fill = torch.zeros_like(x)

        return torch.where(mask, fill, x)

class SharpnessJitter(Transform):
    """Randomly change the sharpness of an image or video.

    Similar to a v2.RandomAdjustSharpness with p=1 and a sharpness_factor sampled randomly.
    While v2.RandomAdjustSharpness applies — with a given probability — a fixed sharpness_factor to an image,
    SharpnessJitter applies a random sharpness_factor each time. This is to have a more diverse set of
    augmentations as a result.

    A sharpness_factor of 0 gives a blurred image, 1 gives the original image while 2 increases the sharpness
    by a factor of 2.

    If the input is a :class:`torch.Tensor`,
    it is expected to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        sharpness: How much to jitter sharpness. sharpness_factor is chosen uniformly from
            [max(0, 1 - sharpness), 1 + sharpness] or the given
            [min, max]. Should be non negative numbers.
    """

    def __init__(self, sharpness: float | Sequence[float]) -> None:
        super().__init__()
        self.sharpness = self._check_input(sharpness)

    def _check_input(self, sharpness):
        if isinstance(sharpness, (int, float)):
            if sharpness < 0:
                raise ValueError(
                    "If sharpness is a single number, it must be non negative."
                )
            sharpness = [1.0 - sharpness, 1.0 + sharpness]
            sharpness[0] = max(sharpness[0], 0.0)
        elif isinstance(sharpness, collections.abc.Sequence) and len(sharpness) == 2:
            sharpness = [float(v) for v in sharpness]
        else:
            raise TypeError(
                f"{sharpness=} should be a single number or a sequence with length 2."
            )

        if not 0.0 <= sharpness[0] <= sharpness[1]:
            raise ValueError(
                f"sharpness values should be between (0., inf), but got {sharpness}."
            )

        return float(sharpness[0]), float(sharpness[1])

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        sharpness_factor = (
            torch.empty(1).uniform_(self.sharpness[0], self.sharpness[1]).item()
        )
        return {"sharpness_factor": sharpness_factor}

    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        sharpness_factor = params["sharpness_factor"]
        return self._call_kernel(
            F.adjust_sharpness, inpt, sharpness_factor=sharpness_factor
        )




class JaxLikeAugmentations(Transform):
    """Apply JAX-like augmentations to a batch of images with separate configs for base and wrist cameras.

    Args:
        config: OmegaConf config with base_camera and wrist_camera transform definitions.
    Notes:
        - Transforms in torchvision v2 are batch-aware and accept (B, C, H, W).
        - Random params are sampled per-call and per-sample even when the
          same Transform/Compose instance is reused across calls.
    """

    def __init__(self, config: Any):
        super().__init__()
        import hydra
        
        self.config = config

        # Instantiate pipelines from Hydra configs
        self._base_pipeline = hydra.utils.instantiate(config.base_camera)
        self._wrist_pipeline = hydra.utils.instantiate(config.wrist_camera)

    @torch.no_grad()
    def forward(
        self,
        image_batch_float_0_1: torch.Tensor,  # (B, C, H, W), float in [0, 1]
        image_key: str,
        augmentations_mask: Optional[torch.Tensor] = None,  # (B,) bool or broadcastable
    ) -> torch.Tensor:
        x = image_batch_float_0_1

        is_wrist = ("wrist" in image_key)
        if is_wrist:
            pipeline = self._wrist_pipeline
        else:
            pipeline = self._base_pipeline

        aug = pipeline(x)

        if augmentations_mask is None:
            return aug

        # Broadcast mask to (B,1,1,1) and select per-sample outputs
        mask = augmentations_mask.to(device=x.device, dtype=torch.bool).view(-1, 1, 1, 1)
        out = torch.where(mask, aug, x)
        return out

    


