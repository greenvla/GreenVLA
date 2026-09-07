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

from pathlib import Path
from typing import Any, Type, TypeVar

from huggingface_hub.utils import validate_hf_hub_args

T = TypeVar("T", bound="HubMixin")


class HubMixin:
    """
    A Mixin containing the functionality to push an object to the hub.

    This is similar to huggingface_hub.ModelHubMixin but is lighter and makes less assumptions about its
    subclasses (in particular, the fact that it's not necessarily a model).

    The inheriting classes must implement '_save_pretrained' and 'from_pretrained'.
    """

    @classmethod
    @validate_hf_hub_args
    def from_pretrained(
        cls: Type[T],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **kwargs,
    ) -> T:
        """
        Download the object from the Huggingface Hub and instantiate it.

        Args:
            pretrained_name_or_path (`str`, `Path`):
                - Either the `repo_id` (string) of the object hosted on the Hub, e.g. `lerobot/diffusion_pusht`.
                - Or a path to a `directory` containing the object files saved using `.save_pretrained`,
                    e.g., `../path/to/my_model_directory/`.
            revision (`str`, *optional*):
                Revision on the Hub. Can be a branch name, a git tag or any commit id.
                Defaults to the latest commit on `main` branch.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether to force (re-)downloading the files from the Hub, overriding the existing cache.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on every request.
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. By default, it will use the token
                cached when running `huggingface-cli login`.
            cache_dir (`str`, `Path`, *optional*):
                Path to the folder where cached files are stored.
            local_files_only (`bool`, *optional*, defaults to `False`):
                If `True`, avoid downloading the file and return the path to the local cached file if it exists.
            kwargs (`Dict`, *optional*):
                Additional kwargs to pass to the object during initialization.
        """
        raise NotImplementedError

