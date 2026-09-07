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
import abc
import logging
import os
from pathlib import Path
from typing import Type, TypeVar

import packaging
import safetensors
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from huggingface_hub.errors import HfHubHTTPError
from safetensors.torch import load_model as load_model_as_safetensor
from safetensors.torch import save_model as save_model_as_safetensor
from torch import Tensor, nn

from lerobot.common.utils.hub import HubMixin
from lerobot.configs.policies import PreTrainedConfig

T = TypeVar("T", bound="PreTrainedPolicy")

DEFAULT_POLICY_CARD = """
---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
{{ card_data }}
---

This policy has been pushed to the Hub using [LeRobot](https://github.com/huggingface/lerobot):
- Docs: {{ docs_url | default("[More Information Needed]", true) }}
"""


class PreTrainedPolicy(nn.Module, HubMixin, abc.ABC):
    """
    Base class for policy models.
    """

    config_class: None
    name: None

    def __init__(self, config: PreTrainedConfig, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, PreTrainedConfig):
            raise ValueError(
                f"Parameter config in `{self.__class__.__name__}(config)` should be an instance of class "
                "`PreTrainedConfig`. To create a model from a pretrained model use "
                f"`model = {self.__class__.__name__}.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.config = config

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "config_class", None):
            raise TypeError(f"Class {cls.__name__} must define 'config_class'")
        if not getattr(cls, "name", None):
            raise TypeError(f"Class {cls.__name__} must define 'name'")

    def _save_pretrained(self, save_directory: Path) -> None:
        self.config._save_pretrained(save_directory)
        model_to_save = self.module if hasattr(self, "module") else self
        save_model_as_safetensor(
            model_to_save, str(save_directory / SAFETENSORS_SINGLE_FILE)
        )

    @classmethod
    def from_pretrained(
        cls: Type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,
        **kwargs,
    ) -> T:
        """
        The policy is set in evaluation mode by default using `policy.eval()` (dropout modules are
        deactivated). To train it, you should first set it back in training mode with `policy.train()`.
        """
        strict=True
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )
        model_id = str(pretrained_name_or_path)
        instance = cls(config, **kwargs)
        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            model_file = os.path.join(model_id, SAFETENSORS_SINGLE_FILE)
            policy = cls._load_as_safetensor(
                instance, model_file, config.device, strict
            )
        else:
            try:
                model_file = hf_hub_download(
                    repo_id=model_id,
                    filename=SAFETENSORS_SINGLE_FILE,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
                policy = cls._load_as_safetensor(
                    instance, model_file, config.device, strict
                )
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{SAFETENSORS_SINGLE_FILE} not found on the HuggingFace Hub in {model_id}"
                ) from e

        policy.to(config.device)
        policy.eval()
        return policy

    @classmethod
    def _load_as_safetensor(
        cls, model: T, model_file: str, map_location: str, strict: bool
    ) -> T:
        needs_filtering = cls._checkpoint_needs_filtering(model, model_file)
        if needs_filtering:
            logging.info(
                "Checkpoint has extra/mismatched keys or dtypes — "
                "using memory-efficient filtered loader."
            )
            cls._load_filtered_state_dict(model, model_file, map_location)
        else:
            cls._load_safetensor_impl(model, model_file, map_location, strict)
        return model

    @classmethod
    def _checkpoint_needs_filtering(
        cls, model: T, model_file: str
    ) -> bool:
        """Quick check whether the checkpoint keys/dtypes match the model.
        Avoids materialising full state_dict — only inspects names and dtypes."""
        from safetensors import safe_open

        model_meta = {}
        for name, param in model.named_parameters():
            model_meta[name] = param.dtype
        for name, buf in model.named_buffers():
            model_meta[name] = buf.dtype
        model_keys = set(model_meta.keys())

        with safe_open(model_file, framework="pt", device="cpu") as f:
            ckpt_keys = set(f.keys())
            if ckpt_keys != model_keys:
                return True
            for key in list(ckpt_keys)[:5]:
                t = f.get_tensor(key)
                if t.dtype != model_meta[key]:
                    return True
        return False

    @classmethod
    def _load_safetensor_impl(
        cls, model: T, model_file: str, map_location: str, strict: bool
    ) -> None:
        if packaging.version.parse(safetensors.__version__) < packaging.version.parse(
            "0.4.3"
        ):
            load_model_as_safetensor(model, model_file, strict=strict)
            if map_location != "cpu":
                logging.warning(
                    "Loading model weights on other devices than 'cpu' is not supported natively in your version of safetensors."
                    " This means that the model is loaded on 'cpu' first and then copied to the device."
                    " This leads to a slower loading time."
                    " Please update safetensors to version 0.4.3 or above for improved performance."
                )
                model.to(map_location)
        else:
            safetensors.torch.load_model(
                model, model_file, strict=strict, device=map_location
            )

    @classmethod
    def _load_filtered_state_dict(
        cls, model: T, model_file: str, map_location: str
    ) -> None:
        """Memory-efficient loader for checkpoints that may have extra keys
        (e.g. untied weights from FSDP) or dtype mismatches (e.g. float32
        saved by FSDP when the model uses bfloat16).

        Uses safe_open to lazily read one tensor at a time, cast it to the
        model's expected dtype, and assign it directly into the model's
        parameters/buffers — avoiding loading the full checkpoint into RAM."""
        from safetensors import safe_open

        model_meta = {}
        for name, param in model.named_parameters():
            model_meta[name] = param.dtype
        for name, buf in model.named_buffers():
            model_meta[name] = buf.dtype
        model_keys = set(model_meta.keys())

        # Non-persistent buffers (e.g. rotary_emb.inv_freq) are recomputed
        # at init and intentionally excluded from checkpoints.
        non_persistent = set()
        for module_name, module in model.named_modules():
            for buf_name in getattr(module, "_non_persistent_buffers_set", set()):
                full = f"{module_name}.{buf_name}" if module_name else buf_name
                non_persistent.add(full)

        loaded_keys = []
        skipped_keys = []

        with safe_open(model_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key not in model_keys:
                    skipped_keys.append(key)
                    continue
                tensor = f.get_tensor(key)
                expected_dtype = model_meta[key]
                if tensor.dtype != expected_dtype:
                    tensor = tensor.to(expected_dtype)
                parts = key.split(".")
                mod = model
                for part in parts[:-1]:
                    mod = getattr(mod, part)
                param_name = parts[-1]
                existing = getattr(mod, param_name, None)
                if isinstance(existing, nn.Parameter):
                    existing.data.copy_(tensor)
                else:
                    setattr(mod, param_name, tensor)
                loaded_keys.append(key)
                del tensor

        missing = model_keys - set(loaded_keys) - non_persistent
        if skipped_keys:
            logging.warning("Skipped %d unexpected checkpoint keys: %s",
                            len(skipped_keys), sorted(skipped_keys)[:5])
        if missing:
            logging.warning("Missing %d keys after filtered load: %s",
                            len(missing), sorted(missing)[:5])

    # def generate_model_card(self, *args, **kwargs) -> ModelCard:
    #     card = ModelCard.from_template(
    #         card_data=self._hub_mixin_info.model_card_data,
    #         template_str=self._hub_mixin_info.model_card_template,
    #         repo_url=self._hub_mixin_info.repo_url,
    #         docs_url=self._hub_mixin_info.docs_url,
    #         **kwargs,
    #     )
    #     return card


    @abc.abstractmethod
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Return one action to run in the environment (potentially in batch mode).

        When the model uses a history of observations, or outputs a sequence of actions, this method deals
        with caching.
        """
        raise NotImplementedError
