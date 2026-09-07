"""GreenVLAv1.1 image/text tokenization, preserving the trained prompt format."""

import logging

import numpy as np
import torch
from transformers import AutoProcessor

from lerobot.common.datasets.policy_metadata import format_policy_metadata_prompt
from lerobot.common.utils.utils import get_local_hf_snapshot_or_repo_id

class GreenVLAv11Tokenizer:

    def __init__(
        self,
        max_len: int = 256,
        state_dim: int = 14,
        control_mode: str | None = None,
        embodiment_name: str | None = None,
        base_vlm_model: str = "Qwen/Qwen3.5-0.8B",
        fast_tokenizer_path: str = "physical-intelligence/fast",
        image_keys: list[str] = ["image"],
        discrete_state_input: bool = False,
        continuous_state_input: bool = False,
        state_dropout_prob: float = 0.0,
        state_special_token_id: int = 10,
        clip_state: bool = False,
        add_control_mode: bool = False,
        add_embodiment_name: bool = False,
        image_shape: tuple[int, int] = (224, 224),
        model_mode: str = "flow_matching",
        n_obs_steps: int = 1,
        obs_stride_seconds: float = 0.0,
    ):
        if model_mode != "flow_matching" or n_obs_steps != 1:
            raise ValueError("GreenVLAv1.1 requires flow_matching with one image per camera")
        if discrete_state_input or continuous_state_input:
            raise ValueError("GreenVLAv1.1 passes state to the action expert, not the text tokenizer")
        self._max_len = max_len
        self._state_dim = state_dim
        self._control_mode = control_mode
        self._embodiment_name = embodiment_name
        self._base_vlm_model = base_vlm_model
        self._image_keys = image_keys
        self._add_control_mode = add_control_mode
        self._add_embodiment_name = add_embodiment_name
        self._image_shape = image_shape
        self._metadata_compaction_warned = False

        self._processor = AutoProcessor.from_pretrained(
            get_local_hf_snapshot_or_repo_id(self._base_vlm_model),
            fix_mistral_regex=True,
        )

    def compute_image_thw(self, img_height: int, img_width: int):
        grid_t = 1
        grid_h = img_height // self._processor.image_processor.patch_size
        grid_w = img_width // self._processor.image_processor.patch_size
        return np.array([grid_t, grid_h, grid_w])

    def _tokenize_prefix(self, prefix: str, num_images: int) -> list[int]:
        visual_type = "image"
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": visual_type, visual_type: "dummy_value"}
                    for _ in range(num_images)
                ]
                + [{"type": "text", "text": prefix}],
            },
        ]
        text_inputs = self._processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )

        image_grid_thw = [
            self.compute_image_thw(self._image_shape[0], self._image_shape[1])
        ] * num_images
        for grid_thw in image_grid_thw:
            text_inputs = text_inputs.replace(
                self._processor.image_token,
                "<|placeholder|>"
                * (
                    grid_thw.prod()
                    // self._processor.image_processor.merge_size**2
                ),
                1,
            )
        text_inputs = text_inputs.replace(
            "<|placeholder|>", self._processor.image_token
        )
        text_inputs = text_inputs.replace("<|im_end|>\n", "")
        return self._processor.tokenizer(
            text_inputs,
            return_tensors=None,
            padding="longest",
        )["input_ids"]

    def _compact_metadata_prefix(
        self,
        prefix_before_metadata: str,
        policy_metadata: dict,
        num_images: int,
        max_prefix_tokens: int,
    ) -> tuple[str, list[int]]:
        metadata = dict(policy_metadata)
        task_tokens = self._processor.tokenizer(
            str(metadata.get("task", "")), add_special_tokens=False
        )["input_ids"]
        subtask_tokens = self._processor.tokenizer(
            str(metadata.get("subtask", "")), add_special_tokens=False
        )["input_ids"]

        def render() -> tuple[str, list[int]]:
            prefix = f"{prefix_before_metadata} {format_policy_metadata_prompt(metadata)}"
            return prefix, self._tokenize_prefix(prefix, num_images)

        prefix, prefix_tokens = render()
        original_length = len(prefix_tokens)
        # The categorical fields are more important than duplicated long-form task
        # text. Preserve a short task/subtask prefix and compact only when required.
        for tokens, key, minimum in (
            (task_tokens, "task", 4),
            (subtask_tokens, "subtask", 6),
            (task_tokens, "task", 0),
            (subtask_tokens, "subtask", 0),
        ):
            while len(prefix_tokens) > max_prefix_tokens and len(tokens) > minimum:
                overflow = len(prefix_tokens) - max_prefix_tokens
                del tokens[-min(overflow, len(tokens) - minimum) :]
                metadata[key] = self._processor.tokenizer.decode(
                    tokens, skip_special_tokens=True
                ).strip()
                prefix, prefix_tokens = render()

        if len(prefix_tokens) < original_length and not self._metadata_compaction_warned:
            logging.warning(
                "Policy metadata exceeded the token budget; compacted Task/Subtask "
                "from %d to %d prefix tokens so Mode/Negative/Error/Speed/Hand/Cameras/Quality are preserved.",
                original_length,
                len(prefix_tokens),
            )
            self._metadata_compaction_warned = True
        return prefix, prefix_tokens

    def tokenize(
        self,
        prompt: str,
        state: np.ndarray,
        image_mask: np.ndarray,
        actions: np.ndarray | None = None,
        subtask: str | None = None,
        next_subtask: str | None = None,
        is_subtask_transition: bool = False,
        policy_metadata: dict | None = None,
    ) -> dict[str, torch.Tensor]:
        cleaned_text = prompt.lower().strip().replace("_", " ")
        prefix = ""

        if self._add_control_mode:
            prefix += f"<|control_mode_start|>{self._control_mode}<|control_mode_end|>"
        if self._add_embodiment_name:
            prefix += f"<|embodiment_name_start|>{self._embodiment_name}<|embodiment_name_end|>"

        prefix_before_metadata = prefix
        if policy_metadata:
            prefix += f" {format_policy_metadata_prompt(policy_metadata)}"
        else:
            # In the end adding the task
            prefix += (
                f" Instruction: <|instruction_start|>{cleaned_text}<|instruction_end|>"
            )

        num_images = (
            torch.tensor([image_mask[key] for key in self._image_keys]).sum().item()
        )
        end_prefix = "<|im_end|>\n"
        end_prefix_tokens = self._processor.tokenizer(
            end_prefix,
            return_tensors=None,
            padding="longest",
        )["input_ids"]
        prefix_tokens = self._tokenize_prefix(prefix, num_images)
        if policy_metadata and len(prefix_tokens) + len(end_prefix_tokens) > self._max_len:
            prefix, prefix_tokens = self._compact_metadata_prefix(
                prefix_before_metadata,
                policy_metadata,
                num_images,
                self._max_len - len(end_prefix_tokens),
            )
        
        end_prefix_loss_mask = False

        if subtask and not policy_metadata:
            end_prefix_loss_mask = True
            predicted_subtask_text = (
                f"Subtask: <|instruction_start|>{subtask}<|instruction_end|>"
            )
            if is_subtask_transition and next_subtask:
                predicted_subtask_text += f"Next subtask: <|instruction_start|>{next_subtask}<|instruction_end|>"
            predicted_subtask_tokens = self._processor.tokenizer(
                predicted_subtask_text,
                return_tensors=None,
                padding="longest",
            )["input_ids"]
        else:
            predicted_subtask_tokens = []

        postfix_tokens = []

        tokens = (
            prefix_tokens
            + predicted_subtask_tokens
            + end_prefix_tokens
            + postfix_tokens
        )
        token_mask = [True] * len(tokens)
        # we do not need here specific attention mask, because we do not use it in the Qwen training

        # loss on predicted subtask and postfix only
        
        
        loss_mask = (
            [False] * len(prefix_tokens)
            + [True] * len(predicted_subtask_tokens)
            + [end_prefix_loss_mask] * len(end_prefix_tokens)
            + [True] * len(postfix_tokens)
        )
        # only 2 would be masked for FM action expert
        # 0 - prefix, 1 - predicted subtask, 0 - end prefix, 2 - postfix; 3 - padding
        # 1 is something on what we would like to compute CE loss, but not mask it for FM action expert
        token_type_ids = (
            [0] * len(prefix_tokens)
            + [1] * len(predicted_subtask_tokens)
            + [0] * len(end_prefix_tokens)
            + [2] * len(postfix_tokens)
        )
        

        # Pad tokens to max length
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [self._processor.tokenizer.pad_token_id] * (
                self._max_len - tokens_len
            )
            padding_mask = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding_mask
            loss_mask = loss_mask + padding_mask
            token_type_ids = token_type_ids + [3] * len(padding)
        else:
            if len(tokens) > self._max_len:
                visual_token = "<|image_pad|>"
                visual_token_id = self._processor.tokenizer.convert_tokens_to_ids(visual_token)
                num_visual_tokens = prefix_tokens.count(visual_token_id)
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently. "
                    f"Num prefix tokens: {len(prefix_tokens)} (including {num_visual_tokens} visual tokens), "
                    f"Num predicted subtask tokens: {len(predicted_subtask_tokens)}, "
                    f"Num end prefix tokens: {len(end_prefix_tokens)}, Num postfix tokens: {len(postfix_tokens)}"
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]
            token_type_ids = token_type_ids[: self._max_len]
            

        tokens = torch.tensor(tokens)
        token_mask = torch.tensor(token_mask).to(torch.long)
        action_loss_mask = torch.zeros(state.shape[0]).to(torch.bool)
        action_loss_mask[: self._state_dim] = True

        return {
            "input_ids": tokens,
            "padded_mask": token_mask,
            "attention_mask": token_mask,
            "loss_mask": torch.tensor(loss_mask),
            "action_loss_mask": action_loss_mask,
            "token_type_ids": torch.tensor(token_type_ids).to(torch.long),
        }

Qwen05Tokenizer = GreenVLAv11Tokenizer
