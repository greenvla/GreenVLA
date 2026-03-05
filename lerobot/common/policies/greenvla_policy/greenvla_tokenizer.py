import logging
import os

import numpy as np
import sentencepiece
from huggingface_hub import snapshot_download
from transformers import AutoProcessor
import torch


from lerobot.common.policies.utils import find_sublist_index
from lerobot.common.datasets.torch_transforms import pad_to_dim
from lerobot.common.utils.utils import get_local_hf_snapshot_or_repo_id



class GreenVLATokenizer:
    def __init__(
        self,
        max_len: int = 256,
        state_dim: int = 14,
        control_mode: str | None = None,
        embodiment_name: str | None = None,
        base_vlm_model: str = "Qwen/Qwen2.5-VL-3B-Instruct",
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
    ):
        self._max_len = max_len
        self._state_dim = state_dim
        self._control_mode = control_mode
        self._embodiment_name = embodiment_name
        self._base_vlm_model = base_vlm_model
        self._fast_tokenizer_path = fast_tokenizer_path
        self._image_keys = image_keys
        self._discrete_state_input = discrete_state_input
        self._continuous_state_input = continuous_state_input
        self._state_dropout_prob = state_dropout_prob
        self._add_control_mode = add_control_mode
        self._add_embodiment_name = add_embodiment_name
        self._model_mode = model_mode
        self._image_shape = image_shape

        vlm_source = get_local_hf_snapshot_or_repo_id(self._base_vlm_model)
        # If the source is still a repo ID (not a local path), pre-download all files
        # to avoid race conditions where AutoProcessor tries to open files that haven't
        # been fully downloaded yet (e.g. vocab.json arriving after tokenizer_config.json).
        if not os.path.isdir(vlm_source):
            vlm_source = snapshot_download(self._base_vlm_model)
        self._processor = AutoProcessor.from_pretrained(
            vlm_source,
            fix_mistral_regex=True,
        )

        if self._model_mode == "flow_matching":
            self._fast_tokenizer = None
        else:
            self._fast_tokenizer = AutoProcessor.from_pretrained(
                get_local_hf_snapshot_or_repo_id(fast_tokenizer_path),
                trust_remote_code=True,
            )

        self.action_seq_start_token = self._processor.tokenizer.encode(
            "<|robot_action_start|>"
        )[0]
        self.action_seq_end_token = self._processor.tokenizer.encode(
            "<|robot_action_end|>"
        )[0]
        self.initial_action_token_in_qwen = self._processor.tokenizer.encode(
            "<|robot_action_0|>"
        )[0]

    def compute_image_thw(self, img_height: int, img_width: int):
        grid_t = 1
        grid_h = img_height // self._processor.image_processor.patch_size
        grid_w = img_width // self._processor.image_processor.patch_size
        return np.array([grid_t, grid_h, grid_w])

    def tokenize(
        self,
        prompt: str,
        state: np.ndarray,
        image_mask: np.ndarray,
        actions: np.ndarray | None = None,
        subtask: str | None = None,
        next_subtask: str | None = None,
        is_subtask_transition: bool = False,
    ) -> dict[str, torch.Tensor]:
        cleaned_text = prompt.lower().strip().replace("_", " ")
        prefix = ""

        if self._discrete_state_input:
            state = state[: self._state_dim]
            # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1])
            discretized_state = (
                np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            )
            state_str = " ".join(map(str, discretized_state))
            prefix += f"State: {state_str};"
        elif self._continuous_state_input:
            raise NotImplementedError(
                "Continuous state input is not implemented for Qwen05Tokenizer"
            )
        if self._add_control_mode:
            prefix += f"<|control_mode_start|>{self._control_mode}<|control_mode_end|>"
        if self._add_embodiment_name:
            prefix += f"<|embodiment_name_start|>{self._embodiment_name}<|embodiment_name_end|>"

        # In the end adding the task
        prefix += (
            f" Instruction: <|instruction_start|>{cleaned_text}<|instruction_end|>"
        )

        num_images = (
            torch.tensor([image_mask[key] for key in self._image_keys]).sum().item()
        )
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "dummy_value"} for _ in range(num_images)
                ]
                + [
                    {"type": "text", "text": prefix}
                ],  # keeped here the ordering images; text because original Qwen works better in zero shot with this ordering
            },
        ]
        text_inputs = self._processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )

        image_grid_thw = [
            self.compute_image_thw(self._image_shape[0], self._image_shape[1])
        ] * num_images
        index = 0
        while self._processor.image_token in text_inputs:
            text_inputs = text_inputs.replace(
                self._processor.image_token,
                "<|placeholder|>"
                * (
                    image_grid_thw[index].prod()
                    // self._processor.image_processor.merge_size**2
                ),
                1,
            )
            index += 1

        text_inputs = text_inputs.replace(
            "<|placeholder|>", self._processor.image_token
        )

        end_prefix = "<|im_end|>\n"
        text_inputs = text_inputs.replace(end_prefix, "")

        prefix_tokens = self._processor.tokenizer(
            text_inputs,
            return_tensors=None,
            padding="longest",
        )["input_ids"]
        
        end_prefix_loss_mask = False

        if subtask:
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

        end_prefix_tokens = self._processor.tokenizer(
            end_prefix,
            return_tensors=None,
            padding="longest",
        )["input_ids"]
        
        if actions is not None and self._model_mode in ["token_prediction", "mixed"]:

            action_prefix = "<|im_start|><|robot_action_start|>"
            actions = actions[:, : self._state_dim]
            action_tokens = self._fast_tokenizer(actions[None])[0]
            action_text_tokens = [f"<|robot_action_{i}|>" for i in action_tokens]
            action_postfix = "<|robot_action_end|><|im_end|>"
            postfix_text = action_prefix + "".join(action_text_tokens) + action_postfix
            postfix_tokens = self._processor.tokenizer(
                postfix_text, return_tensors=None, padding="longest"
            )["input_ids"]
        else:
            # TODO: maybe add here action prefix if mode is token prediction?
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
                image_token_id = self._processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
                num_image_tokens = prefix_tokens.count(image_token_id)
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently. "
                    f"Num prefix tokens: {len(prefix_tokens)} (including {num_image_tokens} image tokens), "
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

    def extract_actions(
        self, tokens: np.ndarray, action_horizon: int, action_dim: int
    ) -> np.ndarray:
        if tokens.ndim == 2:
            tokens = tokens[0]
        try:
            start_pos = np.where(self.action_seq_start_token == tokens)[0][0].item()
            end_pos = np.where(self.action_seq_end_token == tokens)[0][0].item()
            action_seq = tokens[start_pos + 1 : end_pos]
            fast_tokens = (action_seq - self.initial_action_token_in_qwen).tolist()

        except ValueError:
            logging.warning(f"No action sequence found in {tokens}")
            return np.zeros((action_horizon, action_dim), dtype=np.float32)
        actions = self._fast_tokenizer.decode(
            [fast_tokens], time_horizon=action_horizon, action_dim=self._state_dim
        )[0]
        return pad_to_dim(actions, action_dim, axis=-1, value=0.0)
