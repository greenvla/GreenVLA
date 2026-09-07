"""GreenVLAv1.1 inference: multimodal context -> cognition -> flow actions.

Only the released single-observation, context-query architecture is served.
Checkpoint parameter names, tensor operations and sampling order are preserved.
Training, legacy KV bridges, video history and guided/replay sampling are absent.
The actual backbone remains Qwen3.5: its HF identifiers are not branding aliases.
"""
import inspect

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision import transforms
from transformers import AutoConfig, AutoProcessor, Qwen3_5ForConditionalGeneration, Qwen3_5ForCausalLM

from lerobot.common.policies.greenvla_v1_1.expert_utils import create_sinusoidal_pos_embedding
from lerobot.common.policies.greenvla_v1_1.configuration_greenvla_v1_1 import GreenVLAv11Config
from lerobot.common.policies.greenvla_v1_1.identity import POLICY_TYPE
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.utils.utils import get_local_hf_snapshot_or_repo_id


def validate_inference_contract(config: GreenVLAv11Config) -> None:
    """Fail before loading weights if a checkpoint needs a removed code path."""
    required = {
        "cognition_mode": "frozen_vlm_context_queries",
        "model_mode": "flow_matching",
        "inference_mode": "flow_matching",
        "n_obs_steps": 1,
        "n_state_obs_steps": 1,
        "add_state_vlm": False,
        "cognition_layer_mix": True,
        "cognition_layer_mix_per_token": True,
        "cognition_frozen_vlm_reference_kernels": False,
        "cognition_action_expert_reference_kernels": False,
        "suffix_attention_mask": "causal",
    }
    mismatches = [f"{key}={getattr(config, key)!r} (requires {value!r})"
                  for key, value in required.items() if getattr(config, key) != value]
    if mismatches:
        raise ValueError("Unsupported GreenVLAv1.1 inference contract: " + "; ".join(mismatches))

def sample_noise(shape, device):
    return torch.normal(
        mean=0.0,
        std=1.0,
        size=shape,
        dtype=torch.float32,
        device=device,
    )

class GreenVLAv11Policy(PreTrainedPolicy):
    config_class = GreenVLAv11Config
    name = POLICY_TYPE

    def __init__(self, config: GreenVLAv11Config):
        super().__init__(config)
        self.config = config
        self.model_mode = config.model_mode
        self.base_vlm_model = config.base_vlm_model
        self.model = GreenVLAv11Model(config)

        if config.compile_sample_actions:
            self.sample_actions = torch.compile(
                self.sample_actions, backend="inductor", mode="default", fullgraph=False, dynamic=False
            )


    def prepare_images(self, batch):
        """Keep the configured camera order and masks."""
        images = []
        img_masks = []

        # Preprocess image features present in the batch
        for key in self.config.image_keys:
            img = batch["image"][key]
            mask = batch["image_mask"][key]
            images.append(img)
            img_masks.append(mask)
        return images, img_masks

    @torch.no_grad
    def select_action(
        self,
        batch: dict[str, Tensor],
        max_decode_len: int = 128,
        temperature: float | None = None,
    ) -> Tensor:
        """Sample one action chunk. Only flow-matching checkpoints are served here."""
        return self.sample_actions(
            batch,
            batch.get("guidance_traj"),
            batch.get("guidance_weight", 0.0),
            batch.get("guidance_mask"),
            batch.get("guidance_response_weight", 1.0),
        )

    def sample_actions(
        self,
        batch: dict[str, Tensor],
        guidance_traj: Tensor | None = None,
        guidance_weight: float | Tensor = 0.0,
        guidance_mask: Tensor | None = None,
        guidance_response_weight: float | Tensor = 1.0,
    ) -> Tensor:
        self.eval()
        images, img_masks = self.prepare_images(batch)
        lang_tokens, lang_masks = batch["input_ids"], batch["padded_mask"]
        loss_mask = batch.get("loss_mask", None)
        state = batch.get("state_history", batch["state"])
        state_history_is_pad = batch.get("state_history_is_pad")
        return self.model.sample_actions(
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=state,
            state_history_is_pad=state_history_is_pad,
            loss_mask=loss_mask,
            guidance_traj=guidance_traj,
            guidance_weight=guidance_weight,
            guidance_mask=guidance_mask,
            guidance_response_weight=guidance_response_weight,
        )

class GreenVLAv11Model(nn.Module):

    def __init__(self, config: GreenVLAv11Config):
        super().__init__()
        validate_inference_contract(config)
        self.config = config
        self.base_vlm_model = config.base_vlm_model
        self.cognition_mode = config.cognition_mode

        model_source = get_local_hf_snapshot_or_repo_id(self.base_vlm_model)
        base_config = AutoConfig.from_pretrained(model_source)
        self.base_model_type = base_config.model_type

        if self.base_model_type != "qwen3_5":
            raise ValueError(f"GreenVLAv1.1 requires the Qwen3.5 backend, got {self.base_model_type!r}")

        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_source,
            dtype=self.config.precision,
            device_map="cpu",
            attn_implementation=config.attention_implementation,
        )

        self.processor = AutoProcessor.from_pretrained(
            get_local_hf_snapshot_or_repo_id(self.base_vlm_model), fix_mistral_regex=True
        )
        self.image_normalizer = transforms.Normalize(
            mean=self.processor.image_processor.image_mean,
            std=self.processor.image_processor.image_std,
        )
        self._rope_uses_mm_token_type_ids = (
            "mm_token_type_ids" in inspect.signature(self.model.model.get_rope_index).parameters
        )

        self._init_bridge()

    def _init_bridge(self) -> None:
        """Initialize the Qwen3.5 cognition bottleneck and pretrained expert."""
        expert_source = get_local_hf_snapshot_or_repo_id(self.config.action_expert_model)
        expert_lm = Qwen3_5ForCausalLM.from_pretrained(
            expert_source,
            dtype=self.config.precision,
            device_map="cpu",
            attn_implementation=self.config.attention_implementation,
        )
        self.action_expert = expert_lm.model

        # The FM expert consumes inputs_embeds only.  Dropping the tied token
        # embedding/lm-head weight saves roughly 0.5 GB for the 0.8B checkpoint.
        self.action_expert.embed_tokens = None
        self.action_expert.main_input_name = "inputs_embeds"
        expert_lm.model = None
        del expert_lm

        vlm_hidden_size = self.model.config.text_config.hidden_size
        expert_hidden_size = self.action_expert.config.hidden_size
        self.expert_hidden_size = expert_hidden_size

        cognition_width = self.config.cognition_context_width
        self.cognition_tokens = nn.Parameter(
            torch.empty(1, self.config.num_cognition_tokens, cognition_width)
        )
        nn.init.normal_(
            self.cognition_tokens,
            mean=0.0,
            std=getattr(self.model.config.text_config, "initializer_range", 0.02),
        )

        num_vlm_layers = len(self.model.model.language_model.layers)
        num_mix_rows = self.config.num_cognition_tokens
        mix_logits = torch.zeros(num_mix_rows, num_vlm_layers, dtype=torch.float32)
        if self.config.cognition_layer_mix_init == "last_layer":
            # A finite bias keeps gradients alive for every layer while
            # starting close to the previous final-layer-only behavior.
            mix_logits.fill_(-6.0)
            mix_logits[:, -1] = 0.0
        self.cognition_layer_mix_logits = nn.Parameter(mix_logits)

        projection_input_size = cognition_width
        if self.config.cognition_projection == "linear":
            self.cognition_proj = nn.Linear(projection_input_size, expert_hidden_size)
        else:
            self.cognition_proj = nn.Sequential(
                nn.LayerNorm(projection_input_size),
                nn.Linear(projection_input_size, expert_hidden_size),
                nn.SiLU(),
                nn.Linear(expert_hidden_size, expert_hidden_size),
            )

        context_width = self.config.cognition_context_width
        self.cognition_context_kv_proj = nn.Linear(
            vlm_hidden_size,
            context_width * 2,
            bias=False,
        )
        self.cognition_context_query_norm = nn.LayerNorm(context_width)
        self.cognition_context_out_proj = nn.Linear(context_width, context_width)
        self.cognition_context_ffn_norm = nn.LayerNorm(context_width)
        context_ffn_width = context_width * self.config.cognition_context_ffn_multiplier
        self.cognition_context_ffn = nn.Sequential(
            nn.Linear(context_width, context_ffn_width),
            nn.SiLU(),
            nn.Linear(context_ffn_width, context_width),
        )

        self.action_in_proj = nn.Linear(self.config.max_action_dim, expert_hidden_size)
        self.action_out_proj = nn.Linear(expert_hidden_size, self.config.max_action_dim)
        if self.config.add_state_proj_to_action_expert:
            self.state_proj = nn.Linear(self.config.max_state_dim, expert_hidden_size)
        else:
            self.state_proj = None
        self.action_time_mlp_in = nn.Linear(expert_hidden_size * 2, expert_hidden_size)
        self.action_time_mlp_out = nn.Linear(expert_hidden_size, expert_hidden_size)

    def torch_qwen_image_processor(self, images: Tensor) -> tuple[Tensor, Tensor]:
        # images: B * N, C, H, W
        images = self.image_normalizer(images)
        images = images.unsqueeze(1)  # B*N, 1, C, H, W
        # repeating through temporal axis (specific for qwen)
        temporal_patch_size = self.processor.image_processor.temporal_patch_size
        patch_size = self.processor.image_processor.patch_size
        merge_size = self.processor.image_processor.merge_size
        patches = images.repeat(1, temporal_patch_size, 1, 1, 1)
        grid_t = patches.shape[1] // temporal_patch_size
        grid_h = patches.shape[3] // patch_size
        grid_w = patches.shape[4] // patch_size
        batch_size = patches.shape[0]
        channel = patches.shape[2]
        patches = patches.reshape(
            batch_size,
            grid_t,
            temporal_patch_size,
            channel,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        flatten_patches = patches.reshape(
            batch_size * grid_t * grid_h * grid_w,
            channel * temporal_patch_size * patch_size * patch_size,
        )

        image_grid_thw = (torch.tensor([grid_t, grid_h, grid_w]).unsqueeze(0).expand(batch_size, -1)).to(
            flatten_patches.device
        )

        return flatten_patches, image_grid_thw

    def _get_rope_index(
        self,
        lang_tokens: Tensor,
        image_grid_thw: Tensor | None,
        attention_mask: Tensor,
        video_grid_thw: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Call the model-family MRoPE API across transformers 4.x and 5.x."""
        rope_kwargs = {
            "image_grid_thw": image_grid_thw,
            "video_grid_thw": video_grid_thw,
            "attention_mask": attention_mask,
        }
        if self._rope_uses_mm_token_type_ids:
            # transformers>=5.5 requires explicit modality IDs for MRoPE.  The
            # policy expands image placeholders itself, so reproduce the
            # processor's mapping without a device-to-host copy every forward.
            mm_token_type_ids = torch.zeros_like(lang_tokens)
            for token_id in getattr(self.processor, "image_ids", []):
                if token_id is not None:
                    mm_token_type_ids.masked_fill_(lang_tokens == token_id, 1)
            for token_id in getattr(self.processor, "video_ids", []):
                if token_id is not None:
                    mm_token_type_ids.masked_fill_(lang_tokens == token_id, 2)
            rope_kwargs["mm_token_type_ids"] = mm_token_type_ids
        return self.model.model.get_rope_index(lang_tokens, **rope_kwargs)

    def _embed_qwen35_prefix(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Embed a Qwen3.5 multimodal prefix and compute its native MRoPE IDs."""
        inputs_embeds = self.model.model.get_input_embeddings()(lang_tokens)
        images = torch.stack(images, dim=1)
        img_masks = torch.stack(img_masks, dim=1).bool()
        batch_size, num_views, channels, height, width = images.shape
        flat_images = images.reshape(batch_size * num_views, channels, height, width)
        flat_images = flat_images[img_masks.reshape(-1)]
        pixel_values, image_grid_thw = self.torch_qwen_image_processor(flat_images)
        image_outputs = self.model.model.get_image_features(
            pixel_values,
            image_grid_thw=image_grid_thw,
            return_dict=True,
        )
        image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )
        n_image_tokens = (
            lang_tokens == self.model.config.image_token_id
        ).sum().item()
        if n_image_tokens != image_embeds.shape[0]:
            raise ValueError(
                "Qwen3.5 image features and placeholder tokens do not match: "
                f"tokens={n_image_tokens}, features={image_embeds.shape[0]}"
            )
        image_mask, _ = self.model.model.get_placeholder_mask(
            lang_tokens,
            inputs_embeds=inputs_embeds,
            image_features=image_embeds,
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        video_grid_thw = None

        position_ids, rope_deltas = self._get_rope_index(
            lang_tokens,
            image_grid_thw,
            lang_masks,
            video_grid_thw=video_grid_thw,
        )
        self.model.model.rope_deltas = rope_deltas
        return inputs_embeds, lang_masks.to(dtype=torch.long), position_ids

    def _prepare_state_history(
        self,
        state: Tensor,
        state_history_is_pad: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        if state.ndim == 2:
            state = state.unsqueeze(1)
        if state.ndim != 3:
            raise ValueError(
                f"state must be [B, D] or [B, K, D], got {tuple(state.shape)}"
            )
        expected_steps = int(self.config.n_state_obs_steps)
        if state.shape[1] != expected_steps:
            raise ValueError(
                f"Expected {expected_steps} state observations, got {state.shape[1]}"
            )

        if state_history_is_pad is None:
            valid_mask = torch.ones(
                state.shape[:2], dtype=torch.bool, device=state.device
            )
        else:
            state_history_is_pad = state_history_is_pad.to(
                device=state.device, dtype=torch.bool
            )
            if state_history_is_pad.shape != state.shape[:2]:
                raise ValueError(
                    "state_history_is_pad must be [B, K]: "
                    f"mask={tuple(state_history_is_pad.shape)}, state={tuple(state.shape)}"
                )
            valid_mask = ~state_history_is_pad

        # The last entry is the current observation and must never be dropped.
        valid_mask[:, -1] = True
        return state, valid_mask

    def encode_cognition_tokens(
        self, images, img_masks, lang_tokens, lang_masks, state=None, state_mask=None,
    ) -> Tensor:
        """Public hook used by the server's optional cognition cache."""
        return self._encode_frozen_vlm_context_queries(images, img_masks, lang_tokens, lang_masks)

    def _encode_frozen_vlm_context_queries(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
    ) -> Tensor:
        """Encode the multimodal prefix once and query all VLM layer outputs."""
        vlm_context = torch.no_grad() if self.config.freeze_vlm else torch.enable_grad()
        with vlm_context:
            prefix_embeds, prefix_mask, prefix_position_ids = self._embed_qwen35_prefix(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
            )
            text_position_ids = prefix_mask.long().cumsum(dim=-1) - 1
            text_position_ids.masked_fill_(~prefix_mask.bool(), 0)
            position_ids = torch.cat(
                [text_position_ids.unsqueeze(0), prefix_position_ids],
                dim=0,
            )
            outputs = self.model.model.language_model(
                input_ids=None,
                inputs_embeds=prefix_embeds,
                attention_mask=prefix_mask,
                position_ids=position_ids,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = outputs.hidden_states
            num_layers = len(self.model.model.language_model.layers)
            if hidden_states is None or len(hidden_states) != num_layers + 1:
                actual = None if hidden_states is None else len(hidden_states)
                raise RuntimeError(
                    "Qwen3.5 did not return one hidden state per decoder layer: "
                    f"expected {num_layers + 1}, got {actual}"
                )
            final_norm = self.model.model.language_model.norm
            layer_context = torch.stack(
                [final_norm(layer_hidden) for layer_hidden in hidden_states[1:]],
                dim=1,
            )

        return self._cross_attend_frozen_vlm_context(layer_context, prefix_mask)

    def _cross_attend_frozen_vlm_context(
        self,
        layer_context: Tensor,
        prefix_mask: Tensor,
    ) -> Tensor:
        """Apply per-query layer-biased SDPA over flattened VLM context."""
        batch_size, num_layers, prefix_length, _ = layer_context.shape
        context_width = self.config.cognition_context_width
        num_heads = self.config.cognition_context_num_heads
        head_dim = context_width // num_heads
        projection_dtype = self.cognition_context_kv_proj.weight.dtype

        key_value = self.cognition_context_kv_proj(
            layer_context.to(dtype=projection_dtype)
        )
        key, value = key_value.chunk(2, dim=-1)
        key = key.reshape(
            batch_size,
            num_layers * prefix_length,
            num_heads,
            head_dim,
        ).transpose(1, 2)
        value = value.reshape(
            batch_size,
            num_layers * prefix_length,
            num_heads,
            head_dim,
        ).transpose(1, 2)

        query_base = self.cognition_tokens.to(dtype=projection_dtype).expand(
            batch_size,
            -1,
            -1,
        )
        query = self.cognition_context_query_norm(query_base)
        query = query.reshape(
            batch_size,
            self.config.num_cognition_tokens,
            num_heads,
            head_dim,
        ).transpose(1, 2)

        # log-softmax gives every cognition query an explicit normalized prior
        # over depth.  Broadcasting it across prefix tokens turns the prior
        # into an additive attention bias without materializing N copies of
        # the full hidden-state tensor.
        layer_bias = torch.log_softmax(
            self.cognition_layer_mix_logits.float(),
            dim=-1,
        )
        layer_bias = layer_bias.unsqueeze(-1).expand(
            -1,
            -1,
            prefix_length,
        ).reshape(self.config.num_cognition_tokens, num_layers * prefix_length)
        attention_bias = layer_bias.to(dtype=query.dtype)[None, None, :, :].expand(
            batch_size,
            1,
            -1,
            -1,
        ).clone()
        flattened_mask = prefix_mask.bool()[:, None, :].expand(
            batch_size,
            num_layers,
            prefix_length,
        ).reshape(batch_size, num_layers * prefix_length)
        attention_bias.masked_fill_(
            ~flattened_mask[:, None, None, :],
            torch.finfo(attention_bias.dtype).min,
        )

        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_bias,
            dropout_p=0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(
            batch_size,
            self.config.num_cognition_tokens,
            context_width,
        )
        cognition = query_base + self.cognition_context_out_proj(attended)
        cognition = cognition + self.cognition_context_ffn(
            self.cognition_context_ffn_norm(cognition)
        )
        return cognition

    def _forward_expert(
        self,
        cognition: Tensor,
        suffix_embeds: Tensor,
        suffix_mask: Tensor | None = None,
    ) -> Tensor:
        projection_dtype = self.action_in_proj.weight.dtype
        projected_cognition = self.cognition_proj(cognition.to(dtype=projection_dtype))
        expert_inputs = torch.cat([projected_cognition, suffix_embeds.to(dtype=projection_dtype)], dim=1).to(
            dtype=self._expert_dtype
        )
        cognition_mask = torch.ones(
            projected_cognition.shape[:2], dtype=torch.long, device=expert_inputs.device
        )
        if suffix_mask is None:
            suffix_mask = torch.ones(
                suffix_embeds.shape[:2], dtype=torch.long, device=expert_inputs.device
            )
        expert_mask = torch.cat(
            [cognition_mask, suffix_mask.to(device=expert_inputs.device, dtype=torch.long)],
            dim=1,
        )
        outputs = self.action_expert(
            input_ids=None,
            inputs_embeds=expert_inputs,
            attention_mask=expert_mask,
            use_cache=False,
            return_dict=True,
        )
        return outputs.last_hidden_state

    @property
    def _expert_dtype(self) -> torch.dtype:
        """Dtype of the action expert (may differ from projection layers)."""
        return next(self.action_expert.parameters()).dtype

    def embed_suffix(
        self,
        state: Tensor,
        noisy_actions: Tensor,
        timestep: Tensor,
        state_history_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        embs = []
        masks = []

        device = noisy_actions.device
        proj_dtype = self.action_in_proj.weight.dtype
        if self.config.add_state_proj_to_action_expert:
            if state.ndim == 2:
                state = state.unsqueeze(1)
            if state.ndim != 3:
                raise ValueError(f"state must be [B, K, D], got {tuple(state.shape)}")
            if state_history_mask is None:
                state_history_mask = torch.ones(
                    state.shape[:2], dtype=torch.bool, device=state.device
                )
            state_emb = self.state_proj(state.to(proj_dtype))
            state_emb = state_emb.masked_fill(
                ~state_history_mask.unsqueeze(-1), 0.0
            )
            embs.append(state_emb)
            masks.append(state_history_mask.to(dtype=torch.long))
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_expert.config.hidden_size,
            min_period=4e-3,
            max_period=4.0,
            device=device,
        ).to(proj_dtype)
        action_emb = self.action_in_proj(noisy_actions.to(proj_dtype))
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)
        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)
        action_time_emb = self.action_time_mlp_out(action_time_emb)
        embs.append(action_time_emb)
        masks.append(
            torch.ones(
                action_time_emb.shape[:2], dtype=torch.long, device=device
            )
        )
        embs = torch.cat(embs, dim=1)
        attention_mask = torch.cat(masks, dim=1)
        return embs, attention_mask

    def _denoise_step(
        self,
        cognition: Tensor,
        state: Tensor,
        state_history_mask: Tensor,
        x_t: Tensor,
        timestep: Tensor,
    ) -> Tensor:
        suffix_embeds, suffix_mask = self.embed_suffix(
            state, x_t, timestep, state_history_mask
        )
        hidden_states = self._forward_expert(
            cognition, suffix_embeds, suffix_mask
        )
        action_hidden = hidden_states[:, -self.config.n_action_steps :]
        output_dtype = self.action_out_proj.weight.dtype
        return self.action_out_proj(action_hidden.to(output_dtype)).to(torch.float32)

    @torch.no_grad()
    def sample_actions(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        state: Tensor,
        state_history_is_pad: Tensor | None = None,
        loss_mask: Tensor | None = None,
        noise: Tensor | None = None,
        guidance_traj: Tensor | None = None,
        guidance_weight: float | Tensor = 0.0,
        guidance_mask: Tensor | None = None,
        guidance_response_weight: float | Tensor = 1.0,
    ) -> Tensor:
        """Return the full flow-matching chunk; feeding and v1 live outside the model."""
        if guidance_traj is not None or guidance_mask is not None:
            raise ValueError("GreenVLAv1.1 serves native flow actions, not guided/replay sampling")
        batch_size = lang_tokens.shape[0]
        device = lang_tokens.device
        state = state.to(torch.float32)
        if noise is None:
            noise = sample_noise(
                (batch_size, self.config.n_action_steps, self.config.max_action_dim), device
            )
        lang_masks = lang_masks.clone()
        if loss_mask is not None:
            lang_masks[loss_mask > 0] = 0
        state, state_history_mask = self._prepare_state_history(state, state_history_is_pad)
        cognition = self.encode_cognition_tokens(
            images, img_masks, lang_tokens, lang_masks,
            state=state, state_mask=state_history_mask,
        )
        dt = torch.tensor(-1.0 / self.config.num_steps, dtype=torch.float32, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(batch_size)
            velocity = self._denoise_step(cognition, state, state_history_mask, x_t, expanded_time)
            next_time = torch.clamp(time + dt, min=0.0, max=1.0)
            x_t = x_t + dt * velocity
            time = next_time
        return x_t

# Legacy imports share the same implementation and unchanged state_dict keys.
Qwen3VLPolicy = GreenVLAv11Policy
Qwen3VLPolicyModel = GreenVLAv11Model
