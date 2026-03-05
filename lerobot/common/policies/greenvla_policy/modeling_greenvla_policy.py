from typing import Callable, Optional, Union
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from torchvision import transforms
from transformers import (AutoProcessor, Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration,
                          Qwen3VLTextConfig, Qwen3VLTextModel)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb, eager_attention_forward

from lerobot.common.policies.greenvla_policy.greenvla_utils import create_sinusoidal_pos_embedding, sample_beta
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.greenvla_policy.configuration_greenvla_policy import GreenVLAPolicyConfig
from lerobot.common.utils.utils import get_local_hf_snapshot_or_repo_id

def shift_padding_side(
    tokens: torch.Tensor,
    ar_mask: torch.Tensor,
    padding_mask: torch.Tensor,
    padding_side: str = "right",
) -> tuple[torch.Tensor]:
    if padding_side not in ["right", "left"]:
        return tokens, ar_mask, padding_mask

    new_tokens = torch.empty_like(tokens)
    new_ar_masks = torch.empty_like(ar_mask)
    new_padding_mask = torch.empty_like(padding_mask)
    batch_size = tokens.shape[0]
    for i in range(batch_size):
        padding_indices = torch.where(padding_mask[i] == 0)[0]
        non_padding_indices = torch.where(padding_mask[i] == 1)[0]
        if padding_side == "left":
            new_indices = torch.cat((padding_indices, non_padding_indices), dim=0)
        else:
            new_indices = torch.cat((non_padding_indices, padding_indices), dim=0)
        new_tokens[i] = tokens[i].index_select(0, new_indices)
        new_ar_masks[i] = ar_mask[i].index_select(0, new_indices)
        new_padding_mask[i] = padding_mask[i].index_select(0, new_indices)

    return (
        new_tokens,
        new_ar_masks,
        new_padding_mask,
    )




def sample_noise(shape, device):
    return torch.normal(
        mean=0.0,
        std=1.0,
        size=shape,
        dtype=torch.float32,
        device=device,
    )

def sample_time(bsize, device):
    time_beta = sample_beta(1.5, 1.0, bsize, device)
    time = time_beta * 0.999 + 0.001
    return time.to(dtype=torch.float32, device=device)


class GreenVLAPolicy(PreTrainedPolicy):
    config_class = GreenVLAPolicyConfig
    name = "greenvlapolicy"

    def __init__(self, config: GreenVLAPolicyConfig):
        super().__init__(config)
        self.config = config
        self.model_mode = config.model_mode
        assert self.model_mode in [
            "token_prediction",
            "flow_matching",
            "mixed",
        ], f"Invalid training mode: {self.model_mode}"
        self.base_vlm_model = config.base_vlm_model
        self.model = GreenVLAPolicyModel(config)
        
        if config.compile_sample_actions:
            self.sample_actions = torch.compile(self.sample_actions, backend="inductor", mode="default", fullgraph=False, dynamic=False)

    def get_optim_params(self) -> dict:
        return self.parameters()

    def prepare_images(self, batch):
        """Preprocess LeRobot batch into GreenVLA inputs"""
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
        temperature: Union[float, None] = None,
    ) -> Tensor:

        if self.config.model_mode  == "flow_matching":
            return self.sample_actions(batch)
        elif self.config.model_mode == "token_prediction":
            return self.generate(batch, max_decode_len, temperature)
        elif self.config.model_mode == "mixed":
            if self.config.inference_mode == "flow_matching":
                return self.sample_actions(batch)
            elif self.config.inference_mode == "token_prediction":
                return self.generate(batch, max_decode_len, temperature)
            else:
                raise ValueError(f"Invalid inference mode: {self.config.inference_mode}")
        
        else:
            raise ValueError(f"Invalid model mode: {self.config.model_mode}")

    def sample_actions(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        images, img_masks = self.prepare_images(batch)
        lang_tokens, lang_masks = batch["input_ids"], batch["padded_mask"]
        loss_mask = batch.get("loss_mask", None)
        state = batch["state"]
        return self.model.sample_actions(
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=state,
            loss_mask=loss_mask,
        )

    def generate(
        self,
        batch: dict[str, Tensor],
        max_decode_len: int = 128,
        temperature: Union[float, None] = None,
    ) -> Tensor:
        self.eval()
        images, img_masks = self.prepare_images(batch)
        lang_tokens, lang_masks = batch["input_ids"], batch["padded_mask"]
        return self.model.generate(
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            max_decode_len=max_decode_len,
            temperature=temperature,
        )

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        if self.config.model_mode == "flow_matching":
            return self.forward_flow_matching(batch)
        elif self.config.model_mode == "token_prediction":
            return self.forward_cross_entropy(batch)
        elif self.config.model_mode == "mixed":
            return self.forward_flow_matching(batch)
        else:
            raise ValueError(f"Invalid model mode: {self.config.model_mode}")

    def forward_flow_matching(self, batch: dict[str, Tensor]) -> Tensor:
        images, img_masks = self.prepare_images(batch)
        lang_tokens, lang_masks = batch["input_ids"], batch["padded_mask"]
        data_source = batch.get("data_source", None)
        state = batch["state"]
        actions = batch["actions"]
        loss_mask = batch["loss_mask"]
        action_loss_mask = batch["action_loss_mask"]
        token_type_ids = batch["token_type_ids"]
        loss_dict = {}
        loss, loss_dict = self.model.forward_flow_matching(
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            loss_mask=loss_mask,
            state=state,
            actions=actions,
            data_source=data_source,
            action_loss_mask=action_loss_mask,
            token_type_ids=token_type_ids,
        )
        return loss, loss_dict

    def forward_cross_entropy(self, batch: dict[str, Tensor]) -> Tensor:
        images, img_masks = self.prepare_images(batch)
        lang_tokens, lang_masks = batch["input_ids"], batch["padded_mask"]
        token_ar_mask = batch[
            "attention_mask"
        ]  # attention mask not used because we use causal attention and padding pask would be enough
        loss_mask = batch["loss_mask"]
        loss_dict = {}
        data_source = batch.get("data_source", None)

        loss, metrics = self.model.forward_cross_entropy(
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            loss_mask=loss_mask,
            data_source=data_source,
        )
        loss_dict["ce_loss"] = loss.item()
        loss_dict.update(metrics)
        return loss, loss_dict


class GreenVLAPolicyModel(nn.Module):
    def __init__(self, config: GreenVLAPolicyConfig):
        super().__init__()
        self.config = config
        self.base_vlm_model = config.base_vlm_model
        self.is_qwen3 = "Qwen3" in self.base_vlm_model
        
        model_cls = Qwen3VLForConditionalGeneration if self.is_qwen3 else Qwen2_5_VLForConditionalGeneration
        self.model: Qwen3VLForConditionalGeneration = model_cls.from_pretrained(
                self.base_vlm_model,
                dtype=self.config.precision,
                device_map="cpu",
                attn_implementation=config.attention_implementation,
            )
            
        # Try loading the processor from a local HF cache snapshot first.
        # If the cache only contains model weights (no preprocessor_config.json /
        # tokenizer files), fall back to the repo ID so HF downloads them.
        try:
            self.processor = AutoProcessor.from_pretrained(
                get_local_hf_snapshot_or_repo_id(self.base_vlm_model), fix_mistral_regex=True
            )
        except OSError:
            self.processor = AutoProcessor.from_pretrained(
                self.base_vlm_model, fix_mistral_regex=True
            )
        self.image_normalizer = transforms.Normalize(
            mean=self.processor.image_processor.image_mean,
            std=self.processor.image_processor.image_std,
        )

        # Action expert: a smaller text transformer operating in a reduced hidden space
        # of width `proj_width`. We keep the per-head dimension identical to the base
        # VLM text model and derive the number of (KV) heads accordingly. We also
        # downsample layers by `expert_block_stride`: the expert keeps every Nth VLM
        # layer (starting at 0), so KV cache is reused only from those layers.
        if self.config.model_mode in ["flow_matching", "mixed"]:
            base_text_cfg = self.model.config.text_config
            base_hidden_size = base_text_cfg.hidden_size
            base_num_heads = base_text_cfg.num_attention_heads
            base_num_kv_heads = base_text_cfg.num_key_value_heads
            base_num_layers = base_text_cfg.num_hidden_layers
            self.base_num_layers = base_num_layers
            self.base_num_kv_heads = base_num_kv_heads
            proj_width = self.config.proj_width
            stride = max(1, int(self.config.expert_block_stride))

            # Which VLM layers feed KV cache to the expert
            self.expert_layer_indices = list(range(0, base_num_layers, stride))
            num_expert_layers = len(self.expert_layer_indices)
            assert (
                num_expert_layers > 0
            ), "expert_block_stride too large; no layers selected"

            # Sanity checks to avoid shape mismatches.
            assert (
                base_hidden_size % base_num_heads == 0
            ), f"Base hidden size {base_hidden_size} must be divisible by num_attention_heads {base_num_heads}"
            head_dim = base_hidden_size // base_num_heads
            self.head_dim = head_dim

            assert (
                proj_width % head_dim == 0
            ), f"proj_width {proj_width} must be divisible by head_dim {head_dim} to keep per-head dim constant"

            num_attention_heads = proj_width // head_dim
            assert num_attention_heads > 0, "Derived num_attention_heads must be > 0"

            # Keep KV heads identical to the base model so cached prefix KV matches.
            num_key_value_heads = base_num_kv_heads
            assert (
                num_attention_heads >= num_key_value_heads
            ), f"num_attention_heads {num_attention_heads} must be >= num_key_value_heads {num_key_value_heads} to reuse prefix KV cache"
            assert (
                num_attention_heads % num_key_value_heads == 0
            ), f"num_attention_heads {num_attention_heads} must be divisible by num_key_value_heads {num_key_value_heads}"

            # Preserve the MLP expansion ratio from the base model.
            base_intermediate_size = base_text_cfg.intermediate_size
            # Avoid division by zero and ensure at least 1 hidden unit.
            if base_hidden_size > 0:
                mlp_ratio = base_intermediate_size / float(base_hidden_size)
            else:
                mlp_ratio = 4.0
            intermediate_size = max(1, int(round(mlp_ratio * proj_width)))

            action_expert_config = Qwen3VLTextConfig(
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                num_hidden_layers=num_expert_layers,
                hidden_size=proj_width,
                intermediate_size=intermediate_size,
                rope_scaling=base_text_cfg.rope_scaling,
                attn_implementation=self.config.attention_implementation,
            )

            self.action_expert = Qwen3VLTextModel(action_expert_config).to(
                torch.bfloat16
            ).to(device="cpu")
            self.action_expert.embed_tokens = None
            # Projections into / out of the reduced hidden space.
            self.action_in_proj = nn.Linear(self.config.max_action_dim, proj_width)
            self.action_out_proj = nn.Linear(proj_width, self.config.max_action_dim)
            if self.config.add_state_proj_to_action_expert:
                self.state_proj = nn.Linear(self.config.max_state_dim, proj_width)
            else:
                self.state_proj = None
            self.action_time_mlp_in = nn.Linear(proj_width * 2, proj_width)
            self.action_time_mlp_out = nn.Linear(proj_width, proj_width)

            if self.config.enable_learnable_layer_combination:
                init_mode = getattr(self.config, "layer_combination_init", "identity")
                if init_mode not in ["identity", "stride_uniform", "global_uniform"]:
                    raise ValueError(
                        f"Invalid layer_combination_init: {init_mode}. "
                        "Use one of: 'identity', 'stride_uniform', 'global_uniform'."
                    )

                # Initialize weights according to the selected strategy.
                k_init = torch.zeros(num_expert_layers, base_num_layers, dtype=torch.float32)
                v_init = torch.zeros_like(k_init)

                if init_mode == "identity":
                    # Default: match the current behavior (stride-aligned copy).
                    for expert_idx, base_idx in enumerate(self.expert_layer_indices):
                        k_init[expert_idx, base_idx] = 1.0
                        v_init[expert_idx, base_idx] = 1.0
                elif init_mode == "stride_uniform":
                    # Uniform weights within each stride window.
                    for expert_idx, base_idx in enumerate(self.expert_layer_indices):
                        start = base_idx
                        end = min(base_idx + stride, base_num_layers)
                        window_size = max(1, end - start)
                        k_init[expert_idx, start:end] = 1.0 / window_size
                        v_init[expert_idx, start:end] = 1.0 / window_size
                elif init_mode == "global_uniform":
                    # Fully uniform over all VLM layers.
                    k_init.fill_(1.0 / base_num_layers)
                    v_init.fill_(1.0 / base_num_layers)
                self.k_mix_weights = nn.Parameter(k_init)
                self.v_mix_weights = nn.Parameter(v_init)
                bias_shape = (num_expert_layers, base_num_kv_heads, base_text_cfg.head_dim)
                self.k_mix_bias = nn.Parameter(torch.zeros(bias_shape, dtype=torch.float32))
                self.v_mix_bias = nn.Parameter(torch.zeros(bias_shape, dtype=torch.float32))
            else:
                self.k_mix_weights = None
                self.v_mix_weights = None
                self.k_mix_bias = None
                self.v_mix_bias = None

        if self.config.freeze_vlm:
            for param in self.model.parameters():
                param.requires_grad = False

    def embed_prefix(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:

        inputs_embeds = self.model.language_model.embed_tokens(lang_tokens)
        # concating images and choosing only not masked
        images = torch.stack(images, dim=1)  # first dim is batch size
        img_masks = torch.stack(img_masks, dim=1)

        B, N, C, H, W = images.shape
        images = images.reshape(B * N, C, H, W)
        img_masks = img_masks.reshape(B * N)

        # choosing only not masked images
        images = images[img_masks]

        pixel_values, image_grid_thw = self.torch_qwen_image_processor(images)
        if self.is_qwen3:
            image_embeds, deepstack_image_embeds = self.model.model.get_image_features(
                pixel_values, image_grid_thw=image_grid_thw
            )
        else:
            image_embeds = self.model.model.get_image_features(
                pixel_values, image_grid_thw=image_grid_thw
            )
            deepstack_image_embeds = None
        image_embeds = torch.cat(image_embeds, dim=0).to(
            inputs_embeds.device, inputs_embeds.dtype
        )

        n_image_tokens = (lang_tokens == self.model.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        image_mask, _ = self.model.model.get_placeholder_mask(
            lang_tokens, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        # there are already padded everything beside lang and image tokens
        pad_masks = lang_masks
        num_lang_embs = inputs_embeds.shape[1]
        # full attention between image and language inputs (if we need something alike in PaliGemma)
        # att_masks = [0] * num_lang_embs
        # att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        # att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        # setting full causal attention as it was done in pre-train

        att_masks = lang_masks  # pure causal mask (padding inside of a mask)
        # att_masks = [1] * num_lang_embs
        # att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        # att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        image_mask = image_mask[..., 0]

        return (
            inputs_embeds,
            pad_masks,
            att_masks,
            image_grid_thw,
            deepstack_image_embeds,
            image_mask,
        )  # we need to return image_grid_thw for further positional encoding

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

        image_grid_thw = (
            torch.tensor([grid_t, grid_h, grid_w]).unsqueeze(0).expand(batch_size, -1)
        ).to(flatten_patches.device)

        return flatten_patches, image_grid_thw

    def forward(self, images, img_masks, lang_tokens, lang_masks) -> Tensor:
        pass

    def encode_vlm_prefix_with_kv_cache(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        return_cache: bool = True
    ) -> tuple[Tensor, Tensor, Tensor]:
        
        (
            prefix_embs,
            prefix_pad_masks,
            prefix_attn_mask,
            image_grid_thw,
            deepstack_image_embeds,
            image_mask,
        ) = self.embed_prefix(
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
        )
        position_ids, rope_deltas = self.model.model.get_rope_index(
            lang_tokens, image_grid_thw, None, attention_mask=prefix_attn_mask
        )
        self.model.model.rope_deltas = rope_deltas
        prefix_attn_mask = prefix_attn_mask.to(torch.long)
        # prefix_embs, prefix_pad_masks, prefix_attn_mask = shift_padding_side(prefix_embs, prefix_attn_mask, prefix_pad_masks, "left")
        if self.is_qwen3:
            output = self.model.model.language_model.forward(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=prefix_attn_mask,
                inputs_embeds=prefix_embs,
                image_grid_thw=image_grid_thw,
                deepstack_visual_embeds=deepstack_image_embeds,
                visual_pos_masks=image_mask,
                return_dict=True,
                use_cache=return_cache,
            )
        elif self.is_qwen2_5:
            output = self.model.model.language_model.forward(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=prefix_attn_mask,
                inputs_embeds=prefix_embs,
                image_grid_thw=image_grid_thw,
                return_dict=True,
                use_cache=return_cache,
            )
        else:
            raise ValueError(f"Unsupported model mode: {self.config.model_mode}")

        last_hidden_state = output.last_hidden_state
        if return_cache:
            if self.config.is_knowledge_insulation and self.config.model_mode == "mixed":
                past_keys = torch.stack([
                    output.past_key_values[i][0].detach() for i in range(len(output.past_key_values))
                ], dim=1) # B, N_layers, N_heads, N_tokens, head_dim
                past_values = torch.stack([
                    output.past_key_values[i][1].detach() for i in range(len(output.past_key_values))
                ], dim=1) # B, N_layers, N_heads, N_tokens, head_dim
            else:       
                past_keys = torch.stack([
                    output.past_key_values[i][0] for i in range(len(output.past_key_values))
                ], dim=1) # B, N_layers, N_heads, N_tokens, head_dim
                past_values = torch.stack([
                    output.past_key_values[i][1] for i in range(len(output.past_key_values))
                ], dim=1) # B, N_layers, N_heads, N_tokens, head_dim
        else:
            past_keys = None
            past_values = None
        return {
            "past_keys": past_keys,
            "past_values": past_values,
            "last_hidden_state": last_hidden_state,
        }

    def embed_suffix(self,
        state: Tensor,
        noisy_actions: Tensor,
        timestep: Tensor
    ) -> tuple[Tensor, Tensor]:
        embs = []
        
        device = noisy_actions.device
        dtype = torch.bfloat16
        if self.config.add_state_proj_to_action_expert:
            state_emb = self.state_proj(state).type(dtype=dtype).unsqueeze(1)
            embs.append(state_emb)
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_expert.config.hidden_size,
            min_period=4e-3,
            max_period=4.0,
            device=device,
        ).type(dtype=dtype)
        action_emb = self.action_in_proj(noisy_actions)
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)
        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)
        embs.append(action_time_emb)
        embs = torch.cat(embs, dim=1)
        attention_mask = torch.ones(
            embs.shape[0], embs.shape[1], dtype=torch.long, device=device
        ) 
        return embs, attention_mask

    def build_suffix_attention_mask(
        self,
        lang_masks: Tensor,
        suffix_attn_mask: Tensor,
        attn_implementation: str,
        suffix_attention_mode: str = "causal",
    ) -> Tensor:
        """
        Build an attention mask for suffix tokens on top of an existing prefix.

        - `lang_masks` is the padding mask for the prefix tokens (1 = real token, 0 = padding),
          shape `[B, T_prefix]`.
        - `suffix_attn_mask` is the padding mask for the suffix tokens (1 = real token, 0 = padding),
          shape `[B, T_suffix]`.
        - `suffix_attention_mode`:
            - "causal": suffix tokens attend causally to prefix and causally to each other.
            - "block": suffix tokens attend causally to prefix but fully (bidirectionally) to each other.
        - If `attn_implementation == "flash_attention_2"`, returns a 2D padding mask of shape
          `[B, T_prefix + T_suffix]` (True = real token, False = padding), as expected by FA2.
          Note: FA2 only supports causal masks, so "block" mode is not supported with FA2.
        - Otherwise returns a 4D mask of shape `[B, 1, T_suffix, T_prefix + T_suffix]`
          where `True` means that a given (query, key) pair is **allowed** by the mask.
        """
        if suffix_attention_mode not in ["causal", "block"]:
            raise ValueError(
                f"suffix_attention_mode must be 'causal' or 'block', got '{suffix_attention_mode}'"
            )

        # FA2 only supports causal masks; block mode requires a custom 4D mask.
        if (
            attn_implementation == "flash_attention_2"
            and suffix_attention_mode == "block"
        ):
            raise ValueError(
                "flash_attention_2 does not support suffix_attention_mode='block'. "
                "Use attn_implementation='eager' or 'sdpa' instead, or set suffix_attention_mode='causal'."
            )

        device = lang_masks.device
        batch_size, num_prefix_tokens = lang_masks.shape
        _, num_suffix_tokens = suffix_attn_mask.shape

        # Convert padding masks to bool: True = valid token, False = padding.
        prefix_valid = lang_masks.to(dtype=torch.bool)  # [B, T_prefix]
        suffix_valid = suffix_attn_mask.to(dtype=torch.bool)  # [B, T_suffix]

        # For flash_attention_2 we return a standard 2D padding mask [B, T_total]
        # where True/1 = real token, False/0 = padding.
        # (Only reached when suffix_attention_mode == "causal" due to check above.)
        if attn_implementation == "flash_attention_2":
            full_mask = torch.cat(
                [prefix_valid, suffix_valid], dim=1
            )  # [B, T_prefix + T_suffix]
            return full_mask

        # Otherwise, build the 4D mask used by the eager/sdpa attention path.
        # Initialize all positions as masked (False).
        attn_mask = torch.zeros(
            batch_size,
            1,
            num_suffix_tokens,
            num_prefix_tokens + num_suffix_tokens,
            dtype=torch.bool,
            device=device,
        )

        # --- Prefix part: each valid suffix query can attend to all valid prefix tokens. ---
        # (Causal attention to prefix in both modes.)
        # [B, T_suffix, T_prefix]
        prefix_vis = prefix_valid[:, None, :].expand(
            batch_size, num_suffix_tokens, num_prefix_tokens
        )
        # Do not allow padded suffix queries to attend to anything.
        prefix_vis = prefix_vis & suffix_valid[:, :, None].expand(
            batch_size, num_suffix_tokens, num_prefix_tokens
        )
        attn_mask[:, 0, :, :num_prefix_tokens] = prefix_vis

        # --- Suffix-suffix part ---
        if suffix_attention_mode == "causal":
            # Standard causal mask over suffix positions.
            pos = torch.arange(num_suffix_tokens, device=device)
            q_pos = pos.view(1, num_suffix_tokens, 1)  # [1, T_s, 1]
            k_pos = pos.view(1, 1, num_suffix_tokens)  # [1, 1, T_s]
            causal = q_pos >= k_pos  # [1, T_s, T_s], True if key <= query.

            suffix_vis = causal.expand(batch_size, -1, -1)  # [B, T_s, T_s]
        else:
            # "block" mode: full (bidirectional) attention among suffix tokens.
            suffix_vis = torch.ones(
                batch_size,
                num_suffix_tokens,
                num_suffix_tokens,
                dtype=torch.bool,
                device=device,
            )

        # Mask out padded keys.
        suffix_vis = suffix_vis & suffix_valid[:, None, :].expand(
            batch_size, num_suffix_tokens, num_suffix_tokens
        )
        # Mask out padded queries.
        suffix_vis = suffix_vis & suffix_valid[:, :, None].expand(
            batch_size, num_suffix_tokens, num_suffix_tokens
        )

        attn_mask[:, 0, :, num_prefix_tokens:] = suffix_vis

        return attn_mask

    def _prepare_flow_matching_noise_and_time(
        self,
        actions: Tensor,
        noise: Optional[Tensor] = None,
        time: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Prepare noise and time for flow matching, and compute noisy actions x_t and velocity u_t.

        Args:
            actions: Clean action tensor of shape [B, T, D]
            noise: Optional noise tensor. If None, will be sampled.
            time: Optional time tensor. If None, will be sampled.

        Returns:
            x_t: Noisy actions at time t, shape [B, T, D]
            u_t: Velocity field (noise - actions), shape [B, T, D]
            time: Time tensor, shape [B]
        """
        if noise is None:
            noise = sample_noise(actions.shape, actions.device)

        if time is None:
            time = sample_time(actions.shape[0], actions.device)

        # Flow matching interpolation: x_t = t * noise + (1 - t) * actions
        # Reference: Flow matching formulation
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        return x_t, u_t, time

    def _compute_suffix_position_ids(
        self,
        suffix_attn_mask: Tensor,
        lang_masks: Tensor,
    ) -> Tensor:
        """
        Compute position IDs for suffix tokens, accounting for prefix length.

        Position IDs are computed relative to the prefix, so suffix tokens start
        from the end of the prefix sequence.

        Reference: Similar to position_id computation in Qwen3VLTextModel.forward
        (modeling_qwen3_vl.py:823-826)

        Args:
            suffix_attn_mask: Padding mask for suffix tokens, shape [B, T_suffix]
            lang_masks: Padding mask for prefix tokens, shape [B, T_prefix]

        Returns:
            position_ids: Position IDs of shape [3, B, T_suffix] (3D for Qwen3VL's MRoPE)
        """
        # Compute cumulative positions for suffix, offset by prefix length
        prefix_lengths = lang_masks.sum(dim=1, keepdim=True)  # [B, 1]
        suffix_position_ids = (
            suffix_attn_mask.cumsum(dim=1) - 1 + prefix_lengths
        )  # [B, T_suffix]

        # Expand to 3D for Qwen3VL's multi-dimensional RoPE (T, H, W dimensions)
        # Reference: modeling_qwen3_vl.py:824-826
        position_ids = suffix_position_ids[None, ...].expand(
            3, suffix_position_ids.shape[0], -1
        )

        return position_ids

    def _get_prefix_kv(
        self,
        expert_layer_idx: int,
        base_layer_idx: int,
        past_keys: Tensor,
        past_values: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Select (or mix) cached KV for a given action expert layer.

        If learnable layer combination is disabled, falls back to stride-aligned
        layer selection. When enabled, computes a linear combination of all VLM
        layers with per-layer weights (initialized to identity) and optional
        per-head biases.
        """
        if not self.config.enable_learnable_layer_combination:
            return (
                past_keys[:, base_layer_idx],
                past_values[:, base_layer_idx],
            )

        # past_keys shape: [B, L, num_kv_heads, T, head_dim]
        weight_k = self.k_mix_weights[expert_layer_idx].to(past_keys.dtype)
        weight_v = self.v_mix_weights[expert_layer_idx].to(past_values.dtype)

        mixed_keys = (past_keys * weight_k[None, :, None, None, None]).sum(dim=1)
        mixed_values = (past_values * weight_v[None, :, None, None, None]).sum(dim=1)

        if self.k_mix_bias is not None:
            mixed_keys = mixed_keys + self.k_mix_bias[expert_layer_idx].to(
                mixed_keys.dtype
            )[None, :, None, :]
        if self.v_mix_bias is not None:
            mixed_values = mixed_values + self.v_mix_bias[expert_layer_idx].to(
                mixed_values.dtype
            )[None, :, None, :]

        return mixed_keys, mixed_values

    def _apply_attention_with_kv_cache(
        self,
        attention_layer: nn.Module,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask: Tensor,
        past_keys: Tensor,
        past_values: Tensor,
        position_ids: Tensor,
    ) -> Tensor:
        """
        Apply self-attention with cached key-value pairs from the prefix.

        This reimplements Qwen3VLTextAttention.forward but uses pre-computed KV cache
        instead of updating the cache. The cached keys/values from the prefix are
        concatenated with the current suffix's keys/values.

        Reference: Qwen3VLTextAttention.forward (modeling_qwen3_vl.py:416-457)
        Reference: apply_rotary_pos_emb (modeling_qwen3_vl.py:358-382)
        Reference: eager_attention_forward (modeling_qwen3_vl.py:142-165)

        Args:
            attention_layer: Qwen3VLTextAttention layer instance
            hidden_states: Input hidden states, shape [B, T_suffix, H]
            position_embeddings: Tuple of (cos, sin) for RoPE, from rotary_emb
            attention_mask: Attention mask, shape depends on attn_implementation
            past_keys: Cached key states from prefix, shape [B, num_kv_heads, T_prefix, head_dim]
            past_values: Cached value states from prefix, shape [B, num_kv_heads, T_prefix, head_dim]
            position_ids: Position IDs for suffix, shape [3, B, T_suffix]

        Returns:
            attn_output: Attention output, shape [B, T_suffix, H]
        """
        # Project to query, key, value and reshape for multi-head attention
        # Reference: modeling_qwen3_vl.py:425-430
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attention_layer.head_dim)

        query_states = attention_layer.q_norm(
            attention_layer.q_proj(hidden_states).view(hidden_shape)
        ).transpose(
            1, 2
        )  # [B, num_heads, T_suffix, head_dim]

        key_states = attention_layer.k_norm(
            attention_layer.k_proj(hidden_states).view(hidden_shape)
        ).transpose(
            1, 2
        )  # [B, num_kv_heads, T_suffix, head_dim]

        value_states = (
            attention_layer.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        )

        # Apply rotary position embeddings
        # Reference: modeling_qwen3_vl.py:432-433
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        # Concatenate cached prefix KV with current suffix KV
        # This allows suffix queries to attend to prefix keys/values
        key_states = torch.cat(
            [past_keys, key_states], dim=2
        )  # [B, num_kv_heads, T_prefix+T_suffix, head_dim]
        value_states = torch.cat([past_values, value_states], dim=2)

        # Select attention implementation (eager, flash_attention_2, sdpa, etc.)
        # Reference: modeling_qwen3_vl.py:440-442
        attention_interface: Callable = eager_attention_forward
        if attention_layer.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                attention_layer.config._attn_implementation
            ]

        # Apply attention
        # Reference: modeling_qwen3_vl.py:444-453
        attn_output, _ = attention_interface(
            attention_layer,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=attention_layer.scaling,
            dropout=(
                0.0
                if not attention_layer.training
                else attention_layer.attention_dropout
            ),
            position_ids=position_ids[0],  # Use first dimension (text position IDs)
        )

        # Reshape and project output
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attention_layer.o_proj(attn_output)

        return attn_output

    def _process_decoder_layer_with_kv_cache(
        self,
        decoder_layer: nn.Module,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        attention_mask: Tensor,
        past_keys: Tensor,
        past_values: Tensor,
        position_ids: Tensor,
    ) -> Tensor:
        """
        Process a single decoder layer with cached KV from prefix.

        This reimplements Qwen3VLTextDecoderLayer.forward but uses pre-computed
        KV cache instead of updating it. The layer processes suffix tokens while
        attending to both prefix (via cache) and suffix tokens.

        Reference: Qwen3VLTextDecoderLayer.forward (modeling_qwen3_vl.py:488-519)

        Args:
            decoder_layer: Qwen3VLTextDecoderLayer instance
            hidden_states: Input hidden states, shape [B, T_suffix, H]
            position_embeddings: Tuple of (cos, sin) for RoPE
            attention_mask: Attention mask for suffix tokens
            past_keys: Cached key states from prefix
            past_values: Cached value states from prefix
            position_ids: Position IDs for suffix tokens

        Returns:
            hidden_states: Output hidden states after layer processing
        """
        # Self-attention with residual connection
        # Reference: modeling_qwen3_vl.py:499-512
        residual = hidden_states
        hidden_states = decoder_layer.input_layernorm(hidden_states)

        attn_output = self._apply_attention_with_kv_cache(
            attention_layer=decoder_layer.self_attn,
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_keys=past_keys,
            past_values=past_values,
            position_ids=position_ids,
        )
        hidden_states = residual + attn_output

        # Feed-forward MLP with residual connection
        # Reference: modeling_qwen3_vl.py:514-518
        residual = hidden_states
        hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
        hidden_states = decoder_layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def forward_flow_matching(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        loss_mask: Tensor,
        state: Tensor,
        actions: Tensor,
        noise: Tensor | None = None,
        time: Tensor | None = None,
        data_source: Tensor | None = None,
        action_loss_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Forward pass for flow matching training mode.

        This method processes suffix tokens (state + noisy actions) through the decoder
        while reusing cached key-value pairs from the prefix (images + language tokens).
        This avoids recomputing the prefix encoding for each flow matching step.

        The implementation manually processes decoder layers to inject cached KV states,
        similar to Qwen3VLTextModel.forward but without updating the cache.

        Reference: Qwen3VLTextModel.forward (modeling_qwen3_vl.py:784-874)
        Reference: Qwen3VLTextDecoderLayer.forward (modeling_qwen3_vl.py:488-519)

        Args:
            images: List of image tensors
            img_masks: List of image masks
            lang_tokens: Language token IDs, shape [B, T_prefix]
            lang_masks: Language token padding mask, shape [B, T_prefix]
            loss_mask: Loss mask (not used in this function, reserved for future use)
            state: State tensor, shape [B, state_dim]
            actions: Clean action tensor, shape [B, T_action, action_dim]
            noise: Optional noise tensor. If None, will be sampled.
            time: Optional time tensor. If None, will be sampled.

        Returns:
            hidden_states: Final hidden states after all decoder layers, shape [B, T_suffix, H]
        """
       
        # Step 3: Get cached KV states from prefix (images + language tokens)
        # This avoids recomputing prefix encoding for each flow matching step
        vlm_output_with_cache = self.encode_vlm_prefix_with_kv_cache(
            images, img_masks, lang_tokens, lang_masks, return_cache=True
        )
        loss_dict = {}
        if self.config.model_mode == "mixed":
            ce_loss, ce_loss_dict = self.compute_ce_loss(vlm_output_with_cache["last_hidden_state"], lang_tokens, loss_mask, data_source)
            loss_dict.update(ce_loss_dict)
            loss_dict["ce_loss"] = ce_loss.item()
        

        # Step 1: Prepare flow matching inputs (noise, time, noisy actions)
        state = state.to(torch.float32)
        actions = actions.to(torch.float32)
        actions = actions.repeat(self.config.action_head_batch_multiplier, 1, 1)
        state = state.repeat(self.config.action_head_batch_multiplier, 1)

        #applying noise to state
        if self.config.apply_noise_to_state_for_flow_matching:
            state = state + torch.randn_like(state).to(state.device) * self.config.state_noise_amplitude
        
        x_t, u_t, time = self._prepare_flow_matching_noise_and_time(
            actions, noise, time
        )

        # Step 2: Embed suffix tokens (state + noisy actions + timestep)
        suffix_embs, suffix_attn_mask = self.embed_suffix(state, x_t, time)

        vlm_output_with_cache['past_keys'] = vlm_output_with_cache['past_keys'].repeat(self.config.action_head_batch_multiplier, 1, 1, 1, 1)
        vlm_output_with_cache['past_values'] = vlm_output_with_cache['past_values'].repeat(self.config.action_head_batch_multiplier, 1, 1, 1, 1)

        # Step 4: Compute position IDs for suffix tokens
        # Reference: Similar to position_id computation in Qwen3VLTextModel.forward
        
        lang_masks = lang_masks.repeat(self.config.action_head_batch_multiplier, 1)
        loss_mask = loss_mask.repeat(self.config.action_head_batch_multiplier, 1)
        token_type_ids = token_type_ids.repeat(self.config.action_head_batch_multiplier, 1)
        
        # masking out all the targets for CE loss which one we would not like to attend to while training FM action expert
        lang_masks[token_type_ids == 2] = 0 

        position_ids = self._compute_suffix_position_ids(suffix_attn_mask, lang_masks)

        # Step 5: Build attention mask for suffix tokens
        # This mask allows suffix queries to attend to prefix keys and has causal/block mask for suffix-suffix
        attn_impl = getattr(self.model.config, "_attn_implementation", "eager")
        attention_mask = self.build_suffix_attention_mask(
            lang_masks,
            suffix_attn_mask,
            attn_impl,
            suffix_attention_mode=self.config.suffix_attention_mask,
        )

        # Step 6: Prepare hidden states and position embeddings
        hidden_states = suffix_embs.to(torch.bfloat16)

        # Compute rotary position embeddings (shared across all layers)
        # Reference: modeling_qwen3_vl.py:846
        position_embeddings = self.action_expert.rotary_emb(hidden_states, position_ids)

        # Step 7: Process through decoder layers with cached KV
        # Reference: modeling_qwen3_vl.py:849-859
        # Use KV from the corresponding downsampled VLM layers.
        for expert_idx, (decoder_layer, base_layer_idx) in enumerate(
            zip(self.action_expert.layers, self.expert_layer_indices)
        ):
            prefix_keys, prefix_values = self._get_prefix_kv(
                expert_layer_idx=expert_idx,
                base_layer_idx=base_layer_idx,
                past_keys=vlm_output_with_cache["past_keys"],
                past_values=vlm_output_with_cache["past_values"],
            )
            hidden_states = self._process_decoder_layer_with_kv_cache(
                decoder_layer=decoder_layer,
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_keys=prefix_keys,
                past_values=prefix_values,
                position_ids=position_ids,
            )

        # Step 8: Apply final layer norm
        # Reference: modeling_qwen3_vl.py:869
        hidden_states = self.action_expert.norm(
            hidden_states
        )  # TODO: maybe remove final layer norm ????

        suffix_out = hidden_states
        suffix_out = suffix_out[:, -self.config.n_action_steps :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        v_t = self.action_out_proj(suffix_out)

        fm_loss = F.mse_loss(u_t, v_t, reduction="none")
        
        # Build combined mask for loss computation
        # Start with all ones (no masking)
        combined_mask = torch.ones_like(fm_loss)
        
        # Apply action padding mask if enabled: [B, 48] -> [B, 1, 48]
        if self.config.mask_padded_actions and action_loss_mask is not None:
            action_mask = action_loss_mask.float().unsqueeze(1)  # [B, 1, 48]
            if self.config.action_head_batch_multiplier > 1:
                action_mask = action_mask.repeat(self.config.action_head_batch_multiplier, 1, 1)
            combined_mask = combined_mask * action_mask
        
        # Apply robotics-only mask if data_source provided: [B] -> [B, 1, 1]
        if data_source is not None:
            source_mask = (data_source == 0).float().view(-1, 1, 1)
            if self.config.action_head_batch_multiplier > 1:
                source_mask = source_mask.repeat(self.config.action_head_batch_multiplier, 1, 1)
            combined_mask = combined_mask * source_mask
        

        # Compute masked loss
        masked_loss = fm_loss * combined_mask
        n_elements = combined_mask.sum()
        fm_loss = masked_loss.sum() / n_elements.clamp(min=1)
        
        loss_dict["fm_loss"] = fm_loss.item()
        
        if self.config.model_mode == "mixed":
            loss = self.config.ce_loss_weight * ce_loss + (1 - self.config.ce_loss_weight) * fm_loss
        else:
            loss = fm_loss

        return loss, loss_dict
    
    
    def compute_ce_loss(self,
        pre_logits: Tensor,
        lang_tokens: Tensor,
        loss_mask: Tensor,
        data_source: Tensor | None = None
    ) -> tuple[Tensor, dict[str, Tensor]]:

        device = lang_tokens.device
        # Shift for next-token prediction: predict token t+1 from position t
        logits = self.model.lm_head(pre_logits[:, :-1, :])  # [B, T-1, V]
        targets = lang_tokens[:, 1:].to(device=device, dtype=torch.long)  # [B, T-1]

        # Flatten for CrossEntropyLoss
        vocab_size = logits.shape[-1]
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)

        loss_fct = nn.CrossEntropyLoss(reduction="none")
        token_loss = loss_fct(logits_flat, targets_flat)  # [B*(T-1)]

        # Apply loss mask on the *targets* side (ignore prompt / padding)
        loss_mask_targets = loss_mask[:, 1:].to(
            device=device, dtype=torch.float32
        )  # [B, T-1]
        token_loss = token_loss * loss_mask_targets.reshape(-1)

        denom = torch.clamp(loss_mask_targets.sum(), min=1.0)
        loss = token_loss.sum() / denom

        # Metrics: masked accuracy over the same positions
        
        metrics = {}
        
        with torch.no_grad():
            predictions = logits.argmax(dim=-1)  # [B, T-1]
            correct = (predictions == targets).float()
            masked_correct = correct * loss_mask_targets
            if data_source is not None:
                # data_source: 0 = robotics, 1 = vlm (shape: [B])
                robotics_mask = (data_source == 0).unsqueeze(1).float()  # [B, 1]
                vlm_mask = (data_source == 1).unsqueeze(1).float()  # [B, 1]

                # Reshape token_loss for per-source computation
                B, T_minus_1 = loss_mask_targets.shape
                token_loss_2d = token_loss.reshape(B, T_minus_1)  # [B, T-1]

                # Compute loss and accuracy for robotics samples
                robotics_denom = (loss_mask_targets * robotics_mask).sum()
                if robotics_denom > 0:
                    robotics_ce_loss = (token_loss_2d * robotics_mask).sum() / robotics_denom
                    metrics["robotics_ce_loss"] = robotics_ce_loss.item()
                    robotics_accuracy = (masked_correct * robotics_mask).sum() / robotics_denom
                    metrics["robotics_accuracy"] = robotics_accuracy.item()

                # Compute loss and accuracy for VLM samples
                vlm_denom = (loss_mask_targets * vlm_mask).sum()
                if vlm_denom > 0:
                    vlm_ce_loss = (token_loss_2d * vlm_mask).sum() / vlm_denom
                    metrics["vlm_ce_loss"] = vlm_ce_loss.item()
                    vlm_accuracy = (masked_correct * vlm_mask).sum() / vlm_denom
                    metrics["vlm_accuracy"] = vlm_accuracy.item()
            
            accuracy = masked_correct.sum() / denom

        metrics["accuracy"] = accuracy.item()
        metrics["ce_loss"] = loss.item()
        return loss, metrics

    def forward_cross_entropy(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        loss_mask: Tensor,
        data_source: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Compute token-level cross-entropy loss for the language tokens.

        `lang_tokens` is assumed to contain both the prompt and the targets.
        We use standard teacher forcing: at position t we predict token t+1.
        `loss_mask` selects which target positions contribute to the loss.
        """
        device = lang_tokens.device

        # [B, T, H] hidden states from the language model (before lm_head)
        pre_logits = self.encode_vlm_prefix_with_kv_cache(
            images, img_masks, lang_tokens, lang_masks, return_cache=False
        )["last_hidden_state"]

        # Shift for next-token prediction: predict token t+1 from position t
        logits = self.model.lm_head(pre_logits[:, :-1, :])  # [B, T-1, V]
        targets = lang_tokens[:, 1:].to(device=device, dtype=torch.long)  # [B, T-1]

        # Flatten for CrossEntropyLoss
        vocab_size = logits.shape[-1]
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)

        loss_fct = nn.CrossEntropyLoss(reduction="none")
        token_loss = loss_fct(logits_flat, targets_flat)  # [B*(T-1)]

        # Apply loss mask on the *targets* side (ignore prompt / padding)
        loss_mask_targets = loss_mask[:, 1:].to(
            device=device, dtype=torch.float32
        )  # [B, T-1]
        token_loss = token_loss * loss_mask_targets.reshape(-1)

        denom = torch.clamp(loss_mask_targets.sum(), min=1.0)
        loss = token_loss.sum() / denom

        # Metrics: masked accuracy over the same positions
        
        metrics = {}
        
        with torch.no_grad():
            predictions = logits.argmax(dim=-1)  # [B, T-1]
            correct = (predictions == targets).float()
            masked_correct = correct * loss_mask_targets
            if data_source is not None:
                # data_source: 0 = robotics, 1 = vlm (shape: [B])
                robotics_mask = (data_source == 0).unsqueeze(1).float()  # [B, 1]
                vlm_mask = (data_source == 1).unsqueeze(1).float()  # [B, 1]

                # Reshape token_loss for per-source computation
                B, T_minus_1 = loss_mask_targets.shape
                token_loss_2d = token_loss.reshape(B, T_minus_1)  # [B, T-1]

                # Compute loss and accuracy for robotics samples
                robotics_denom = (loss_mask_targets * robotics_mask).sum()
                if robotics_denom > 0:
                    robotics_ce_loss = (token_loss_2d * robotics_mask).sum() / robotics_denom
                    metrics["robotics_ce_loss"] = robotics_ce_loss.item()
                    robotics_accuracy = (masked_correct * robotics_mask).sum() / robotics_denom
                    metrics["robotics_accuracy"] = robotics_accuracy.item()

                # Compute loss and accuracy for VLM samples
                vlm_denom = (loss_mask_targets * vlm_mask).sum()
                if vlm_denom > 0:
                    vlm_ce_loss = (token_loss_2d * vlm_mask).sum() / vlm_denom
                    metrics["vlm_ce_loss"] = vlm_ce_loss.item()
                    vlm_accuracy = (masked_correct * vlm_mask).sum() / vlm_denom
                    metrics["vlm_accuracy"] = vlm_accuracy.item()
            
            accuracy = masked_correct.sum() / denom

        metrics["accuracy"] = accuracy.item()
        metrics["ce_loss"] = loss.item()
        return loss, metrics

    def generate(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        max_decode_len: int = 128,
        temperature: float | None = None,
    ) -> Tensor:

        attention_mask = lang_masks
        images = torch.stack(images, dim=1)  # first dim is batch size
        img_masks = torch.stack(img_masks, dim=1)
        B, N, C, H, W = images.shape
        images = images.reshape(B * N, C, H, W)
        img_masks = img_masks.reshape(B * N)

        # choosing only not masked images
        images = images[img_masks]
        pixel_values, image_grid_thw = self.torch_qwen_image_processor(images)

        do_sample = False if temperature is None else True

        # So, we can not easily pass deepstack_image_embeds into the generate method, so we would use here the classical generate
        lang_tokens, attention_mask, _ = shift_padding_side(
            lang_tokens, attention_mask, attention_mask, "left"
        )

        output = self.model.generate(
            input_ids=lang_tokens,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            max_new_tokens=max_decode_len,
            temperature=temperature,
            do_sample=do_sample,
            use_cache=False,
        )
        return output

    @torch.no_grad()
    def sample_actions(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        state: Tensor,
        loss_mask: Tensor | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = lang_tokens.shape[0]
        device = lang_tokens.device
        state = state.to(torch.float32)

        if noise is None:
            actions_shape = (
                bsize,
                self.config.n_action_steps,
                self.config.max_action_dim,
            )
            noise = sample_noise(actions_shape, device)
        
        if loss_mask is not None:
            lang_masks[loss_mask > 0] = 0 # masking out all the targets for CE loss 
        
        vlm_output_with_cache = self.encode_vlm_prefix_with_kv_cache(
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            return_cache=True,
        )
        # position_ids = self._compute_suffix_position_ids(suffix_attn_mask, lang_masks)

        dt = -1.0 / self.config.num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)

        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state=state,
                prefix_pad_masks=lang_masks,
                vlm_output_with_cache=vlm_output_with_cache,
                x_t=x_t,
                timestep=expanded_time,
            )
            x_t += dt * v_t
            time += dt
        return x_t

    def denoise_step(
        self,
        state: Tensor,
        prefix_pad_masks: Tensor,
        vlm_output_with_cache: dict[str, Tensor],
        x_t: Tensor,
        timestep: Tensor,
    ) -> Tensor:
        suffix_embs, suffix_attn_mask = self.embed_suffix(state, x_t, timestep)
        position_ids = self._compute_suffix_position_ids(
            suffix_attn_mask, prefix_pad_masks
        )
        attn_impl = getattr(self.model.config, "_attn_implementation", "eager")
        attention_mask = self.build_suffix_attention_mask(
            prefix_pad_masks,
            suffix_attn_mask,
            attn_implementation=attn_impl,
            suffix_attention_mode=self.config.suffix_attention_mask,
        )
        hidden_states = suffix_embs.to(dtype=torch.bfloat16)
        position_embeddings = self.action_expert.rotary_emb(hidden_states, position_ids)
        for expert_idx, (decoder_layer, base_layer_idx) in enumerate(
            zip(self.action_expert.layers, self.expert_layer_indices)
        ):
            prefix_keys, prefix_values = self._get_prefix_kv(
                expert_layer_idx=expert_idx,
                base_layer_idx=base_layer_idx,
                past_keys=vlm_output_with_cache["past_keys"],
                past_values=vlm_output_with_cache["past_values"],
            )
            hidden_states = self._process_decoder_layer_with_kv_cache(
                decoder_layer=decoder_layer,
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_keys=prefix_keys,
                past_values=prefix_values,
                position_ids=position_ids,
            )

        # Step 8: Apply final layer norm
        # Reference: modeling_qwen3_vl.py:869
        hidden_states = self.action_expert.norm(
            hidden_states
        )  # TODO: maybe remove final layer norm ????
        suffix_out = hidden_states
        suffix_out = suffix_out[:, -self.config.n_action_steps :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return v_t
