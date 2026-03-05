import torch
from transformers.cache_utils import HybridCache, StaticCache
import torch.nn.functional as F  # noqa: N812
from transformers.cache_utils import HybridCache, StaticCache
from torch import Tensor, nn
import math

def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype



def block_causal_update_causal_mask(
    attention_mask,
    token_type_ids=None,
    past_key_values=None,
    cache_position=None,
    input_tensor=None,
    attn_implementation: str = "eager",
    dtype: torch.dtype = "float32",
):
    """
    Update the causal mask during training and generation. It can be customized to different attention masks.
    """
    if attn_implementation == "flash_attention_2":
        if attention_mask is not None and 0.0 in attention_mask:
            return attention_mask
        return None
    using_static_cache = isinstance(past_key_values, StaticCache)
    min_dtype = torch.finfo(dtype).min

    if input_tensor is None:
        input_tensor = attention_mask

    inputs_lead_dim, sequence_length = input_tensor.shape[:2]

    if using_static_cache or isinstance(past_key_values, HybridCache):
        target_length = past_key_values.get_max_cache_shape()
    else:
        target_length = (
            attention_mask.shape[-1]
            if isinstance(attention_mask, torch.Tensor)
            else cache_position[0] + sequence_length + 1
        )

    # Handle precomputed attention masks
    if attention_mask is not None and attention_mask.dim() == 4:
        return attention_mask

    # Causal mask initialization
    causal_mask = torch.full(
        (sequence_length, target_length),
        fill_value=min_dtype,
        dtype=dtype,
        device=cache_position.device,
    )

    # Standard causal masking (triu ensures tokens can only attend to past)
    if sequence_length != 1:
        causal_mask = torch.triu(causal_mask, diagonal=1)

        # Apply block causal mask
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(causal_mask.device).bool()
            B, S = token_type_ids.size()
            cumsum = token_type_ids.cumsum(dim=1)  # [B, S]
            # block_causal_mask full 3D
            block_causal_mask = cumsum[:, None, :] <= cumsum[:, :, None]  # [B, S, S]
            # ensure causal_mask shape = [B, 1, S, S]
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(B, 1, S, S)
            # apply block masking
            causal_mask = torch.where(block_causal_mask.unsqueeze(1), 0.0, causal_mask)
        else:
            # Apply past cache position constraint
            causal_mask *= torch.arange(
                target_length, device=cache_position.device
            ) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(
                inputs_lead_dim, 1, -1, -1
            )
    else:
        # Apply past cache position constraint
        causal_mask *= torch.arange(
            target_length, device=cache_position.device
        ) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(inputs_lead_dim, 1, -1, -1)

    if attention_mask is not None:
        causal_mask = (
            causal_mask.clone()
        )  # Copy to contiguous memory for in-place edits
        mask_length = attention_mask.shape[-1]

        # Apply padding mask
        padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[
            :, None, None, :
        ].to(causal_mask.device)
        padding_mask = padding_mask == 0
        causal_mask[:, :, :, :mask_length] = causal_mask[
            :, :, :, :mask_length
        ].masked_fill(padding_mask, min_dtype)

    return causal_mask

def prepare_inputs_for_generation(
    input_ids,
    past_key_values=None,
    inputs_embeds=None,
    cache_position=None,
    position_ids=None,
    pixel_values=None,
    attention_mask=None,
    token_type_ids=None,
    use_cache=True,
    num_logits_to_keep=None,
    labels=None,
    self=None,
    **kwargs,
):
    # create block causal attention
    if cache_position[0] > 0 and input_ids.shape[1] > 0:
        input_tensor = input_ids[:, -1:]
        new_positions = (
            torch.ones(
                (position_ids.shape[0], input_ids.shape[1]),
                dtype=position_ids.dtype,
                device=position_ids.device,
            ).cumsum(-1)
            + position_ids[:, -1:]
        )
        position_ids = torch.cat([position_ids, new_positions], dim=-1)
    else:
        input_tensor = inputs_embeds
    attention_mask = block_causal_update_causal_mask(
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        cache_position=cache_position,
        input_tensor=input_tensor,
        token_type_ids=token_type_ids,
        dtype=self.dtype,
        attn_implementation=self.config.text_config._attn_implementation,
    )
    # Overwritten -- custom `position_ids` and `pixel_values` handling
    
    
    #creating model inputs without using prepare_inputs_for_generation
    
    
    model_inputs = self.legacy_prepare_inputs_for_generation(
        input_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        cache_position=cache_position,
        use_cache=use_cache,
        num_logits_to_keep=num_logits_to_keep,
        token_type_ids=token_type_ids,
        **kwargs,
    )
    # Position_ids in Qwen are 1-indexed
    if model_inputs.get("position_ids") is not None:
        model_inputs["position_ids"] += 1
    # If we're in cached decoding stage, pixel values should be None because input ids do not contain special image token anymore
    # Otherwise we need pixel values to be passed to model. NOTE: use_cache=False needs pixel_values always
    if cache_position[0] == 0:
        model_inputs["pixel_values"] = pixel_values
    is_training = token_type_ids is not None and labels is not None
    if cache_position[0] == 0 and isinstance(past_key_values, HybridCache):
        input_tensor = inputs_embeds if inputs_embeds is not None else input_ids
        causal_mask = self._update_causal_mask(
            attention_mask,
            token_type_ids,
            past_key_values,
            cache_position,
            input_tensor,
            is_training,
        )
        model_inputs["attention_mask"] = causal_mask

    return model_inputs


def put_along_last_axis(arr, indices, values):
    """JAX's put_along_axis(..., axis=-1) implemented in PyTorch."""
    onehot = F.one_hot(indices, arr.shape[-1]).to(dtype=values.dtype)
    put_mask = (torch.ones_like(values, dtype=torch.int32).unsqueeze(-1) * onehot).sum(
        -2
    )
    put_values = values.unsqueeze(-1) * onehot
    return torch.where(put_mask.bool(), put_values.sum(-2), arr)


def left_to_right_align(x, input_mask, attn_mask):
    """Converts input from left-align to right-aligned."""
    # x: [B, S, E], input_mask: [B, S], attn_mask: [B, S, S]
    batch_size = x.shape[0]

    # Equivalent to JAX's `jnp.max(input_mask * jnp.arange(input_mask.shape[0])) + 1`
    # under vmap.
    seq_lengths = torch.sum(input_mask, dim=1, dtype=torch.long)

    rolled_x = torch.zeros_like(x)
    rolled_input_mask = torch.zeros_like(input_mask)
    rolled_attn_mask = torch.zeros_like(attn_mask)

    for i in range(batch_size):
        shift = -seq_lengths[i]
        rolled_x[i] = torch.roll(x[i], shifts=shift.item(), dims=0)
        rolled_input_mask[i] = torch.roll(input_mask[i], shifts=shift.item(), dims=0)
        rolled_attn_mask[i] = torch.roll(
            attn_mask[i], shifts=(shift.item(), shift.item()), dims=(0, 1)
        )

    return rolled_x, rolled_input_mask, rolled_attn_mask


def create_sinusoidal_pos_embedding(
    time: torch.tensor,
    dimension: int,
    min_period: float,
    max_period: float,
    device="cpu",
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks

def shift_padding_side(
    tokens: torch.Tensor,
    ar_mask: torch.Tensor,
    padding_mask: torch.Tensor,
    token_type_ids: torch.Tensor,
    padding_side: str = "right",
) -> tuple[torch.Tensor]:
    if padding_side not in ["right", "left"]:
        return tokens, ar_mask, padding_mask, token_type_ids

    new_tokens = torch.empty_like(tokens)
    new_ar_masks = torch.empty_like(ar_mask)
    new_padding_mask = torch.empty_like(padding_mask)
    new_token_type_ids = torch.empty_like(token_type_ids)
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
        new_token_type_ids[i] = token_type_ids[i].index_select(0, new_indices)

    return (
        new_tokens,
        new_ar_masks,
        new_padding_mask,
        new_token_type_ids,
    )