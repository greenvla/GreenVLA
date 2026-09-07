from dataclasses import dataclass
from dataclasses import field
from lerobot.configs.policies import PreTrainedConfig
from .identity import POLICY_TYPE, POLICY_ALIASES


@PreTrainedConfig.register_subclass("qwen3vlpolicy")
@PreTrainedConfig.register_subclass("greenvla_v1_1")
@PreTrainedConfig.register_subclass(POLICY_TYPE)
@dataclass
class GreenVLAv11Config(PreTrainedConfig):
    """The config a GreenVLAv1.1 checkpoint carries in its config.json.

    Some fields below only meant anything while the checkpoint was being trained
    (loss weights, dropout, freezing). They stay because a trained config.json names
    them and has to parse; inference reads the rest.
    """

    # Input/output sequence lengths
    n_obs_steps: int = 1
    # Historical fields remain parseable; the serving contract requires one observation.
    obs_stride_seconds: float = 0.0
    # Proprioception is sampled independently from video. A dense history can
    # therefore cover robot dynamics without increasing visual token count.
    n_state_obs_steps: int = 1
    state_obs_stride_seconds: float = 0.0
    n_action_steps: int = 50
    model_type: str = POLICY_TYPE
    num_steps: int = 10
    # Shorter state and action vectors will be padded
    max_state_dim: int = 50
    max_action_dim: int = 50
    map_to_unified_space: bool = False
    unified_space_dim: int = 64
    add_action_space_factorization: bool = True
    
    image_keys: list[str] = field(
        default_factory=lambda: ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
    )


    # Tokenizer
    tokenizer_max_length: int = 768
    
    proj_width: int = 1024
    # How often action expert attends over VLM layers: 1 = every layer,
    # 2 = every second VLM layer, 3 = every third, etc.
    expert_block_stride: int = 4

    # Base VLM model
    base_vlm_model: str = "Qwen/Qwen3.5-0.8B"
    # Released architecture: external queries over VLM context -> action expert.
    cognition_mode: str = "frozen_vlm_context_queries"
    action_expert_model: str | None = "Qwen/Qwen3.5-0.8B"
    num_cognition_tokens: int = 64
    cognition_vlm_layer: int = -1
    # When enabled, collect cognition-token states after every VLM decoder
    # layer and combine them with a learned softmax over depth.
    cognition_layer_mix: bool = True
    cognition_layer_mix_per_token: bool = True
    cognition_layer_mix_init: str = "uniform"
    cognition_projection: str = "linear"
    # Run the frozen Qwen3.5 backbone as an immutable context encoder and
    # learn cognition queries in a small external cross-attention adapter.
    # This avoids backpropagating through Gated DeltaNet while preserving an
    # explicit, per-query learned preference over every VLM decoder layer.
    cognition_context_width: int = 512
    cognition_context_num_heads: int = 8
    cognition_context_ffn_multiplier: int = 4
    cognition_frozen_vlm_reference_kernels: bool = False
    cognition_action_expert_reference_kernels: bool = False
    num_trainable_vlm_layers: int | None = None
    inputs_embeds_gradient_checkpointing: bool = False
    precision: str = "bfloat16"
    normalization_mode: str | None = "mean_std"
    suffix_attention_mask: str = "causal"  # "causal" or "block" (block = full attention among suffix tokens)
    model_mode: str = "flow_matching"
    inference_mode: str = "flow_matching" # flow_matching / token_prediction (works only for model_mode = mixed)
    image_shape: tuple[int, int] = (448, 448)
    default_temperature: float = 0.7 # default temperature for token generation
    discrete_state_input: bool = False
    continuous_state_input: bool = False
    # Encode one continuous state token per history step in the frozen Qwen3.5
    # prefix immediately before learned cognition queries.
    add_state_vlm: bool = False
    state_history_dropout_prob: float = 0.0
    add_state_proj_to_action_expert: bool = True
    apply_noise_to_state_for_flow_matching: bool = False
    state_noise_amplitude: float = 0.1
    state_dropout_prob: float = 0.5
    attention_implementation: str = "flash_attention_2"
    mask_padded_actions: bool = False #whether to mask padded actions in the loss
    hold_at_subtask_boundary: bool = False #whether to repeat the last action past a subtask boundary
    # Fraction of the full action horizon that may be trained as repeated final
    # actions after a subtask boundary. 1.0 preserves legacy behavior.
    hold_at_subtask_boundary_loss_fraction: float = 1.0
    mask_temporally_padded_actions: bool = False #whether to mask temporally padded actions in the loss
    # When true, action expert layers attend to learnable linear combinations
    # of all VLM layers instead of a single stride-aligned layer.
    enable_learnable_layer_combination: bool = False
    # Initialization strategy for learnable layer combination weights:
    # - "identity": (default) copy stride-aligned VLM layer as-is
    # - "stride_uniform": uniform over layers within the stride window
    # - "global_uniform": uniform over all VLM layers
    layer_combination_init: str = "identity"

    is_knowledge_insulation: bool = False # if True, KI is applied to the action expert, no grad flow through KV cache
    ce_loss_weight: float = 0.5 # if mode is mixed, this is the weight of the CE loss, FM loss weight will be 1 - ce_loss_weight

    freeze_vlm: bool = False
    compile_sample_actions: bool = False


    #Below there are not used currently settings
    state_special_token_id: int = 228
    clip_state: bool = False
    

    # Finetuning settings
    freeze_vision_encoder: bool = False
    use_cache: bool = False
    add_control_mode: bool = True
    add_embodiment_name: bool = True
    
    
    action_head_batch_multiplier: int = 1


    def __post_init__(self):
        super().__post_init__()
        if self.model_type not in POLICY_ALIASES:
            raise ValueError(f"Unsupported GreenVLAv1.1 model_type: {self.model_type!r}")
        # Normalize in memory only; never rewrite a supplied checkpoint config.
        self.model_type = POLICY_TYPE
        if self.n_obs_steps < 1:
            raise ValueError("n_obs_steps must be positive")
        if self.n_obs_steps > 1 and self.obs_stride_seconds <= 0:
            raise ValueError(
                "obs_stride_seconds must be positive when n_obs_steps > 1"
            )
        if self.n_state_obs_steps < 1:
            raise ValueError("n_state_obs_steps must be positive")
        if self.n_state_obs_steps > 1 and self.state_obs_stride_seconds <= 0:
            raise ValueError(
                "state_obs_stride_seconds must be positive when n_state_obs_steps > 1"
            )
        if not 0.0 <= self.state_history_dropout_prob < 1.0:
            raise ValueError("state_history_dropout_prob must be in [0.0, 1.0)")
        qwen35_cognition_modes = {
            "learned_query_tokens",
            "frozen_vlm_context_queries",
        }
        if self.cognition_mode not in {"legacy_kv_cache", *qwen35_cognition_modes}:
            raise ValueError(
                "cognition_mode must be 'legacy_kv_cache', "
                "'learned_query_tokens', or 'frozen_vlm_context_queries'; "
                f"got {self.cognition_mode!r}"
            )
        if self.cognition_projection not in {"linear", "mlp"}:
            raise ValueError(
                "cognition_projection must be 'linear' or 'mlp', "
                f"got {self.cognition_projection!r}"
            )
        if self.num_cognition_tokens <= 0:
            raise ValueError("num_cognition_tokens must be positive")
        if self.num_trainable_vlm_layers is not None and self.num_trainable_vlm_layers < 0:
            raise ValueError("num_trainable_vlm_layers must be non-negative or None")
        if self.cognition_layer_mix_init not in {"uniform", "last_layer"}:
            raise ValueError(
                "cognition_layer_mix_init must be 'uniform' or 'last_layer', "
                f"got {self.cognition_layer_mix_init!r}"
            )
        if self.cognition_layer_mix and self.cognition_mode not in qwen35_cognition_modes:
            raise ValueError(
                "cognition_layer_mix is only supported with a Qwen3.5 cognition mode"
            )
        if self.cognition_mode in qwen35_cognition_modes:
            if not self.action_expert_model:
                raise ValueError(
                    "action_expert_model is required when "
                    f"cognition_mode={self.cognition_mode!r}"
                )
            if self.model_mode != "flow_matching":
                raise ValueError(
                    "Qwen3.5 cognition modes currently support model_mode='flow_matching' only"
                )
            if self.suffix_attention_mask != "causal":
                raise ValueError(
                    "Qwen3.5's Gated DeltaNet action expert is causal; set "
                    "suffix_attention_mask='causal'"
                )
        elif self.n_state_obs_steps > 1:
            raise ValueError(
                "State history is currently supported only by Qwen3.5 cognition modes"
            )
        if self.add_state_vlm and self.cognition_mode != "learned_query_tokens":
            raise ValueError(
                "add_state_vlm requires cognition_mode='learned_query_tokens' so gradients can flow "
                "from frozen Qwen3.5 operations into the state encoder"
            )
        if self.cognition_mode == "frozen_vlm_context_queries":
            if not self.cognition_layer_mix:
                raise ValueError(
                    "frozen_vlm_context_queries requires cognition_layer_mix=True"
                )
            if not self.cognition_layer_mix_per_token:
                raise ValueError(
                    "frozen_vlm_context_queries requires per-token layer mixing"
                )
            if self.cognition_context_num_heads <= 0:
                raise ValueError("cognition_context_num_heads must be positive")
            if self.cognition_context_width <= 0:
                raise ValueError("cognition_context_width must be positive")
            if self.cognition_context_width % self.cognition_context_num_heads != 0:
                raise ValueError(
                    "cognition_context_width must be divisible by "
                    "cognition_context_num_heads"
                )
            if self.cognition_context_ffn_multiplier <= 0:
                raise ValueError("cognition_context_ffn_multiplier must be positive")
        if not 0.0 <= self.hold_at_subtask_boundary_loss_fraction <= 1.0:
            raise ValueError(
                "hold_at_subtask_boundary_loss_fraction must be in [0.0, 1.0], "
                f"got {self.hold_at_subtask_boundary_loss_fraction}"
            )


# Import compatibility for existing clients; not a second config implementation.
Qwen3VLPolicyConfig = GreenVLAv11Config
