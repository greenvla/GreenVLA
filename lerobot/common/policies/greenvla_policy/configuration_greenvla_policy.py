from dataclasses import dataclass
from dataclasses import field
from lerobot.configs.policies import PreTrainedConfig


@PreTrainedConfig.register_subclass("greenvlapolicy")
@dataclass
class GreenVLAPolicyConfig(PreTrainedConfig):
    # Input/output sequence lengths
    n_obs_steps: int = 1
    n_action_steps: int = 10
    model_type: str = "greenvlapolicy"
    num_steps: int = 10
    # Shorter state and action vectors will be padded
    max_state_dim: int = 32
    max_action_dim: int = 32
    map_to_unified_space: bool = False
    unified_space_dim: int = 64
    add_action_space_factorization: bool = True
    
    image_keys: list[str] = field(
        default_factory=lambda: ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
    )


    # Tokenizer
    tokenizer_max_length: int = 296
    
    proj_width: int = 1024
    # How often action expert attends over VLM layers: 1 = every layer,
    # 2 = every second VLM layer, 3 = every third, etc.
    expert_block_stride: int = 4

    # Base VLM model
    base_vlm_model: str = "Qwen/Qwen3-VL-2B-Instruct"
    precision: str = "bfloat16"
    normalization_mode: str | None = "quantile"
    suffix_attention_mask: str = "causal"  # "causal" or "block" (block = full attention among suffix tokens)
    model_mode: str = "token_prediction" # flow_matching / token_prediction / mixed
    inference_mode: str = "flow_matching" # flow_matching / token_prediction (works only for model_mode = mixed)
    image_shape: tuple[int, int] = (448, 448)
    default_temperature: float = 0.7 # default temperature for token generation
    discrete_state_input: bool = False
    continuous_state_input: bool = False
    add_state_proj_to_action_expert: bool = True
    apply_noise_to_state_for_flow_matching: bool = False
    state_noise_amplitude: float = 0.1
    state_dropout_prob: float = 0.5
    attention_implementation: str = "flash_attention_2"
    mask_padded_actions: bool = False #whether to mask padded actions in the loss
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
