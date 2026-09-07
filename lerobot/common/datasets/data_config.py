import abc
import dataclasses
import logging
import pathlib
from collections.abc import Sequence
from typing import Any, Protocol

import etils.epath as epath
import tyro
from typing_extensions import override

import lerobot.common.utils.normalize as _normalize
from lerobot.common.datasets.torch_transforms import (AbsoluteActions, DeltaActions, ExtractQwen05ActionsTorch, Group, InjectDefaultPromptTorch, InterpolateActions, RepackTransform, ResizeImagesTorch, SimpleOutputsTransform, SmoothActions, TokenizeQwen05InputsTransform, make_bool_mask)
from lerobot.common.policies.greenvla_v1_1.tokenizer import GreenVLAv11Tokenizer
from lerobot.common.policies.greenvla_v1_1.identity import POLICY_ALIASES


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="s3://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Optional asset id for explicitly configured asset bundles.
    # Robotics dataset stats use the dataset factory's top-level `asset_id`.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    root_dir: str | None = None
    task_file_path: str | None = None
    episodes_list_file: str | None = None
    sample_weights_file: str | None = None
    return_subtasks: bool = False
    return_subtasks_mode: str | None = None
    exclude_negative_subtasks: bool = False
    action_offset: int = 0
    action_sample_step: int = 1
    action_horizon: int = 16

    # used only for tokenization in prompt
    state_dim: int | None = None
    control_mode: str | None = None
    # optional string describing how the action space is factorized
    action_space_factorization: str | None = None
    map_to_unified_space: bool = True
    validation_episodes: str | None = None 
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _normalize.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: Group = dataclasses.field(default_factory=Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: Group = dataclasses.field(default_factory=Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: Group = dataclasses.field(default_factory=Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)
    # Raw dataset camera columns that should be queried over the observation
    # history when the policy requests more than one observation step.
    observation_sequence_keys: Sequence[str] = ()
    # Raw proprioception columns queried independently from camera history.
    # Empty means infer the source mapped to canonical observation/state.
    state_sequence_keys: Sequence[str] = ()

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # If true, will disable syncing the dataset from the Hugging Face Hub. Allows training on local-only datasets.
    local_files_only: bool = False
    policy_metadata_source_root: str | None = None
    quality_annotation_root: str | None = None
    quality_annotation_version: str | None = None
    quality_annotation_horizon: int | None = None
    quality_annotation_allow_missing_episodes_file: str | None = None
    quality_dropout_probabilities: dict[str, float] | None = None

    mixture_configs: Sequence["DataConfig"] | None = None
    mixture_weights: Sequence[float] | None = None
    # Per-dataset weight schedules: list of dicts with keys start_weight, end_weight,
    # start_step_percent (default 0.0), end_step_percent (default 1.0).
    # If None or empty, weights are static (use mixture_weights).
    mixture_weight_schedules: Sequence[dict] | None = None
    mixture_sample_category_weights: dict[str, float] | None = None
    mixture_metadata_dropout_prob: float = 0.0
    mixture_stop_on_empty: bool = False

    def __post_init__(self):
        if self.mixture_configs is None:
            object.__setattr__(self, "mixture_configs", ())

        else:
            object.__setattr__(self, "mixture_configs", tuple(self.mixture_configs))

        if self.mixture_weights is None:
            object.__setattr__(self, "mixture_weights", ())
        else:
            object.__setattr__(self, "mixture_weights", tuple(self.mixture_weights))

        if self.mixture_weight_schedules is None:
            object.__setattr__(self, "mixture_weight_schedules", ())
        else:
            object.__setattr__(self, "mixture_weight_schedules", tuple(self.mixture_weight_schedules))

        if (
            self.mixture_configs
            and self.mixture_weights
            and len(self.mixture_configs) != len(self.mixture_weights)
        ):
            raise ValueError(
                f"Length of mixture_configs ({len(self.mixture_configs)}) does not match "
                f"length of mixture_weights ({len(self.mixture_weights)})."
            )


class GroupFactory(Protocol):
    def __call__(self, model_config: Any) -> Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates the model transforms for the GreenVLAv1.1 whole-body policy."""

    default_prompt: str | None = None

    # If provided, will determine the default prompt that be used by the model.

    def __call__(self, model_config, data_config) -> Group:
        if model_config.model_type not in ("qwen05", *POLICY_ALIASES):
            raise ValueError(f"Model type {model_config.model_type} not supported")
        return Group(
            inputs=[
                InjectDefaultPromptTorch(self.default_prompt),
                ResizeImagesTorch(*model_config.image_shape),
                TokenizeQwen05InputsTransform(
                    GreenVLAv11Tokenizer(max_len=model_config.tokenizer_max_length,
                                    state_dim=data_config.state_dim,
                                    control_mode=data_config.control_mode,
                                    embodiment_name=data_config.name,
                                    image_keys=model_config.image_keys,
                                    base_vlm_model=model_config.base_vlm_model,
                                    discrete_state_input=model_config.discrete_state_input,
                                    continuous_state_input=model_config.continuous_state_input,
                                    state_dropout_prob=model_config.state_dropout_prob,
                                    state_special_token_id=model_config.state_special_token_id,
                                    clip_state=model_config.clip_state,
                                    add_control_mode=model_config.add_control_mode,
                                    add_embodiment_name=model_config.add_embodiment_name,
                                    model_mode=model_config.model_mode,
                                    image_shape=model_config.image_shape,
                                    n_obs_steps=model_config.n_obs_steps,
                                    obs_stride_seconds=model_config.obs_stride_seconds,
                                    ),
                ),
            ],
            outputs=[
                ExtractQwen05ActionsTorch(
                    GreenVLAv11Tokenizer(max_len=model_config.tokenizer_max_length,
                                    state_dim=data_config.state_dim,
                                    control_mode=data_config.control_mode,
                                    embodiment_name=data_config.name,
                                    image_keys=model_config.image_keys,
                                    base_vlm_model=model_config.base_vlm_model,
                                    discrete_state_input=model_config.discrete_state_input,
                                    continuous_state_input=model_config.continuous_state_input,
                                    state_dropout_prob=model_config.state_dropout_prob,
                                    state_special_token_id=model_config.state_special_token_id,
                                    clip_state=model_config.clip_state,
                                    add_control_mode=model_config.add_control_mode,
                                    add_embodiment_name=model_config.add_embodiment_name,
                                    model_mode=model_config.model_mode,
                                    image_shape=model_config.image_shape,
                                    n_obs_steps=model_config.n_obs_steps,
                                    obs_stride_seconds=model_config.obs_stride_seconds,
                                    ),
                    inference_mode=model_config.inference_mode,
                    action_horizon=data_config.action_horizon,
                    action_dim=model_config.max_action_dim,
                    model_mode=model_config.model_mode,
                )
            ],
        )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    name: str = ""
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None
    # Optional replacement for <root_dir>/meta/tasks.jsonl.
    task_file_path: str | None = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: Any) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = getattr(self, "asset_id", None)
        assert asset_id is not None, (
            f"{type(self).__name__} must define asset_id in the robotics dataset config."
        )
        base_config = self.base_config or DataConfig()
        return dataclasses.replace(
            base_config,
            repo_id=repo_id,
            asset_id=asset_id,
            task_file_path=self.task_file_path or base_config.task_file_path,
            norm_stats=self._load_norm_stats(
                epath.Path(self.assets.assets_dir or assets_dirs), asset_id
            ),
        )

    def _load_norm_stats(
        self, assets_dir: epath.Path, asset_id: str | None
    ) -> dict[str, _normalize.NormStats] | None:
        """Load the statistics named by `asset_id`, or refuse to build the config.

        Missing statistics used to be swallowed here and reported as one INFO line
        ("skipping"), leaving `norm_stats=None` to travel down the pipeline. That is the
        one failure this runtime must never absorb: unnormalized state reaches the model
        and unnormalized actions reach the robot, and both look entirely plausible on the
        wire. A checkpoint whose statistics cannot be found is a checkpoint this runtime
        cannot serve, so it fails here, at load time, naming the path it looked in.
        """
        if asset_id is None:
            return None
        data_assets_dir = pathlib.Path(str(assets_dir / asset_id))
        if not data_assets_dir.is_dir():
            raise FileNotFoundError(
                f"Normalization statistics not found at {data_assets_dir}. They ship "
                "inside the checkpoint, under norm_stats/<asset_id>/, and `asset_id` "
                "comes from the data config selected with --data-config "
                f"(this one asks for {asset_id!r})."
            )
        norm_stats = _normalize.load(data_assets_dir.resolve())
        logging.info(f"Loaded norm stats from {data_assets_dir}")
        return norm_stats



@dataclasses.dataclass(frozen=True)
class LeRobotHumanoidS0S1DataConfig(DataConfigFactory):
    """S0S1 whole-body data mapped to the 50D WBC state/action layout."""

    asset_id: str | None = None
    repo_id: str | None = None
    root_dir: str | None = None
    episodes_list_file: str | None = None
    sample_weights_file: str | None = None
    action_sequence_keys: Sequence[str] = ("action",)
    action_horizon: int = 50
    action_offset: int = 0
    action_sample_step: float = 1.0
    state_dim: int = 50
    control_mode: str = "joint"
    smooth_actions: bool = True
    action_space_factorization: str = (
        "Num hands: 2. Control mode: joints. Num joints per hand: 7. "
        "Hand type: dexhand. Torso: True. Legs: True. Head: True. "
        "Base velocity: 6-DOF."
    )
    validation_episodes: str | None = None
    has_velocities: bool = True
    map_to_closed_kinematic: bool = True
    input_kinematic_space: str = "open"
    model_kinematic_space: str = "closed"
    return_subtasks: bool = False
    return_subtasks_mode: str | None = None
    prompt_from_subtask: bool = False
    exclude_negative_subtasks: bool = False

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: Any
    ) -> DataConfig:
        from lerobot.common.datasets.data_transforms.robots.humanoid import (
            HumanoidS0S1InputsTransform,
            S0S1_OUTPUT_DIM,
        )

        current_action_dim = (
            model_config.max_action_dim
            if getattr(model_config, "max_action_dim", None) is not None
            else S0S1_OUTPUT_DIM
        )
        if current_action_dim < S0S1_OUTPUT_DIM:
            raise ValueError(
                f"S0/S1 WBC requires max_action_dim >= {S0S1_OUTPUT_DIM}, "
                f"got {current_action_dim}"
            )

        effective_subtasks_mode = self.return_subtasks_mode or (
            "optional" if self.return_subtasks else "disabled"
        )
        if self.prompt_from_subtask and effective_subtasks_mode == "disabled":
            raise ValueError(
                "prompt_from_subtask requires return_subtasks or return_subtasks_mode"
            )
        if self.exclude_negative_subtasks and effective_subtasks_mode != "strict":
            raise ValueError(
                "exclude_negative_subtasks requires return_subtasks_mode=strict"
            )

        prompt_source = "subtask" if self.prompt_from_subtask else "prompt"
        repack_transform = Group(
            inputs=[
                RepackTransform(
                    structure={
                        "observation/head_image": "observation.images.cam_head",
                        "observation/left_wrist_image": "observation.images.cam_left_wrist",
                        "observation/right_wrist_image": "observation.images.cam_right_wrist",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": prompt_source,
                        "loss_mask_index": "loss_mask_index",
                        "action_valid_mask": "action_valid_mask",
                        "state_valid_mask": "state_valid_mask",
                        "episode_index": "episode_index",
                    }
                )
            ]
        )

        data_transforms = Group(
            inputs=[
                HumanoidS0S1InputsTransform(
                    action_dim=current_action_dim,
                    has_velocities=self.has_velocities,
                    map_to_closed_kinematic=self.map_to_closed_kinematic,
                    input_kinematic_space=self.input_kinematic_space,
                    model_kinematic_space=self.model_kinematic_space,
                )
            ],
            outputs=[SimpleOutputsTransform(action_dim=S0S1_OUTPUT_DIM)],
        )
        if self.smooth_actions:
            data_transforms = data_transforms.push(inputs=[SmoothActions()])
        data_transforms = data_transforms.push(
            outputs=[
                InterpolateActions(
                    sample_step=1.0 / self.action_sample_step,
                    actions_type="absolute",
                )
            ]
        )
        delta_action_mask = make_bool_mask(25, -9, 16)
        data_transforms = data_transforms.push(
            inputs=[DeltaActions(delta_action_mask)],
            outputs=[AbsoluteActions(delta_action_mask)],
        )

        base_data_config = self.create_base_config(assets_dirs)
        return dataclasses.replace(
            base_data_config,
            asset_id=self.asset_id,
            repo_id=self.repo_id,
            root_dir=self.root_dir,
            episodes_list_file=self.episodes_list_file,
            sample_weights_file=self.sample_weights_file,
            return_subtasks=self.return_subtasks,
            return_subtasks_mode=self.return_subtasks_mode,
            exclude_negative_subtasks=self.exclude_negative_subtasks,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=ModelTransformFactory()(model_config, self),
            action_sequence_keys=self.action_sequence_keys,
            control_mode=self.control_mode,
            action_space_factorization=self.action_space_factorization,
            state_dim=S0S1_OUTPUT_DIM,
            action_horizon=self.action_horizon,
            action_offset=self.action_offset,
            action_sample_step=self.action_sample_step,
            validation_episodes=self.validation_episodes,
        )
