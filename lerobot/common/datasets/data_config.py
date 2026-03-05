import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Protocol, Union, List

import etils.epath as epath
from sympy import false
from typing_extensions import override
import tyro

# Import NormStats from the new torch_transforms.py
from .torch_transforms import (
    Group as TorchGroup,
    InjectDefaultPromptTorch,
    ResizeImagesTorch,
    TokenizeGreenVLAInputsTransform,
    ExtractGreenVLAActionsTorch,
)
from lerobot.common.policies.greenvla_policy.greenvla_tokenizer import GreenVLATokenizer
import lerobot.common.utils.normalize as _normalize


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location.
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    root_dir: str | None = None
    episodes_list_file: str | None = None
    return_subtasks: bool = False
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
    repack_transforms: TorchGroup = dataclasses.field(default_factory=TorchGroup)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: TorchGroup = dataclasses.field(default_factory=TorchGroup)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: TorchGroup = dataclasses.field(default_factory=TorchGroup)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # If true, will disable syncing the dataset from the Hugging Face Hub. Allows training on local-only datasets.
    local_files_only: bool = False

    mixture_configs: Sequence["DataConfig"] | None = None
    mixture_weights: Sequence[float] | None = None
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
    def __call__(self, model_config: Any) -> TorchGroup:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for policy."""

    default_prompt: str | None = None

    # If provided, will determine the default prompt that be used by the model.

    def __call__(self, model_config, data_config) -> TorchGroup:
        match model_config.model_type:
                
            case "greenvlapolicy":
                return TorchGroup(
                    inputs=[
                        InjectDefaultPromptTorch(self.default_prompt),
                        ResizeImagesTorch(*model_config.image_shape),
                        TokenizeGreenVLAInputsTransform(
                            GreenVLATokenizer(max_len=model_config.tokenizer_max_length,
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
                                            ),
                        ),
                    ],
                    outputs=[
                        ExtractGreenVLAActionsTorch(
                            GreenVLATokenizer(max_len=model_config.tokenizer_max_length,
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
                                           
                                            ),
                            inference_mode=model_config.inference_mode,
                            action_horizon=data_config.action_horizon,
                            action_dim=model_config.max_action_dim,
                            model_mode=model_config.model_mode,
                        )
                    ],
                )


            case _:
                raise ValueError(f"Model type {model_config.model_type} not supported")


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    name: str = ""
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: Any) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(
                epath.Path(self.assets.assets_dir or assets_dirs), asset_id
            ),
        )

    def _load_norm_stats(
        self, assets_dir: epath.Path, asset_id: str | None
    ) -> dict[str, _normalize.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(data_assets_dir)
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: Any) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(
        default_factory=GroupFactory
    )
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(
        default_factory=ModelTransformFactory
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: Any) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
            use_quantile_norm=getattr(model_config, "model_type", None)
            == "greenvlapolicy",  # Example adaptation
        )

from omegaconf import DictConfig, OmegaConf
from typing import Union

@dataclasses.dataclass(frozen=True)
class MixtureDataConfigFactory(DataConfigFactory):
    """Factory for creating a data config that samples from multiple datasets with weighted probabilities."""

    repo_id: str = "mixture"

    data_configs: Sequence[Union[DataConfigFactory, DictConfig]] = ()
    weights: Sequence[float] | None = None

    # data_configs: dict[str, DataConfigFactory] | None = None

    primary_dataset_id: str | None = None

    @property
    def mixture_asset_id(self) -> str:
        """Generate a consistent asset ID for this mixture dataset."""
        if self.assets.asset_id:
            return self.assets.asset_id

        import hashlib

        repo_ids = [
            str(config.repo_id)
            for config in self.data_configs
            if hasattr(config, "repo_id") and config.repo_id is not tyro.MISSING
        ]

        weights_str = (
            ",".join(str(w) for w in self.weights) if self.weights else ""
        )
        mixture_str = f"{'-'.join(repo_ids)}_{weights_str}"

        hash_obj = hashlib.md5(mixture_str.encode())
        hash_str = hash_obj.hexdigest()[:8]  # Use first 8 chars of hash

        return f"mixture_{hash_str}"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: Any) -> DataConfig:
        """Create a data config for mixing multiple datasets."""
        from hydra.utils import instantiate

        # 1) Ensure each entry in self.data_configs is a DataConfigFactory:
        factories: list[DataConfigFactory] = []
        for maybe_cfg in self.data_configs:
            if isinstance(maybe_cfg, DataConfigFactory):
                factories.append(maybe_cfg)
            elif isinstance(maybe_cfg, DictConfig):
                factories.append(instantiate(maybe_cfg, _recursive_=True))
            else:
                # Could be a DictConfig or raw dict -> instantiate exactly one level deep
                factories.append(instantiate(maybe_cfg, _recursive_=False))

        # 2) Sanity checks
        if not factories:
            raise ValueError("At least one data config must be provided.")
        if self.weights is not None and len(self.weights) != len(factories):
            raise ValueError(
                f"Number of weights ({len(self.weights)}) must match number of data configs ({len(factories)})."
            )

        # 3) Build each sub‑DataConfig by calling its factory:
        configs = [factory.create(assets_dirs, model_config) for factory in factories]


        # 5) Assemble and return the final mixture DataConfig:
        asset_id = self.assets.asset_id or self.mixture_asset_id
        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            asset_id=asset_id,
            mixture_configs=configs,
            mixture_weights=self.weights,
        )

    @override
    def _load_norm_stats(
        self, assets_dir: epath.Path, asset_id: str | None
    ) -> dict[str, _normalize.NormStats] | None:
        """Load normalization statistics from the primary dataset."""
        # if self.primary_dataset_id not in [
        #     cfg.assets.asset_id or cfg.repo_id for cfg in self.data_configs
        # ]:
        #     logging.warning(
        #         f"Primary dataset ID {self.primary_dataset_id} not found in data configs. Using first dataset."
        #     )
        #     primary_dataset_id = (
        #         self.data_configs[0].assets.asset_id or self.data_configs[0].repo_id
        #     )
        # else:
        #     primary_dataset_id = self.primary_dataset_id

        # try:
        #     data_assets_dir = str(assets_dir / primary_dataset_id)
        #     norm_stats = _normalize.load(data_assets_dir)
        #     logging.info(f"Loaded norm stats from {data_assets_dir}")
        #     return norm_stats
        # except FileNotFoundError:
        #     logging.warning(
        #         f"Norm stats for {primary_dataset_id} not found at {data_assets_dir}"
        #     )
        return None


from lerobot.common.datasets.torch_transforms import (
    RepackTransform,
    DeltaActions,
    AbsoluteActions,
    SmoothActions,
    InterpolateActions,
    make_bool_mask,
    Group as TorchGroup,  # Ensure Group is imported as TorchGroup if that's the alias used
)


@dataclasses.dataclass(frozen=True)
class LeRobotBridgeDataConfig(DataConfigFactory):
    asset_id: str | None = None
    repo_id: str | None = None
    root_dir: str | None = None
    episodes_list_file: str | None = None
    action_sequence_keys: Sequence[str] = ("action",)
    action_horizon: int = 16  # Bridge was collected with ~5 hz, but we have no choise
    action_offset: int = 0
    action_sample_step: int = 1
    state_dim: int = 7
    control_mode: str = "cartesian"
    action_space_factorization: str = "Num hands: 1. Control mode: cartesian. Hand type: 2 fingers. Torso: False. Head: False."
    map_to_unified_space: bool = True
    validation_episodes: str | None = None 
    
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: Any) -> DataConfig:
        from lerobot.common.datasets.data_transforms.robots.bridge import (
            BridgeInputsTransform,
            BridgeOutputsTransform,
        )

        current_action_dim = (
            model_config.max_action_dim
            if hasattr(model_config, "max_action_dim")
            and model_config.max_action_dim is not None
            else 7
        )
        

        map_to_unified_space = (model_config.map_to_unified_space
                                if hasattr(model_config, "map_to_unified_space")
                                and model_config.map_to_unified_space is not None
                                else self.map_to_unified_space)
        
        repack_structure_corrected = {
            "observation/image": "observation.images.image_0",
            "observation/state": "observation.state",
            "actions": "action",
            "prompt": "prompt",
        }
        repack_transform = TorchGroup(
            inputs=[RepackTransform(structure=repack_structure_corrected)]
        )
        data_transforms = TorchGroup(
            inputs=[BridgeInputsTransform(action_dim=current_action_dim, map_to_unified_space=map_to_unified_space)],
            outputs=[BridgeOutputsTransform()],
        )
        data_transforms = data_transforms.push(
            outputs=[InterpolateActions(sample_step=1. / self.action_sample_step, actions_type="delta")],
        )
        model_transforms_factory = ModelTransformFactory()
        model_transforms = model_transforms_factory(model_config, self)

        base_data_config = self.create_base_config(assets_dirs)

        # Create the final DataConfig
        final_config = dataclasses.replace(
            base_data_config,
            asset_id=self.asset_id,
            repo_id=self.repo_id,
            root_dir=self.root_dir,
            episodes_list_file=self.episodes_list_file,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,  # from PyTorchModelTransformFactory
            action_sequence_keys=self.action_sequence_keys,
            action_space_factorization=self.action_space_factorization,
            control_mode=self.control_mode,
            state_dim=self.state_dim,
            action_horizon=self.action_horizon,
            action_offset=self.action_offset,
            action_sample_step=self.action_sample_step,
            validation_episodes=self.validation_episodes, 
            # norm_stats are loaded by create_base_config
        )
        return final_config


@dataclasses.dataclass(frozen=True)
class LeRobotFractalDataConfig(DataConfigFactory):
    asset_id: str | None = None
    repo_id: str | None = None
    root_dir: str | None = None
    episodes_list_file: str | None = None
    action_sequence_keys: Sequence[str] = ("action",)
    action_horizon: int = 16  # fractal was collected with ~5hz, but we have no choise
    action_offset: int = 0
    action_sample_step: int = 1
    state_dim: int = 8
    control_mode: str = "cartesian"
    action_space_factorization: str = "Num hands: 1. Control mode: cartesian. Hand type: 2 fingers. Torso: False. Head: False."
    map_to_unified_space: bool = True
    validation_episodes: str | None = None 

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: Any) -> DataConfig:
        from lerobot.common.datasets.data_transforms.robots.fractal import (
            FractalInputsTransform,
            FractalOutputsTransform,
        )

        current_action_dim = (
            model_config.max_action_dim
            if hasattr(model_config, "max_action_dim")
            and model_config.max_action_dim is not None
            else 7
        )

        map_to_unified_space = (model_config.map_to_unified_space
                                if hasattr(model_config, "map_to_unified_space")
                                and model_config.map_to_unified_space is not None
                                else self.map_to_unified_space)
        
        repack_structure_corrected = {
            "observation/image": "observation.images.image",
            "observation/state": "observation.state",
            "actions": "action",
            "prompt": "prompt",
        }

        repack_transform = TorchGroup(
            inputs=[RepackTransform(structure=repack_structure_corrected)]
        )

        data_transforms = TorchGroup(
            inputs=[FractalInputsTransform(action_dim=current_action_dim, map_to_unified_space=map_to_unified_space)],
            outputs=[FractalOutputsTransform()],
        )
        data_transforms = data_transforms.push(
            outputs=[InterpolateActions(sample_step=1. / self.action_sample_step, actions_type="delta")],
        )

        model_transforms_factory = ModelTransformFactory()
        model_transforms = model_transforms_factory(model_config, self)

        base_data_config = self.create_base_config(assets_dirs)

        # Create the final DataConfig
        # Create the final DataConfig
        final_config = dataclasses.replace(
            base_data_config,
            asset_id=self.asset_id,
            repo_id=self.repo_id,
            root_dir=self.root_dir,
            episodes_list_file=self.episodes_list_file,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,  # from PyTorchModelTransformFactory
            action_sequence_keys=self.action_sequence_keys,
            action_space_factorization=self.action_space_factorization,
            control_mode=self.control_mode,
            state_dim=self.state_dim,
            action_horizon=self.action_horizon,
            action_offset=self.action_offset,
            action_sample_step=self.action_sample_step,
            validation_episodes=self.validation_episodes, 
            # norm_stats are loaded by create_base_config
        )
        return final_config


@dataclasses.dataclass(frozen=True)
class LeRobotDroidDataConfigOld(DataConfigFactory):
    asset_id: str | None = None
    repo_id: str | None = None
    root_dir: str | None = None
    episodes_list_file: str | None = None
    action_sequence_keys: Sequence[str] = ("action",)
    action_horizon: int = 16  # for 15 hz it is okay
    action_offset: int = 0
    action_sample_step: int = 1
    state_dim: int = 7
    control_mode: str = "joint"
    action_space_factorization: str = "Num hands: 1. Control mode: joint. Hand type: 2 fingers. Torso: False. Head: False."
    map_to_unified_space: bool = True
    validation_episodes: str | None = None 
    
    # def __init__(self, *args, **kwargs):
    #     raise DeprecationWarning("Use LeRobotDroidDataConfig instead")

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: Any) -> DataConfig:
        from lerobot.common.datasets.data_transforms.robots.droid import (
            DroidInputsTransformOld,
            DroidOutputsTransformOld,
        )

        current_action_dim = (
            model_config.max_action_dim
            if hasattr(model_config, "max_action_dim")
            and model_config.max_action_dim is not None
            else 7
        )

                
        map_to_unified_space = (model_config.map_to_unified_space
                                if hasattr(model_config, "map_to_unified_space")
                                and model_config.map_to_unified_space is not None
                                else self.map_to_unified_space)
        
        repack_structure_corrected = {
            "observation/exterior_image_1": "observation.images.exterior_image_1_left",
            "observation/exterior_image_2": "observation.images.exterior_image_2_left",
            "observation/wrist_image": "observation.images.wrist_image_left",
            "observation/state": "observation.state",
            "actions": "action",
            "prompt": "prompt",
        }

        repack_transform = TorchGroup(
            inputs=[RepackTransform(structure=repack_structure_corrected)]
        )
        data_transforms = TorchGroup(
            inputs=[DroidInputsTransformOld(action_dim=current_action_dim, map_to_unified_space=map_to_unified_space)],
            outputs=[DroidOutputsTransformOld()],
        )
        data_transforms = data_transforms.push(
            outputs=[InterpolateActions(sample_step=1. / self.action_sample_step, actions_type="delta")],
        )
        model_transforms_factory = ModelTransformFactory()
        model_transforms = model_transforms_factory(model_config, self)

        base_data_config = self.create_base_config(assets_dirs)

        # Create the final DataConfig
        final_config = dataclasses.replace(
            base_data_config,
            asset_id=self.asset_id,
            repo_id=self.repo_id,
            root_dir=self.root_dir,
            episodes_list_file=self.episodes_list_file,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,  # from PyTorchModelTransformFactory
            action_sequence_keys=self.action_sequence_keys,
            action_space_factorization=self.action_space_factorization,
            control_mode=self.control_mode,
            state_dim=self.state_dim,
            action_horizon=self.action_horizon,
            action_offset=self.action_offset,
            action_sample_step=self.action_sample_step,
            validation_episodes=self.validation_episodes, 
            # norm_stats are loaded by create_base_config
        )
        return final_config
    
    
@dataclasses.dataclass(frozen=True)
class LeRobotDroidDataConfig(DataConfigFactory):
    asset_id: str | None = None
    repo_id: str | None = None
    root_dir: str | None = None
    action_sequence_keys: Sequence[str] = ("action",)
    action_horizon: int = 16  # for 15 hz it is okay
    action_offset: int = 0
    action_sample_step: int = 1
    state_dim: int = 7
    control_mode: str = "cartesian"
    action_space_factorization: str = "Num hands: 1. Control mode: cartesian (joints). Hand type: 2 fingers. Torso: False. Head: False."
    map_to_unified_space: bool = True
    validation_episodes: str | None = None 
    

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: Any) -> DataConfig:
        from lerobot.common.datasets.data_transforms.robots.droid import (
            DroidInputsTransform,
            DroidOutputsTransform,
        )

        current_action_dim = (
            model_config.max_action_dim
            if hasattr(model_config, "max_action_dim")
            and model_config.max_action_dim is not None
            else (7 if self.control_mode == 'cartesian' else 8)
        )

        repack_structure_corrected = {
            "observation/exterior_image_1": "observation.images.exterior_1",
            "observation/exterior_image_2": "observation.images.exterior_2",
            "observation/wrist_image": "observation.images.wrist",
            "observation/state": "observation.state",
            "actions": "action",
            "prompt": "prompt",
        }

        repack_transform = TorchGroup(
            inputs=[RepackTransform(structure=repack_structure_corrected)]
        )
        data_transforms = TorchGroup(
            inputs=[DroidInputsTransform(action_dim=current_action_dim, control_type=self.control_mode)],
            outputs=[DroidOutputsTransform(control_type=self.control_mode)],
        )
        
        delta_action_mask_torch = make_bool_mask(
            6, 1, 6, 1
        ) #left joints, left gripper, right joints, right gripper
        
        data_transforms = data_transforms.push(
            outputs=[InterpolateActions(sample_step=1. / self.action_sample_step, actions_type="absolute")],
        )
        data_transforms = data_transforms.push(
            inputs=[DeltaActions(delta_action_mask_torch)],
            outputs=[AbsoluteActions(delta_action_mask_torch)],
        )
        
        model_transforms_factory = ModelTransformFactory()
        model_transforms = model_transforms_factory(model_config, self)

        base_data_config = self.create_base_config(assets_dirs)

        # Create the final DataConfig
        final_config = dataclasses.replace(
            base_data_config,
            asset_id=self.asset_id,
            repo_id=self.repo_id,
            root_dir=self.root_dir,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,  # from PyTorchModelTransformFactory
            action_sequence_keys=self.action_sequence_keys,
            control_mode=self.control_mode,
            state_dim=self.state_dim,
            action_horizon=self.action_horizon,
            action_offset=self.action_offset,
            action_sample_step=self.action_sample_step,
            action_space_factorization=self.action_space_factorization,
            map_to_unified_space=self.map_to_unified_space,
            validation_episodes=self.validation_episodes,
            # norm_stats are loaded by create_base_config
        )
        return final_config


@dataclasses.dataclass(frozen=True)
class LeRobotRobosetDataConfig(DataConfigFactory):
    asset_id: str | None = None
    repo_id: str | None = None
    root_dir: str | None = None
    episodes_list_file: str | None = None

    action_sequence_keys: Sequence[str] = ("action",)
    action_horizon: int = 16  # for 15 hz it is okay
    action_offset: int = 0
    action_sample_step: int = 1
    state_dim: int = 8
    control_mode: str = "joint"
    action_space_factorization: str = "Num hands: 1. Control mode: joints. Hand type: 2 fingers. Torso: False. Head: False."
    map_to_unified_space: bool = True
    validation_episodes: str | None = None 
    
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: Any) -> DataConfig:
        from lerobot.common.datasets.data_transforms.robots.roboset import (
            RobosetInputsTransform,
            RobosetOutputsTransform,
        )

        current_action_dim = (
            model_config.max_action_dim
            if hasattr(model_config, "max_action_dim")
            and model_config.max_action_dim is not None
            else 8
        )

        map_to_unified_space = (model_config.map_to_unified_space
                                if hasattr(model_config, "map_to_unified_space")
                                and model_config.map_to_unified_space is not None
                                else self.map_to_unified_space)

        repack_structure_corrected = {
            "observation/cam_left": "observation.images.cam_left",
            "observation/cam_right": "observation.images.cam_right",
            "observation/cam_top": "observation.images.cam_top",
            "observation/cam_wrist": "observation.images.cam_wrist",
            "observation/state": "observation.state",
            "actions": "action",
            "prompt": "prompt",
        }

        repack_transform = TorchGroup(
            inputs=[RepackTransform(structure=repack_structure_corrected)]
        )
        data_transforms = TorchGroup(
            inputs=[RobosetInputsTransform(action_dim=current_action_dim, map_to_unified_space=map_to_unified_space)],
            outputs=[RobosetOutputsTransform()],
        )
        delta_action_mask_torch = make_bool_mask(
           7,1 
        )  # Actions were sampled from states
        
        data_transforms = data_transforms.push(
            outputs=[InterpolateActions(sample_step=1. / self.action_sample_step, actions_type="absolute")],
        )
        data_transforms = data_transforms.push(
            inputs=[DeltaActions(delta_action_mask_torch)],
            outputs=[AbsoluteActions(delta_action_mask_torch)],
        )

        model_transforms_factory = ModelTransformFactory()
        model_transforms = model_transforms_factory(model_config, self)

        base_data_config = self.create_base_config(assets_dirs)

        # Create the final DataConfig
        final_config = dataclasses.replace(
            base_data_config,
            asset_id=self.asset_id,
            repo_id=self.repo_id,
            root_dir=self.root_dir,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,  # from PyTorchModelTransformFactory
            action_sequence_keys=self.action_sequence_keys,
            action_space_factorization=self.action_space_factorization,
            control_mode=self.control_mode,
            state_dim=self.state_dim,
            action_horizon=self.action_horizon,
            action_offset=self.action_offset,
            action_sample_step=self.action_sample_step,
            validation_episodes=self.validation_episodes, 
            # norm_stats are loaded by create_base_config
        )
        return final_config

@dataclasses.dataclass(frozen=True)
class LeRobotCobotMagicLegacyDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline
    for CobotMagic datasets, using PyTorch-native transforms.
    This dataset uses non intuitive naming for cameras
    """

    asset_id: str | None = None
    repo_id: str | None = None
    root_dir: str | None = None
    episodes_list_file: str | None = None
    action_sequence_keys: Sequence[str] = ("action",)
    action_horizon: int = 16  # for 15 hz it is okay
    action_offset: int = 0
    action_sample_step: int = 1
    state_dim: int = 14
    control_mode: str = "joint"
    action_space_factorization: str = "Num hands: 2. Control mode: joints. Num joints per hand: 6. Hand type: 2 fingers. Torso: False. Head: False."
    map_to_unified_space: bool = True
    validation_episodes: str | None = None 

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: Any
    ) -> DataConfig:  # model_config is more like cfg.policy
        from lerobot.common.datasets.data_transforms.robots.cobot_magic import (
            CobotMagicInputsTransform,
            CobotMagicOutputsTransform,
        )

        current_action_dim = (
            model_config.max_action_dim
            if hasattr(model_config, "max_action_dim")
            and model_config.max_action_dim is not None
            else 14
        )
        
                
        map_to_unified_space = (model_config.map_to_unified_space
                                if hasattr(model_config, "map_to_unified_space")
                                and model_config.map_to_unified_space is not None
                                else self.map_to_unified_space)

        repack_structure_corrected = {
            "observation/base_image": "observation.images.camera_2",
            "observation/left_wrist_image": "observation.images.camera_1",
            "observation/right_wrist_image": "observation.images.camera_0",
            "observation/state": "observation.state",
            "actions": "action",  # Map dataset 'action' (singular) to 'actions' (plural) for the transform
            "prompt": "prompt",
        }

        repack_transform = TorchGroup(
            inputs=[RepackTransform(structure=repack_structure_corrected)]
        )

        # Data transforms specific to CobotMagic, using PyTorch versions
        data_transforms = TorchGroup(
            inputs=[CobotMagicInputsTransform(action_dim=current_action_dim, map_to_unified_space=map_to_unified_space)],
            outputs=[CobotMagicOutputsTransform()],
        )

        delta_action_mask_torch = make_bool_mask(
            6, 1, 6, 1
        )  # left joints, left gripper, right joints, right gripper
        
        data_transforms = data_transforms.push(
            outputs=[InterpolateActions(sample_step=1. / self.action_sample_step, actions_type="absolute")],
        )
        data_transforms = data_transforms.push(
            inputs=[DeltaActions(delta_action_mask_torch)],
            outputs=[AbsoluteActions(delta_action_mask_torch)],
        )


        model_transforms_factory = ModelTransformFactory()
        model_transforms = model_transforms_factory(model_config, self)

        base_data_config = self.create_base_config(assets_dirs)

        # Create the final DataConfig
        final_config = dataclasses.replace(
            base_data_config,
            asset_id=self.asset_id,
            repo_id=self.repo_id,
            root_dir=self.root_dir,
            episodes_list_file=self.episodes_list_file,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,  # from PyTorchModelTransformFactory
            action_sequence_keys=self.action_sequence_keys,
            control_mode=self.control_mode,
            action_space_factorization=self.action_space_factorization,
            state_dim=self.state_dim,
            action_horizon=self.action_horizon,
            action_offset=self.action_offset,
            action_sample_step=self.action_sample_step,
            validation_episodes=self.validation_episodes, 
            # norm_stats are loaded by create_base_config
        )
        return final_config






@dataclasses.dataclass(frozen=True)
class LeRobotRDTDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline
    for CobotMagic datasets, using PyTorch-native transforms.
    """

    asset_id: str | None = None
    repo_id: str | None = None
    root_dir: str | None = None
    episodes_list_file: str | None = None
    action_sequence_keys: Sequence[str] = ("action",)
    action_horizon: int = 16  # for 15 hz it is okay
    action_offset: int = 0
    action_sample_step: int = 1
    state_dim: int = 14
    control_mode: str = "joint"
    action_space_factorization: str = "Num hands: 2. Control mode: joints. Num joints per hand: 6. Hand type: 2 fingers. Torso: False. Head: False."
    map_to_unified_space: bool = True
    validation_episodes: str | None = None 
    
    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: Any
    ) -> DataConfig:  # model_config is more like cfg.policy
        from lerobot.common.datasets.data_transforms.robots.cobot_magic import (
            CobotMagicInputsTransform,
            CobotMagicOutputsTransform,
        )

        current_action_dim = (
            model_config.max_action_dim
            if hasattr(model_config, "max_action_dim")
            and model_config.max_action_dim is not None
            else 14
        )

                
        map_to_unified_space = (model_config.map_to_unified_space
                                if hasattr(model_config, "map_to_unified_space")
                                and model_config.map_to_unified_space is not None
                                else self.map_to_unified_space)

        repack_structure_corrected = {
            "observation/base_image": "observation.images.cam_high",
            "observation/left_wrist_image": "observation.images.cam_left_wrist",
            "observation/right_wrist_image": "observation.images.cam_right_wrist",
            "observation/state": "observation.state",
            "actions": "action",  # Map dataset 'action' (singular) to 'actions' (plural) for the transform
            "prompt": "prompt",
        }

        repack_transform = TorchGroup(
            inputs=[RepackTransform(structure=repack_structure_corrected)]
        )

        # Data transforms specific to CobotMagic, using PyTorch versions
        data_transforms = TorchGroup(
            inputs=[CobotMagicInputsTransform(action_dim=current_action_dim, map_to_unified_space=map_to_unified_space)],
            outputs=[CobotMagicOutputsTransform()],
        )

        delta_action_mask_torch = make_bool_mask(
            6, 1, 6, 1
        )  # left joints, left gripper, right joints, right gripper
        
        data_transforms = data_transforms.push(
            outputs=[InterpolateActions(sample_step=1. / self.action_sample_step, actions_type="absolute")],
        )
        data_transforms = data_transforms.push(
            inputs=[DeltaActions(delta_action_mask_torch)],
            outputs=[AbsoluteActions(delta_action_mask_torch)],
        )


        model_transforms_factory = ModelTransformFactory()
        model_transforms = model_transforms_factory(model_config, self)

        base_data_config = self.create_base_config(assets_dirs)

        # Create the final DataConfig
        final_config = dataclasses.replace(
            base_data_config,
            asset_id=self.asset_id,
            repo_id=self.repo_id,
            root_dir=self.root_dir,
            episodes_list_file=self.episodes_list_file,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,  # from PyTorchModelTransformFactory
            action_sequence_keys=self.action_sequence_keys,
            action_space_factorization=self.action_space_factorization,
            control_mode=self.control_mode,
            state_dim=self.state_dim,
            action_horizon=self.action_horizon,
            action_offset=self.action_offset,
            action_sample_step=self.action_sample_step,
            validation_episodes=self.validation_episodes, 
            # norm_stats are loaded by create_base_config
        )
        return final_config



@dataclasses.dataclass(frozen=True)
class LeRobotGalaxeaR1LiteDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline
    for CobotMagic datasets, using PyTorch-native transforms.
    """

    asset_id: str | None = None
    repo_id: str | None = None
    root_dir: str | None = None
    episodes_list_file: str | None = None
    action_sequence_keys: Sequence[str] = ("action",)
    action_horizon: int = 16  # for 15 hz it is okay
    action_offset: int = 0
    action_sample_step: int = 1
    state_dim: int = 20
    control_mode: str = "joint"
    action_space_factorization: str = "Num hands: 2. Control mode: joints. Num joints per hand: 6. Hand type: 2 fingers. Torso: True. Head: True."
    map_to_unified_space: bool = True
    validation_episodes: str | None = None 

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: Any
    ) -> DataConfig:  # model_config is more like cfg.policy
        from lerobot.common.datasets.data_transforms.robots.galaxea_r1_lite import (
            GalaxeaR1LiteInputsTransform,
            GalaxeaR1LiteOutputsTransform,
        )

        current_action_dim = (
            model_config.max_action_dim
            if hasattr(model_config, "max_action_dim")
            and model_config.max_action_dim is not None
            else 20
        )

                
        map_to_unified_space = (model_config.map_to_unified_space
                                if hasattr(model_config, "map_to_unified_space")
                                and model_config.map_to_unified_space is not None
                                else self.map_to_unified_space)
        
        repack_structure_corrected = {
            "observation/head_image": "observation.images.head",
            "observation/left_wrist_image": "observation.images.left_wrist",
            "observation/right_wrist_image": "observation.images.right_wrist",
            "observation/state": "observation.state",
            "actions": "action",  # Map dataset 'action' (singular) to 'actions' (plural) for the transform
            "prompt": "prompt",
        }

        repack_transform = TorchGroup(
            inputs=[RepackTransform(structure=repack_structure_corrected)]
        )

        # Data transforms specific to CobotMagic, using PyTorch versions
        data_transforms = TorchGroup(
            inputs=[GalaxeaR1LiteInputsTransform(action_dim=current_action_dim, map_to_unified_space=map_to_unified_space)],
            outputs=[GalaxeaR1LiteOutputsTransform()],
        )

        delta_action_mask_torch = make_bool_mask(
            6, 1, 6, 1, 3
        )  # left arm; left gripper; right arm; right gripper; torso
        
        data_transforms = data_transforms.push(
            outputs=[InterpolateActions(sample_step=1. / self.action_sample_step, actions_type="absolute")],
        )
        data_transforms = data_transforms.push(
            inputs=[DeltaActions(delta_action_mask_torch)],
            outputs=[AbsoluteActions(delta_action_mask_torch)],
        )

        model_transforms_factory = ModelTransformFactory()
        model_transforms = model_transforms_factory(model_config, self)

        base_data_config = self.create_base_config(assets_dirs)

        # Create the final DataConfig
        final_config = dataclasses.replace(
            base_data_config,
            asset_id=self.asset_id,
            repo_id=self.repo_id,
            root_dir=self.root_dir,
            episodes_list_file=self.episodes_list_file, 
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,  # from PyTorchModelTransformFactory
            action_sequence_keys=self.action_sequence_keys,
            control_mode=self.control_mode,
            action_space_factorization=self.action_space_factorization,
            state_dim=self.state_dim,
            action_horizon=self.action_horizon,
            action_offset=self.action_offset,
            action_sample_step=self.action_sample_step,
            validation_episodes=self.validation_episodes, 
            # norm_stats are loaded by create_base_config
        )
        return final_config

@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    asset_id: str | None = None
    repo_id: str | None = None
    root_dir: str | None = None
    episodes_list_file: str | None = None
    action_sequence_keys: Sequence[str] = ("action",)
    action_horizon: int = 16  # data collected with 15 hz
    action_offset: int = 0
    action_sample_step: int = 1
    state_dim: int = 14
    adapt_to_pi: bool = True
    control_mode: str = "joint"
    action_space_factorization: str = "Num hands: 2. Control mode: joints. Num joints per hand: 6. Hand type: 2 fingers. Torso: False. Head: False."
    map_to_unified_space: bool = True
    validation_episodes: str | None = None 

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: Any
    ) -> DataConfig:  # model_config is more like cfg.policy
        from lerobot.common.datasets.data_transforms.robots.aloha import (
            AlohaInputsTransform,
            AlohaOutputsTransform,
        )

        current_action_dim = (
            model_config.max_action_dim
            if hasattr(model_config, "max_action_dim")
            and model_config.max_action_dim is not None
            else 14
        )

                
        map_to_unified_space = (model_config.map_to_unified_space
                                if hasattr(model_config, "map_to_unified_space")
                                and model_config.map_to_unified_space is not None
                                else self.map_to_unified_space)
        
        repack_structure_corrected = {
            "observation/base_image": "observation.images.cam_low",
            "observation/left_wrist_image": "observation.images.cam_left_wrist",
            "observation/right_wrist_image": "observation.images.cam_right_wrist",
            "observation/state": "observation.state",
            "actions": "action",
            "prompt": "prompt",
        }
        repack_transform = TorchGroup(
            inputs=[RepackTransform(structure=repack_structure_corrected)]
        )

        # Data transforms specific to CobotMagic, using PyTorch versions
        data_transforms = TorchGroup(
            inputs=[
                AlohaInputsTransform(
                    action_dim=current_action_dim, adapt_to_pi=self.adapt_to_pi, map_to_unified_space=map_to_unified_space
                )
            ],
            outputs=[AlohaOutputsTransform(adapt_to_pi=self.adapt_to_pi)],
        )

        delta_action_mask_torch = make_bool_mask(
            6, 1, 6, 1
        )  # left joints, left gripper, right joints, right gripper
        
        data_transforms = data_transforms.push(
            outputs=[InterpolateActions(sample_step=1. / self.action_sample_step, actions_type="absolute")],
        )
        data_transforms = data_transforms.push(
            inputs=[DeltaActions(delta_action_mask_torch)],
            outputs=[AbsoluteActions(delta_action_mask_torch)],
        )

        model_transforms_factory = ModelTransformFactory()
        model_transforms = model_transforms_factory(model_config, self)

        base_data_config = self.create_base_config(assets_dirs)

        # Create the final DataConfig
        final_config = dataclasses.replace(
            base_data_config,
            asset_id=self.asset_id,
            repo_id=self.repo_id,
            root_dir=self.root_dir,
            episodes_list_file=self.episodes_list_file, 
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,  # from PyTorchModelTransformFactory
            action_sequence_keys=self.action_sequence_keys,
            control_mode=self.control_mode,
            action_space_factorization=self.action_space_factorization,
            state_dim=self.state_dim,
            action_horizon=self.action_horizon,
            action_offset=self.action_offset,
            action_sample_step=self.action_sample_step,
            validation_episodes=self.validation_episodes, 
            # norm_stats are loaded by create_base_config
        )
        return final_config


@dataclasses.dataclass(frozen=True)
class LeRobotAgibotTwoFingerDataConfig(DataConfigFactory):

    asset_id: str | None = None
    repo_id: str | None = None
    root_dir: str | None = None
    episodes_list_file: str | None = None
    action_sequence_keys: Sequence[str] = ("action",)
    action_horizon: int = 16
    action_offset: int = 1
    action_sample_step: int = (
        5  # agibot was collected to slow, so we need to sample it with 5x
    )
    state_dim: int = 20
    control_mode: str = "joint"
    action_space_factorization: str = "Num hands: 2. Control mode: joints. Num joints per hand: 7. Hand type: 2 fingers. Torso: True. Head: True."
    map_to_unified_space: bool = True
    map_to_humanoid: bool = True
    validation_episodes: str | None = None 
    return_subtasks: bool = False
    
    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: Any
    ) -> DataConfig:  # model_config is more like cfg.policy
        from lerobot.common.datasets.data_transforms.robots.agibot_twofinger import (
            AgibotTwoFingerInputsTransform,
            AgibotTwoFingerOutputsTransform,
        )

        current_action_dim = (
            model_config.max_action_dim
            if hasattr(model_config, "max_action_dim")
            and model_config.max_action_dim is not None
            else 14
        )

        repack_structure_corrected = {
            "observation/head_image": "observation.images.head",
            "observation/left_wrist_image": "observation.images.hand_left",
            "observation/right_wrist_image": "observation.images.hand_right",
            "observation/state": "observation.state",
            "actions": "action",
            "prompt": "prompt",
            "subtask": "subtask",
            "next_subtask": "next_subtask",
            "is_subtask_transition": "is_subtask_transition",
        }

                
        map_to_unified_space = (model_config.map_to_unified_space
                                if hasattr(model_config, "map_to_unified_space")
                                and model_config.map_to_unified_space is not None
                                else self.map_to_unified_space)
        repack_transform = TorchGroup(
            inputs=[RepackTransform(structure=repack_structure_corrected)]
        )

        # Data transforms specific to CobotMagic, using PyTorch versions
        data_transforms = TorchGroup(
            inputs=[AgibotTwoFingerInputsTransform(action_dim=current_action_dim, 
                                                   map_to_unified_space=map_to_unified_space,
                                                   map_to_humanoid=self.map_to_humanoid)],
            outputs=[AgibotTwoFingerOutputsTransform()],
        )

        delta_action_mask_torch = make_bool_mask(
            7, -1, 7, -1, -2, -2
        )  # left arm, left gripper, right arm, right gripper, torso, head
        
        data_transforms = data_transforms.push(
            outputs=[InterpolateActions(sample_step=1. / self.action_sample_step, actions_type="absolute")],
        )
        data_transforms = data_transforms.push(
            inputs=[DeltaActions(delta_action_mask_torch)],
            outputs=[AbsoluteActions(delta_action_mask_torch)],
        )

        model_transforms_factory = ModelTransformFactory()
        model_transforms = model_transforms_factory(model_config, self)

        base_data_config = self.create_base_config(assets_dirs)

        final_config = dataclasses.replace(
            base_data_config,
            asset_id=self.asset_id,
            repo_id=self.repo_id,
            root_dir=self.root_dir,
            episodes_list_file=self.episodes_list_file, 
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,  # from PyTorchModelTransformFactory
            action_sequence_keys=self.action_sequence_keys,
            control_mode=self.control_mode,
            action_space_factorization=self.action_space_factorization,
            state_dim=self.state_dim,
            action_horizon=self.action_horizon,
            action_offset=self.action_offset,
            action_sample_step=self.action_sample_step,
            validation_episodes=self.validation_episodes, 
            return_subtasks=self.return_subtasks,
            # norm_stats are loaded by create_base_config
        )
        return final_config


@dataclasses.dataclass(frozen=True)
class LeRobotAgibotDexHandDataConfig(DataConfigFactory):

    asset_id: str | None = None
    repo_id: str | None = None
    root_dir: str | None = None
    episodes_list_file: str | None = None
    action_sequence_keys: Sequence[str] = ("action",)
    action_horizon: int = 16
    action_offset: int = 1
    action_sample_step: int = (
        5  # agibot was collected to slow, so we need to sample at 2x
    )
    state_dim: int = 30
    control_mode: str = "joint"
    action_space_factorization: str = "Num hands: 2. Control mode: joints. Num joints per hand: 7. Hand type: dexhand. Torso: True. Head: True."
    map_to_unified_space: bool = True
    map_to_humanoid: bool = True
    validation_episodes: str | None = None 
    return_subtasks: bool = False
    
    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: Any
    ) -> DataConfig:  # model_config is more like cfg.policy
        from lerobot.common.datasets.data_transforms.robots.agibot_dexhand import (
            AgibotDexHandInputsTransform,
            AgibotDexHandOutputsTransform,
        )

        current_action_dim = (
            model_config.max_action_dim
            if hasattr(model_config, "max_action_dim")
            and model_config.max_action_dim is not None
            else 30
        )
        
                
        map_to_unified_space = (model_config.map_to_unified_space
                                if hasattr(model_config, "map_to_unified_space")
                                and model_config.map_to_unified_space is not None
                                else self.map_to_unified_space)

        repack_structure_corrected = {
            "observation/head_image": "observation.images.head",
            "observation/left_wrist_image": "observation.images.hand_left_fisheye",
            "observation/right_wrist_image": "observation.images.hand_right_fisheye",
            "observation/state": "observation.state",
            "actions": "action",
            "prompt": "prompt",
            "subtask": "subtask",
            "next_subtask": "next_subtask",
            "is_subtask_transition": "is_subtask_transition",
            
        }

        repack_transform = TorchGroup(
            inputs=[RepackTransform(structure=repack_structure_corrected)]
        )

        data_transforms = TorchGroup(
            inputs=[AgibotDexHandInputsTransform(action_dim=current_action_dim, 
                                                 map_to_unified_space=map_to_unified_space,
                                                 map_to_humanoid=self.map_to_humanoid)],
            outputs=[AgibotDexHandOutputsTransform()],
        )   
        delta_action_mask_torch = make_bool_mask(
            7,
            -6,
            7,
            -6,
            -2,
            -2,
        )  # left arm, left gripper, right arm, right gripper, torso, head, velocity
        
        data_transforms = data_transforms.push(
            outputs=[InterpolateActions(sample_step=1. / self.action_sample_step, actions_type="absolute")],
        )
        data_transforms = data_transforms.push(
            inputs=[DeltaActions(delta_action_mask_torch)],
            outputs=[AbsoluteActions(delta_action_mask_torch)],
        )

        model_transforms_factory = ModelTransformFactory()
        model_transforms = model_transforms_factory(model_config, self)

        base_data_config = self.create_base_config(assets_dirs)

        final_config = dataclasses.replace(
            base_data_config,
            asset_id=self.asset_id,
            repo_id=self.repo_id,
            root_dir=self.root_dir,
            episodes_list_file=self.episodes_list_file, 
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,  # from PyTorchModelTransformFactory
            action_sequence_keys=self.action_sequence_keys,
            control_mode=self.control_mode,
            state_dim=self.state_dim,
            action_horizon=self.action_horizon,
            action_offset=self.action_offset,
            action_sample_step=self.action_sample_step,
            action_space_factorization=self.action_space_factorization,
            return_subtasks=self.return_subtasks,
            validation_episodes=self.validation_episodes, 
            # norm_stats are loaded by create_base_config
        )
        return final_config

@dataclasses.dataclass(frozen=True)
class LeRobotFourierGR1DataConfig(DataConfigFactory):

    asset_id: str | None = None
    repo_id: str | None = None
    root_dir: str | None = None
    episodes_list_file: str | None = None
    action_sequence_keys: Sequence[str] = ("action",)
    action_horizon: int = 16
    action_offset: int = 1
    action_sample_step: int = (
        2  # agibot was collected with 30 hz frequency;
    )
    state_dim: int = 26
    control_mode: str = "joint"
    action_space_factorization: str = "Num hands: 2. Control mode: joints. Num joints per hand: 7. Hand type: dexhand. Torso: False. Head: False."
    map_to_unified_space: bool = True
    map_to_humanoid: bool = True
    validation_episodes: str | None = None 

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: Any
    ) -> DataConfig:  # model_config is more like cfg.policy
        from lerobot.common.datasets.data_transforms.robots.fourier_gr1 import (
            FourierGR1InputsTransform,
            FourierGR1OutputsTransform,
        )

        current_action_dim = (
            model_config.max_action_dim
            if hasattr(model_config, "max_action_dim")
            and model_config.max_action_dim is not None
            else 26
        )

        repack_structure_corrected = {
            "observation/head_image": "observation.images.top",
            "observation/state": "observation.state",
            "actions": "action",
            "prompt": "prompt",
        }

                
        map_to_unified_space = (model_config.map_to_unified_space
                                if hasattr(model_config, "map_to_unified_space")
                                and model_config.map_to_unified_space is not None
                                else self.map_to_unified_space)
        
        repack_transform = TorchGroup(
            inputs=[RepackTransform(structure=repack_structure_corrected)]
        )

        data_transforms = TorchGroup(
            inputs=[FourierGR1InputsTransform(action_dim=current_action_dim, 
                                               map_to_unified_space=map_to_unified_space,
                                               map_to_humanoid=self.map_to_humanoid)],
            outputs=[FourierGR1OutputsTransform()],
        )
        delta_action_mask_torch = make_bool_mask(
            7,
            6,
            7,
            6,
        )  # left arm, left gripper, right arm, right gripper
        
        data_transforms = data_transforms.push(
            outputs=[InterpolateActions(sample_step=1. / self.action_sample_step, actions_type="absolute")],
        )
        data_transforms = data_transforms.push(
            inputs=[DeltaActions(delta_action_mask_torch)],
            outputs=[AbsoluteActions(delta_action_mask_torch)],
        )

        model_transforms_factory = ModelTransformFactory()
        model_transforms = model_transforms_factory(model_config, self)

        base_data_config = self.create_base_config(assets_dirs)

        final_config = dataclasses.replace(
            base_data_config,
            asset_id=self.asset_id,
            repo_id=self.repo_id,
            root_dir=self.root_dir,
            episodes_list_file=self.episodes_list_file, 
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,  # from PyTorchModelTransformFactory
            action_sequence_keys=self.action_sequence_keys,
            control_mode=self.control_mode,
            action_space_factorization=self.action_space_factorization,
            state_dim=self.state_dim,
            action_horizon=self.action_horizon,
            action_offset=self.action_offset,
            action_sample_step=self.action_sample_step,
            validation_episodes=self.validation_episodes, 
            # norm_stats are loaded by create_base_config
        )
        return final_config

@dataclasses.dataclass(frozen=True)
class LeRobotCalvinDataConfig(DataConfigFactory):
    asset_id: str | None = None
    repo_id: str | None = None
    root_dir: str | None = None
    episodes_list_file: str | None = None
    action_sequence_keys: Sequence[str] = ("action",)
    action_horizon: int = 16  # for 15 hz it is okay
    action_offset: int = 0
    action_sample_step: int = 1
    state_dim: int = 7
    control_mode: str = "cartesian"
    action_space_factorization: str = "Num hands: 1. Control mode: cartesian. Hand type: 2 fingers. Torso: False. Head: False."
    map_to_unified_space: bool = True
    validation_episodes: str | None = None 
    
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: Any) -> DataConfig:
        from lerobot.common.datasets.data_transforms.robots.calvin import (
            CalvinInputsTransform,
            CalvinOutputsTransform,
        )
        current_action_dim = (
            model_config.max_action_dim
            if hasattr(model_config, "max_action_dim")
            and model_config.max_action_dim is not None
            else 7
        )
        
                
        map_to_unified_space = (model_config.map_to_unified_space
                                if hasattr(model_config, "map_to_unified_space")
                                and model_config.map_to_unified_space is not None
                                else self.map_to_unified_space) 
        
        repack_structure_corrected = {
            "observation/rgb_static": "observation.images.rgb_static",
            "observation/rgb_gripper": "observation.images.rgb_gripper",
            "observation/state": "observation.state",
            "actions": "action",
            "prompt": "prompt",
        }
        
        repack_transform = TorchGroup(
            inputs=[RepackTransform(structure=repack_structure_corrected)]
        )
        
        data_transforms = TorchGroup(
            inputs=[CalvinInputsTransform(action_dim=current_action_dim, map_to_unified_space=map_to_unified_space)],
            outputs=[CalvinOutputsTransform()],
        )
        model_transforms_factory = ModelTransformFactory()
        model_transforms = model_transforms_factory(model_config, self)

        base_data_config = self.create_base_config(assets_dirs)
                # Create the final DataConfig
        final_config = dataclasses.replace(
            base_data_config,
            asset_id=self.asset_id,
            repo_id=self.repo_id,
            root_dir=self.root_dir,
            episodes_list_file=self.episodes_list_file, 
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,  # from PyTorchModelTransformFactory
            action_sequence_keys=self.action_sequence_keys,
            control_mode=self.control_mode,
            action_space_factorization=self.action_space_factorization,
            state_dim=self.state_dim,
            action_horizon=self.action_horizon,
            action_offset=self.action_offset,
            action_sample_step=self.action_sample_step,
            validation_episodes=self.validation_episodes, 
            # norm_stats are loaded by create_base_config
        )
        return final_config



@dataclasses.dataclass(frozen=True)
class LeRobotCentaurDexHandDataConfig(DataConfigFactory):

    asset_id: str | None = None
    repo_id: str | None = None
    root_dir: str | None = None
    episodes_list_file: str | None = None
    action_sequence_keys: Sequence[str] = ("action",)
    action_horizon: int = 16
    action_offset: int = 1
    action_sample_step: int = 2 # data collected with low frequency via teleop
    state_dim: int = 26
    control_mode: str = "joint"
    validation_episodes: str | None = None 

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: Any
    ) -> DataConfig:
        from lerobot.common.datasets.data_transforms.robots.centaur_dexhand import (
            CentaurDexHandInputsTransform,
            CentaurDexHandOutputsTransform,
        )

        current_action_dim = (
            model_config.max_action_dim
            if hasattr(model_config, "max_action_dim")
            and model_config.max_action_dim is not None
            else 26
        )
        

        repack_structure_corrected = {
            "observation/head_image": "observation.images.head_left_eye",
            "observation/left_wrist_image": "observation.images.left_wrist",
            "observation/right_wrist_image": "observation.images.right_wrist",
            "observation/state": "observation.state",
            "actions": "action",
            "prompt": "task",
        }

        repack_transform = TorchGroup(
            inputs=[RepackTransform(structure=repack_structure_corrected)]
        )

        data_transforms = TorchGroup(
            inputs=[CentaurDexHandInputsTransform(action_dim=current_action_dim)],
            outputs=[CentaurDexHandOutputsTransform()],
        )
        delta_action_mask_torch = make_bool_mask(
            7,
            6,
            7,
            6,
        )  # left arm joints + left gripper + right arm joints + right gripper
        
        data_transforms = data_transforms.push(
            outputs=[InterpolateActions(sample_step=1. / self.action_sample_step, actions_type="absolute")],
        )
        data_transforms = data_transforms.push(
            inputs=[DeltaActions(delta_action_mask_torch)],
            outputs=[AbsoluteActions(delta_action_mask_torch)],
        )

        model_transforms_factory = ModelTransformFactory()
        model_transforms = model_transforms_factory(model_config, self)

        base_data_config = self.create_base_config(assets_dirs)

        final_config = dataclasses.replace(
            base_data_config,
            asset_id=self.asset_id,
            repo_id=self.repo_id,
            root_dir=self.root_dir,
            episodes_list_file=self.episodes_list_file, 
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,  # from PyTorchModelTransformFactory
            action_sequence_keys=self.action_sequence_keys,
            control_mode=self.control_mode,
            state_dim=self.state_dim,
            action_horizon=self.action_horizon,
            action_offset=self.action_offset,
            action_sample_step=self.action_sample_step,
            validation_episodes=self.validation_episodes, 
            # norm_stats are loaded by create_base_config
        )
        return final_config

@dataclasses.dataclass(frozen=True)
class LeRobotRobomindTienkungDataConfig(DataConfigFactory):

    asset_id: str | None = None
    repo_id: str | None = None
    root_dir: str | None = None
    episodes_list_file: str | None = None

    action_sequence_keys: Sequence[str] = ("action",)
    action_horizon: int = 16
    action_offset: int = 0
    action_sample_step: int = 1 # data collected with low frequency via teleop
    state_dim: int = 26
    control_mode: str = "joint"
    smooth_actions: bool = True
    action_space_factorization: str = "Num hands: 2. Control mode: joints. Num joints per hand: 7. Hand type: dexhand. Torso: False. Head: False."
    map_to_unified_space: bool = True
    map_to_humanoid: bool = True
    validation_episodes: str | None = None 

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: Any
    ) -> DataConfig:
        from lerobot.common.datasets.data_transforms.robots.robomind_tienkung import (
            RobomindTienkungInputsTransform,
            RobomindTienkungOutputsTransform,
        )

        current_action_dim = (
            model_config.max_action_dim
            if hasattr(model_config, "max_action_dim")
            and model_config.max_action_dim is not None
            else 26
        )
                
        map_to_unified_space = (model_config.map_to_unified_space
                                if hasattr(model_config, "map_to_unified_space")
                                and model_config.map_to_unified_space is not None
                                else self.map_to_unified_space)

        repack_structure_corrected = {
            "observation/head_image": "observation.images.camera_top",
            "observation/state": "observation.state",
            "actions": "action",
            "prompt": "prompt",
        }

        repack_transform = TorchGroup(
            inputs=[RepackTransform(structure=repack_structure_corrected)]
        )

        data_transforms = TorchGroup(
            inputs=[RobomindTienkungInputsTransform(action_dim=current_action_dim, 
                                                    map_to_unified_space=map_to_unified_space,
                                                    map_to_humanoid=self.map_to_humanoid)],
            outputs=[RobomindTienkungOutputsTransform()],
        )
        if self.smooth_actions:
            data_transforms = data_transforms.push(
                inputs=[SmoothActions()],
            )
        delta_action_mask_torch = make_bool_mask(
            7,
            6,
            7,
            6,
        )  # left arm joints + left gripper + right arm joints + right gripper
        
        data_transforms = data_transforms.push(
            outputs=[InterpolateActions(sample_step=1. / self.action_sample_step, actions_type="absolute")],
        )
        data_transforms = data_transforms.push(
            inputs=[DeltaActions(delta_action_mask_torch)],
            outputs=[AbsoluteActions(delta_action_mask_torch)],
        )

        model_transforms_factory = ModelTransformFactory()
        model_transforms = model_transforms_factory(model_config, self)

        base_data_config = self.create_base_config(assets_dirs)

        final_config = dataclasses.replace(
            base_data_config,
            asset_id=self.asset_id,
            repo_id=self.repo_id,
            root_dir=self.root_dir,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,  # from PyTorchModelTransformFactory
            action_sequence_keys=self.action_sequence_keys,
            control_mode=self.control_mode,
            action_space_factorization=self.action_space_factorization,
            state_dim=self.state_dim,
            action_horizon=self.action_horizon,
            action_offset=self.action_offset,
            action_sample_step=self.action_sample_step,
            validation_episodes=self.validation_episodes, 
            # norm_stats are loaded by create_base_config
        )
        return final_config


@dataclasses.dataclass(frozen=True)
class LeRobotRobocoinAgibotDataConfig(DataConfigFactory):

    asset_id: str | None = None
    repo_id: str | None = None
    root_dir: str | None = None
    episodes_list_file: str | None = None

    action_sequence_keys: Sequence[str] = ("action",)
    possible_head_image_keys: Union[None, List[str]] = dataclasses.field(default_factory=lambda: ["observation/head_image_fisheye", "observation/head_image_regular"])  # We add it here for further changing the chosen head image on inference just by changing config
    action_horizon: int = 16
    action_offset: int = 0
    action_sample_step: int = 1 # data collected with low frequency via teleop
    state_dim: int = 16
    control_mode: str = "joint"
    smooth_actions: bool = False
    action_space_factorization: str = ""
    map_to_unified_space: bool = False
    map_to_humanoid: bool = False
    validation_episodes: str | None = None 
    return_subtasks: bool = False

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: Any
    ) -> DataConfig:
        from lerobot.common.datasets.data_transforms.robots.robocoin_agibot import (
            RobocoinAgibotInputsTransform,
            RobocoinAgibotOutputsTransform,
        )

        current_action_dim = (
            model_config.max_action_dim
            if hasattr(model_config, "max_action_dim")
            and model_config.max_action_dim is not None
            else 16
        )
                
        map_to_unified_space = (model_config.map_to_unified_space
                                if hasattr(model_config, "map_to_unified_space")
                                and model_config.map_to_unified_space is not None
                                else self.map_to_unified_space)

        repack_structure_corrected = {
            "observation/head_image_fisheye": "observation.images.cam_high_center_fisheye_rgb",
            "observation/head_image_regular": "observation.images.cam_high_rgb",
            "observation/left_wrist_image": "observation.images.cam_left_wrist_rgb",
            "observation/right_wrist_image": "observation.images.cam_right_wrist_rgb",
            "observation/state": "observation.state",
            "actions": "action",
            "prompt": "prompt",
            "subtask": "subtask",
            "next_subtask": "next_subtask",
            "is_subtask_transition": "is_subtask_transition",
        }

        repack_transform = TorchGroup(
            inputs=[RepackTransform(structure=repack_structure_corrected)]
        )

        data_transforms = TorchGroup(
            inputs=[RobocoinAgibotInputsTransform(max_action_dim=current_action_dim, 
                                                    map_to_unified_space=map_to_unified_space,
                                                    map_to_humanoid=self.map_to_humanoid,
                                                    head_image_keys=self.possible_head_image_keys,
                                                    action_space=self.control_mode)],
            outputs=[RobocoinAgibotOutputsTransform(action_dim=self.state_dim)],
        )
        if self.smooth_actions:
            data_transforms = data_transforms.push(
                inputs=[SmoothActions()],
            )
        delta_action_mask_torch = make_bool_mask(
            self.state_dim,
        )  # full relative actions
        #in case of joint we have 7joints + 1 gripper for each hand = 16
        #in case of cartesian we have pose (7 values, xyz, quat) + 1 gripper for each hand = 16
        
        data_transforms = data_transforms.push(
            outputs=[InterpolateActions(sample_step=1. / self.action_sample_step, actions_type="absolute")],
        )
        data_transforms = data_transforms.push(
            inputs=[DeltaActions(delta_action_mask_torch)],
            outputs=[AbsoluteActions(delta_action_mask_torch)],
        )

        model_transforms_factory = ModelTransformFactory()
        model_transforms = model_transforms_factory(model_config, self)

        base_data_config = self.create_base_config(assets_dirs)

        final_config = dataclasses.replace(
            base_data_config,
            asset_id=self.asset_id,
            repo_id=self.repo_id,
            root_dir=self.root_dir,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,  # from PyTorchModelTransformFactory
            action_sequence_keys=self.action_sequence_keys,
            control_mode=self.control_mode,
            action_space_factorization=self.action_space_factorization,
            state_dim=self.state_dim,
            action_horizon=self.action_horizon,
            action_offset=self.action_offset,
            action_sample_step=self.action_sample_step,
            validation_episodes=self.validation_episodes, 
            return_subtasks=self.return_subtasks,
            # norm_stats are loaded by create_base_config
        )
        return final_config
    
    
@dataclasses.dataclass(frozen=True)
class LeRobotRobocoinAirbotDataConfig(DataConfigFactory):

    asset_id: str | None = None
    repo_id: str | None = None
    root_dir: str | None = None
    episodes_list_file: str | None = None

    action_sequence_keys: Sequence[str] = ("action",)
    possible_head_image_keys: Union[None, List[str]] = dataclasses.field(default_factory=lambda: ["observation/third_view", "observation/head_image"])  # We add it here for further changing the chosen head image on inference just by changing config
    action_horizon: int = 16
    action_offset: int = 0
    action_sample_step: int = 1 # data collected with low frequency via teleop
    state_dim: int = 36
    control_mode: str = "joint"
    smooth_actions: bool = False
    action_space_factorization: str = ""
    map_to_unified_space: bool = False
    map_to_humanoid: bool = False
    validation_episodes: str | None = None 
    return_subtasks: bool = False

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: Any
    ) -> DataConfig:
        from lerobot.common.datasets.data_transforms.robots.robocoin_airbot import (
            RobocoinAirbotInputsTransform,
            RobocoinAirbotOutputsTransform,
        )

        current_action_dim = (
            model_config.max_action_dim
            if hasattr(model_config, "max_action_dim")
            and model_config.max_action_dim is not None
            else 36
        )
                
        map_to_unified_space = (model_config.map_to_unified_space
                                if hasattr(model_config, "map_to_unified_space")
                                and model_config.map_to_unified_space is not None
                                else self.map_to_unified_space)

        repack_structure_corrected = {
            "observation/third_view": "observation.images.cam_third_view",
            "observation/head_image": "observation.images.cam_high_rgb",
            "observation/left_wrist_image": "observation.images.cam_left_wrist_rgb",
            "observation/right_wrist_image": "observation.images.cam_right_wrist_rgb",
            "observation/state": "observation.state",
            "actions": "action",
            "prompt": "prompt",
            "subtask": "subtask",
            "next_subtask": "next_subtask",
            "is_subtask_transition": "is_subtask_transition",
        }

        repack_transform = TorchGroup(
            inputs=[RepackTransform(structure=repack_structure_corrected)]
        )

        data_transforms = TorchGroup(
            inputs=[RobocoinAirbotInputsTransform(max_action_dim=current_action_dim, 
                                                    map_to_unified_space=map_to_unified_space,
                                                    map_to_humanoid=self.map_to_humanoid,
                                                    head_image_keys=self.possible_head_image_keys,
                                                    action_space=self.control_mode)],
            outputs=[RobocoinAirbotOutputsTransform(action_dim=self.state_dim)],
        )
        if self.smooth_actions:
            data_transforms = data_transforms.push(
                inputs=[SmoothActions()],
            )
        delta_action_mask_torch = make_bool_mask(
            self.state_dim,
        )  # full relative actions
        #in case of joint we have 7joints + 1 gripper for each hand = 16
        #in case of cartesian we have pose (7 values, xyz, quat) + 1 gripper for each hand = 16
        
        data_transforms = data_transforms.push(
            outputs=[InterpolateActions(sample_step=1. / self.action_sample_step, actions_type="absolute")],
        )
        data_transforms = data_transforms.push(
            inputs=[DeltaActions(delta_action_mask_torch)],
            outputs=[AbsoluteActions(delta_action_mask_torch)],
        )

        model_transforms_factory = ModelTransformFactory()
        model_transforms = model_transforms_factory(model_config, self)

        base_data_config = self.create_base_config(assets_dirs)

        final_config = dataclasses.replace(
            base_data_config,
            asset_id=self.asset_id,
            repo_id=self.repo_id,
            root_dir=self.root_dir,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,  # from PyTorchModelTransformFactory
            action_sequence_keys=self.action_sequence_keys,
            control_mode=self.control_mode,
            action_space_factorization=self.action_space_factorization,
            state_dim=self.state_dim,
            action_horizon=self.action_horizon,
            action_offset=self.action_offset,
            action_sample_step=self.action_sample_step,
            validation_episodes=self.validation_episodes, 
            return_subtasks=self.return_subtasks,
            # norm_stats are loaded by create_base_config
        )
        return final_config
    
    
@dataclasses.dataclass(frozen=True)
class LeRobotRobocoinAlphaBotDataConfig(DataConfigFactory):

    asset_id: str | None = None
    repo_id: str | None = None
    root_dir: str | None = None
    episodes_list_file: str | None = None

    action_sequence_keys: Sequence[str] = ("action",)
    action_horizon: int = 16
    action_offset: int = 0
    action_sample_step: int = 1 # data collected with low frequency via teleop
    state_dim: int = 16
    control_mode: str = "joint"
    smooth_actions: bool = False
    action_space_factorization: str = ""
    map_to_unified_space: bool = False
    map_to_humanoid: bool = False
    validation_episodes: str | None = None 
    return_subtasks: bool = False

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: Any
    ) -> DataConfig:
        from lerobot.common.datasets.data_transforms.robots.robocoin_alpha_bot import (
            RobocoinAlphaBotInputsTransform,
            RobocoinAlphaBotOutputsTransform,
        )

        current_action_dim = (
            model_config.max_action_dim
            if hasattr(model_config, "max_action_dim")
            and model_config.max_action_dim is not None
            else 32
        )
                
        map_to_unified_space = (model_config.map_to_unified_space
                                if hasattr(model_config, "map_to_unified_space")
                                and model_config.map_to_unified_space is not None
                                else self.map_to_unified_space)

        repack_structure_corrected = {
            "observation/head_image": "observation.images.cam_chest_rgb",
            "observation/left_wrist_image": "observation.images.cam_left_wrist_rgb",
            "observation/right_wrist_image": "observation.images.cam_right_wrist_rgb",
            "observation/state": "observation.state",
            "actions": "action",
            "prompt": "prompt",
            "subtask": "subtask",
            "next_subtask": "next_subtask",
            "is_subtask_transition": "is_subtask_transition",
        }

        repack_transform = TorchGroup(
            inputs=[RepackTransform(structure=repack_structure_corrected)]
        )

        data_transforms = TorchGroup(
            inputs=[RobocoinAlphaBotInputsTransform(max_action_dim=current_action_dim, 
                                                    map_to_unified_space=map_to_unified_space,
                                                    map_to_humanoid=self.map_to_humanoid,
                                                    action_space=self.control_mode)],
            outputs=[RobocoinAlphaBotOutputsTransform(action_dim=self.state_dim)],
        )
        if self.smooth_actions:
            data_transforms = data_transforms.push(
                inputs=[SmoothActions()],
            )
        delta_action_mask_torch = make_bool_mask(
            self.state_dim,
        )  # full relative actions
        #in case of joint we have 7joints + 1 gripper for each hand = 16
        #in case of cartesian we have pose (7 values, xyz, quat) + 1 gripper for each hand = 16
        
        data_transforms = data_transforms.push(
            outputs=[InterpolateActions(sample_step=1. / self.action_sample_step, actions_type="absolute")],
        )
        data_transforms = data_transforms.push(
            inputs=[DeltaActions(delta_action_mask_torch)],
            outputs=[AbsoluteActions(delta_action_mask_torch)],
        )

        model_transforms_factory = ModelTransformFactory()
        model_transforms = model_transforms_factory(model_config, self)

        base_data_config = self.create_base_config(assets_dirs)

        final_config = dataclasses.replace(
            base_data_config,
            asset_id=self.asset_id,
            repo_id=self.repo_id,
            root_dir=self.root_dir,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,  # from PyTorchModelTransformFactory
            action_sequence_keys=self.action_sequence_keys,
            control_mode=self.control_mode,
            action_space_factorization=self.action_space_factorization,
            state_dim=self.state_dim,
            action_horizon=self.action_horizon,
            action_offset=self.action_offset,
            action_sample_step=self.action_sample_step,
            validation_episodes=self.validation_episodes, 
            return_subtasks=self.return_subtasks,
            # norm_stats are loaded by create_base_config
        )
        return final_config
    
    
@dataclasses.dataclass(frozen=True)
class LeRobotRobocoinCobotMagicDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline
    for CobotMagic datasets, using PyTorch-native transforms.
    """

    asset_id: str | None = None
    repo_id: str | None = None
    root_dir: str | None = None
    episodes_list_file: str | None = None
    action_sequence_keys: Sequence[str] = ("action",)
    action_horizon: int = 16  # for 15 hz it is okay
    action_offset: int = 0
    action_sample_step: int = 1
    state_dim: int = 14
    control_mode: str = "joint"
    action_space_factorization: str = "Num hands: 2. Control mode: joints. Num joints per hand: 6. Hand type: 2 fingers. Torso: False. Head: False."
    map_to_unified_space: bool = True
    validation_episodes: str | None = None 
    return_subtasks: bool = False
    
    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: Any
    ) -> DataConfig:  # model_config is more like cfg.policy
        from lerobot.common.datasets.data_transforms.robots.cobot_magic import (
            CobotMagicInputsTransform,
            CobotMagicOutputsTransform,
        )

        current_action_dim = (
            model_config.max_action_dim
            if hasattr(model_config, "max_action_dim")
            and model_config.max_action_dim is not None
            else 14
        )

                
        map_to_unified_space = (model_config.map_to_unified_space
                                if hasattr(model_config, "map_to_unified_space")
                                and model_config.map_to_unified_space is not None
                                else self.map_to_unified_space)

        repack_structure_corrected = {
            "observation/base_image": "observation.images.cam_high_rgb",
            "observation/left_wrist_image": "observation.images.cam_left_wrist_rgb",
            "observation/right_wrist_image": "observation.images.cam_right_wrist_rgb",
            "observation/state": "observation.state",
            "actions": "action",  # Map dataset 'action' (singular) to 'actions' (plural) for the transform
            "prompt": "prompt",
            "subtask": "subtask",
            "next_subtask": "next_subtask",
            "is_subtask_transition": "is_subtask_transition",
        }

        repack_transform = TorchGroup(
            inputs=[RepackTransform(structure=repack_structure_corrected)]
        )

        # Data transforms specific to CobotMagic, using PyTorch versions
        data_transforms = TorchGroup(
            inputs=[CobotMagicInputsTransform(action_dim=current_action_dim, map_to_unified_space=map_to_unified_space)],
            outputs=[CobotMagicOutputsTransform()],
        )

        delta_action_mask_torch = make_bool_mask(
            6, 1, 6, 1
        )  # left joints, left gripper, right joints, right gripper
        
        data_transforms = data_transforms.push(
            outputs=[InterpolateActions(sample_step=1. / self.action_sample_step, actions_type="absolute")],
        )
        data_transforms = data_transforms.push(
            inputs=[DeltaActions(delta_action_mask_torch)],
            outputs=[AbsoluteActions(delta_action_mask_torch)],
        )


        model_transforms_factory = ModelTransformFactory()
        model_transforms = model_transforms_factory(model_config, self)

        base_data_config = self.create_base_config(assets_dirs)

        # Create the final DataConfig
        final_config = dataclasses.replace(
            base_data_config,
            asset_id=self.asset_id,
            repo_id=self.repo_id,
            root_dir=self.root_dir,
            episodes_list_file=self.episodes_list_file,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,  # from PyTorchModelTransformFactory
            action_sequence_keys=self.action_sequence_keys,
            action_space_factorization=self.action_space_factorization,
            control_mode=self.control_mode,
            state_dim=self.state_dim,
            action_horizon=self.action_horizon,
            action_offset=self.action_offset,
            action_sample_step=self.action_sample_step,
            validation_episodes=self.validation_episodes, 
            return_subtasks=self.return_subtasks,
            # norm_stats are loaded by create_base_config
        )
        return final_config
    
@dataclasses.dataclass(frozen=True)
class LeRobotRobocoinLejuDataConfig(DataConfigFactory):

    asset_id: str | None = None
    repo_id: str | None = None
    root_dir: str | None = None
    episodes_list_file: str | None = None

    action_sequence_keys: Sequence[str] = ("action",)
    action_horizon: int = 16
    action_offset: int = 0
    action_sample_step: int = 1 # data collected with low frequency via teleop
    state_dim: int = 26
    control_mode: str = "joint"
    smooth_actions: bool = False
    action_space_factorization: str = ""
    map_to_unified_space: bool = False
    map_to_humanoid: bool = False
    validation_episodes: str | None = None 
    return_subtasks: bool = False

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: Any
    ) -> DataConfig:
        from lerobot.common.datasets.data_transforms.robots.robocoin_leju import (
            RobocoinLejuInputsTransform,
            RobocoinLejuOutputsTransform,
        )

        current_action_dim = (
            model_config.max_action_dim
            if hasattr(model_config, "max_action_dim")
            and model_config.max_action_dim is not None
            else 48
        )
                
        map_to_unified_space = (model_config.map_to_unified_space
                                if hasattr(model_config, "map_to_unified_space")
                                and model_config.map_to_unified_space is not None
                                else self.map_to_unified_space)

        repack_structure_corrected = {
            "observation/head_image": "observation.images.camera_head_rgb",
            "observation/left_wrist_image": "observation.images.camera_left_wrist_rgb",
            "observation/right_wrist_image": "observation.images.camera_right_wrist_rgb",
            "observation/state": "observation.state",
            "actions": "action",
            "prompt": "prompt",
            "subtask": "subtask",
            "next_subtask": "next_subtask",
            "is_subtask_transition": "is_subtask_transition",
        }

        repack_transform = TorchGroup(
            inputs=[RepackTransform(structure=repack_structure_corrected)]
        )

        data_transforms = TorchGroup(
            inputs=[RobocoinLejuInputsTransform(max_action_dim=current_action_dim, 
                                                    map_to_unified_space=map_to_unified_space,
                                                    map_to_humanoid=self.map_to_humanoid,
                                                    action_space=self.control_mode)],
            outputs=[RobocoinLejuOutputsTransform(action_dim=self.state_dim)],
        )
        if self.smooth_actions:
            data_transforms = data_transforms.push(
                inputs=[SmoothActions()],
            )
        delta_action_mask_torch = make_bool_mask(
            self.state_dim,
        )  # full relative actions
        #in case of joint we have 7joints + 6 gripper for each hand = 26
        #in case of cartesian we have pose (6 values, xyz, euler) + 6 gripper for each hand = 24
        
        data_transforms = data_transforms.push(
            outputs=[InterpolateActions(sample_step=1. / self.action_sample_step, actions_type="absolute")],
        )
        data_transforms = data_transforms.push(
            inputs=[DeltaActions(delta_action_mask_torch)],
            outputs=[AbsoluteActions(delta_action_mask_torch)],
        )

        model_transforms_factory = ModelTransformFactory()
        model_transforms = model_transforms_factory(model_config, self)

        base_data_config = self.create_base_config(assets_dirs)

        final_config = dataclasses.replace(
            base_data_config,
            asset_id=self.asset_id,
            repo_id=self.repo_id,
            root_dir=self.root_dir,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,  # from PyTorchModelTransformFactory
            action_sequence_keys=self.action_sequence_keys,
            control_mode=self.control_mode,
            action_space_factorization=self.action_space_factorization,
            state_dim=self.state_dim,
            action_horizon=self.action_horizon,
            action_offset=self.action_offset,
            action_sample_step=self.action_sample_step,
            validation_episodes=self.validation_episodes, 
            return_subtasks=self.return_subtasks,
            # norm_stats are loaded by create_base_config
        )
        return final_config
    
@dataclasses.dataclass(frozen=True)
class LeRobotRobocoinR1LiteDataConfig(DataConfigFactory):

    asset_id: str | None = None
    repo_id: str | None = None
    root_dir: str | None = None
    episodes_list_file: str | None = None

    action_sequence_keys: Sequence[str] = ("action",)
    action_horizon: int = 16
    action_offset: int = 0
    action_sample_step: int = 1 # data collected with low frequency via teleop
    state_dim: int = 14
    control_mode: str = "joint"
    smooth_actions: bool = False
    action_space_factorization: str = ""
    map_to_unified_space: bool = False
    map_to_humanoid: bool = False
    validation_episodes: str | None = None 
    return_subtasks: bool = False

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: Any
    ) -> DataConfig:
        from lerobot.common.datasets.data_transforms.robots.robocoin_r1_lite import (
            RobocoinR1LiteInputsTransform,
            RobocoinR1LiteOutputsTransform,
        )

        current_action_dim = (
            model_config.max_action_dim
            if hasattr(model_config, "max_action_dim")
            and model_config.max_action_dim is not None
            else 48
        )
                
        map_to_unified_space = (model_config.map_to_unified_space
                                if hasattr(model_config, "map_to_unified_space")
                                and model_config.map_to_unified_space is not None
                                else self.map_to_unified_space)

        repack_structure_corrected = {
            "observation/head_image": "observation.images.cam_high_rgb",
            "observation/left_wrist_image": "observation.images.cam_left_wrist_rgb",
            "observation/right_wrist_image": "observation.images.cam_right_wrist_rgb",
            "observation/state": "observation.state",
            "actions": "action",
            "prompt": "prompt",
            "subtask": "subtask",
            "next_subtask": "next_subtask",
            "is_subtask_transition": "is_subtask_transition",
        }

        repack_transform = TorchGroup(
            inputs=[RepackTransform(structure=repack_structure_corrected)]
        )

        data_transforms = TorchGroup(
            inputs=[RobocoinR1LiteInputsTransform(max_action_dim=current_action_dim, 
                                                    map_to_unified_space=map_to_unified_space,
                                                    map_to_humanoid=self.map_to_humanoid,
                                                    action_space=self.control_mode)],
            outputs=[RobocoinR1LiteOutputsTransform(action_dim=self.state_dim)],
        )
        if self.smooth_actions:
            data_transforms = data_transforms.push(
                inputs=[SmoothActions()],
            )
        delta_action_mask_torch = make_bool_mask(
            self.state_dim,
        )  # full relative actions
        #e have 6joints + 1 gripper for each hand = 14
        
        data_transforms = data_transforms.push(
            outputs=[InterpolateActions(sample_step=1. / self.action_sample_step, actions_type="absolute")],
        )
        data_transforms = data_transforms.push(
            inputs=[DeltaActions(delta_action_mask_torch)],
            outputs=[AbsoluteActions(delta_action_mask_torch)],
        )

        model_transforms_factory = ModelTransformFactory()
        model_transforms = model_transforms_factory(model_config, self)

        base_data_config = self.create_base_config(assets_dirs)

        final_config = dataclasses.replace(
            base_data_config,
            asset_id=self.asset_id,
            repo_id=self.repo_id,
            root_dir=self.root_dir,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,  # from PyTorchModelTransformFactory
            action_sequence_keys=self.action_sequence_keys,
            control_mode=self.control_mode,
            action_space_factorization=self.action_space_factorization,
            state_dim=self.state_dim,
            action_horizon=self.action_horizon,
            action_offset=self.action_offset,
            action_sample_step=self.action_sample_step,
            validation_episodes=self.validation_episodes, 
            return_subtasks=self.return_subtasks,
            # norm_stats are loaded by create_base_config
        )
        return final_config
    

@dataclasses.dataclass(frozen=True)
class LeRobotRobocoinRMCAidaDataConfig(DataConfigFactory):

    asset_id: str | None = None
    repo_id: str | None = None
    root_dir: str | None = None
    episodes_list_file: str | None = None

    action_sequence_keys: Sequence[str] = ("action",)
    action_horizon: int = 16
    action_offset: int = 0
    action_sample_step: int = 1 # data collected with low frequency via teleop
    state_dim: int = 16
    control_mode: str = "joint"
    smooth_actions: bool = False
    action_space_factorization: str = ""
    map_to_unified_space: bool = False
    map_to_humanoid: bool = False
    validation_episodes: str | None = None 
    

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: Any
    ) -> DataConfig:
        from lerobot.common.datasets.data_transforms.robots.robocoin_rmc_aida import (
            RobocoinRMCAidaInputsTransform,
            RobocoinRMCAidaOutputsTransform,
        )

        current_action_dim = (
            model_config.max_action_dim
            if hasattr(model_config, "max_action_dim")
            and model_config.max_action_dim is not None
            else 48
        )
                
        map_to_unified_space = (model_config.map_to_unified_space
                                if hasattr(model_config, "map_to_unified_space")
                                and model_config.map_to_unified_space is not None
                                else self.map_to_unified_space)

        repack_structure_corrected = {
            "observation/head_image": "observation.images.cam_high_rgb",
            "observation/left_wrist_image": "observation.images.cam_left_wrist_rgb",
            "observation/right_wrist_image": "observation.images.cam_right_wrist_rgb",
            "observation/state": "observation.state",
            "actions": "action",
            "prompt": "prompt",
        }

        repack_transform = TorchGroup(
            inputs=[RepackTransform(structure=repack_structure_corrected)]
        )

        data_transforms = TorchGroup(
            inputs=[RobocoinRMCAidaInputsTransform(max_action_dim=current_action_dim, 
                                                    map_to_unified_space=map_to_unified_space,
                                                    map_to_humanoid=self.map_to_humanoid,
                                                    action_space=self.control_mode)],
            outputs=[RobocoinRMCAidaOutputsTransform(action_dim=self.state_dim)],
        )
        if self.smooth_actions:
            data_transforms = data_transforms.push(
                inputs=[SmoothActions()],
            )
        delta_action_mask_torch = make_bool_mask(
            self.state_dim,
        )  # full relative actions
        #we have 7joints + 1 gripper for each hand = 16
        #in case of cartesian we have pose (6 values, xyz, euler) + 1 gripper for each hand = 14
        
        data_transforms = data_transforms.push(
            outputs=[InterpolateActions(sample_step=1. / self.action_sample_step, actions_type="absolute")],
        )
        data_transforms = data_transforms.push(
            inputs=[DeltaActions(delta_action_mask_torch)],
            outputs=[AbsoluteActions(delta_action_mask_torch)],
        )

        model_transforms_factory = ModelTransformFactory()
        model_transforms = model_transforms_factory(model_config, self)

        base_data_config = self.create_base_config(assets_dirs)

        final_config = dataclasses.replace(
            base_data_config,
            asset_id=self.asset_id,
            repo_id=self.repo_id,
            root_dir=self.root_dir,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,  # from PyTorchModelTransformFactory
            action_sequence_keys=self.action_sequence_keys,
            control_mode=self.control_mode,
            action_space_factorization=self.action_space_factorization,
            state_dim=self.state_dim,
            action_horizon=self.action_horizon,
            action_offset=self.action_offset,
            action_sample_step=self.action_sample_step,
            validation_episodes=self.validation_episodes, 
            # norm_stats are loaded by create_base_config
        )
        return final_config
    
    
@dataclasses.dataclass(frozen=True)
class LeRobotRealsourceWorldDataConfig(DataConfigFactory):

    asset_id: str | None = None
    repo_id: str | None = None
    root_dir: str | None = None
    episodes_list_file: str | None = None

    action_sequence_keys: Sequence[str] = ("action",)
    action_horizon: int = 16
    action_offset: int = 0
    action_sample_step: int = 1 # data collected with low frequency via teleop
    state_dim: int = 17
    control_mode: str = "joint"
    smooth_actions: bool = False
    action_space_factorization: str = ""
    map_to_unified_space: bool = False
    map_to_humanoid: bool = False
    validation_episodes: str | None = None 
    return_subtasks: bool = False

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: Any
    ) -> DataConfig:
        from lerobot.common.datasets.data_transforms.robots.realsource_world import (
            RealsourceWorldInputsTransform,
            RealsourceWorldOutputsTransform,
        )

        current_action_dim = (
            model_config.max_action_dim
            if hasattr(model_config, "max_action_dim")
            and model_config.max_action_dim is not None
            else 48
        )
                
        map_to_unified_space = (model_config.map_to_unified_space
                                if hasattr(model_config, "map_to_unified_space")
                                and model_config.map_to_unified_space is not None
                                else self.map_to_unified_space)

        repack_structure_corrected = {
            "observation/head_image": "observation.images.head_camera",
            "observation/left_wrist_image": "observation.images.left_hand_camera",
            "observation/right_wrist_image": "observation.images.right_hand_camera",
            "observation/state": "observation.state",
            "actions": "action",
            "prompt": "prompt",
            "subtask": "subtask",
            "next_subtask": "next_subtask",
            "is_subtask_transition": "is_subtask_transition",
        }

        repack_transform = TorchGroup(
            inputs=[RepackTransform(structure=repack_structure_corrected)]
        )

        data_transforms = TorchGroup(
            inputs=[RealsourceWorldInputsTransform(max_action_dim=current_action_dim, 
                                                    map_to_unified_space=map_to_unified_space,
                                                    map_to_humanoid=self.map_to_humanoid,
                                                    action_space=self.control_mode)],
            outputs=[RealsourceWorldOutputsTransform(action_dim=self.state_dim)],
        )
        if self.smooth_actions:
            data_transforms = data_transforms.push(
                inputs=[SmoothActions()],
            )
        delta_action_mask_torch = make_bool_mask(
            self.state_dim,
        )  # full relative actions
        #we have 7joints + 1 gripper for each hand = 16
        #in case of cartesian we have pose (6 values, xyz, euler) + 1 gripper for each hand = 14
        
        data_transforms = data_transforms.push(
            outputs=[InterpolateActions(sample_step=1. / self.action_sample_step, actions_type="absolute")],
        )
        data_transforms = data_transforms.push(
            inputs=[DeltaActions(delta_action_mask_torch)],
            outputs=[AbsoluteActions(delta_action_mask_torch)],
        )

        model_transforms_factory = ModelTransformFactory()
        model_transforms = model_transforms_factory(model_config, self)

        base_data_config = self.create_base_config(assets_dirs)

        final_config = dataclasses.replace(
            base_data_config,
            asset_id=self.asset_id,
            repo_id=self.repo_id,
            root_dir=self.root_dir,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,  # from PyTorchModelTransformFactory
            action_sequence_keys=self.action_sequence_keys,
            return_subtasks=self.return_subtasks,
            control_mode=self.control_mode,
            action_space_factorization=self.action_space_factorization,
            state_dim=self.state_dim,
            action_horizon=self.action_horizon,
            action_offset=self.action_offset,
            action_sample_step=self.action_sample_step,
            validation_episodes=self.validation_episodes, 
            # norm_stats are loaded by create_base_config
        )
        return final_config