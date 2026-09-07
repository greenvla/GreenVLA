import dataclasses
import json
from pathlib import Path

import numpy as np
import torch
from lerobot.common.datasets.s0s1_kinematics import (
    convert_s0s1_kinematic_space,
)
from lerobot.common.datasets.torch_transforms import (
    DataTransformFn,
    pad_to_dim,
    parse_image_helper,
    passthrough_defaults,
    BaseModelConfigPlaceholder as ModelConfig,  # Use the placeholder
)

BASE_MASKS = torch.tensor(
    [
        [True] * 26 + [False] * 6,
        [True] * 26 + [False] * 2 + [True] + [False] * 3,
        [True] * 32,
    ],
    dtype=torch.bool,
)







# ──────────────────────────────────────────────────────────────────────
# S0/S1 humanoid transform with legs, full velocity, and WBC layout
# ──────────────────────────────────────────────────────────────────────
#
# OUTPUT LAYOUT (50 dims) — matches WBC ref_obs expected by Anton's controller:
#
# ┌─ ref_joint (25) ──────────────────────────────────────────────────┐
# │  [ 0] left_hip_pitch          [ 6] right_hip_pitch               │
# │  [ 1] left_hip_roll           [ 7] right_hip_roll                │
# │  [ 2] left_hip_yaw            [ 8] right_hip_yaw                 │
# │  [ 3] left_knee_pitch         [ 9] right_knee_pitch              │
# │  [ 4] left_ankle_pitch        [10] right_ankle_pitch             │
# │  [ 5] left_ankle_roll         [11] right_ankle_roll              │
# │  [12] torso_yaw                                                  │
# │  [13] left_shoulder_pitch     [18] right_shoulder_pitch          │
# │  [14] left_shoulder_roll      [19] right_shoulder_roll           │
# │  [15] left_shoulder_yaw       [20] right_shoulder_yaw            │
# │  [16] left_elbow_pitch        [21] right_elbow_pitch             │
# │  [17] left_elbow_yaw          [22] right_elbow_yaw               │
# │  [23] neck_yaw                [24] neck_pitch                    │
# └───────────────────────────────────────────────────────────────────┘
# [25]    root_height                    (absolute, from odom z)
# ┌─ lin_vel (3) ─┐  ┌─ ang_vel (3) ─┐
# │ [26] vx       │  │ [29] wx       │  (absolute, body-frame)
# │ [27] vy       │  │ [30] wy       │
# │ [28] vz       │  │ [31] wz       │
# └───────────────┘  └───────────────┘
# [32]    body_roll                      (absolute)
# [33]    body_pitch                     (absolute)
# ┌─ left hand (8) ─────────────────────────────────────────────────┐
# │ [34] left_pinky     [35] left_ring     [36] left_middle        │
# │ [37] left_index     [38] left_thumb_pitch  [39] left_thumb_yaw │
# │ [40] left_wrist_roll  [41] left_wrist_pitch                    │
# └─────────────────────────────────────────────────────────────────┘
# ┌─ right hand (8) ────────────────────────────────────────────────┐
# │ [42] right_pinky    [43] right_ring    [44] right_middle       │
# │ [45] right_index    [46] right_thumb_pitch [47] right_thumb_yaw│
# │ [48] right_wrist_roll  [49] right_wrist_pitch                  │
# └─────────────────────────────────────────────────────────────────┘
#
# DELTA MASK: joints [0-24] and fingers+wrists [34-49] are DELTA;
#             height [25], velocities [26-31], roll/pitch [32-33] are ABSOLUTE.
#
# VELOCITY LOSS MASK: when has_velocities=False (old dataset),
#             positions [26-31] are masked out in the action loss.

# Reorder indices: from 51-dim parquet -> 50-dim output
# (torso_yaw at index 44 is dropped, always 0)
_S0S1_REORDER_51 = (
    list(range(0, 25))          # [0-24]  legs(12) + torso_yaw(1) + arms(10) + neck(2) = 25
    + [43]                      # [25]    torso_z -> root_height
    + [45, 46, 47]              # [26-28] lin_vel x,y,z
    + [48, 49, 50]              # [29-31] ang_vel x,y,z
    + [41, 42]                  # [32-33] roll, pitch
    + [29, 30, 31, 32, 33, 34]  # [34-39] left fingers
    + [25, 26]                  # [40-41] left wrist
    + [35, 36, 37, 38, 39, 40]  # [42-47] right fingers
    + [27, 28]                  # [48-49] right wrist
)

# For 44-dim old datasets: same joint layout but no velocity fields (indices 44-50 don't exist).
# We pick the available ones and zero-pad velocities.
_S0S1_REORDER_44 = (
    list(range(0, 25))          # [0-24]  legs(12) + torso_yaw(1) + arms(10) + neck(2)
    + [43]                      # [25]    torso_z -> root_height
    # indices 26-31 will be zero-padded (no velocities in 44-dim)
    + [41, 42]                  # mapped to [32-33] roll, pitch (after zero padding)
    + [29, 30, 31, 32, 33, 34]  # [34-39] left fingers
    + [25, 26]                  # [40-41] left wrist
    + [35, 36, 37, 38, 39, 40]  # [42-47] right fingers
    + [27, 28]                  # [48-49] right wrist
)

# For old 32-dim datasets (upper body only, after old sampled_state_idxs).
# Old 32-dim layout:
#   [0-4]   left arm (5)        [5-6]   left wrist (2)
#   [7-12]  left fingers (6)
#   [13-17] right arm (5)       [18-19] right wrist (2)
#   [20-25] right fingers (6)
#   [26]    torso_roll  [27] torso_pitch  [28] torso_yaw
#   [29]    torso_z     [30] neck_yaw     [31] neck_pitch
#
# Picked into: torso_yaw(1) + left_arm(5) + right_arm(5) + neck(2)
#              + root_height(1) + roll(1) + pitch(1) + left_fingers(6)
#              + left_wrist(2) + right_fingers(6) + right_wrist(2) = 32
# Then assembled as: zeros(12) + picked[:14] + zeros(6) + picked[14:] = 50
# Legs [0-11] and velocities [26-31] become zeros (masked out).
_S0S1_REORDER_32 = (
    [28]                        # torso_yaw      -> output[12]
    + list(range(0, 5))         # left arm (5)   -> output[13-17]
    + list(range(13, 18))       # right arm (5)  -> output[18-22]
    + [30, 31]                  # neck (2)       -> output[23-24]
    + [29]                      # torso_z        -> output[25]
    + [26, 27]                  # roll, pitch    -> output[32-33]
    + list(range(7, 13))        # left fingers   -> output[34-39]
    + [5, 6]                    # left wrist     -> output[40-41]
    + list(range(20, 26))       # right fingers  -> output[42-47]
    + [18, 19]                  # right wrist    -> output[48-49]
)

S0S1_OUTPUT_DIM = 50
_S0S1_NO_LEG_KEEP = tuple(range(12, S0S1_OUTPUT_DIM))
S0S1_NO_LEG_OUTPUT_DIM = len(_S0S1_NO_LEG_KEEP)

_LEFT_WRIST_OUTPUT_INDICES = (40, 41)
_RIGHT_WRIST_OUTPUT_INDICES = (48, 49)
_LEFT_LEADSHINE_HAND_TO_INSPIRE_INDICES = (34, 35, 36, 37, 38, 39)
_LEFT_LEADSHINE_HAND_TO_INSPIRE_A = (
    1.266993,
    1.335953,
    1.312314,
    1.372722,
    0.608624,
    1.149267,
)
_LEFT_LEADSHINE_HAND_TO_INSPIRE_B = (
    0.000380,
    0.000802,
    0.000787,
    0.000824,
    0.304312,
    -0.099196,
)


# Real S0/S1 mean state in the 50-dim WBC layout, used as the neutral fill for
# state channels that a source dataset does not actually observe.
S0S1_REAL_ROBOT_NEUTRAL_STATE = (
    -0.2611356785, -0.02198516932, -0.06928652597, 0.4240251104, -0.2439087,
    0.02476310947, -0.2840679865, -0.02136309433, 0.1047440507, 0.4596222091,
    -0.2697068164, -0.003430663452, -0.03088243036, -0.2129020776, 0.1428004881,
    -0.2095366157, -0.3564485948, -0.06663051718, -0.2763304607, -0.2100845684,
    0.2784674832, -0.7167083551, -0.02424113079, -0.03537133993, -0.05487289832,
    0.870236088, 0.3141623126, 0.0, 0.0, 0.0,
    0.0, 0.02861263568, 0.01025960818, 0.0758500269, 0.07256481689,
    0.07169568314, 0.06347022281, 0.04518236726, 0.2089236319, 1.094756647,
    0.1149236675, 0.02930031503, 0.5112941225, 0.4621726901, 0.437535676,
    0.4066613054, 0.154660785, 1.190956468, 0.1636076821, 0.02108809863,
)

S0S1_JOINT_NAMES = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw",
    "left_knee_pitch", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw",
    "right_knee_pitch", "right_ankle_pitch", "right_ankle_roll",
    "torso_yaw",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow_pitch", "left_elbow_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow_pitch", "right_elbow_yaw",
    "neck_yaw", "neck_pitch",
    "root_height",
    "linear_vel_x", "linear_vel_y", "linear_vel_z",
    "angular_vel_x", "angular_vel_y", "angular_vel_z",
    "body_roll", "body_pitch",
    "left_pinky", "left_ring", "left_middle",
    "left_index", "left_thumb_pitch", "left_thumb_yaw",
    "left_wrist_roll", "left_wrist_pitch",
    "right_pinky", "right_ring", "right_middle",
    "right_index", "right_thumb_pitch", "right_thumb_yaw",
    "right_wrist_roll", "right_wrist_pitch",
]


@dataclasses.dataclass(frozen=True)
class HumanoidS0S1InputsTransform(DataTransformFn):
    """Transform for S0/S1 humanoid with full WBC layout (legs + velocities).

    Handles:
      - 51-dim (new S0S1, with velocity)
      - 52-dim (51 + episode_progress in actions)
      - 44-dim (old parquet with legs but no velocity)
      - 45-dim (44 + episode_progress in actions)
      - 32-dim (old upper-body-only after sampled_state_idxs, no legs/vel)
      - 33-dim (32 + episode_progress in actions)

    Produces a unified 50-dim output matching the WBC ref_obs layout.
    """
    action_dim: int
    has_velocities: bool = True
    map_to_closed_kinematic: bool = False
    input_kinematic_space: str | None = None
    model_kinematic_space: str | None = None
    image_dropout_prob: float = 0.0  # probability of dropping all images (set image_mask=False)
    exclude_legs: bool = False

    def _pick_and_pad(self, x, indices, n_leg_zeros, n_vel_zeros, split_vel, axis=-1):
        """Pick elements by *indices*, then insert zero blocks for legs and velocities.

        Args:
            x: source tensor/array.
            indices: fancy-index array into x.
            n_leg_zeros: how many zeros to prepend (leg positions).
            n_vel_zeros: how many zeros to insert at *split_vel* (velocity positions).
            split_vel: position inside the picked array where velocity zeros go
                       (right after root_height).
            axis: working axis.
        """
        picked = x[..., indices] if axis == -1 else x[:, indices]
        if n_leg_zeros == 0 and n_vel_zeros == 0:
            return picked

        def _zeros(n):
            shape = list(picked.shape)
            shape[axis] = n
            if isinstance(x, torch.Tensor):
                return torch.zeros(shape, dtype=x.dtype, device=x.device)
            return np.zeros(shape, dtype=x.dtype)

        def _cat(parts):
            if isinstance(x, torch.Tensor):
                return torch.cat(parts, dim=axis)
            return np.concatenate(parts, axis=axis)

        before = picked[..., :split_vel] if axis == -1 else picked[:, :split_vel]
        after = picked[..., split_vel:] if axis == -1 else picked[:, split_vel:]
        return _cat([_zeros(n_leg_zeros), before, _zeros(n_vel_zeros), after])

    def _reorder(self, x, axis=-1):
        """Reorder a vector from any supported layout to 50-dim WBC output."""
        src_dim = x.shape[axis]

        if src_dim == S0S1_OUTPUT_DIM:
            return x

        if src_dim in (51, 52):
            return x[..., _S0S1_REORDER_51] if axis == -1 else x[:, _S0S1_REORDER_51]

        if src_dim in (44, 45):
            return self._pick_and_pad(x, _S0S1_REORDER_44, n_leg_zeros=0, n_vel_zeros=6, split_vel=26, axis=axis)

        if src_dim in (32, 33, 34):
            return self._pick_and_pad(x, _S0S1_REORDER_32, n_leg_zeros=12, n_vel_zeros=6, split_vel=14, axis=axis)

        raise ValueError(
            f"Unexpected dimension {src_dim}, expected {S0S1_OUTPUT_DIM}, 32-34, 44-45, or 51-52"
        )

    def _missing_dims(self, src_dim: int) -> tuple[int, ...]:
        dims: set[int] = set()
        if not self.has_velocities:
            dims.update(range(26, 32))
        if src_dim in (32, 33, 34):
            dims.update(range(0, 12))
            dims.update(range(26, 32))
        return tuple(sorted(dims))

    def _neutral_like(self, x):
        if isinstance(x, torch.Tensor):
            return torch.as_tensor(S0S1_REAL_ROBOT_NEUTRAL_STATE, dtype=x.dtype, device=x.device)
        return np.asarray(S0S1_REAL_ROBOT_NEUTRAL_STATE, dtype=x.dtype)

    def _fill_state_dims(self, state, dims: tuple[int, ...]):
        if not dims:
            return state
        state = state.clone() if isinstance(state, torch.Tensor) else state.copy()
        neutral = self._neutral_like(state)
        for dim in dims:
            state[..., dim] = neutral[dim]
        return state

    def _model_neutral_state(self, reference):
        """Return the real-robot neutral pose in the model-facing state layout."""
        neutral = self._neutral_like(reference)
        neutral = self._map_to_model_kinematic_space(neutral)
        neutral = self._maybe_remove_legs(neutral)
        return pad_to_dim(neutral, self.action_dim, axis=-1, value=0.0)

    def _fill_action_dims_from_state(self, actions, state, dims: tuple[int, ...]):
        if not dims:
            return actions
        actions = actions.clone() if isinstance(actions, torch.Tensor) else actions.copy()
        if isinstance(actions, torch.Tensor):
            state_values = state.to(dtype=actions.dtype, device=actions.device)
        else:
            state_values = np.asarray(state, dtype=actions.dtype)
        for dim in dims:
            actions[..., dim] = state_values[..., dim] if state_values.ndim == actions.ndim else state_values[dim]
        return actions

    def _map_to_model_kinematic_space(self, x):
        if (
            self.input_kinematic_space is not None
            or self.model_kinematic_space is not None
        ):
            if self.input_kinematic_space is None or self.model_kinematic_space is None:
                raise ValueError(
                    "input_kinematic_space and model_kinematic_space must be set together"
                )
            return convert_s0s1_kinematic_space(
                x,
                self.input_kinematic_space,
                self.model_kinematic_space,
            )
        if not self.map_to_closed_kinematic:
            return x
        return _map_s0s1_ankles_to_closed_kinematic(x)


    def _maybe_remove_legs(self, x):
        if not self.exclude_legs:
            return x
        return x[..., _S0S1_NO_LEG_KEEP]

    def _build_loss_mask(self, src_dim: int, data: dict | None = None) -> torch.Tensor:
        mask = torch.ones(S0S1_OUTPUT_DIM, dtype=torch.bool)
        if not self.has_velocities:
            mask[26:32] = False
        if src_dim in (32, 33, 34):
            mask[0:12] = False
            mask[26:32] = False
        if self.exclude_legs:
            mask = mask[list(_S0S1_NO_LEG_KEEP)]
        return mask
    def _build_valid_mask(
        self, data: dict, key: str, missing_dims: tuple[int, ...]
    ) -> torch.Tensor | None:
        raw_mask = data.get(key)
        if raw_mask is None:
            return None
        if isinstance(raw_mask, np.ndarray):
            raw_mask = torch.from_numpy(raw_mask)
        elif not isinstance(raw_mask, torch.Tensor):
            raw_mask = torch.as_tensor(raw_mask)
        raw_mask = raw_mask.bool()
        if raw_mask.ndim > 1:
            raw_mask = raw_mask[-1]

        mask = self._reorder(raw_mask).bool()
        if missing_dims:
            mask = mask.clone()
            for dim in missing_dims:
                mask[dim] = False
        mask = self._maybe_remove_legs(mask)
        return pad_to_dim(mask, self.action_dim, axis=-1, value=False)


    def __call__(self, data: dict) -> dict:
        raw_state = data["observation/state"]
        if isinstance(raw_state, np.ndarray):
            raw_state = torch.from_numpy(raw_state).float()

        # Handle multi-timestep state: (state_history, state_dim) or (state_dim,)
        if raw_state.ndim == 2:
            # Multiple timesteps from delta_timestamps — reorder each, keep last as current
            src_dim = raw_state.shape[-1]
            missing_dims = self._missing_dims(src_dim)
            state_50 = self._reorder(raw_state)
            state_history_50 = self._map_to_model_kinematic_space(
                self._fill_state_dims(state_50, missing_dims)
            )
            state_full = state_history_50[-1]  # current timestep for flow matching head
            state_history = pad_to_dim(
                self._maybe_remove_legs(state_history_50), self.action_dim
            )  # (H, action_dim)
        else:
            src_dim = raw_state.shape[-1]
            missing_dims = self._missing_dims(src_dim)
            state_50 = self._reorder(raw_state)
            state_full = self._map_to_model_kinematic_space(
                self._fill_state_dims(state_50, missing_dims)
            )
            state_history = None
        state = pad_to_dim(self._maybe_remove_legs(state_full), self.action_dim)
        state_valid_mask = self._build_valid_mask(data, "state_valid_mask", missing_dims)

        # Some legacy episodes declare a channel valid in episodes.jsonl while the
        # parquet values are NaN/Inf.  Masking only the loss is too late: non-finite
        # state/action values have already entered normalization and the forward pass.
        # Sanitize the values here and conservatively invalidate the affected channel.
        state_finite = torch.isfinite(state)
        state = torch.where(state_finite, state, torch.zeros_like(state))
        if state_history is not None:
            history_finite = torch.isfinite(state_history)
            state_history = torch.where(history_finite, state_history, torch.zeros_like(state_history))
            state_finite = state_finite & history_finite.all(dim=-2)
        if state_valid_mask is not None:
            state_valid_mask = state_valid_mask & state_finite
        elif not bool(state_finite.all()):
            state_valid_mask = state_finite

        # Episode-level masks describe source channels that are absent or
        # untrustworthy even when the parquet has a full 51D layout. Excluding
        # them from statistics is not sufficient: with validity conditioning
        # disabled the model would still consume their zero/garbage values.
        # Impute them with the same real-robot neutral pose used for structural
        # missing dimensions, including every state-history timestep.
        if state_valid_mask is not None:
            neutral_state = self._model_neutral_state(state)
            state = torch.where(state_valid_mask, state, neutral_state)
            if state_history is not None:
                state_history = torch.where(
                    state_valid_mask.unsqueeze(0), state_history, neutral_state
                )

        head_image = parse_image_helper(data["observation/head_image"])
        left_wrist_image = parse_image_helper(data["observation/left_wrist_image"])
        right_wrist_image = parse_image_helper(data["observation/right_wrist_image"])

        if isinstance(head_image, np.ndarray):
            head_image = torch.from_numpy(head_image).permute(2, 0, 1)
            left_wrist_image = torch.from_numpy(left_wrist_image).permute(2, 0, 1)
            right_wrist_image = torch.from_numpy(right_wrist_image).permute(2, 0, 1)
        elif head_image.ndim == 3 and head_image.shape[2] == 3:
            head_image = head_image.permute(2, 0, 1)
            left_wrist_image = left_wrist_image.permute(2, 0, 1)
            right_wrist_image = right_wrist_image.permute(2, 0, 1)

        # Random image dropout: with probability image_dropout_prob, mask all images
        # so the VLM processes text-only. This teaches the model to rely on language
        # when visual input is unavailable (important for cross-dataset generalization).
        images_available = not (self.image_dropout_prob > 0 and torch.rand(1).item() < self.image_dropout_prob)

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": head_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": torch.tensor(images_available),
                "left_wrist_0_rgb": torch.tensor(images_available),
                "right_wrist_0_rgb": torch.tensor(images_available),
            },
        }

        if state_history is not None:
            inputs["state_history"] = state_history  # (H, action_dim)
        if state_valid_mask is not None:
            inputs["state_valid_mask"] = state_valid_mask

        if "actions" in data:
            action_src_dim = data["actions"].shape[-1]
            actions_reordered = self._reorder(data["actions"], axis=-1)
            if isinstance(actions_reordered, np.ndarray):
                actions_reordered = torch.from_numpy(actions_reordered).float()
            actions_reordered = self._map_to_model_kinematic_space(
                actions_reordered
            )
            actions_reordered = self._fill_action_dims_from_state(
                actions_reordered,
                state_full,
                self._missing_dims(action_src_dim),
            )
            actions = pad_to_dim(
                self._maybe_remove_legs(actions_reordered), self.action_dim, axis=-1, value=0.0
            )
            actions_finite = torch.isfinite(actions)
            finite_action_dims = (
                actions_finite.all(dim=tuple(range(actions_finite.ndim - 1)))
                if actions_finite.ndim > 1
                else actions_finite
            )
            inputs["actions"] = torch.where(
                actions_finite, actions, torch.zeros_like(actions)
            )
            loss_mask = pad_to_dim(
                self._build_loss_mask(action_src_dim, data=data), self.action_dim, axis=-1, value=False
            )
            action_valid_mask = self._build_valid_mask(
                data, "action_valid_mask", self._missing_dims(action_src_dim)
            )
            if action_valid_mask is not None:
                action_valid_mask = action_valid_mask & finite_action_dims
                inputs["action_valid_mask"] = action_valid_mask
            elif not bool(finite_action_dims.all()):
                action_valid_mask = finite_action_dims
                inputs["action_valid_mask"] = action_valid_mask
            loss_mask = loss_mask & finite_action_dims
            if action_valid_mask is not None:
                loss_mask = loss_mask & action_valid_mask
            inputs["action_loss_mask"] = loss_mask

        passthrough_defaults(data, inputs)
        if "prompt" not in data and "task" in data:
            inputs["prompt"] = data["task"]

        return inputs


# ──────────────────────────────────────────────────────────────────────
# S0/S1 humanoid locomotion transform (MoCap-retargeted, no cameras).
# ──────────────────────────────────────────────────────────────────────
#
# Produces the same 50-dim WBC output layout as HumanoidS0S1InputsTransform,
# but:
#   - image_mask is False for every camera key, so the VLM skips the vision
#     encoder entirely for these samples (dummy black videos upstream are
#     still decoded by the dataloader but never embedded).
#   - action_loss_mask zeros out dims that are NOT recoverable from the
#     locomotion source:
#         * output [23, 24] neck_yaw / neck_pitch   (no neck channel in source)
#         * output [39]     left_thumb_yaw         (uncertain source mapping)
#         * output [47]     right_thumb_yaw        (uncertain source mapping)
#
# The transform is a thin subclass of HumanoidS0S1InputsTransform. All the
# reordering (_reorder) and the absolute/delta velocity handling are inherited
# unchanged.

# Output-dim indices in the 50-dim S0S1 layout that the locomotion dataset
# leaves unsupervised, near-constant, or closed-kinematic-coupled.
# Masking these prevents the FM expert from getting contradictory gradients
# between locomotion (coupled/empty) and humanoid_s0s1 (independent joints).
#
# Verified against source .npz files and parquet data:
#   - ankles (4, 5, 10, 11): source has ankle_pitch = -knee_pitch and
#     ankle_roll = +knee_pitch EXACTLY (|diff|=0). Mask so the FM expert
#     doesn't unlearn s0s1's independent ankle behaviour.
#   - elbows (16, 21): std ≈ 0.003, not exercised in MoCap retargeting.
#   - neck (23, 24): std = 0, source has no neck channel.
#   - thumb_yaw (39, 47): std = 0, uncertain source mapping.
#   - wrist_pitch (41, 49): std ≈ 0.004, closed-kinematic artifact.
# Output dims (50-dim WBC layout) NOT present in the locomotion MoCap source.
# These are always 0 in the converted data, so their loss should be masked out.
# With the corrected joint mapping, legs/arms/neck ARE real MoCap values.
# Only wrists, fingers, and neck_pitch are missing/constant in source.
_LOCO_UNMAPPED_DIMS: tuple[int, ...] = (
    24,  # neck_pitch              (constant -0.324 in source, std≈0)
    # All wrists and fingers (not in MoCap source — always 0)
    34,  # left_pinky
    35,  # left_ring
    36,  # left_middle
    37,  # left_index
    38,  # left_thumb_pitch
    39,  # left_thumb_yaw
    40,  # left_wrist_roll
    41,  # left_wrist_pitch
    42,  # right_pinky
    43,  # right_ring
    44,  # right_middle
    45,  # right_index
    46,  # right_thumb_pitch
    47,  # right_thumb_yaw
    48,  # right_wrist_roll
    49,  # right_wrist_pitch
)




# Module-level cache so the pool is loaded once per worker
_IMAGE_POOL_CACHE: dict[str, list] = {}






# ──────────────────────────────────────────────────────────────────────
# Navigation dataset transform (ego camera, 5fps, head cam only)
# ──────────────────────────────────────────────────────────────────────
#
# Same 50-dim WBC output as S0S1, but:
#   - Only head camera (wrist cameras get dummy black + mask=False)
#   - Wrists [25-28] and fingers [29-40] always zero in source → masked
#   - linear_z, angular_x, angular_y always zero → masked
#
# In the 50-dim output layout after reorder:
_NAV_UNMAPPED_DIMS: tuple[int, ...] = (
    # Wrists and fingers (not present in nav data — always 0)
    34,  # left_pinky
    35,  # left_ring
    36,  # left_middle
    37,  # left_index
    38,  # left_thumb_pitch
    39,  # left_thumb_yaw
    40,  # left_wrist_roll
    41,  # left_wrist_pitch
    42,  # right_pinky
    43,  # right_ring
    44,  # right_middle
    45,  # right_index
    46,  # right_thumb_pitch
    47,  # right_thumb_yaw
    48,  # right_wrist_roll
    49,  # right_wrist_pitch
    # Velocities that are always 0
    28,  # linear_vel_z
    29,  # angular_vel_x
    30,  # angular_vel_y
)

_NAV_WRIST_HAND_SLICE = slice(34, 50)


