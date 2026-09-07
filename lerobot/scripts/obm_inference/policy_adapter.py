"""Raw 51-column state adapter for the Green Challenge WebSocket server.

This file is the narrow compatibility layer between the simulator wire format
from simulation/aij/green-vla and the native whole-body contract implemented by
lerobot-fork commit 9027db256089d476a5498349489f8ec0e2b54602.
"""

from __future__ import annotations

import importlib.util
import logging
import json
import os
import types
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from scipy.interpolate import PchipInterpolator

from lerobot.common.datasets.torch_transforms import compose
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.greenvla_v1_1.identity import POLICY_TYPE, POLICY_ALIASES
from lerobot.common.utils.inference_transforms import (
    get_torch_input_transforms,
    get_torch_output_transforms,
)
from lerobot.common.utils.torch_observation import (
    move_dict_to_batch_for_inference,
    torch_preprocess_dict_inference,
)
from lerobot.configs.policies import PreTrainedConfig
from lerobot.common.robot_safety.runtime import RuntimeGMMGuard
from lerobot.common.datasets.s0s1_kinematics import convert_s0s1_kinematic_space
from lerobot.scripts.obm_inference.action_feed import (
    FeedConfig,
    NumericActionFeed,
)

LOGGER = logging.getLogger(__name__)
CONTROL_DT = 0.004
# Which data contract to serve. A checkpoint is trained under exactly one, and the
# contract carries the embodiment name that goes into the prompt, the statistics folder
# to normalize with, and the action sample step. `--data-config` selects it per run;
# this is only the default, so that a container started without the flag can still be
# pointed at another contract through the environment.
DATA_CONFIG = os.environ.get("S0S1_DATA_CONFIG", "track27")
CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "conf"
SOURCE_COMMIT = "ca8b8db031f4292977b28204cb2d5e005a2a78e9"
DEFAULT_ROOT_HEIGHT = 0.87
RAW_STATE_DIM = 51
MODEL_WBC_DIM = 50
MODEL_TO_RAW_51 = np.asarray(
    (
        *range(0, 25),
        43,
        45,
        46,
        47,
        48,
        49,
        50,
        41,
        42,
        29,
        30,
        31,
        32,
        33,
        34,
        25,
        26,
        35,
        36,
        37,
        38,
        39,
        40,
        27,
        28,
    ),
    dtype=np.int64,
)

# (scale, offset): simulator = scale * raw + offset.
# These are the established S0/S1 hardware/simulator thumb calibrations, moved
# from the old packed-50 indices to the native raw-51 indices.
_THUMB_HW = {
    33: (0.998, 0.0),
    34: (0.6464, 0.2596),
    39: (0.672, 0.046),
    40: (0.6014, 0.2781),
}


class _VectorizedInterpolateActions:
    """Numerically identical batched PCHIP for the output contract.

    The stock transform constructs one SciPy interpolator per joint (50 Python
    calls per policy request).  SciPy supports the joint dimension natively;
    evaluating all joints in one call produces bitwise-identical float32 output
    while avoiding that Python loop.
    """

    def __init__(self, *, sample_step: float, actions_type: str) -> None:
        if float(sample_step) <= 0.0:
            raise ValueError("sample_step must be positive")
        if actions_type not in {"absolute", "delta"}:
            raise ValueError(
                "Qwen3.5 vectorized output supports absolute/delta actions, "
                f"got {actions_type!r}"
            )
        self.sample_step = float(sample_step)
        self.actions_type = actions_type

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data
        state, actions = data["state"], data["actions"]
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if self.actions_type == "absolute":
            zero_action = np.expand_dims(state, axis=-2)
        else:
            zero_action = actions[..., :1, :]
        actions = np.concatenate([zero_action, actions], axis=-2)
        batch_mode = actions.ndim == 3
        if not batch_mode:
            actions = actions[None, ...]
        query_indices = np.arange(
            0,
            actions.shape[1] - 1 + 1e-6,
            self.sample_step,
        )
        interpolated = PchipInterpolator(
            np.arange(actions.shape[1]),
            actions,
            axis=1,
        )(query_indices)[:, 1:, :].astype(actions.dtype, copy=False)
        data["actions"] = interpolated if batch_mode else interpolated[0]
        return data


def _enabled(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) not in ("", "0", "false", "False")


def _zero_velocity_domain(domain: str) -> bool:
    """Whether the stationary runtime disables one velocity domain."""
    configured = {
        item.strip().lower()
        for item in os.environ.get("S0S1_ZERO_VEL_DIMS", "").split(",")
        if item.strip()
    }
    return domain.lower() in configured


def _vector(value, *, name: str, size: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def _roll_pitch_yaw(state: Mapping) -> tuple[float, float, float]:
    quaternion = state.get("root_quat")
    if quaternion is not None:
        w, x, y, z = _vector(quaternion, name="state.root_quat", size=4)
        roll = np.arctan2(
            2.0 * (w * x + y * z),
            1.0 - 2.0 * (x * x + y * y),
        )
        pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
        yaw = np.arctan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
        return float(roll), float(pitch), float(yaw)

    gravity = _vector(
        state.get("projected_gravity", (0.0, 0.0, -1.0)),
        name="state.projected_gravity",
        size=3,
    )
    roll = np.arctan2(gravity[1], -gravity[2])
    pitch = -np.arcsin(np.clip(gravity[0], -1.0, 1.0))
    yaw = float(state.get("root_yaw", 0.0))
    if not np.isfinite(yaw):
        raise ValueError("state.root_yaw must be finite")
    return float(roll), float(pitch), yaw


def _split_upper_body(state: Mapping) -> tuple[np.ndarray, ...]:
    torso = np.asarray(state.get("torso_joint_pos"), dtype=np.float32).reshape(-1)
    fingers = np.asarray(state.get("finger_joint_pos"), dtype=np.float32).reshape(-1)
    wrists_value = state.get("wrist_joint_pos")

    if wrists_value is not None:
        torso = _vector(torso, name="state.torso_joint_pos", size=13)
        fingers = _vector(fingers, name="state.finger_joint_pos", size=12)
        wrists = _vector(wrists_value, name="state.wrist_joint_pos", size=4)
        left_fingers, right_fingers = fingers[:6], fingers[6:]
        left_wrists, right_wrists = wrists[:2], wrists[2:]
    elif torso.shape == (17,) and fingers.shape == (12,):
        wrists = torso[13:].copy()
        torso = torso[:13].copy()
        left_fingers, right_fingers = fingers[:6], fingers[6:]
        left_wrists, right_wrists = wrists[:2], wrists[2:]
    elif torso.shape == (13,) and fingers.shape == (16,):
        # The challenge client interleaves six fingers, two wrists, six
        # fingers, two wrists in finger_joint_pos.
        left_fingers = fingers[0:6]
        left_wrists = fingers[6:8]
        right_fingers = fingers[8:14]
        right_wrists = fingers[14:16]
    else:
        raise ValueError(
            "state must provide torso=13/fingers=12/wrists=4, "
            "torso=17/fingers=12, or torso=13/interleaved-fingers=16"
        )

    for name, value in (
        ("torso", torso),
        ("left_fingers", left_fingers),
        ("right_fingers", right_fingers),
        ("left_wrists", left_wrists),
        ("right_wrists", right_wrists),
    ):
        if not np.isfinite(value).all():
            raise ValueError(f"state.{name} contains NaN or Inf")
    return torso, left_wrists, right_wrists, left_fingers, right_fingers


def build_raw_state(state) -> np.ndarray:
    """Convert a simulator observation into the native raw-51 order."""
    if not isinstance(state, Mapping):
        return _vector(state, name="state", size=RAW_STATE_DIM).copy()

    legs = _vector(state.get("legs_joint_pos"), name="state.legs_joint_pos", size=12)
    torso, left_wrists, right_wrists, left_fingers, right_fingers = (
        _split_upper_body(state)
    )
    velocity = (
        _vector(state["velocity"], name="state.velocity", size=6)
        if "velocity" in state
        else np.concatenate(
            [
                _vector(
                    state.get("base_lin_vel", (0.0, 0.0, 0.0)),
                    name="state.base_lin_vel",
                    size=3,
                ),
                _vector(
                    state.get("base_ang_vel", (0.0, 0.0, 0.0)),
                    name="state.base_ang_vel",
                    size=3,
                ),
            ]
        )
    )
    root_height = float(state.get("root_height", DEFAULT_ROOT_HEIGHT))
    if not np.isfinite(root_height):
        raise ValueError("state.root_height must be finite")
    roll, pitch, yaw = _roll_pitch_yaw(state)

    raw = np.zeros(RAW_STATE_DIM, dtype=np.float32)
    raw[0:12] = legs
    raw[12:25] = torso
    raw[25:27] = left_wrists
    raw[27:29] = right_wrists
    raw[29:35] = left_fingers
    raw[35:41] = right_fingers
    raw[41:45] = (roll, pitch, root_height, yaw)
    raw[45:51] = velocity

    if _enabled("S0S1_SWAP_WRISTS"):
        raw[[25, 26]] = raw[[26, 25]]
        raw[[27, 28]] = raw[[28, 27]]
    if _enabled("S0S1_INVERT_R_WRIST_ROLL"):
        raw[27] = -raw[27]
    if _enabled("S0S1_INVERT_R_THUMB_YAW"):
        raw[40] = -raw[40]
    if _enabled("S0S1_THUMB_HW_MAP"):
        for index, (scale, offset) in _THUMB_HW.items():
            raw[index] = (float(raw[index]) - offset) / scale

    return raw


def _hwc_uint8(image, *, name: str) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"{name} must be a 3D image, got {array.shape}")
    if array.shape[0] in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] not in (1, 3, 4):
        raise ValueError(f"{name} has unsupported channel count {array.shape[-1]}")
    if np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or Inf")
        scale = 255.0 if array.size and float(array.max()) <= 1.0 else 1.0
        array = np.clip(array * scale, 0.0, 255.0).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def prepare_images(images: Mapping, *, rotate: bool) -> dict[str, np.ndarray]:
    required = {
        "top_head": "head",
        "hand_left": "left wrist",
        "hand_right": "right wrist",
    }
    missing = sorted(set(required) - set(images))
    if missing:
        raise ValueError(f"Missing image keys: {missing}")
    prepared = {
        key: _hwc_uint8(images[key], name=name) for key, name in required.items()
    }
    if rotate:
        prepared["top_head"] = np.rot90(prepared["top_head"], k=-1).copy()
        prepared["hand_left"] = np.rot90(prepared["hand_left"], k=2).copy()
        prepared["hand_right"] = np.rot90(prepared["hand_right"], k=2).copy()

    quality = int(os.environ.get("S0S1_JPEG_Q", "0") or 0)
    if quality > 0:
        import io

        from PIL import Image

        for key in list(prepared):
            buffer = io.BytesIO()
            Image.fromarray(prepared[key]).save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            prepared[key] = np.asarray(Image.open(buffer).convert("RGB"))
    return prepared


_PROMPT_FROM_SUBTASK = os.environ.get("S0S1_PROMPT_FROM_SUBTASK", "") not in ("", "0", "false", "False")
_PROMPT_SWITCH_SOFT = os.environ.get("S0S1_PROMPT_SWITCH_SOFT", "") not in ("", "0", "false", "False")
_PROMPT_SCHEDULE = sorted(
    (float(entry["time"]), str(entry["instruction"]))
    for entry in json.loads(os.environ.get("S0S1_PROMPT_SCHEDULE_JSON", "") or "[]")
)


def _effective_prompt(observation, fallback: str) -> str:
    """Prompt the checkpoint is conditioned on for this request.

    By default it is whatever the client sent. S0S1_PROMPT_FROM_SUBTASK follows the
    benchmark instead: the client already reports its current subtask, so the
    evaluation can be driven by the simulator's own subtask monitor. A
    S0S1_PROMPT_SCHEDULE_JSON of {"time", "instruction"} entries overrides both from
    the episode clock, which is how the reference evaluation replays dataset timings.
    """
    prompt = fallback
    if _PROMPT_FROM_SUBTASK:
        subtask = observation.get("subtask")
        if subtask:
            prompt = str(subtask).strip()
    if _PROMPT_SCHEDULE and observation.get("t") is not None:
        clock = float(observation["t"])
        for at, instruction in _PROMPT_SCHEDULE:
            if clock >= at:
                prompt = instruction
    return prompt


class _ActionOverrides:
    """Optional, env-gated edits of the action the policy returns.

    They exist so that an evaluation can steer the robot without patching the
    simulator: the same run script then works against any checkpoint. Every knob
    defaults to "off", in which case the action is returned untouched.

      S0S1_TORSO_PITCH_OFFSET  rad added to the WBC pitch command (forward lean)
      S0S1_ROOT_HEIGHT_OFFSET  metres added to the WBC root height command
      S0S1_LEG_SQUAT_FRAC      blend of the leg targets toward a recorded squat
      S0S1_SQUAT_POSE_JSON     file with {"stand": [12], "squat": [12],
                               "root_stand": float, "root_squat": float}
      S0S1_ARM_OFFSETS         "index:rad,..." added to the 13 torso targets
      S0S1_FINGER_GAIN         scale of every finger target
      S0S1_THUMB_PITCH_MIN     minimum thumb pitch once that hand closes
    """

    _NUM_LEGS = slice(0, 12)
    _NUM_TORSO = slice(12, 25)
    _NUM_FINGERS = slice(25, 37)
    _NUM_ROOT_HEIGHT = 47
    _NUM_PITCH = 49
    _THUMB_PITCH = (4, 10)   # inside the 12 finger targets
    _PINKY = (0, 6)

    def __init__(self) -> None:
        self.pitch = float(os.environ.get("S0S1_TORSO_PITCH_OFFSET", "0") or 0)
        self.root_height = float(os.environ.get("S0S1_ROOT_HEIGHT_OFFSET", "0") or 0)
        self.finger_gain = float(os.environ.get("S0S1_FINGER_GAIN", "1") or 1)
        self.thumb_pitch_min = float(os.environ.get("S0S1_THUMB_PITCH_MIN", "0") or 0)
        self.squat_frac = float(os.environ.get("S0S1_LEG_SQUAT_FRAC", "0") or 0)
        self.arm_offsets: dict[int, float] = {}
        for item in (os.environ.get("S0S1_ARM_OFFSETS", "") or "").split(","):
            item = item.strip()
            if not item:
                continue
            index, _, value = item.partition(":")
            self.arm_offsets[int(index)] = float(value)
        self.squat_delta = [0.0] * 12
        self.squat_drop = 0.0
        if self.squat_frac:
            path = os.environ.get("S0S1_SQUAT_POSE_JSON", "")
            if not path:
                raise ValueError("S0S1_LEG_SQUAT_FRAC needs S0S1_SQUAT_POSE_JSON")
            with open(path) as handle:
                pose = json.load(handle)
            self.squat_delta = [b - a for a, b in zip(pose["stand"], pose["squat"])]
            self.squat_drop = float(pose["root_stand"]) - float(pose["root_squat"])

    @property
    def active(self) -> bool:
        return bool(
            self.pitch
            or self.root_height
            or self.squat_frac
            or self.arm_offsets
            or self.finger_gain != 1.0
            or self.thumb_pitch_min
        )

    def describe(self) -> str:
        return (
            "pitch=%+.3f root_height=%+.3f squat_frac=%.2f arm_offsets=%s "
            "finger_gain=%.2f thumb_pitch_min=%.2f"
            % (
                self.pitch,
                self.root_height,
                self.squat_frac,
                self.arm_offsets or "-",
                self.finger_gain,
                self.thumb_pitch_min,
            )
        )

    def apply_dict(self, action: dict) -> dict:
        if not self.active:
            return action
        legs = list(action["legs_joint_pos"])
        torso = list(action["torso_joint_pos"])
        fingers = list(action["finger_joint_pos"])
        base = dict(action["base_command"])
        if self.squat_frac:
            legs = [v + self.squat_frac * d for v, d in zip(legs, self.squat_delta)]
            base["root_height"] = float(base["root_height"]) - self.squat_frac * self.squat_drop
        for index, value in self.arm_offsets.items():
            torso[index] += value
        fingers = self._fingers(fingers)
        base["pitch"] = float(base["pitch"]) + self.pitch
        base["root_height"] = float(base["root_height"]) + self.root_height
        action = dict(action)
        action["legs_joint_pos"] = legs
        action["torso_joint_pos"] = torso
        action["finger_joint_pos"] = fingers
        action["base_command"] = base
        return action

    def apply_numeric(self, numeric):
        if not self.active:
            return numeric
        numeric = np.array(numeric, dtype=numeric.dtype, copy=True)
        if self.squat_frac:
            numeric[:, self._NUM_LEGS] += self.squat_frac * np.asarray(
                self.squat_delta, dtype=numeric.dtype
            )
            numeric[:, self._NUM_ROOT_HEIGHT] -= self.squat_frac * self.squat_drop
        for index, value in self.arm_offsets.items():
            numeric[:, self._NUM_TORSO.start + index] += value
        for row in range(numeric.shape[0]):
            numeric[row, self._NUM_FINGERS] = self._fingers(
                numeric[row, self._NUM_FINGERS].tolist()
            )
        numeric[:, self._NUM_PITCH] += self.pitch
        numeric[:, self._NUM_ROOT_HEIGHT] += self.root_height
        return numeric

    def _fingers(self, fingers: list) -> list:
        fingers = [float(v) for v in fingers]
        if self.thumb_pitch_min:
            for pinky, thumb in zip(self._PINKY, self._THUMB_PITCH):
                if fingers[pinky] > 0.5 and fingers[thumb] < self.thumb_pitch_min:
                    fingers[thumb] = self.thumb_pitch_min
        if self.finger_gain != 1.0:
            fingers = [v * self.finger_gain for v in fingers]
        return fingers


_ACTION_OVERRIDES: "_ActionOverrides | None" = None


def _action_overrides() -> "_ActionOverrides":
    global _ACTION_OVERRIDES
    if _ACTION_OVERRIDES is None:
        _ACTION_OVERRIDES = _ActionOverrides()
    return _ACTION_OVERRIDES


def model50_to_sim_dict(action) -> dict:
    """Map the checkpoint's audited 50D WBC order to the simulator wire order."""
    model = _vector(action, name="action", size=MODEL_WBC_DIM)
    raw = np.zeros(RAW_STATE_DIM, dtype=np.float32)
    raw[MODEL_TO_RAW_51] = model
    # Raw index 44 is global root yaw. It was deliberately dropped during
    # training, so keep it neutral; torso yaw remains the learned raw index 12.
    raw[44] = 0.0
    if _zero_velocity_domain("action"):
        # Stationary manipulation episodes mark every raw velocity channel
        # invalid. Never let unsupervised locomotion-head output reach WBC.
        raw[45:51] = 0.0

    if _enabled("S0S1_INVERT_R_THUMB_YAW"):
        raw[40] = -raw[40]
    if _enabled("S0S1_THUMB_HW_MAP"):
        for index, (scale, offset) in _THUMB_HW.items():
            raw[index] = scale * float(raw[index]) + offset

    if _enabled("S0S1_SWAP_WRISTS"):
        wrists = raw[[26, 25, 28, 27]].tolist()
        if _enabled("S0S1_INVERT_R_WRIST_ROLL"):
            wrists[3] = -wrists[3]
    else:
        wrists = raw[25:29].tolist()

    return _action_overrides().apply_dict({
        "legs_joint_pos": raw[0:12].tolist(),
        "torso_joint_pos": raw[12:25].tolist(),
        "finger_joint_pos": np.concatenate([raw[29:35], raw[35:41]]).tolist(),
        "wrist_joint_pos": wrists,
        "velocity": raw[45:51].tolist(),
        "base_command": {
            "root_height": float(raw[43]),
            "roll": float(raw[41]),
            "pitch": float(raw[42]),
            "yaw": float(raw[44]),
        },
    })


def model50_to_numeric_rows(actions) -> np.ndarray:
    """Vectorized equivalent of ``model50_to_sim_dict`` for the numeric feed."""
    model = np.asarray(actions, dtype=np.float32)
    if model.ndim != 2 or model.shape[1] != MODEL_WBC_DIM:
        raise ValueError(
            f"actions must have shape (N, {MODEL_WBC_DIM}), got {model.shape}"
        )
    if not np.isfinite(model).all():
        raise ValueError("actions contain NaN or Inf")
    raw = np.zeros((model.shape[0], RAW_STATE_DIM), dtype=np.float32)
    raw[:, MODEL_TO_RAW_51] = model
    raw[:, 44] = 0.0
    if _zero_velocity_domain("action"):
        raw[:, 45:51] = 0.0

    if _enabled("S0S1_INVERT_R_THUMB_YAW"):
        raw[:, 40] = -raw[:, 40]
    if _enabled("S0S1_THUMB_HW_MAP"):
        # Compute each calibration as Python float64 and assign to float32,
        # matching model50_to_sim_dict exactly rather than relying on NumPy's
        # scalar promotion rules.
        for index, (scale, offset) in _THUMB_HW.items():
            raw[:, index] = np.asarray(
                [scale * float(value) + offset for value in raw[:, index]],
                dtype=np.float32,
            )

    if _enabled("S0S1_SWAP_WRISTS"):
        wrists = raw[:, (26, 25, 28, 27)].copy()
        if _enabled("S0S1_INVERT_R_WRIST_ROLL"):
            wrists[:, 3] = -wrists[:, 3]
    else:
        wrists = raw[:, 25:29]

    numeric = np.concatenate(
        [
            raw[:, 0:12],
            raw[:, 12:25],
            raw[:, 29:41],
            wrists,
            raw[:, 45:51],
            raw[:, (43, 41, 42, 44)],
        ],
        axis=1,
    )
    if numeric.shape != (model.shape[0], RAW_STATE_DIM):
        raise RuntimeError(f"unexpected numeric feed shape {numeric.shape}")
    return _action_overrides().apply_numeric(numeric)


def available_data_configs() -> list[str]:
    """The data contracts this runtime ships, by `--data-config` name."""
    return sorted(path.stem for path in CONFIG_DIR.glob("*.yaml"))


def load_data_config(name: str):
    """Instantiate the data config factory named by `--data-config`.

    The name becomes a file path, so it is a bare stem or nothing: no directory
    separators, no traversal, no absolute paths. An unknown name is refused with the
    list of the ones that exist, because the alternative -- serving a checkpoint under
    a contract that does not match it -- produces plausible, wrong actions.
    """
    if not name or name != Path(name).name or name in (".", ".."):
        raise ValueError(
            f"--data-config must be one of the config names in {CONFIG_DIR}, "
            f"not a path: {name!r}"
        )
    config_path = CONFIG_DIR / f"{name}.yaml"
    if not config_path.is_file():
        raise ValueError(
            f"Unknown data config {name!r}. This runtime ships: "
            f"{', '.join(available_data_configs())}."
        )
    return instantiate(OmegaConf.load(config_path))


def resolve_attention_implementation(requested: str) -> str:
    """The attention kernel to actually run with, given what the checkpoint asks for.

    `attention_implementation` records the kernel the checkpoint was *trained* with, not
    a property of the weights: every implementation computes the same attention, they
    differ in how the reduction is tiled. The dtwin checkpoint asks for
    `flash_attention_2`, whose package is a long source build and is not in the image;
    transformers refuses to load rather than fall back. On CUDA, torch's own `sdpa`
    dispatches to a fused flash kernel anyway, so the fallback costs precision no one
    can measure and makes the checkpoint servable as shipped. Install `flash-attn` if
    you want the exact training kernel; the log says which one is in force either way.
    """
    if requested not in ("flash_attention_2", "flash_attention_3"):
        return requested
    package = "flash_attn" if requested == "flash_attention_2" else "flash_attn_3"
    if importlib.util.find_spec(package) is not None:
        return requested
    LOGGER.warning(
        "Checkpoint asks for attention_implementation=%s but %s is not installed; "
        "falling back to sdpa, which uses torch's own fused flash kernel on CUDA. "
        "Install %s to run the exact training kernel.",
        requested,
        package,
        package.replace("_", "-"),
    )
    return "sdpa"


class HumanoidPolicyAdapter:
    """GreenVLAv1.1 behind the Green Challenge WebSocket protocol."""

    def __init__(
        self,
        checkpoint: str,
        *,
        data_config: str = DATA_CONFIG,
        device: str = "cuda:0",
        rotate_images: bool = True,
        feed: FeedConfig | None = None,
        control_dt: float = CONTROL_DT,
        compile_sample_actions: bool = False,
        inference_steps: int | None = None,
    ) -> None:
        # The data contract first: it names the statistics folder to look for inside the
        # checkpoint, so it has to be resolved before the checkpoint is checked. One
        # object answers both questions from here on -- the folder that is verified to
        # exist at startup and the folder the statistics are then loaded from cannot
        # drift apart, which is what an independently configured asset id allowed.
        data_factory = load_data_config(data_config)
        norm_asset_id = getattr(data_factory, "asset_id", None)
        if not norm_asset_id:
            raise ValueError(f"Data config {data_config!r} defines no asset_id")
        self.device = device
        self.data_config = data_config
        self.norm_asset_id = norm_asset_id
        # The handshake publishes this and a client multiplies the action horizon by it
        # to know how many rows a chunk carries, so a fractional step would advertise a
        # row count the output transforms never produce.
        sample_step = float(data_factory.action_sample_step)
        if sample_step < 1 or sample_step != int(sample_step):
            raise ValueError(
                f"Data config {data_config!r} has action_sample_step={sample_step!r}; "
                "it must be a positive whole number of rows"
            )
        self.action_sample_step = int(sample_step)
        self.rotate_images = rotate_images
        self.task_description: str | None = None
        if control_dt <= 0.0:
            raise ValueError(f"control_dt must be positive, got {control_dt!r}")
        self.control_dt = float(control_dt)
        self._feed = (
            None
            if feed is None
            else NumericActionFeed(feed, self.control_dt)
        )
        self._feed_awaiting_seed = True
        self._feed_last_clock = 0.0
        self._cognition_cache_calls = int(
            os.environ.get("QWEN35_COGNITION_CACHE_CALLS", "1") or 1
        )
        if self._cognition_cache_calls < 1:
            raise ValueError("QWEN35_COGNITION_CACHE_CALLS must be at least one")
        self._cognition_cache_value = None
        self._cognition_cache_age = 0
        self._cognition_cache_hits = 0
        self._cognition_cache_misses = 0
        self._last_cognition_prompt: str | None = None
        self._last_prompt_in_force: str | None = None
        # When visual cognition is cached, the model intentionally ignores the
        # new camera tensors until the next refresh.  Keep the matching
        # preprocessed batch as well so cache-hit calls only transform and copy
        # the current proprioceptive state instead of pointlessly resizing
        # three images and tokenizing the same prompt again.
        self._cached_prepared_images: dict[str, np.ndarray] | None = None
        self._cached_full_batch: dict | None = None
        self._state_input_transforms = None
        self._last_policy_seed: int | None = None
        # S0/S1 GMM state guard, off unless S0S1_GMM_GUARD_MODE says otherwise.
        self._gmm_guard = RuntimeGMMGuard.from_env(LOGGER)

        checkpoint_path = Path(checkpoint)
        pretrained_model = checkpoint_path / "pretrained_model"
        assets_path = checkpoint_path / "norm_stats"
        for required in (
            pretrained_model / "config.json",
            pretrained_model / "model.safetensors",
            assets_path / norm_asset_id / "norm_stats.json",
        ):
            if not required.is_file() or required.stat().st_size == 0:
                raise FileNotFoundError(f"Missing checkpoint component: {required}")

        cfg = PreTrainedConfig.from_pretrained(pretrained_model)
        if cfg.type not in POLICY_ALIASES:
            raise ValueError(f"Expected {POLICY_TYPE}, got {cfg.type!r}")
        if cfg.max_state_dim != MODEL_WBC_DIM or cfg.max_action_dim != MODEL_WBC_DIM:
            raise ValueError(
                f"Expected 50D WBC checkpoint, got state={cfg.max_state_dim} "
                f"action={cfg.max_action_dim}"
            )
        checkpoint_inference_steps = int(cfg.num_steps)
        self.checkpoint_inference_steps = checkpoint_inference_steps
        if inference_steps is not None:
            if not 1 <= int(inference_steps) <= checkpoint_inference_steps:
                raise ValueError(
                    "inference_steps must be in [1,%d], got %r"
                    % (checkpoint_inference_steps, inference_steps)
                )
            cfg.num_steps = int(inference_steps)
        flow_steps_cycle_text = os.environ.get(
            "QWEN35_FLOW_STEPS_CYCLE", ""
        ).strip()
        if flow_steps_cycle_text:
            try:
                flow_steps_cycle = tuple(
                    int(item.strip())
                    for item in flow_steps_cycle_text.split(",")
                    if item.strip()
                )
            except ValueError as error:
                raise ValueError(
                    "QWEN35_FLOW_STEPS_CYCLE must be comma-separated integers"
                ) from error
            if not flow_steps_cycle:
                raise ValueError("QWEN35_FLOW_STEPS_CYCLE must not be empty")
            invalid_steps = [
                item
                for item in flow_steps_cycle
                if not 1 <= item <= checkpoint_inference_steps
            ]
            if invalid_steps:
                raise ValueError(
                    "QWEN35_FLOW_STEPS_CYCLE entries must be in [1,%d], got %r"
                    % (checkpoint_inference_steps, invalid_steps)
                )
        else:
            flow_steps_cycle = (int(cfg.num_steps),)
        self._flow_steps_cycle = flow_steps_cycle
        self._flow_steps_call_index = 0
        self._flow_steps_last = int(flow_steps_cycle[0])
        self._flow_steps_counts = {
            item: 0 for item in sorted(set(flow_steps_cycle))
        }
        cfg.pretrained_path = pretrained_model
        cfg.device = device
        cfg.compile_sample_actions = bool(compile_sample_actions)
        cfg.attention_implementation = resolve_attention_implementation(
            str(cfg.attention_implementation)
        )
        # Match the proven Qwen3-VL serving contract even when torch.compile is
        # disabled.  Qwen3.5 compile is unstable in this runtime, while TF32 is
        # an independent CUDA matmul setting.  It changes only the execution
        # kernels used for float32 matrix products; checkpoint weights, flow
        # integration count and the action contract stay intact.
        torch.set_float32_matmul_precision("high")
        self.policy = make_policy(cfg)
        self.policy.to(device)
        self.policy.eval()
        if self._cognition_cache_calls > 1:
            model = self.policy.model
            if str(model.config.cognition_mode) != "frozen_vlm_context_queries":
                raise ValueError(
                    "cognition caching requires frozen_vlm_context_queries mode"
                )
            if bool(getattr(model.config, "add_state_vlm", False)):
                raise ValueError("cognition caching requires state-free VLM cognition")
            uncached_encode = model.encode_cognition_tokens

            def cached_encode(_model, *args, **kwargs):
                if (
                    self._cognition_cache_value is None
                    or self._cognition_cache_age + 1 >= self._cognition_cache_calls
                ):
                    value = uncached_encode(*args, **kwargs)
                    self._cognition_cache_value = value.detach()
                    self._cognition_cache_age = 0
                    self._cognition_cache_misses += 1
                    return value
                self._cognition_cache_age += 1
                self._cognition_cache_hits += 1
                return self._cognition_cache_value

            model.encode_cognition_tokens = types.MethodType(cached_encode, model)

        data_cfg = data_factory.create(assets_path, cfg)
        if data_cfg.state_dim != MODEL_WBC_DIM:
            raise ValueError(
                f"Data config is not 50D WBC: state_dim={data_cfg.state_dim}"
            )

        map_to_unified_space = bool(getattr(cfg, "map_to_unified_space", False))
        input_transform_sequence = tuple(
            get_torch_input_transforms(
                policy_config=cfg,
                data_config_factory=data_factory,
                assets_dirs=assets_path,
                normalization_mode=cfg.normalization_mode,
                map_to_unified_space=map_to_unified_space,
            )
        )
        self.input_transforms = compose(input_transform_sequence)
        if self._cognition_cache_calls > 1:
            resize_indices = [
                index
                for index, transform in enumerate(input_transform_sequence)
                if type(transform).__name__ == "ResizeImagesTorch"
            ]
            if len(resize_indices) != 1:
                raise RuntimeError(
                    "Expected exactly one ResizeImagesTorch transform for the "
                    f"Qwen3.5 fast path, got {resize_indices}"
                )
            # Everything before image resize is the audited raw simulator state
            # reorder/kinematic conversion/normalization path.  Running this
            # prefix on every call preserves live state conditioning exactly.
            self._state_input_transforms = compose(
                input_transform_sequence[: resize_indices[0]]
            )
        output_transform_sequence = tuple(
            get_torch_output_transforms(
                policy_config=cfg,
                data_config_factory=data_factory,
                assets_dirs=assets_path,
                normalization_mode=cfg.normalization_mode,
                map_to_unified_space=map_to_unified_space,
            )
        )
        interpolation_indices = [
            index
            for index, transform in enumerate(output_transform_sequence)
            if type(transform).__name__ == "InterpolateActions"
        ]
        if len(interpolation_indices) != 1:
            raise RuntimeError(
                "Expected exactly one InterpolateActions transform for the "
                f"output contract, got {interpolation_indices}"
            )
        interpolation_index = interpolation_indices[0]
        interpolation = output_transform_sequence[interpolation_index]
        vectorized_interpolation = _VectorizedInterpolateActions(
            sample_step=float(interpolation.sample_step),
            actions_type=str(interpolation.actions_type),
        )
        output_transform_sequence = (
            output_transform_sequence[:interpolation_index]
            + (vectorized_interpolation,)
            + output_transform_sequence[interpolation_index + 1 :]
        )
        self.output_transforms = compose(output_transform_sequence)
        LOGGER.info(
            "Loaded model=%s family=GreenVLAv1.1 checkpoint=%s commit=%s "
            "data_config=%s state_dim=50 action_dim=50 horizon=%d device=%s",
            type(self.policy).__name__,
            checkpoint_path,
            SOURCE_COMMIT,
            data_config,
            cfg.n_action_steps,
            device,
        )
        LOGGER.info(
            "Contract=raw51 -> audited 50D closed-WBC reorder "
            "embodiment=%s norm_stats=%s action_sample_step=%d "
            "attention_implementation=%s",
            data_factory.name,
            assets_path / norm_asset_id / "norm_stats.json",
            self.action_sample_step,
            cfg.attention_implementation,
        )
        LOGGER.info(
            "Simulator mapping rotate_images=%s wrist_swap=%s thumb_hw=%s "
            "jpeg_q=%s feed_by_server=%s",
            rotate_images,
            _enabled("S0S1_SWAP_WRISTS"),
            _enabled("S0S1_THUMB_HW_MAP"),
            os.environ.get("S0S1_JPEG_Q", "0"),
            self._feed is not None,
        )
        LOGGER.info(
            "Inference optimization compile_sample_actions=%s matmul_precision=%s",
            bool(compile_sample_actions),
            torch.get_float32_matmul_precision(),
        )
        LOGGER.info(
            "Flow integration steps runtime=%d cycle=%s checkpoint=%d",
            int(cfg.num_steps),
            ",".join(str(item) for item in self._flow_steps_cycle),
            checkpoint_inference_steps,
        )
        LOGGER.info(
            "Visual cognition refresh every %d policy call(s)",
            self._cognition_cache_calls,
        )
        if self._state_input_transforms is not None:
            LOGGER.info(
                "Cache-hit preprocessing fast path enabled: live state every call, "
                "camera resize/tokenization only on cognition refresh"
            )
        LOGGER.info(
            "Vectorized PCHIP output enabled: sample_step=%s actions_type=%s",
            vectorized_interpolation.sample_step,
            vectorized_interpolation.actions_type,
        )
        overrides = _action_overrides()
        if overrides.active:
            LOGGER.info("action overrides: %s", overrides.describe())
        if _PROMPT_FROM_SUBTASK or _PROMPT_SCHEDULE:
            LOGGER.info(
                "prompt source: from_subtask=%s schedule=%d entries soft_switch=%s",
                _PROMPT_FROM_SUBTASK,
                len(_PROMPT_SCHEDULE),
                _PROMPT_SWITCH_SOFT,
            )

    @property
    def metadata(self) -> dict:
        metadata = {
            "policy": POLICY_TYPE,
            "model_family": "GreenVLAv1.1",
            "robot": self.data_config,
            "source_commit": SOURCE_COMMIT,
            "raw_state_dim": RAW_STATE_DIM,
            "state_dim": MODEL_WBC_DIM,
            "action_dim": MODEL_WBC_DIM,
            "action_horizon": self.policy.config.n_action_steps,
            # Both of these come from the data contract that is actually serving, so the
            # handshake cannot advertise a row count the output transforms do not produce.
            "action_sample_step": self.action_sample_step,
            "norm_asset_id": self.norm_asset_id,
            "feed_by_server": self._feed is not None,
            "control_dt": self.control_dt,
            "compile_sample_actions": bool(self.policy.config.compile_sample_actions),
            "inference_steps": int(self.policy.config.num_steps),
            "flow_steps_cycle": list(self._flow_steps_cycle),
            "flow_steps_last": self._flow_steps_last,
            "flow_steps_counts": {
                str(key): value for key, value in self._flow_steps_counts.items()
            },
            "checkpoint_inference_steps": self.checkpoint_inference_steps,
            "cognition_cache_calls": self._cognition_cache_calls,
            "cognition_cache_hits": self._cognition_cache_hits,
            "cognition_cache_misses": self._cognition_cache_misses,
            "last_policy_seed": self._last_policy_seed,
            "gmm_guard_mode": self._gmm_guard.mode,
        }
        return metadata

    def reset(self, task_description: str | None = None) -> None:
        self.task_description = task_description
        self._feed_awaiting_seed = True
        self._cognition_cache_value = None
        self._cognition_cache_age = 0
        self._last_cognition_prompt = None
        self._last_prompt_in_force = None
        self._cached_prepared_images = None
        self._cached_full_batch = None
        self._flow_steps_call_index = 0
        self._flow_steps_last = int(self._flow_steps_cycle[0])
        self._gmm_guard.reset()

    def _raw_observation(
        self, observation: Mapping, *, reuse_cached_images: bool = False
    ) -> dict:
        # Use the same effective instruction for tokenization and feed/cache resets.
        # Previously step() followed the native subtask but this path still tokenized
        # the top-level task. This is the former serve_checkpoint.py routing fix.
        prompt = _effective_prompt(
            observation,
            str(observation.get("prompt") or self.task_description or "").strip(),
        ).strip()
        if not prompt:
            raise ValueError("A non-empty prompt is required")
        if reuse_cached_images:
            if self._cached_prepared_images is None:
                raise RuntimeError("No cached images are available for a cache-hit call")
            images = self._cached_prepared_images
        else:
            images = prepare_images(
                observation.get("images", {}), rotate=self.rotate_images
            )
            self._cached_prepared_images = images
        raw_state = build_raw_state(observation.get("state"))
        state_valid_mask = np.ones(RAW_STATE_DIM, dtype=bool)
        if _zero_velocity_domain("state"):
            # The stationary-manipulation episodes used for these
            # prompts carry state_valid_mask=False at raw indices 45:51. The
            # previous runtime advertised all six channels as valid even though
            # four have exactly zero training std; simulator noise was therefore
            # normalized by 1e-6 and entered the model at |z| up to 3.3e4.
            raw_state = raw_state.copy()
            raw_state[45:51] = 0.0
            state_valid_mask[45:51] = False
        raw = {
            "observation/state": raw_state,
            "state_valid_mask": state_valid_mask,
            "observation/head_image": images["top_head"],
            "observation/left_wrist_image": images["hand_left"],
            "observation/right_wrist_image": images["hand_right"],
            "prompt": prompt,
        }
        return raw

    # Qwen3.5's fused causal-conv kernel internally uses an autograd Function
    # even during inference.  no_grad keeps gradients disabled without
    # creating inference tensors that the fused kernel is unable to save.
    @torch.no_grad()
    def step(self, observation: Mapping) -> dict:
        if observation.get("reset", False):
            self.reset(observation.get("prompt"))
            policy_seed = observation.get("policy_seed")
            if policy_seed is not None:
                policy_seed = int(policy_seed)
                if not 0 <= policy_seed < 2**63:
                    raise ValueError("policy_seed must be in [0, 2**63)")
                torch.manual_seed(policy_seed)
                torch.cuda.manual_seed_all(policy_seed)
                self._last_policy_seed = policy_seed

        prompt = _effective_prompt(
            observation,
            str(observation.get("prompt") or self.task_description or "").strip(),
        ).strip()
        if not prompt:
            raise ValueError("A non-empty prompt is required")
        if prompt != self._last_prompt_in_force:
            if self._last_prompt_in_force is not None:
                LOGGER.info(
                    "instruction boundary at t=%s: %r -> %r%s",
                    observation.get("t"),
                    self._last_prompt_in_force,
                    prompt,
                    "" if _PROMPT_SWITCH_SOFT else " (feed reseeded)",
                )
                if not _PROMPT_SWITCH_SOFT and self._feed is not None:
                    self._feed_awaiting_seed = True
            self._last_prompt_in_force = prompt
        if prompt != self._last_cognition_prompt:
            self._cognition_cache_value = None
            self._cognition_cache_age = 0
            self._cached_prepared_images = None
            self._cached_full_batch = None
            self._last_cognition_prompt = prompt
        cognition_refresh = (
            self._cognition_cache_calls <= 1
            or self._cognition_cache_value is None
            or self._cognition_cache_age + 1 >= self._cognition_cache_calls
        )
        raw = self._raw_observation(
            observation, reuse_cached_images=not cognition_refresh
        )
        if cognition_refresh:
            transformed = self.input_transforms(raw)
            preprocessed = torch_preprocess_dict_inference(transformed)
            batch = move_dict_to_batch_for_inference(
                preprocessed, device=self.device
            )
            if self._cognition_cache_calls > 1:
                self._cached_full_batch = batch
        else:
            if self._state_input_transforms is None or self._cached_full_batch is None:
                raise RuntimeError("Incomplete Qwen3.5 cache-hit preprocessing state")
            current = self._state_input_transforms(raw)
            batch = dict(self._cached_full_batch)
            for key in (
                "state",
                "state_history",
                "state_history_is_pad",
                "state_valid_mask",
            ):
                if key not in current:
                    continue
                value = current[key]
                if isinstance(value, np.ndarray):
                    value = torch.from_numpy(value)
                if not isinstance(value, torch.Tensor):
                    raise TypeError(
                        f"Cache-hit field {key!r} must be a tensor, got {type(value)}"
                    )
                if key in ("state", "state_history"):
                    value = value.float()
                batch[key] = value.unsqueeze(0).to(self.device)
        flow_steps = int(
            self._flow_steps_cycle[
                self._flow_steps_call_index % len(self._flow_steps_cycle)
            ]
        )
        # A two-step plan on each visual-cognition refresh anchors the three
        # intervening one-step plans.  The server-side feed blends four live
        # plans, so this general schedule preserves a high-quality plan without
        # any task, phase, or simulator-state branching.
        self.policy.config.num_steps = flow_steps
        self.policy.model.config.num_steps = flow_steps
        raw_chunk = self.policy.select_action(batch)
        self._flow_steps_call_index += 1
        self._flow_steps_last = flow_steps
        self._flow_steps_counts[flow_steps] += 1
        if not torch.isfinite(raw_chunk).all():
            raise RuntimeError("Policy returned NaN or Inf")

        denormalized = self.output_transforms(
            {
                "actions": raw_chunk.detach().cpu().numpy(),
                "state": batch["state"].detach().cpu().numpy(),
            }
        )["actions"]
        if denormalized.ndim == 3:
            denormalized = denormalized[0]
        if denormalized.ndim != 2 or denormalized.shape[1] != MODEL_WBC_DIM:
            raise RuntimeError(
                f"Unexpected action shape after output transforms: "
                f"{denormalized.shape}, expected (*, {MODEL_WBC_DIM})"
            )
        if not np.isfinite(denormalized).all():
            raise RuntimeError("Output transforms returned NaN or Inf")
        gmm_summary = None
        if self._gmm_guard.mode != "off":
            # Same 50-D absolute layout the guard was fitted on; the current state in that
            # layout gives the independent state score the diagnostics report.
            # The guard was fitted in the model's closed-kinematic space (ankles converted
            # open -> closed by the training input transform); the wire state is open.
            closed51 = np.asarray(
                convert_s0s1_kinematic_space(
                    np.asarray(raw["observation/state"], dtype=np.float32).copy(), "open", "closed"
                ),
                dtype=np.float32,
            )
            current_m50 = np.ascontiguousarray(closed51[MODEL_TO_RAW_51], dtype=np.float32)
            guarded, _ = self._gmm_guard.apply(
                np.asarray(denormalized, dtype=np.float32), current_state=current_m50
            )
            denormalized = np.asarray(guarded, dtype=np.float32)
            gmm_summary = dict(self._gmm_guard.last_summary)
        if self._feed is None:
            rows = [model50_to_sim_dict(action) for action in denormalized]
            response = {"actions_list": rows}
            if gmm_summary is not None:
                response["gmm"] = gmm_summary
            return response

        if "t" not in observation:
            raise ValueError("Server-side feed requires simulator model time t")
        clock = float(observation["t"])
        if not self._feed_awaiting_seed and clock < self._feed_last_clock - 1e-9:
            LOGGER.warning(
                "Episode clock moved backwards %.3f -> %.3f; resetting feed",
                self._feed_last_clock,
                clock,
            )
            self._feed_awaiting_seed = True
        if self._feed_awaiting_seed:
            initial_model = raw["observation/state"][MODEL_TO_RAW_51]
            self._feed.reset(model50_to_sim_dict(initial_model))
            self._feed_awaiting_seed = False
        self._feed_last_clock = clock
        numeric_rows = model50_to_numeric_rows(denormalized)
        response = {"actions_list": self._feed.update_numeric(clock, numeric_rows)}
        if gmm_summary is not None:
            response["gmm"] = gmm_summary
        return response
