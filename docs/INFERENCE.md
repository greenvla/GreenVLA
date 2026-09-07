# Inference contract

Use [the launcher](../README.md) or [the infrastructure container](INFRA.md).
Calling `serve_policy` directly bypasses the required simulator wire conversion
and is not equivalent to the packaged endpoint.

## Default configuration

| Setting | Packaged default |
|---|---|
| Profile / data config | `014814` / `scripted_014814` |
| Embodiment in the prompt | `scripted_vla_s0s1_step1_state_action_subtasks_shared` |
| Normalization | `norm_stats/scripted_vla/norm_stats.json` in the checkpoint |
| Instruction | Nonempty simulator `subtask`; otherwise `prompt` / model fallback |
| Adapter / GMM / neck compensation | Disabled |
| Action horizon / sample step | 50 / 1 for the supported profiles |
| Action feed | Enabled: scheduling, blending and limiting remain active |
| Plan row interval / reply period / control tick | `1/30` s / `0.080` s / `0.004` s |
| Reply | 20 commands per request, not 20 independent model predictions |
| Warmup | 2 calls |

Flow integration steps come from the checkpoint. Native subtask switching is
driven by the simulator, not a timer. Do not set `S0S1_PROMPT_SCHEDULE_JSON`
when using native subtasks: an explicit schedule overrides the selected instruction.

Profile `017799` selects config `dtwin_017799`, embodiment
`dtwin_v102_filtered_s0s1_step1_state_action_episode_task`, and statistics under
`norm_stats/dtwin_v102/`. A profile must match the checkpoint; finding a statistics
directory alone does not prove semantic compatibility.

The low-level Python service has different defaults (`track27`, feed disabled,
`row_dt=0.025`). Do not copy those defaults into the packaged deployment.

## Checkpoint requirements

The server supports GreenVLAv1.1 models with 50 state and 50 action channels.
Provide the model weights, saved policy configuration and training normalization
statistics in this layout:

```text
checkpoint/
  pretrained_model/
    config.json
    model.safetensors
  norm_stats/
    <asset_id>/
      norm_stats.json
```

The policy configuration must describe the trained architecture and reference
the correct backbone assets. Accepted configuration identities include
`GreenVLAv1.1`, `greenvla_v1_1` and `qwen3vlpolicy`. These are aliases for the
same implementation, not interchangeable model architectures.

For a checkpoint matching a supplied profile, pass its local directory to the
launcher; the directory name does not need to be a checkpoint step number.
For a different training contract, add a YAML data config under `lerobot/conf/`,
rebuild the Docker image and pass `--data-config <config-name>` after the image
name in the [Docker command](INFRA.md#direct-docker-launch). The configuration
name is the YAML filename without its extension.

Match these settings to training:

- Embodiment `name`, included in the tokenized instruction.
- `asset_id`, selecting the normalization directory.
- State/action ordering, delta mask, kinematic transforms and validity masks.
- `action_horizon`, `action_sample_step` and instruction granularity.
- Camera orientation and the feed's output row interval (`--feed.row-dt`).

Select a transform implementation matching the trained representation; changing
the YAML name alone does not convert between action spaces. The supplied stationary
profiles are not suitable for a checkpoint that needs nonzero base velocities.
For global-instruction training, set `S0S1_PROMPT_FROM_SUBTASK=0` on the Docker
container and supply `prompt`; for subtask training, use the default native
`subtask` field. Provide all backbone assets in the HF cache for offline use.

Check the startup logs and handshake before applying commands. The server checks
required files and dimensions, but cannot prove that your training semantics match
the simulator. Evaluate each custom checkpoint in the intended environment.

## Simulator connection

Start the simulator separately with `ARM_FF=0 PROPRIO_CMD=all`. Verify these values
reach its actual process; setting them on the VLA container cannot configure another
container. This repository does not change the simulator's `make up` defaults.

Connect to the public address printed by the launcher (8999 for direct Docker),
not the internal policy port. Health check: `GET /healthz`. The protocol is
WebSocket + msgpack using
[`msgpack_numpy.py`](../lerobot/scripts/obm_inference/msgpack_numpy.py).
The server sends metadata on connection; requests are `{"obs": observation}`.

Use a **state dictionary**, not a packed vector, at the public wire boundary:

```python
observation = {
    "state": {
        "legs_joint_pos": legs12,
        "torso_joint_pos": torso13,
        "finger_joint_pos": fingers_and_wrists16,
        "base_lin_vel": linear_velocity3,
        "base_ang_vel": angular_velocity3,
        "projected_gravity": gravity3,
        "root_height": height_scalar,
        "root_quat": quaternion_wxyz4,
    },
    "images": {"top_head": head_rgb, "hand_left": left_rgb, "hand_right": right_rgb},
    "subtask": "Raise your left hand",  # current instruction from the simulator
    "prompt": "Take the plate",        # fallback if subtask is empty
    "reset": True,                     # first request of each episode
    "t": 0.0,                          # simulator time in seconds
}
```

Images are RGB arrays, CHW or HWC. The policy rotates the head camera 90 degrees
clockwise and wrist cameras 180 degrees, then resizes to 448×448. Do not rotate
them twice. `S0S1_SWAP_WRISTS=1` maps wrist state/action ordering; thumb hardware
mapping and JPEG re-encoding are not enabled by the launcher.

Each `actions_list` command contains legs (12), torso (13), fingers (12), wrists
(4), velocity (6), and `base_command` (height, roll, pitch). For the default launch,
check handshake fields `robot=scripted_014814`, `action_sample_step=1`,
`feed_by_server=true`, `control_dt=0.004`, `gmm_guard_mode=off`,
`joint_tracking_calibration.enabled=false`, and
`wire_boundary.empirical_neck_calibration=false`.

## State/action conversion

| Model channels | Contents | Action interpretation |
|---|---|---|
| `0:25` | Legs, torso, arms, neck | Delta from the input state |
| `25:34` | Root height, base twist, roll, pitch | Absolute |
| `34:50` | Fingers and wrists | Delta from the input state |

The policy unnormalizes predictions and adds the input state to delta channels.
The supported stationary profiles mask input base velocities and zero output
base twist (`26:32` in the model layout). Joint targets and root height/roll/pitch
are retained. Use a different data contract for locomotion checkpoints.

The feed schedules, blends and limits targets. The outer wire boundary subtracts
robot finger defaults for the simulator's relative finger action terms. With
`PROPRIO_CMD=all`, it decodes echoed commands back to model coordinates and restores
previously issued body intent, including legs where the simulator echoes idle-leg
commands. Reset/time rewind clears history. Measured root attitude is not replaced.

These conversions are part of raw VLA. With corrections disabled, body, neck,
wrist and base commands pass through the outer boundary unchanged.

## Optional corrections

`--adapter v1` enables packaged joint tracking weights; `--neck-compensation`
independently enables a calibrated neck offset. Both are off by default and
supported by the launcher and Docker entrypoint.

For custom direct-container deployments, GMM settings are defined in
[`runtime.py`](../lerobot/common/robot_safety/runtime.py):

- `S0S1_GMM_GUARD_MODE`: `off`, `monitor`, or `correct`.
- `S0S1_GMM_GUARD_PATH`: model file required in either enabled mode.
- `S0S1_GMM_GRADIENT_STEP`, `S0S1_GMM_MAX_CORRECTION`.
- `S0S1_GMM_RAMP_ROWS`, `S0S1_GMM_HARD_Z_LIMIT`, `S0S1_GMM_LOG_EVERY`.
- `S0S1_GMM_CORRECT_FINGERS`: opt in to finger correction.

The convenience launcher explicitly starts GMM off. Monitoring computes density
scores and has overhead; it is not a zero-cost mode.

## Code and dependencies

The model implementation is `lerobot/common/policies/greenvla_v1_1/`.
Configuration aliases are listed under [checkpoint requirements](#checkpoint-requirements).

Dependencies and CUDA kernel groups are declared in `pyproject.toml` and installed
by `docker/Dockerfile`. Use that build rather than a separate manually pinned
recipe. See [offline requirements](INFRA.md#hugging-face-and-offline-startup).
