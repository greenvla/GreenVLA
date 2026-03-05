"""
Example: single-step inference with GreenVLA on the CALVIN embodiment.

This script demonstrates the minimal end-to-end pipeline:
    1. Load a pretrained policy and its pre-/post-processing transforms.
    2. Prepare a raw observation dict (state + images + language prompt).
    3. Run the model to predict actions.
    4. Post-process (apply output transforms) the actions to obtain real-world commands.

Usage:
    python examples/example_inference_calvin.py
    python examples/example_inference_calvin.py --model /path/to/local/checkpoint
    python examples/example_inference_calvin.py --device cpu --prompt "open the drawer"

Notes:
    - The CALVIN environment uses a 7-dim proprioceptive state:
        [x, y, z, roll, pitch, yaw, gripper_width]
    - Predicted actions are 7-dim relative actions:
        [dx, dy, dz, droll, dpitch, dyaw, gripper_action]
    - Two camera views are used:
        - ``observation/rgb_static``:  static third-person camera (448×448).
        - ``observation/rgb_gripper``: wrist-mounted gripper camera (224×224).
      Images are expected in HWC uint8 format at any resolution; the
      transforms handle resizing internally.
"""

import argparse

import numpy as np
import torch
from rich.console import Console

from lerobot.common.policies.factory import load_pretrained_policy
from lerobot.common.utils.torch_observation import (
    move_dict_to_batch_for_inference,
    torch_preprocess_dict_inference,
)

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run single-step inference with GreenVLA on CALVIN.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="SberRoboticsCenter/GreenVLA-5b-stride-1-R2-calvin",
        help="HuggingFace Hub model ID or local checkpoint path.",
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default="calvin",
        help="Name of the data config for transforms (default: calvin).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (default: cuda if available).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="open the drawer",
        help="Natural-language task instruction.",
    )
    return parser.parse_args()


def build_dummy_observation(prompt: str) -> dict:
    """Construct a dummy observation mimicking a real CALVIN environment input.

    In a real deployment this would come from the simulator or robot:
      - ``observation/state``:       proprioceptive reading (7-dim float32):
            [x, y, z, roll, pitch, yaw, gripper_width]
      - ``observation/rgb_static``:  static third-person camera (H, W, 3) uint8.
      - ``observation/rgb_gripper``: wrist-mounted gripper camera (H, W, 3) uint8.
      - ``prompt``:                  natural-language task instruction.
    """
    return {
        "observation/state": np.random.rand(7).astype(np.float32),
        "observation/rgb_static": np.random.randint(256, size=(448, 448, 3), dtype=np.uint8),
        "observation/rgb_gripper": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": prompt,
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()

    # 1. Load policy, input transforms (normalize + repack), and output
    #    transforms (denormalize actions back to real-world space).
    console.print(f"\n[bold]Loading policy from[/bold] [cyan]{args.model}[/cyan] …")
    policy, input_transforms, output_transforms = load_pretrained_policy(
        args.model,
        data_config_name=args.data_config,
    )
    policy.to(args.device).eval()
    console.print(f"[green]✓[/green] Policy loaded on [bold]{args.device}[/bold].\n")

    # 2. Build a raw observation and apply input transforms.
    raw_obs = build_dummy_observation(args.prompt)
    transformed_obs = input_transforms(raw_obs)

    # 3. Convert images to float, add a batch dimension, and move to device.
    preprocessed = torch_preprocess_dict_inference(transformed_obs)
    batch = move_dict_to_batch_for_inference(preprocessed, device=args.device)

    # 4. Run a forward pass to predict normalised action tokens / flow.
    raw_actions = policy.select_action(batch).cpu().numpy()

    # 5. Denormalize and post-process into real-world action space.
    #    The output transforms expect both ``actions`` and ``state``
    #    (the latter is needed for delta-to-absolute conversion).
    actions = output_transforms(
        {
            "actions": raw_actions,
            "state": batch["state"].cpu().numpy(),
        }
    )["actions"]

    # 6. Display results.
    # actions shape: (action_horizon, 7) — [dx, dy, dz, droll, dpitch, dyaw, gripper_action]
    console.print(f"[bold]Prompt:[/bold]  {args.prompt}")
    console.print(f"[bold]Actions shape:[/bold] {actions.shape}  "
                  f"[dim](action_horizon × 7)[/dim]")
    console.print(f"[bold]First action:[/bold]  {actions[0]}")


if __name__ == "__main__":
    main()
