"""Run GreenVLAv1.1 as a msgpack WebSocket inference service."""

from __future__ import annotations

import dataclasses
import logging
import os
import time

import numpy as np
import tyro

from lerobot.scripts.obm_inference.action_feed import FeedConfig
from lerobot.scripts.obm_inference.policy_adapter import (
    CONTROL_DT,
    DATA_CONFIG,
    HumanoidPolicyAdapter,
    available_data_configs,
)
from lerobot.scripts.obm_inference.websocket_policy_server import (
    WebsocketPolicyServer,
)


# Default server port, overridable at runtime via the POLICY_PORT env var
# (the same variable the green-challenge-sim client reads).
_DEFAULT_PORT = int(os.environ.get("POLICY_PORT", "8999"))


@dataclasses.dataclass
class Args:
    checkpoint: str

    data_config: str = DATA_CONFIG
    """The data contract to serve the checkpoint under -- one of the config names in
    lerobot/conf/. It fixes the embodiment name written into the prompt, the
    norm_stats/<asset_id>/ folder read from the checkpoint, and the action sample step,
    so it has to be the one the checkpoint was trained under; the wrong one loads and
    returns plausible, wrong actions. Defaults to $S0S1_DATA_CONFIG, else
    track27 (Track27)."""

    host: str = "0.0.0.0"
    port: int = _DEFAULT_PORT
    device: str = "cuda:0"
    rotate_images: bool = True
    # torch.compile of the sampling loop is unstable on this model in this runtime
    # (dynamo fails inside the vision rotary embedding), so it is opt-in.
    compile_sample_actions: bool = False
    inference_steps: int | None = None
    warmup_steps: int = 1
    """Compile/warm all fixed-shape inference kernels before opening the port."""

    feed_by_server: bool = False
    """Return commands ready to execute instead of the raw action chunk.

    Off by default: a client that schedules the chunk itself must keep receiving it.
    Turn it on to own the scheduling, blending and limiting here -- see action_feed.py
    for what each of them does and why its default is what it is.
    """

    feed: FeedConfig = dataclasses.field(default_factory=FeedConfig)
    """Action feed settings, e.g. --feed.blend-life 1.5. Ignored unless enabled above."""

    control_dt: float = CONTROL_DT
    """Control tick the feed schedules its commands for, seconds.

    The default is the reference simulator's tick, and it is announced in the handshake
    so a client can check it: a client whose own tick differs should refuse to run rather
    than execute a schedule that is wrong by a factor, since every command still looks
    plausible. Change this only to match a client that ticks at a different rate.
    Ignored without --feed-by-server, which is what schedules against it.
    """

    log_level: str = "INFO"
    ssl_certfile: str | None = None
    ssl_keyfile: str | None = None


def main(args: Args) -> None:
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )
    logging.getLogger(__name__).info(
        "Serving data config %s (available: %s)",
        args.data_config,
        ", ".join(available_data_configs()),
    )
    policy = HumanoidPolicyAdapter(
        args.checkpoint,
        data_config=args.data_config,
        device=args.device,
        rotate_images=args.rotate_images,
        feed=args.feed if args.feed_by_server else None,
        control_dt=args.control_dt,
        compile_sample_actions=args.compile_sample_actions,
        inference_steps=args.inference_steps,
    )
    if args.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if args.warmup_steps:
        warmup_state = np.zeros(51, dtype=np.float32)
        warmup_state[43] = 0.87
        warmup_image = np.zeros((448, 448, 3), dtype=np.uint8)
        warmup_prompt = (
            policy.metadata.get("task_adapter", {}).get("instruction")
            or "Raise your right hand"
        )
        warmup_observation = {
            "state": warmup_state,
            "images": {
                "top_head": warmup_image,
                "hand_left": warmup_image,
                "hand_right": warmup_image,
            },
            "prompt": warmup_prompt,
            "reset": True,
            "t": 0.0,
        }
        for warmup_index in range(args.warmup_steps):
            started = time.perf_counter()
            result = policy.step(warmup_observation)
            logging.getLogger(__name__).info(
                "Policy warmup %d/%d complete in %.1f ms (%d action rows)",
                warmup_index + 1,
                args.warmup_steps,
                1000.0 * (time.perf_counter() - started),
                len(result.get("actions_list", ())),
            )
            warmup_observation["reset"] = False
        policy.reset(None)
    WebsocketPolicyServer(
        policy,
        host=args.host,
        port=args.port,
        ssl_certfile=args.ssl_certfile,
        ssl_keyfile=args.ssl_keyfile,
    ).serve_forever()


if __name__ == "__main__":
    main(tyro.cli(Args))
