"""Serve the simulator wire boundary; learned tracking and neck bias are opt-in.

The two-server topology deliberately preserves the original WebSocket boundary.
Start the upstream with the checkpoint's data config and S0S1_PROMPT_FROM_SUBTASK=1. Simulator
PROPRIO_CMD must be all; fingers must use default-relative action semantics.
No runtime access to datasets, object poses, phase clocks or teacher trajectories.
"""
import argparse
import json
import logging
from pathlib import Path

import websockets.sync.client

from lerobot.common.robot_safety.tracking.joint_calibrated_boundary import JointCalibratedBoundary
from lerobot.common.robot_safety.tracking.joint_tracking_calibrator import JointTrackingCalibrator
from lerobot.common.robot_safety.tracking.wire_boundary import WireBoundary
from lerobot.scripts.obm_inference.msgpack_numpy import Packer, unpackb
from lerobot.scripts.obm_inference.websocket_policy_server import WebsocketPolicyServer

ASSETS = Path(__file__).resolve().parents[2] / "common/robot_safety/tracking/assets"


class TrackingProxy:
    def __init__(self, args):
        self.packer = Packer()
        self.ws = websockets.sync.client.connect(
            f"ws://127.0.0.1:{args.upstream_port}", compression=None, max_size=None
        )
        calibration = json.loads(Path(args.calibration).read_text())
        self.boundary = JointCalibratedBoundary(
            WireBoundary(calibration, proprio="all", neck=args.neck_compensation),
            JointTrackingCalibrator(args.joint_calibration) if args.adapter == "v1" else None,
        )
        self.metadata = dict(unpackb(self.ws.recv()), wire_boundary={
            "finger_absolute_to_relative": True, "proprio": "all",
            "empirical_neck_calibration": args.neck_compensation, "calibration": calibration,
            "upstream_port": args.upstream_port, "image_or_prompt_changes": False,
        })
        self.metadata["joint_tracking_calibration"] = {
            "enabled": args.adapter == "v1", "adapter": args.adapter,
            "weights": json.loads(Path(args.joint_calibration).read_text()) if args.adapter == "v1" else None,
            "input_leg_feedback": "previous uncorrected model command",
            "input_body_correction_echo_decoded": True,
            "measured_root_attitude_untouched": True, "runtime_dataset_access": False,
        }
        self.log = Path(args.log)
        self.log.parent.mkdir(parents=True, exist_ok=True)

    def step(self, observation):
        forwarded = self.boundary.decode_observation(observation)
        assert forwarded.get("images") is observation.get("images")
        assert forwarded.get("subtask") == observation.get("subtask")
        self.ws.send(self.packer.pack(forwarded))
        reply = self.ws.recv()
        if isinstance(reply, str):
            raise RuntimeError(reply)
        raw = unpackb(reply)
        result = self.boundary.encode_response(raw)
        with self.log.open("a") as stream:
            stream.write(json.dumps({
                "t": float(observation["t"]), "subtask": observation.get("subtask"),
                "last_original": raw["actions_list"][-1],
                "last_wire": result["actions_list"][-1],
                "decoded_command_feedback": forwarded is not observation,
            }) + "\n")
        return result


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-port", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--calibration", type=Path, default=ASSETS / "wire_calibration.json")
    parser.add_argument("--joint-calibration", type=Path,
                        default=ASSETS / "joint_tracking_calibration_v1.json")
    parser.add_argument("--adapter", choices=("none", "v1"), default="none",
                        help="Optional learned joint correction; default: raw VLA (no weights loaded).")
    parser.add_argument("--neck-compensation", action="store_true",
                        help="Opt in to the empirical neck bias; raw neck commands are the default.")
    parser.add_argument("--log", required=True)
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    if args.upstream_port == args.port:
        parser.error("Upstream and boundary ports must differ")
    logging.basicConfig(level=logging.INFO)
    policy = TrackingProxy(args)
    print(json.dumps(policy.metadata), flush=True)
    WebsocketPolicyServer(policy, port=args.port).serve_forever()


if __name__ == "__main__":
    main()
