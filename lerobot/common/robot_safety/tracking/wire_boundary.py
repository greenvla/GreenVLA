"""Task-independent absolute-state / simulator-command boundary.

Finger action terms add the robot's default joint position. The model predicts
absolute joints. Optional neck calibration is a fitted, non-integrating bias;
it is NOT part of raw VLA. No scene, phase, clock, image or trajectory lookup.
"""
import copy
import numpy as np


class WireBoundary:
    def __init__(self, calibration, proprio="all", neck=False):
        if proprio not in ("all", "measured"):
            raise ValueError(proprio)
        self.defaults = np.asarray(calibration["finger_default_position"], dtype=float)
        assert self.defaults.shape == (12,) and np.isfinite(self.defaults).all()
        self.neck_bias = float(calibration["neck_pitch_bias_rad"]) if neck else 0.
        self.neck_limits = calibration["neck_pitch_limits"]
        assert abs(self.neck_bias) <= .18
        self.proprio = proprio
        self.last_delta = None
        self.last_t = None

    def decode_observation(self, observation):
        t = float(observation["t"])
        if observation.get("reset") or self.last_t is None or t < self.last_t:
            self.last_delta = None
        self.last_t = t
        if self.proprio != "all" or self.last_delta is None:
            return observation
        # PROPRIO_CMD=all echoes the previous *wire* command; decode back to
        # model absolute coordinates. On first/reset input the state is measured.
        result = dict(observation)
        state = dict(observation["state"])
        fingers = np.asarray(state["finger_joint_pos"], dtype=float).copy()
        if fingers.shape == (16,):
            fingers[np.r_[0:6, 8:14]] += self.defaults
        elif fingers.shape == (12,):
            fingers += self.defaults
        else:
            raise ValueError(f"Unexpected finger state shape {fingers.shape}")
        state["finger_joint_pos"] = fingers
        torso = np.asarray(state["torso_joint_pos"], dtype=float).copy()
        if torso.shape != (13,):
            raise ValueError(f"Unexpected torso shape {torso.shape}")
        torso[12] -= self.last_delta
        state["torso_joint_pos"] = torso
        result["state"] = state
        return result

    def encode_response(self, response):
        result = dict(response)
        rows = []
        for row in response["actions_list"]:
            action = copy.deepcopy(row)
            fingers = np.asarray(row["finger_joint_pos"], dtype=float)
            torso = np.asarray(row["torso_joint_pos"], dtype=float).copy()
            if fingers.shape != (12,) or torso.shape != (13,):
                raise ValueError("Unexpected output joint order")
            if not np.isfinite(fingers).all() or not np.isfinite(torso).all():
                raise ValueError("Nonfinite actions")
            action["finger_joint_pos"] = (fingers - self.defaults).tolist()
            before = float(torso[12])
            # Only the optional empirical neck compensation is limited here.
            if self.neck_bias:
                torso[12] = np.clip(before + self.neck_bias, *self.neck_limits)
            self.last_delta = float(torso[12] - before)
            action["torso_joint_pos"] = torso.tolist()
            rows.append(action)
        result["actions_list"] = rows
        return result
