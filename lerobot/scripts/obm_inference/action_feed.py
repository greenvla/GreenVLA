"""Turning an action chunk into the commands a 50 Hz robot actually executes.

The policy returns a *plan*: fifty rows describing where the robot should be at a
sequence of future moments.  Nothing in that plan says which row to send on the next
control tick, what to do while two plans overlap, or how fast a joint is allowed to
move between ticks.  Those decisions are the *action feed*, and they matter as much as
the model: on the reference task the same checkpoint scores about 70 % with the feed
below and 7 % without it.

The feed lives here, next to the policy, so that it can be read, measured and replaced.
A client that wants the raw chunk instead simply leaves the feed disabled and gets the
untouched fifty rows, exactly as before.

Three mechanisms, in the order they run
---------------------------------------

1. **Scheduling.**  Row *j* of a plan describes the moment ``t_obs + (j+1)*row_dt +
   lead`` on the plan's own time scale, where ``t_obs`` is the model time of the
   observation the plan was computed from.  The command for control tick ``t`` is
   therefore the plan sampled at fractional row ``(t - t_obs - lead)/row_dt - 1``,
   linearly interpolated between neighbours.  Rows are bound to *time*, not to a tick
   counter: a plan is replaced long before it is played out, and the deep rows -- where
   a chunk's largest excursions live -- are never reached.

2. **Blending.**  Replacing the plan once per request does not by itself remove the
   seam: two adjacent plans disagree about the present moment and the setpoint jumps
   between their opinions at the request rate.  So the setpoint is not the newest plan
   but the weighted mean of *every plan still alive*, each evaluated at the current
   moment on its own time scale -- that is, each plan's forecast for *now*, not its own
   past value.  Averaging forecasts cancels the disagreement without adding phase lag,
   which is what separates this from a low-pass filter.

3. **Clamping.**  The blended setpoint is finally limited in velocity and acceleration
   against the previously issued command, with the braking distance taken into account
   so that the limiter settles onto the target instead of ringing around it.  At a
   0.02 s tick the default limits allow a finger 0.202 rad of travel and 0.602 rad/s of
   velocity gain per tick, and an arm joint 0.076 of each.

Why the plan's own clock is needed
----------------------------------

Every formula above is written in model seconds since the start of the episode.  The
simulator owns that clock, so it must send it: the ``t`` field of the observation.  The
control tick is a fixed property of the simulator and is not transmitted -- it is
passed to :class:`ActionFeed` once, at construction.

Numbers
-------

The defaults are measured optima on the reference pick task, not guesses:

* ``row_dt = 0.025`` s, ``lead = 0.030`` s -- the plan's row step and how far ahead row
  0 sits.  Together they set the moment a fresh plan takes effect: until its first row
  comes due, the previous plan keeps playing.
* ``period = 0.080`` s -- observation to observation.  The number of commands returned
  per request is ``period / dt``, so the request rate is a consequence of this value.
* ``blend_life = 1.20`` s -- how long a plan takes part in the blend.  With the period
  above that admits sixteen live plans.  This is *not* the value one would derive from
  the chunk horizon (``50*0.025 + 0.030 = 1.28`` s); shorter lifetimes measured worse
  (73.2 % at 1.20 against 64.7 % at 0.78, p = 0.0096 over 866 episodes), and longer ones
  fall off a cliff because a plan older than the horizon does not expire -- it *freezes*
  on its last row and feeds a stale pose into the blend.
* ``blend_tau = 1.0`` s -- the weight time constant, ``w = exp(-age/tau)``.  At a fixed
  plan period this and ``blend_life`` move the same lever; tune one, not both.

The peak number of live plans is logged whenever it grows, and it is the one number
worth checking after a change: with the defaults it settles at **15**.  (The naive
count ``1 + floor(life/period) = 16`` is one too many -- it forgets that a plan waits
``lead + row_dt = 0.055`` s before it takes effect at all.)  A log that stops at one
means the blend never engaged, and whatever was measured was measured without it.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class FeedConfig:
    """Every knob of the action feed.  The defaults are the measured configuration."""

    # ── scheduling ────────────────────────────────────────────────────────────────
    row_dt: float = 0.025
    """Plan row step, model seconds."""

    lead: float = 0.030
    """How far ahead of the observation row 0 is scheduled, model seconds."""

    period: float = 0.080
    """Observation-to-observation period, model seconds.  Sets the reply length."""

    latency: Optional[float] = None
    """When a fresh plan takes effect, measured from its own observation.

    ``None`` derives it as ``lead + row_dt`` -- the moment row 0 comes due, i.e. "keep
    playing the previous plan until the new one has something to say".  Values at or
    above ``period`` are impossible: the plan would never be replaced at all.
    """

    # ── blending ──────────────────────────────────────────────────────────────────
    blend: bool = True
    """Average the forecasts of every live plan instead of following the newest one."""

    blend_tau: float = 1.0
    """Weight time constant, seconds: a plan of age ``a`` weighs ``exp(-a/tau)``."""

    blend_life: float = 1.20
    """How long a plan keeps taking part in the blend, seconds."""

    blend_skip: tuple = ()
    """Actuator groups excluded from the blend, taken from the newest plan instead.

    Names are the row keys, e.g. ``("finger_joint_pos",)``.  Averaging suits joints
    that travel; it does not suit a short impulse.  Blending the finger command can
    average a grasp away entirely -- the hand reaches the fruit and never closes.
    """

    # ── clamping ──────────────────────────────────────────────────────────────────
    clamp: bool = True
    """Limit how far and how fast a command may move between control ticks."""

    clamp_channels: str = "both"
    """Which joints are clamped: ``rhand``, ``hands``, ``arm``, ``both`` or ``all``.

    ``both`` -- the default -- is both hands, both wrists and the right arm: twenty-one
    channels.  Clamping a joint the policy does not drive costs nothing but noise.
    """

    # Speed and acceleration ceilings per joint group, rad/s and rad/s^2.  These are the
    # limits of the physical arm the reference data was recorded on.  The acceleration
    # ceiling is the one that does the work: on a synthetic chunk a lone spike over
    # three rows passes at 33 percent with it and at 80 percent without, while a
    # sustained step passes in full either way -- the limiter rounds off the leading
    # edge over 0.1-0.15 s rather than cutting the target down.
    max_speed_fingers: float = 10.1
    max_speed_arm: float = 3.8
    max_speed_neck: float = 4.5
    max_speed_legs: float = 3.0
    max_acceleration_fingers: float = 30.1
    max_acceleration_arm: float = 3.8
    max_acceleration_neck: float = 10.5
    max_acceleration_legs: float = 3.0

    def speed_limit(self, group: str) -> float:
        return float(getattr(self, "max_speed_" + group))

    def acceleration_limit(self, group: str) -> float:
        return float(getattr(self, "max_acceleration_" + group))

    def resolved_latency(self) -> float:
        lat = self.lead + self.row_dt if self.latency is None else float(self.latency)
        if lat >= self.period:
            logger.warning(
                "latency %.4f s >= period %.4f s: a fresh plan could never take effect "
                "before the next one arrives, so the feed would degenerate into playing "
                "one chunk to its end. Clamping to %.4f s.",
                lat, self.period, self.period * 0.5,
            )
            lat = self.period * 0.5
        return lat


class _Plan:
    """One chunk, with the model time of the observation it was computed from."""

    __slots__ = ("rows", "t_obs", "index")

    def __init__(self, rows: Sequence[Dict[str, Any]], t_obs: float, index: int) -> None:
        self.rows = rows
        self.t_obs = float(t_obs)
        self.index = int(index)


def lerp_rows(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]],
              w: float) -> Dict[str, Any]:
    """Linearly interpolate two command rows, ``a + w*(b - a)``.

    A row is what the policy adapter produces: float lists keyed by actuator group,
    plus a nested ``base_command`` mapping.  Everything numeric is interpolated; keys
    that are not numeric are taken from ``b``, the newer row.
    """
    if a is None:
        return dict(b or {})
    if b is None:
        return dict(a)
    out: Dict[str, Any] = {}
    for key, vb in b.items():
        va = a.get(key)
        if isinstance(vb, dict):
            if isinstance(va, dict):
                out[key] = {
                    k: (float(va[k]) + w * (float(v) - float(va[k])))
                    if k in va and isinstance(v, (int, float)) else v
                    for k, v in vb.items()
                }
            else:
                out[key] = dict(vb)
        elif isinstance(vb, (list, tuple)):
            if isinstance(va, (list, tuple)) and len(va) == len(vb):
                out[key] = [float(x) + w * (float(y) - float(x)) for x, y in zip(va, vb)]
            else:
                out[key] = list(vb)
        elif isinstance(vb, (int, float)) and isinstance(va, (int, float)):
            out[key] = float(va) + w * (float(vb) - float(va))
        else:
            out[key] = vb
    return out


# ── which joint sits where inside a command row ───────────────────────────────────
# The names are only for diagnostics; the group picks the limits.
_TORSO_NAMES = ["torso_yaw",
                "l_shoulder_pitch", "l_shoulder_roll", "l_shoulder_yaw",
                "l_elbow_pitch", "l_elbow_yaw",
                "r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw",
                "r_elbow_pitch", "r_elbow_yaw",
                "neck_yaw", "neck_pitch"]
_FINGER_NAMES = ["l_pinky", "l_ring", "l_middle", "l_index", "l_thumb_pitch", "l_thumb_yaw",
                 "r_pinky", "r_ring", "r_middle", "r_index", "r_thumb_pitch", "r_thumb_yaw"]
_WRIST_NAMES = ["l_wrist_crank", "l_wrist_roll", "r_wrist_crank", "r_wrist_roll"]


def _channels(selection: str) -> List[tuple]:
    """The ``(row key, index, limit group, name)`` list for a channel selection."""
    fingers = [("finger_joint_pos", k, "fingers", _FINGER_NAMES[k]) for k in range(12)]
    fingers_right = fingers[6:]
    wrists = [("wrist_joint_pos", k, "arm", _WRIST_NAMES[k]) for k in range(4)]
    wrists_right = wrists[2:]
    arm_right = [("torso_joint_pos", k, "arm", _TORSO_NAMES[k]) for k in range(6, 11)]
    arm_left = [("torso_joint_pos", k, "arm", _TORSO_NAMES[k]) for k in range(1, 6)]
    torso_yaw = [("torso_joint_pos", 0, "arm", _TORSO_NAMES[0])]
    neck = [("torso_joint_pos", k, "neck", _TORSO_NAMES[k]) for k in (11, 12)]
    legs = [("legs_joint_pos", k, "legs", "leg%02d" % k) for k in range(12)]
    if selection == "rhand":
        return fingers_right + wrists_right
    if selection == "hands":
        return fingers + wrists
    if selection == "arm":
        return arm_right
    if selection == "all":
        return legs + torso_yaw + arm_left + arm_right + neck + wrists + fingers
    return fingers + wrists + arm_right          # "both", the default


class _Clamp:
    """Velocity and acceleration limiter, stateful, one per episode.

    The state is kept on the command *issued*, never on a measured pose: the limiter
    has to reason about what it asked for, not about how well the joint tracked it.
    """

    def __init__(self, config: FeedConfig, dt: float) -> None:
        self.config = config
        self.dt = float(dt)
        self.channels = _channels(config.clamp_channels)
        self.position: Dict[tuple, float] = {}
        self.velocity: Dict[tuple, float] = {}

    def seed(self, command: Dict[str, Any]) -> int:
        """Seat the state on a command, before the first chunk row is issued.

        Without this the very first command of an episode is unconstrained and jumps
        straight out of the start pose; from then on the limiter tracks its own output.
        """
        seated = 0
        for key, idx, _group, _name in self.channels:
            values = command.get(key)
            if isinstance(values, (list, tuple)) and len(values) > idx:
                self.position[(key, idx)] = float(values[idx])
                self.velocity[(key, idx)] = 0.0
                seated += 1
        return seated

    def apply(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Return a copy of ``row`` with the selected channels limited."""
        out = dict(row)
        for key in ("legs_joint_pos", "torso_joint_pos", "wrist_joint_pos",
                    "finger_joint_pos"):
            if isinstance(row.get(key), (list, tuple)):
                out[key] = [float(x) for x in row[key]]
        for key, idx, group, _name in self.channels:
            target = out.get(key)
            if target is None or len(target) <= idx:
                continue
            channel = (key, idx)
            goal = target[idx]
            if channel not in self.position:      # not seated: adopt the goal as-is
                self.position[channel] = goal
                self.velocity[channel] = 0.0
                continue
            position, velocity = self.position[channel], self.velocity[channel]
            vmax = self.config.speed_limit(group)
            amax = self.config.acceleration_limit(group)
            gap = goal - position
            # The speed from which the joint can still brake exactly onto the target.
            # The continuous form sqrt(2*a*|gap|) overshoots on a discrete tick and the
            # limiter starts ringing (measured on a synthetic chunk: a peak of 1.566
            # against the requested 1.500, and nineteen sign reversals). Discretely:
            # braking by a*dt per tick covers dt*(n*v - a*dt*n*(n-1)/2) in n = v/(a*dt)
            # ticks; inverting for v gives the expression below.
            if amax > 0:
                adt = amax * self.dt
                brake = 0.5 * adt * (math.sqrt(1.0 + 8.0 * abs(gap) / (adt * self.dt)) - 1.0)
            else:
                brake = vmax
            wanted = gap / self.dt
            wanted = max(-vmax, min(vmax, wanted))
            wanted = max(-brake, min(brake, wanted))
            wanted = max(velocity - amax * self.dt, min(velocity + amax * self.dt, wanted))
            moved = position + wanted * self.dt
            if (goal - moved) * (goal - position) < 0.0:   # never step past the target
                moved, wanted = goal, 0.0
            self.position[channel] = moved
            self.velocity[channel] = wanted
            target[idx] = moved
        return out


class ActionFeed:
    """Stateful translator from action chunks to per-tick commands.

    One instance per episode-stream.  Call :meth:`reset` at an episode boundary and
    :meth:`update` once per policy request; ``update`` returns the commands for the
    ticks up to the next request, oldest first.
    """

    def __init__(self, config: FeedConfig, dt: float) -> None:
        self.config = config
        self.dt = float(dt)
        self.latency = config.resolved_latency()
        self.ticks_per_period = max(1, int(round(config.period / self.dt)))
        self._active: Optional[_Plan] = None
        self._pending: Optional[_Plan] = None
        self._pending_effective_at = 0.0
        self._n_chunks = 0
        self._live: List[_Plan] = []
        self._live_peak = 0
        self._clamp: Optional[_Clamp] = None
        self._last_t_obs: Optional[float] = None
        self.reset()

    # ── episode boundary ──────────────────────────────────────────────────────────
    def reset(self, current_command: Optional[Dict[str, Any]] = None) -> None:
        """Start a new episode: forget every plan and re-seat the limiter.

        Blending must not leak across an episode boundary, and the limiter has to know
        where the robot is standing before it issues the first command.  Pass the pose
        the robot is holding -- in command form -- as ``current_command``.
        """
        self._active = None
        self._pending = None
        self._pending_effective_at = 0.0
        self._n_chunks = 0
        self._live = []
        self._last_t_obs = None
        self._clamp = _Clamp(self.config, self.dt) if self.config.clamp else None
        if self._clamp is not None and current_command is not None:
            seated = self._clamp.seed(current_command)
            logger.info("action feed: limiter seated on %d channels", seated)

    # ── the plan, sampled at one moment ───────────────────────────────────────────
    def _evaluate(self, plan: _Plan, t: float) -> Dict[str, Any]:
        """The plan's forecast for model time ``t``.

        Before row 0 comes due the plan can only offer row 0 itself.  Past its last row
        the plan *freezes* rather than expiring -- there is no continuation to
        extrapolate into, and inventing one is worse than repeating the last pose.  The
        freeze is why a plan lifetime longer than the chunk horizon hurts.
        """
        rows = plan.rows
        elapsed = t - plan.t_obs
        first_row_due = self.config.row_dt + self.config.lead
        if elapsed <= first_row_due - 1e-9:
            return dict(rows[0])
        x = (elapsed - self.config.lead) / self.config.row_dt - 1.0
        if x >= len(rows) - 1:
            return dict(rows[-1])
        j = int(x)
        return lerp_rows(rows[j], rows[j + 1], x - j)

    def _promote_pending(self) -> None:
        self._active = self._pending
        self._pending = None
        if self.config.blend:
            self._live.append(self._active)

    # ── the blend ─────────────────────────────────────────────────────────────────
    def _blend(self, t: float, active_command: Dict[str, Any]) -> Dict[str, Any]:
        """Weighted mean of every live plan's forecast for model time ``t``.

        The active plan never expires.  If requests stop -- no camera frames, a server
        hiccup -- the blend has to degenerate into the plan currently in force rather
        than go empty.
        """
        life, tau = self.config.blend_life, self.config.blend_tau
        self._live = [p for p in self._live
                      if p is self._active or (t - p.t_obs) <= life + 1e-9]
        n = len(self._live)
        if n > self._live_peak:
            self._live_peak = n
            logger.info("action feed: %d plans live in the blend "
                        "(lifetime %.2f s, tau %.2f s)", n, life, tau)
        if n < 2:
            return active_command

        accumulated: Optional[Dict[str, Any]] = None
        weight_sum = 0.0
        for plan in self._live:                     # oldest first, so that non-numeric
            age = t - plan.t_obs                    # fields end up from the newest plan
            w = math.exp(-age / max(tau, 1e-9))
            forecast = (active_command if plan is self._active
                        else self._evaluate(plan, t))
            if accumulated is None:
                accumulated, weight_sum = dict(forecast), w
            else:
                weight_sum += w
                # The mean is accumulated by successive interpolation:
                # acc += (w/sum w)*(c - acc), which by induction is exactly
                # sum(w_i c_i)/sum(w_i) -- and lerp_rows already knows the row layout.
                accumulated = lerp_rows(accumulated, forecast,
                                        w / weight_sum if weight_sum > 0 else 1.0)

        # Skipped groups are painted back over the average from the newest plan, which
        # is the active one -- its forecast arrived here as `active_command`.
        for key in self.config.blend_skip:
            if key in active_command:
                value = active_command[key]
                accumulated[key] = (list(value) if isinstance(value, (list, tuple))
                                    else dict(value) if isinstance(value, dict)
                                    else value)
        return accumulated

    # ── one request ───────────────────────────────────────────────────────────────
    def update(self, t_obs: float, rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Register a fresh chunk and return the commands to play before the next one.

        ``t_obs`` is the model time of the observation this chunk answers; ``rows`` is
        the chunk itself, straight out of the policy.
        """
        if not rows:
            raise ValueError("the action feed was handed an empty chunk")
        t_obs = float(t_obs)
        if self._last_t_obs is not None and t_obs < self._last_t_obs - 1e-9:
            # An episode clock only moves forward. Backwards means a new episode that
            # did not announce itself, and carrying the old plans across would be worse
            # than useless: their age is then negative, which neither expires them nor
            # reduces their weight -- exp(-age/tau) makes a stale plan dominate every
            # fresh one, and the setpoint sticks to a chunk from a run that has ended.
            raise ValueError(
                "The episode clock went backwards, from %.3f s to %.3f s. Call reset() "
                "at the episode boundary before feeding a chunk from the new one."
                % (self._last_t_obs, t_obs))
        self._last_t_obs = t_obs

        # A requested chunk must not be dropped: if the previous one has still not taken
        # effect by the time the next arrives, bring it in now, or the active plan would
        # never be replaced at all.
        if self._pending is not None:
            self._promote_pending()

        self._n_chunks += 1
        self._pending = _Plan(rows, t_obs, self._n_chunks)
        # The first chunk of an episode takes effect at once: until it does there is no
        # setpoint to issue, and the robot would just sit in its start pose.
        self._pending_effective_at = t_obs + (self.latency if self._active is not None else 0.0)

        # The client's clock is a whole number of control ticks -- that is the contract,
        # and the simulator refuses to run on any other grid.  Rebuilding the tick index
        # and multiplying back reproduces the client's own arithmetic exactly, instead of
        # accumulating t_obs + j*dt, which differs from (i+j)*dt in the last bits and
        # drifts by ~1e-11 rad over a few hundred ticks.
        first_tick = round(t_obs / self.dt)
        commands: List[Dict[str, Any]] = []
        for tick in range(self.ticks_per_period):
            t = (first_tick + tick) * self.dt
            if self._pending is not None and t >= self._pending_effective_at - 1e-9:
                self._promote_pending()
            if self._active is None:  # unreachable: the first chunk promotes at once
                raise RuntimeError("the action feed has no active plan at %.3f s" % t)
            command = self._evaluate(self._active, t)
            if self.config.blend:
                command = self._blend(t, command)
            if self._clamp is not None:
                command = self._clamp.apply(command)
            commands.append(command)
        return commands


# Policy rows have one fixed, all-numeric wire layout.  Keeping that
# layout flat while plans are sampled and blended avoids thousands of tiny
# Python list/dict operations per request.  The generic ActionFeed above stays
# available for other robots and for reference comparisons.
_NUMERIC_ROW_FIELDS = (
    ("legs_joint_pos", 12),
    ("torso_joint_pos", 13),
    ("finger_joint_pos", 12),
    ("wrist_joint_pos", 4),
    ("velocity", 6),
)
_NUMERIC_ROW_SLICES: Dict[str, slice] = {}
_numeric_offset = 0
for _numeric_name, _numeric_size in _NUMERIC_ROW_FIELDS:
    _NUMERIC_ROW_SLICES[_numeric_name] = slice(
        _numeric_offset, _numeric_offset + _numeric_size
    )
    _numeric_offset += _numeric_size
_NUMERIC_ROW_SLICES["base_command"] = slice(_numeric_offset, _numeric_offset + 4)
_NUMERIC_ROW_DIM = _numeric_offset + 4


def _row_to_numeric(row: Dict[str, Any]) -> np.ndarray:
    values = np.empty(_NUMERIC_ROW_DIM, dtype=np.float64)
    for key, size in _NUMERIC_ROW_FIELDS:
        source = row.get(key)
        if not isinstance(source, (list, tuple)) or len(source) != size:
            raise ValueError(f"action row {key!r} must contain {size} values")
        values[_NUMERIC_ROW_SLICES[key]] = source
    base = row.get("base_command")
    if not isinstance(base, dict):
        raise ValueError("action row 'base_command' must be a mapping")
    values[_NUMERIC_ROW_SLICES["base_command"]] = (
        float(base["root_height"]),
        float(base["roll"]),
        float(base["pitch"]),
        float(base["yaw"]),
    )
    if not np.isfinite(values).all():
        raise ValueError("action row contains NaN or Inf")
    return values


def _numeric_to_row(values: np.ndarray) -> Dict[str, Any]:
    row = {
        key: values[_NUMERIC_ROW_SLICES[key]].tolist()
        for key, _size in _NUMERIC_ROW_FIELDS
    }
    base = values[_NUMERIC_ROW_SLICES["base_command"]]
    row["base_command"] = {
        "root_height": float(base[0]),
        "roll": float(base[1]),
        "pitch": float(base[2]),
        "yaw": float(base[3]),
    }
    return row


class _NumericPlan:
    __slots__ = ("rows", "t_obs", "index")

    def __init__(self, rows: np.ndarray, t_obs: float, index: int) -> None:
        self.rows = rows
        self.t_obs = float(t_obs)
        self.index = int(index)


class NumericActionFeed:
    """Vectorized, contract-specific equivalent of :class:`ActionFeed`."""

    def __init__(self, config: FeedConfig, dt: float) -> None:
        self.config = config
        self.dt = float(dt)
        self.latency = config.resolved_latency()
        self.ticks_per_period = max(1, int(round(config.period / self.dt)))
        self._skip_slices = tuple(
            _NUMERIC_ROW_SLICES[key] for key in config.blend_skip
        )
        self.reset()

    def reset(self, current_command: Optional[Dict[str, Any]] = None) -> None:
        self._active: Optional[_NumericPlan] = None
        self._pending: Optional[_NumericPlan] = None
        self._pending_effective_at = 0.0
        self._n_chunks = 0
        self._live: List[_NumericPlan] = []
        self._live_peak = 0
        self._last_t_obs: Optional[float] = None
        self._clamp = _Clamp(self.config, self.dt) if self.config.clamp else None
        if self._clamp is not None and current_command is not None:
            seated = self._clamp.seed(current_command)
            logger.info("numeric action feed: limiter seated on %d channels", seated)

    def _evaluate(self, plan: _NumericPlan, t: float) -> np.ndarray:
        rows = plan.rows
        elapsed = t - plan.t_obs
        first_row_due = self.config.row_dt + self.config.lead
        if elapsed <= first_row_due - 1e-9:
            return rows[0].copy()
        x = (elapsed - self.config.lead) / self.config.row_dt - 1.0
        if x >= len(rows) - 1:
            return rows[-1].copy()
        index = int(x)
        return rows[index] + (x - index) * (rows[index + 1] - rows[index])

    def _promote_pending(self) -> None:
        self._active = self._pending
        self._pending = None
        if self.config.blend:
            self._live.append(self._active)

    def _blend(self, t: float, active_command: np.ndarray) -> np.ndarray:
        life, tau = self.config.blend_life, self.config.blend_tau
        self._live = [
            plan
            for plan in self._live
            if plan is self._active or (t - plan.t_obs) <= life + 1e-9
        ]
        count = len(self._live)
        if count > self._live_peak:
            self._live_peak = count
            logger.info(
                "numeric action feed: %d plans live in the blend "
                "(lifetime %.2f s, tau %.2f s)",
                count,
                life,
                tau,
            )
        if count < 2:
            return active_command

        accumulated = None
        weight_sum = 0.0
        for plan in self._live:
            age = t - plan.t_obs
            weight = math.exp(-age / max(tau, 1e-9))
            forecast = (
                active_command if plan is self._active else self._evaluate(plan, t)
            )
            if accumulated is None:
                accumulated = forecast.copy()
                weight_sum = weight
            else:
                weight_sum += weight
                blend_weight = weight / weight_sum if weight_sum > 0 else 1.0
                accumulated = accumulated + blend_weight * (forecast - accumulated)
        for field_slice in self._skip_slices:
            accumulated[field_slice] = active_command[field_slice]
        return accumulated

    def update(
        self, t_obs: float, rows: Sequence[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not rows:
            raise ValueError("the action feed was handed an empty chunk")
        numeric_rows = np.stack([_row_to_numeric(row) for row in rows], axis=0)
        return self.update_numeric(t_obs, numeric_rows)

    def update_numeric(
        self, t_obs: float, numeric_rows: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Register already-flattened action rows without a dict round-trip."""
        numeric_rows = np.asarray(numeric_rows)
        if numeric_rows.ndim != 2 or numeric_rows.shape[1] != _NUMERIC_ROW_DIM:
            raise ValueError(
                "numeric action rows must have shape (N, %d), got %r"
                % (_NUMERIC_ROW_DIM, numeric_rows.shape)
            )
        if numeric_rows.shape[0] == 0:
            raise ValueError("the action feed was handed an empty chunk")
        if not np.isfinite(numeric_rows).all():
            raise ValueError("numeric action rows contain NaN or Inf")
        # The reference feed converts every source scalar to a Python float
        # before interpolation, i.e. float64.  Preserve that arithmetic exactly.
        numeric_rows = np.asarray(numeric_rows, dtype=np.float64).copy()
        t_obs = float(t_obs)
        if self._last_t_obs is not None and t_obs < self._last_t_obs - 1e-9:
            raise ValueError(
                "The episode clock moved backwards, from %.3f s to %.3f s. "
                "Call reset() at the episode boundary before feeding a chunk "
                "from the new one." % (self._last_t_obs, t_obs)
            )
        self._last_t_obs = t_obs
        if self._pending is not None:
            self._promote_pending()

        self._n_chunks += 1
        self._pending = _NumericPlan(numeric_rows, t_obs, self._n_chunks)
        self._pending_effective_at = t_obs + (
            self.latency if self._active is not None else 0.0
        )

        first_tick = round(t_obs / self.dt)
        commands: List[Dict[str, Any]] = []
        for tick in range(self.ticks_per_period):
            t = (first_tick + tick) * self.dt
            if self._pending is not None and t >= self._pending_effective_at - 1e-9:
                self._promote_pending()
            if self._active is None:
                raise RuntimeError(f"the action feed has no active plan at {t:.3f} s")
            command = self._evaluate(self._active, t)
            if self.config.blend:
                command = self._blend(t, command)
            row = _numeric_to_row(command)
            if self._clamp is not None:
                row = self._clamp.apply(row)
            commands.append(row)
        return commands
