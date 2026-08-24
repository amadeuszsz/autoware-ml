# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reachability time-to-collision for the criticality metrics (B1, B2).

One planner-independent model. Every agent, ego included, travels at the
maximum map-legal speed for its class. The recorded path and observed
velocities are never used. Time-to-collision is the earliest look-ahead time
``t`` at which ego and the object can occupy a common point at the same time
``t``, each having travelled exactly ``speed * t`` along a feasible path
(reachable at ``t``, not "arrive early and wait"). That is what makes
same-direction traffic at matched speed non-critical: the two fronts stay a
constant gap apart and never meet, so TTC = inf and its collision weight is 0.

Reachable-at-``t`` set by class, in the map frame (metres):

* wheeled (car / truck / bus / train / motorcycle, and ego): the endpoints
  of every feasible constant-curvature arc of length ``v * t`` under bounded
  steering (minimum turn radius), a curved front, clipped to the drivable
  area (an arc leaving the road is infeasible) and given the vehicle's body
  half-width.
* VRU (pedestrian / animal / bicycle): the disc of radius ``v * t`` about the
  current position, free to move in any direction, over any surface.
* static (barrier / traffic_cone / debris / bicycle_rack / vehicle_extension):
  the fixed footprint, for every ``t``.

The two steps match the metric's contract:

1. prescreen: distance bounds prove most objects can never meet ego within
   the horizon (their largest reachable set misses ego's reachable region, or
   the straight-line gap cannot close in time), giving TTC = inf with no
   stepping.
2. otherwise step ``t`` up from the earliest feasible step and return the
   first ``t`` at which the two reachable-at-``t`` sets overlap, the fastest
   path to collision.

Wheeled max speed comes from the lanelet map: the ``speed_limit`` of the lanelet
the agent is in (``LaneletMap.speed_at``, km/h converted to m/s), falling back
to a spec-versioned constant off-map. VRUs use a per-class run speed.

:class:`EgoReachability` holds everything that depends only on ego (the
localized drivable area, the filled reachable region used by the prescreen, and
the per-step fronts), so one frame's many objects share it instead of rebuilding
identical ego geometry per box.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, cos, hypot, inf, sin

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon, box
from shapely.geometry.base import BaseGeometry

WHEELED = "wheeled"
VRU = "vru"
STATIC = "static"
_KINDS = frozenset({WHEELED, VRU, STATIC})


@dataclass(frozen=True)
class ReachabilityParams:
    """Spec-versioned parameters shared by every reachability TTC evaluation.

    The wheeled turn radius is bounded by the lateral-acceleration (friction)
    limit, ``R = v^2 / max_lateral_accel``, floored at ``min_radius_m`` for low
    speed, so a fast vehicle cannot turn implausibly tight.
    """

    horizon_s: float = 4.0
    dt_s: float = 0.1
    max_lateral_accel_mps2: float = 3.0
    min_radius_m: float = 3.0
    arc_samples: int = 21

    def __post_init__(self) -> None:
        if self.horizon_s <= 0.0 or self.dt_s <= 0.0:
            raise ValueError("horizon_s and dt_s must be > 0.")
        if self.max_lateral_accel_mps2 <= 0.0 or self.min_radius_m <= 0.0:
            raise ValueError("max_lateral_accel_mps2 and min_radius_m must be > 0.")
        if self.arc_samples < 3:
            raise ValueError("arc_samples must be >= 3 to trace a front.")

    def turn_radius(self, speed: float) -> float:
        """Minimum feasible turn radius at ``speed`` (friction-limited, floored)."""
        return max(speed * speed / self.max_lateral_accel_mps2, self.min_radius_m)


@dataclass(frozen=True)
class Agent:
    """One collision participant in the map frame.

    ``kind`` selects the reachable-set shape. ``heading`` and ``speed`` are used
    by the wheeled kind, the VRU kind uses ``speed`` isotropically, and the
    static kind ignores both and uses ``footprint``. ``body_radius`` is the
    half-extent added so a collision is a footprint overlap, not a point
    coincidence.
    """

    kind: str
    x: float
    y: float
    heading: float = 0.0
    speed: float = 0.0
    body_radius: float = 0.5
    footprint: Polygon | None = None

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"kind must be one of {sorted(_KINDS)}, got {self.kind!r}.")
        if self.kind == STATIC and self.footprint is None:
            raise ValueError("a static agent needs a footprint polygon.")
        if self.kind != STATIC and self.speed < 0.0:
            raise ValueError("speed must be >= 0.")


def _arc_endpoint(
    x: float, y: float, heading: float, kappa: float, length: float
) -> tuple[float, float]:
    """Endpoint of a constant-curvature arc of signed curvature ``kappa`` and ``length``.

    ``kappa > 0`` turns left of the heading, ``kappa == 0`` is straight. The map
    frame is right-handed in (x, y) and the caller supplies ``heading`` in radians.
    """
    if abs(kappa) < 1e-9:
        return (x + length * cos(heading), y + length * sin(heading))
    radius = 1.0 / kappa
    # Turn centre is 90 deg to the left of the heading (left of travel for kappa>0).
    cx = x - radius * sin(heading)
    cy = y + radius * cos(heading)
    phi = length * kappa  # signed swept angle
    sx, sy = x - cx, y - cy
    ex = cx + sx * cos(phi) - sy * sin(phi)
    ey = cy + sx * sin(phi) + sy * cos(phi)
    return (ex, ey)


def wheeled_front(agent: Agent, t: float, params: ReachabilityParams) -> BaseGeometry:
    """The wheeled reachable-at-``t`` front: buffered curved endpoint locus.

    The body buffer tapers at the start: the sweep begins as a single point at
    the agent's reference position (the rear axle for ego) and widens to the
    full body width within the first body radius of travel, an approximation
    of the vehicle expanding from its reference point to its width at the
    front bumper.
    """
    length = agent.speed * t
    if length <= 1e-6:
        return Point(agent.x, agent.y).buffer(1e-3)
    kmax = 1.0 / params.turn_radius(agent.speed)
    kappas = np.linspace(-kmax, kmax, params.arc_samples)
    points = [_arc_endpoint(agent.x, agent.y, agent.heading, float(k), length) for k in kappas]
    return LineString(points).buffer(min(agent.body_radius, length))


def reachable_region(
    agent: Agent, params: ReachabilityParams, drivable: BaseGeometry | None
) -> BaseGeometry:
    """Filled region a wheeled agent can reach within the horizon (the "hat").

    Union of its fronts over ``t`` in ``(0, T]``, whose boundary is the two
    extreme max-curvature arcs plus the far arc at ``t = T``, clipped to
    ``drivable``. The body buffer tapers at the start exactly like
    :func:`wheeled_front`: the region begins at the agent's reference point and
    widens to the body width within the first body radius of travel. Used by
    the collision filter's in-the-ego-path membership test.
    """
    length = agent.speed * params.horizon_s
    if length <= 1e-6:
        region = Point(agent.x, agent.y).buffer(1e-3)
    else:
        kmax = 1.0 / params.turn_radius(agent.speed)
        n = params.arc_samples
        x, y, heading = agent.x, agent.y, agent.heading
        left = [_arc_endpoint(x, y, heading, kmax, float(s)) for s in np.linspace(0.0, length, n)]
        far = [_arc_endpoint(x, y, heading, float(k), length) for k in np.linspace(kmax, -kmax, n)]
        right = [_arc_endpoint(x, y, heading, -kmax, float(s)) for s in np.linspace(length, 0.0, n)]
        shell = Polygon(left + far + right)
        if not shell.is_valid:
            # At low speed the max-curvature arcs sweep past pi and the shell
            # ring self-touches, repair it the same way _safe_polygon does.
            shell = shell.buffer(0)
        # Full body width everywhere except within one body radius of the seed
        # point, so the region starts at the point and widens to the body.
        widened = shell.buffer(agent.body_radius).difference(
            Point(agent.x, agent.y).buffer(agent.body_radius)
        )
        region = shell.union(widened)
    if drivable is not None:
        region = region.intersection(drivable)
    return region


def reachable_set(
    agent: Agent, t: float, params: ReachabilityParams, drivable: BaseGeometry | None
) -> BaseGeometry:
    """The agent's reachable-at-``t`` set in the map frame.

    Wheeled fronts are intersected with ``drivable`` (off-road arcs are dropped),
    so a wheeled evaluation needs a drivable polygon. VRU discs and static
    footprints ignore ``drivable``.
    """
    if agent.kind == STATIC:
        return agent.footprint
    if agent.kind == VRU:
        return Point(agent.x, agent.y).buffer(agent.speed * t + agent.body_radius)
    front = wheeled_front(agent, t, params)
    if drivable is not None:
        front = front.intersection(drivable)
    return front


class EgoReachability:
    """Ego reachable sets for one frame, shared across that frame's objects.

    Everything that depends only on ego is computed once: the drivable area localized to ego's
    horizon disc (overlap can only happen there, so clipping against the full map union per step
    is wasted work), the filled reachable region ("hat") the prescreen tests against, and, lazily
    one per step, the prepared reachable-at-``t`` fronts. :meth:`time_to_collision` then runs the
    two-step contract per object.

    The prescreen bounds are supersets of the true reachable sets (every front lies inside the
    hat, and inside the disc of radius ``speed * t + body``), so skipping a step or an object
    they exclude never misses a real overlap.
    """

    def __init__(self, ego: Agent, drivable: BaseGeometry, params: ReachabilityParams) -> None:
        """Localize the drivable area and build the prescreen region for one ego frame.

        Args:
            ego: The ego agent, must be wheeled.
            drivable: Road-region union in the map frame that wheeled fronts are clipped to.
            params: Propagation parameters shared by every evaluation.
        """
        if ego.kind != WHEELED:
            raise ValueError("ego must be a wheeled agent.")
        if drivable is None:
            raise ValueError("ego reachability needs a drivable polygon to clip against.")
        self.ego = ego
        self.params = params
        self.steps = int(round(params.horizon_s / params.dt_s))
        reach = ego.speed * params.horizon_s + ego.body_radius
        self._drivable = drivable.intersection(
            box(ego.x - reach, ego.y - reach, ego.x + reach, ego.y + reach)
        )
        self._hat = reachable_region(ego, params, self._drivable)
        shapely.prepare(self._hat)
        self._fronts: list[BaseGeometry | None] = [None] * (self.steps + 1)

    def _front(self, index: int) -> BaseGeometry:
        """Ego's prepared reachable-at-``index * dt`` front (built once per step)."""
        front = self._fronts[index]
        if front is None:
            front = wheeled_front(self.ego, index * self.params.dt_s, self.params)
            front = front.intersection(self._drivable)
            shapely.prepare(front)
            self._fronts[index] = front
        return front

    def _first_step(self, distance: float, speed: float, body: float) -> int | None:
        """Earliest step at which a set growing ``speed * t + body`` spans ``distance``.

        ``None`` means never within the horizon, the caller returns ``inf``.
        """
        if distance <= body:
            return 1
        if speed <= 0.0:
            return None
        t_min = (distance - body) / speed
        if t_min > self.params.horizon_s:
            return None
        return max(1, ceil(t_min / self.params.dt_s - 1e-9))

    def time_to_collision(self, obj: Agent) -> float:
        """Earliest ``t`` at which ego and ``obj`` can occupy a common point at ``t``."""
        if self._hat.is_empty:
            return inf
        ego, params = self.ego, self.params

        if obj.kind == STATIC:
            # A static set never grows: it must already meet the hat.
            if not self._hat.intersects(obj.footprint):
                return inf
            # Ego's front is within speed * t + body of ego's position.
            distance = obj.footprint.distance(Point(ego.x, ego.y))
            start = self._first_step(distance, ego.speed, ego.body_radius)
            if start is None:
                return inf
            for index in range(start, self.steps + 1):
                if self._front(index).intersects(obj.footprint):
                    return index * params.dt_s
            return inf

        # Moving object: its set at t is within speed * t + body of its position, so
        # it must be able to reach the hat, and jointly close the gap to ego, in time.
        hat_distance = self._hat.distance(Point(obj.x, obj.y))
        start_hat = self._first_step(hat_distance, obj.speed, obj.body_radius)
        gap = hypot(obj.x - ego.x, obj.y - ego.y)
        start_gap = self._first_step(
            gap, ego.speed + obj.speed, ego.body_radius + obj.body_radius
        )
        if start_hat is None or start_gap is None:
            return inf
        for index in range(max(start_hat, start_gap), self.steps + 1):
            t = index * params.dt_s
            obj_set = reachable_set(obj, t, params, self._drivable)
            if not obj_set.is_empty and self._front(index).intersects(obj_set):
                return t
        return inf


def time_to_collision(
    ego: Agent,
    obj: Agent,
    drivable: BaseGeometry | None,
    params: ReachabilityParams,
) -> float:
    """Earliest ``t`` at which ego and ``obj`` can occupy a common point at time ``t``.

    Returns ``inf`` when no such ``t`` exists within the horizon. ``ego`` must be a wheeled agent
    and needs ``drivable`` (fronts are clipped to it), pass the road/road_shoulder/crosswalk
    union in the map frame. One-pair form of :class:`EgoReachability`, which callers evaluating
    many objects against the same ego frame should build once and query instead.
    """
    return EgoReachability(ego, drivable, params).time_to_collision(obj)


def collision_weights(ttc: np.ndarray, decay: float) -> np.ndarray:
    """Risk weight ``e^(-decay * TTC)`` per entry. An unreachable object (``inf``) weighs 0."""
    if decay < 0.0:
        raise ValueError("decay must be >= 0.")
    ttc = np.asarray(ttc, dtype=np.float64)
    weights = np.zeros_like(ttc)
    finite = np.isfinite(ttc)
    weights[finite] = np.exp(-decay * ttc[finite])
    return weights
