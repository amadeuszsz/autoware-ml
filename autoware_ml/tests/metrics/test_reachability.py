"""Unit tests for the reachability time-to-collision engine."""

from __future__ import annotations

from math import inf, isclose, pi

import numpy as np
import pytest
from shapely.geometry import Point, box

from autoware_ml.metrics.geometry.reachability import (
    STATIC,
    VRU,
    WHEELED,
    Agent,
    EgoReachability,
    ReachabilityParams,
    collision_weights,
    reachable_region,
    reachable_set,
    time_to_collision,
)

PARAMS = ReachabilityParams(horizon_s=4.0, dt_s=0.1)
# A wide open drivable region so wheeled fronts are not clipped in these unit cases.
ROAD = box(-80.0, -50.0, 500.0, 50.0)


def _footprint(x: float, y: float, size: float = 2.0):
    return box(x - size / 2, y - size / 2, x + size / 2, y + size / 2)


def test_same_speed_lead_never_collides() -> None:
    # Following a vehicle at matched speed: fronts stay a constant gap apart -> inf.
    ego = Agent(WHEELED, 0.0, 0.0, heading=0.0, speed=10.0, body_radius=1.2)
    lead = Agent(WHEELED, 25.0, 0.0, heading=0.0, speed=10.0, body_radius=1.2)
    assert time_to_collision(ego, lead, ROAD, PARAMS) == inf


def test_stationary_object_ahead_finite_near_distance_over_speed() -> None:
    ego = Agent(WHEELED, 0.0, 0.0, heading=0.0, speed=10.0, body_radius=1.0)
    obj = Agent(STATIC, 30.0, 0.0, footprint=_footprint(30.0, 0.0), body_radius=1.0)
    ttc = time_to_collision(ego, obj, ROAD, PARAMS)
    assert ttc != inf
    assert 2.4 <= ttc <= 3.1  # ~ (30 - body - half-footprint) / 10


def test_oncoming_closes_at_combined_speed() -> None:
    ego = Agent(WHEELED, 0.0, 0.0, heading=0.0, speed=10.0, body_radius=1.0)
    obj = Agent(WHEELED, 40.0, 0.0, heading=pi, speed=10.0, body_radius=1.0)
    ttc = time_to_collision(ego, obj, ROAD, PARAMS)
    assert ttc != inf
    assert 1.7 <= ttc <= 2.1  # ~ 40 / (10 + 10)


def test_oncoming_beyond_ego_reach_still_collides() -> None:
    # The object's approach path is checked on the full drivable surface localized to
    # its own reach, so an incoming vehicle starting outside ego's reach clip is found.
    ego = Agent(WHEELED, 0.0, 0.0, heading=0.0, speed=10.0, body_radius=1.0)
    obj = Agent(WHEELED, 50.0, 0.0, heading=pi, speed=10.0, body_radius=1.0)
    ttc = time_to_collision(ego, obj, ROAD, PARAMS)
    assert 2.3 <= ttc <= 2.6  # ~ 50 / (10 + 10)


def test_crossing_vru_is_finite_within_horizon() -> None:
    ego = Agent(WHEELED, 0.0, 0.0, heading=0.0, speed=10.0, body_radius=1.0)
    ped = Agent(VRU, 18.0, 6.0, speed=4.0, body_radius=0.4)
    ttc = time_to_collision(ego, ped, ROAD, PARAMS)
    assert ttc != inf and ttc <= PARAMS.horizon_s


def test_far_object_is_rejected_cheaply() -> None:
    ego = Agent(WHEELED, 0.0, 0.0, heading=0.0, speed=10.0, body_radius=1.0)
    obj = Agent(STATIC, 200.0, 0.0, footprint=_footprint(200.0, 0.0))
    assert time_to_collision(ego, obj, ROAD, PARAMS) == inf


def test_wheeled_front_needs_drivable() -> None:
    ego = Agent(WHEELED, 0.0, 0.0, heading=0.0, speed=10.0)
    obj = Agent(STATIC, 20.0, 0.0, footprint=_footprint(20.0, 0.0))
    with pytest.raises(ValueError, match="drivable"):
        time_to_collision(ego, obj, None, PARAMS)


def test_disconnected_road_is_unreachable() -> None:
    # Two drivable strips separated by a non-drivable gap: arcs whose endpoints land
    # on the other strip cross the gap, so the strips can never meet. A wheeled agent
    # off the surface entirely has no feasible arc at all.
    split_road = box(-80.0, -10.0, 200.0, 10.0).union(box(-80.0, 20.0, 200.0, 40.0))
    ego = Agent(WHEELED, 0.0, 0.0, heading=0.0, speed=10.0, body_radius=1.0)
    oncoming_across = Agent(WHEELED, 5.0, 30.0, heading=-pi / 2, speed=10.0, body_radius=1.0)
    assert time_to_collision(ego, oncoming_across, split_road, PARAMS) == inf
    off_road = Agent(WHEELED, 0.0, 60.0, heading=0.0, speed=10.0, body_radius=1.0)
    assert time_to_collision(ego, off_road, ROAD, PARAMS) == inf
    # The filled reachable region keeps only the strip the agent is on.
    region = reachable_region(ego, PARAMS, split_road)
    assert region.intersection(box(-80.0, 20.0, 200.0, 40.0)).is_empty


def test_params_reject_step_exceeding_horizon() -> None:
    with pytest.raises(ValueError, match="dt_s must not exceed horizon_s"):
        ReachabilityParams(horizon_s=0.5, dt_s=0.6)


def test_steps_stay_within_horizon() -> None:
    # A non-divisible horizon/dt floors to the last step inside the horizon, while an
    # exact multiple keeps its final step despite floating point.
    ego = Agent(WHEELED, 0.0, 0.0, heading=0.0, speed=1.0, body_radius=0.5)
    assert EgoReachability(ego, ROAD, ReachabilityParams(horizon_s=1.0, dt_s=0.6)).steps == 1
    assert EgoReachability(ego, ROAD, ReachabilityParams(horizon_s=3.0, dt_s=0.1)).steps == 30
    # A meeting first reachable at 1.1 s lies beyond the 1.0 s horizon, so it stays inf.
    vru = Agent(VRU, 7.6, 0.0, speed=5.0, body_radius=0.5)
    assert time_to_collision(ego, vru, ROAD, ReachabilityParams(horizon_s=1.0, dt_s=0.6)) == inf


def test_collision_weights_monotone_and_bounds() -> None:
    weights = collision_weights([inf, 0.0, 1.0, 3.0], 0.1)
    assert weights[0] == 0.0
    assert isclose(weights[1], 1.0)
    assert weights[2] > weights[3] > 0.0


def test_low_speed_region_stays_valid_past_pi_sweep() -> None:
    # At lanelet speeds around 10 km/h the max-curvature arcs sweep past pi and
    # the raw hat ring self-touches, the region must still come out valid with
    # sane membership (regression for the buffer(0) repair).
    for speed in (0.83, 2.78, 3.0):
        agent = Agent(WHEELED, 0.0, 0.0, heading=0.0, speed=speed, body_radius=1.0)
        region = reachable_region(agent, PARAMS, ROAD)
        assert region.is_valid and not region.is_empty
        assert region.contains(Point(min(speed * PARAMS.horizon_s * 0.9, 10.0), 0.0))


def test_ego_reachability_matches_bruteforce_stepping() -> None:
    # The prescreen bounds are supersets of the true sets, so the optimized
    # engine must return exactly what naive stepping over every t returns.
    def brute_force(ego: Agent, obj: Agent) -> float:
        steps = int(PARAMS.horizon_s / PARAMS.dt_s + 1e-9)
        for index in range(1, steps + 1):
            t = index * PARAMS.dt_s
            ego_set = reachable_set(ego, t, PARAMS, ROAD)
            obj_set = reachable_set(obj, t, PARAMS, ROAD)
            if not ego_set.is_empty and not obj_set.is_empty and ego_set.intersects(obj_set):
                return t
        return inf

    rng = np.random.default_rng(7)
    ego = Agent(WHEELED, 0.0, 0.0, heading=0.0, speed=13.9, body_radius=1.0)
    frame = EgoReachability(ego, ROAD, PARAMS)
    for index in range(40):
        x, y = float(rng.uniform(-30, 130)), float(rng.uniform(-45, 45))
        kind = (WHEELED, VRU, STATIC)[index % 3]
        if kind == STATIC:
            obj = Agent(STATIC, x, y, footprint=_footprint(x, y))
        elif kind == VRU:
            obj = Agent(VRU, x, y, speed=float(rng.uniform(0.0, 6.0)), body_radius=0.4)
        else:
            obj = Agent(
                WHEELED, x, y, heading=float(rng.uniform(0, 2 * pi)),
                speed=float(rng.uniform(0.5, 16.7)), body_radius=1.0,
            )
        assert frame.time_to_collision(obj) == brute_force(ego, obj), f"agent {index}: {obj}"
