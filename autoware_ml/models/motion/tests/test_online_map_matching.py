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

"""Tests for the order invariant matching of predicted polylines to map elements."""

from __future__ import annotations

import unittest

import torch

from autoware_ml.dataclasses.batch.online_map import OnlineMapGTBatch
from autoware_ml.models.detection3d.task_modules.match_costs import ClassificationCost
from autoware_ml.models.motion.online_map_matching import (
    assign_polylines,
    point_orders,
    polyline_cost,
)

ROI = (-10.0, -10.0, 10.0, 10.0)
LINE = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
SQUARE = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])


class TestOnlineMapMatching(unittest.TestCase):
    def test_an_open_element_matches_in_both_directions_and_a_ring_from_every_start(self) -> None:
        """
        Input: a 3-point line and a closed 5-point square ring.
        Expected: two valid orders of the line, forward and reversed, and eight orders of the
          ring, each a closed ring over the same corners.
        """
        line_orders, line_valid = point_orders(LINE[None], torch.tensor([False]))
        ring_orders, ring_valid = point_orders(SQUARE[None], torch.tensor([True]))

        self.assertEqual(line_valid.tolist(), [[True, True, False, False]])
        torch.testing.assert_close(line_orders[0, 0], LINE)
        torch.testing.assert_close(line_orders[0, 1], LINE.flip(0))
        self.assertTrue(ring_valid.all())
        self.assertEqual(ring_orders.shape, (1, 8, 5, 2))
        corners = {tuple(point) for point in SQUARE[:-1].tolist()}
        for order in ring_orders[0]:
            torch.testing.assert_close(order[0], order[-1])
            self.assertEqual({tuple(point) for point in order[:-1].tolist()}, corners)
        self.assertEqual(len({tuple(order.flatten().tolist()) for order in ring_orders[0]}), 8)

    def test_the_cost_takes_the_closest_order(self) -> None:
        """
        Input: a prediction equal to the reversed line.
        Expected: zero cost in the reversed order.
        """
        orders, valid = point_orders(LINE[None], torch.tensor([False]))

        cost, order = polyline_cost(LINE.flip(0)[None], orders, valid, beta=0.01)

        self.assertEqual(cost.item(), 0.0)
        self.assertEqual(order.item(), 1)

    def test_queries_match_the_elements_they_trace_in_their_order(self) -> None:
        """
        Input: three queries, one tracing the reversed line, one tracing the ring from its third
          corner, one far away, and a second sample without elements.
        Expected: the tracing queries match with targets in their own point order, the far one and
          the empty sample stay background.
        """
        ring = torch.cat([SQUARE[2:4], SQUARE[:3]])
        far = torch.full((5, 2), 8.0)
        pred = torch.stack([torch.cat([LINE, LINE[-1:].expand(2, 2)]).flip(0), ring, far])
        line = torch.cat([LINE, LINE[-1:].expand(2, 2)])
        ground_truth = OnlineMapGTBatch(
            points=torch.stack([torch.stack([line, SQUARE]), torch.zeros(2, 5, 2)]),
            labels=torch.tensor([[0, 2], [-1, -1]]),
            closed=torch.tensor([[False, True], [False, False]]),
        )

        targets = assign_polylines(
            torch.zeros(2, 3, 3),
            torch.stack([pred, pred]),
            ground_truth,
            ROI,
            ClassificationCost(),
            line_cost_weight=10.0,
            beta=0.01,
        )

        self.assertEqual(targets.matched.tolist(), [[True, True, False], [False, False, False]])
        self.assertEqual(targets.num_pos, 2)
        self.assertEqual(targets.cls_targets[0, :2].argmax(dim=-1).tolist(), [0, 2])
        expected = (torch.stack([pred[0], pred[1]]) + 10.0) / 20.0
        torch.testing.assert_close(targets.point_targets[0, :2], expected)
