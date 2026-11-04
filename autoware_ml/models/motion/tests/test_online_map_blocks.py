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

"""Tests for the query blocks of the polyline map head."""

from __future__ import annotations

import unittest

import torch

from autoware_ml.models.motion.online_map_blocks import (
    SparsePolylineKeyPointsGenerator,
    SparsePolylineRefinementModule,
)


class TestOnlineMapBlocks(unittest.TestCase):
    def test_keypoints_surround_every_point_at_every_height(self) -> None:
        """
        Input: a generator with zeroed offsets, 3 points, heights 0 and 0.5, 2 keypoints each.
        Expected: 12 keypoints per query, each on its polyline point at its height.
        """
        generator = SparsePolylineKeyPointsGenerator(
            embed_dims=8, num_points=3, fix_heights=(0.0, 0.5), num_learnable_pts=2
        )
        torch.nn.init.zeros_(generator.learnable_fc.weight)
        anchor = torch.tensor([[[0.0, 0.0, 1.0, 0.0, 2.0, 1.0]]])

        key_points = generator(anchor, torch.randn(1, 1, 8))

        self.assertEqual(generator.num_pts, 12)
        self.assertEqual(key_points.shape, (1, 1, 12, 3))
        expected_xy = anchor.reshape(3, 2).repeat_interleave(4, dim=0)
        torch.testing.assert_close(key_points[0, 0, :, :2], expected_xy)
        self.assertEqual(key_points[0, 0, :, 2].tolist(), [0.0, 0.0, 0.5, 0.5] * 3)

    def test_refinement_moves_the_anchor_and_starts_at_a_low_class_prior(self) -> None:
        """
        Input: a refinement module over 4-point anchors and 3 classes.
        Expected: refined anchors of the anchor shape and class probabilities of 0.01.
        """
        module = SparsePolylineRefinementModule(embed_dims=8, num_points=4, num_classes=3)
        torch.nn.init.zeros_(module.cls_layers[-1].weight)

        refined, logits = module(torch.randn(2, 5, 8), torch.zeros(2, 5, 8), torch.zeros(2, 5, 8))

        self.assertEqual(refined.shape, (2, 5, 8))
        self.assertEqual(logits.shape, (2, 5, 3))
        torch.testing.assert_close(logits.sigmoid(), torch.full((2, 5, 3), 0.01))
