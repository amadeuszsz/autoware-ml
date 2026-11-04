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

"""Tests for the vectorization and collation of the map elements."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from autoware_ml.dataclasses.batch.online_map import (
    OnlineMapGTBatch,
    OnlineMapGTSample,
    VectorMapSpec,
)
from autoware_ml.dataclasses.geometry.transformation import LiDARTransformationSample
from autoware_ml.geometry.vector_map import FrameMapElements

SPEC = VectorMapSpec(
    roi=(-50.0, -25.0, 50.0, 25.0), num_points=5, min_length=1.0, crop_half_size=70.0
)


def elements(*shapes: tuple[list[tuple[float, float]], int, bool]) -> FrameMapElements:
    """Elements from (vertices, label, area) triples."""
    return FrameMapElements(
        points=np.array([point for vertices, _, _ in shapes for point in vertices], dtype=float),
        lengths=np.array([len(vertices) for vertices, _, _ in shapes], dtype=np.int64),
        labels=np.array([label for _, label, _ in shapes], dtype=np.int64),
        areas=np.array([area for _, _, area in shapes], dtype=bool),
    )


# A line crossing the region along x, a square area inside it, an area cut by the region border
# and a line whose part inside the region is 0.5 m long
FRAME = elements(
    ([(-60.0, 0.0), (60.0, 0.0)], 0, False),
    ([(10.0, 10.0), (14.0, 10.0), (14.0, 14.0), (10.0, 14.0), (10.0, 10.0)], 2, True),
    ([(45.0, 0.0), (55.0, 0.0), (55.0, 4.0), (45.0, 4.0), (45.0, 0.0)], 2, True),
    ([(49.5, -20.0), (60.0, -20.0)], 1, False),
)


def rotation(yaw: float) -> LiDARTransformationSample:
    """Augmentation turning the cloud by yaw about the origin."""
    matrix = torch.eye(4)
    matrix[:2, :2] = torch.tensor(
        [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]], dtype=torch.float32
    )
    return LiDARTransformationSample(transformation_matrix=matrix, transformation_order=[])


class TestOnlineMap(unittest.TestCase):
    def test_elements_are_cut_to_the_region_and_resampled(self) -> None:
        """
        Input: the frame without augmentation.
        Expected: the crossing line is cut to x -50 to 50 and resampled every 25 m. The inner square
          stays a closed ring, the cut area closes along the region border, the 0.5 m piece is
          dropped.
        """
        batch = OnlineMapGTBatch.collate_gt_samples([OnlineMapGTSample(FRAME, SPEC)], [None])

        self.assertEqual(batch.labels.tolist(), [[0, 2, 2]])
        self.assertEqual(batch.closed.tolist(), [[False, True, True]])
        torch.testing.assert_close(
            batch.points[0, 0, :, 0], torch.tensor([-50.0, -25.0, 0.0, 25.0, 50.0])
        )
        torch.testing.assert_close(batch.points[0, 0, :, 1], torch.zeros(5))
        for ring in batch.points[0, 1:]:
            torch.testing.assert_close(ring[0], ring[-1])
        self.assertAlmostEqual(batch.points[0, 2, :, 0].max().item(), 50.0, places=5)

    def test_the_augmentation_moves_the_elements_before_the_cut(self) -> None:
        """
        Input: the frame turned by 90 degrees.
        Expected: the crossing line runs along y and is cut to y -25 to 25 by the region.
        """
        batch = OnlineMapGTBatch.collate_gt_samples(
            [OnlineMapGTSample(FRAME, SPEC)], [rotation(np.pi / 2)]
        )

        line = batch.points[0, batch.labels[0].tolist().index(0)]
        torch.testing.assert_close(line[:, 0], torch.zeros(5), atol=1e-5, rtol=0.0)
        torch.testing.assert_close(torch.sort(line[[0, -1], 1]).values, torch.tensor([-25.0, 25.0]))

    def test_samples_pad_to_the_most_elements(self) -> None:
        """
        Input: a frame with three kept elements and a frame with one.
        Expected: three slots per frame, the free ones labeled -1.
        """
        single = elements(([(0.0, 0.0), (10.0, 0.0)], 1, False))

        batch = OnlineMapGTBatch.collate_gt_samples(
            [OnlineMapGTSample(FRAME, SPEC), OnlineMapGTSample(single, SPEC)], [None, None]
        )

        self.assertEqual(batch.points.shape, (2, 3, 5, 2))
        self.assertEqual(batch.labels[1].tolist(), [1, -1, -1])
        self.assertEqual(batch.closed[1].tolist(), [False, False, False])

    def test_a_region_leaving_the_crop_square_raises(self) -> None:
        """
        Input: a crop square of half size 50 while the region corners lie 56 m from the origin.
        Expected: a ValueError naming the margin.
        """
        spec = SPEC._replace(crop_half_size=50.0)

        with self.assertRaisesRegex(ValueError, "Raise the crop margin"):
            OnlineMapGTBatch.collate_gt_samples(
                [OnlineMapGTSample(FRAME, spec)], [rotation(np.pi / 4)]
            )

    def test_samples_with_different_specs_raise(self) -> None:
        """
        Input: two samples with different point counts.
        Expected: a ValueError.
        """
        with self.assertRaisesRegex(ValueError, "one map spec"):
            OnlineMapGTBatch.collate_gt_samples(
                [
                    OnlineMapGTSample(FRAME, SPEC),
                    OnlineMapGTSample(FRAME, SPEC._replace(num_points=7)),
                ],
                [None, None],
            )
