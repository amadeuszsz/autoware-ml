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

"""Tests for the polyline map head on the shared sparse query decoder."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import torch

from autoware_ml.dataclasses.batch.online_map import OnlineMapGTBatch
from autoware_ml.dataclasses.models.motion.online_map_outputs import OnlineMapHeadOutputs
from autoware_ml.models.detection3d.ptv3 import SparseStage
from autoware_ml.models.motion.online_map_blocks import PolylineInstanceBank
from autoware_ml.models.motion.online_map_head import OnlineMapHead
from autoware_ml.models.motion.tests.online_map_fixtures import NUM_POINTS, build_head


def stage(batch_size: int, num_voxels: int = 64) -> SparseStage:
    """Random voxels of every sample over the 8 x 8 x 4 test grid."""
    generator = torch.Generator().manual_seed(0)
    total = batch_size * num_voxels
    grid_coord = torch.stack(
        [torch.randint(0, size, (total,), generator=generator) for size in (8, 8, 4)], dim=1
    )
    feat = torch.randn((total, 32), generator=generator)
    return SparseStage(feat, grid_coord, torch.arange(1, batch_size + 1) * num_voxels)


def run_frame(
    head: OnlineMapHead, stages: list[SparseStage], timestamp: float = 0.0
) -> OnlineMapHeadOutputs:
    """Run the head on one frame of every lane, each lane streaming its own scene."""
    batch_size = stages[0].offset.shape[0]
    return head(
        stages,
        timestamps_seconds=torch.full((batch_size,), timestamp, dtype=torch.float64),
        lidar2global=torch.eye(4, dtype=torch.float64).repeat(batch_size, 1, 1),
        scene_tokens=[f"scene_{lane}" for lane in range(batch_size)],
    )


def ground_truth() -> OnlineMapGTBatch:
    """Two elements in the first sample, a line and a closed ring, none in the second."""
    line = torch.stack([torch.linspace(1.0, 7.0, NUM_POINTS), torch.full((NUM_POINTS,), 2.0)], -1)
    ring = torch.tensor([[5.0, 5.0], [6.0, 5.0], [6.0, 6.0], [5.0, 5.0]])
    return OnlineMapGTBatch(
        points=torch.stack([torch.stack([line, ring]), torch.zeros(2, NUM_POINTS, 2)]),
        labels=torch.tensor([[0, 2], [-1, -1]]),
        closed=torch.tensor([[False, True], [False, False]]),
    )


class TestOnlineMapHead(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp_path = Path(tmp.name)

    def test_training_steps_fit_the_elements(self) -> None:
        """
        Input: 10 Adam steps on one batch with two elements in one sample and none in the other.
        Expected: two matches per layer and the summed loss falls below half its start.
        """
        torch.manual_seed(0)
        head = build_head(self.tmp_path)
        optimizer = torch.optim.Adam(head.parameters(), lr=5e-3)
        stages = [stage(2)]

        losses = []
        for _ in range(10):
            loss_dict = head.loss(run_frame(head, stages), ground_truth())
            optimizer.zero_grad()
            loss_dict["loss"].backward()
            optimizer.step()
            losses.append(float(loss_dict["loss"]))

        self.assertEqual(float(loss_dict["num_pos"]), 2.0)
        self.assertLess(losses[-1], losses[0] / 2)

    def test_a_temporal_head_carries_its_polylines_through_the_frame_motion(self) -> None:
        """
        Input: a polyline cached in a frame that then turns by 90 degrees and moves 1 m along x,
          and a temporal head run over two frames of one scene.
        Expected: the cached points expressed in the new frame, and the second frame continues the
          stream of the first.
        """
        previous_to_current = torch.eye(4)[None].clone()
        previous_to_current[0, :2, :2] = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
        previous_to_current[0, 0, 3] = 1.0
        anchor = torch.tensor([[[1.0, 0.0, 2.0, 0.0]]])

        moved = PolylineInstanceBank.project_anchor(anchor, torch.ones(1), previous_to_current)

        torch.testing.assert_close(moved, torch.tensor([[[1.0, 1.0, 1.0, 2.0]]]))
        head = build_head(self.tmp_path).eval()
        for frame in range(2):
            with torch.no_grad():
                run_frame(head, [stage(1)], 0.1 * frame)
        self.assertEqual(head.instance_bank.lane_continues.tolist(), [True])

    def test_decoding_keeps_the_best_query_and_class_pairs(self) -> None:
        """
        Input: last layer logits where query 1 is class 2 at score 0.9 and query 0 class 0 at 0.6.
        Expected: those two pairs lead the decoded elements with their polylines.
        """
        head = OnlineMapHead.__new__(OnlineMapHead)
        head.num_classes, head.num_points, head.num_output = 3, NUM_POINTS, 2
        logits = torch.full((1, 1, 2, 3), -10.0)
        logits[0, 0, 1, 2] = math.log(0.9 / 0.1)
        logits[0, 0, 0, 0] = math.log(0.6 / 0.4)
        prediction = torch.arange(16.0).reshape(1, 1, 2, 8)
        outputs = OnlineMapHeadOutputs(
            classification=logits, prediction=prediction, instance_feature=torch.zeros(1, 2, 4)
        )

        (decoded,) = head.decode_outputs(outputs)

        self.assertEqual(decoded.labels.tolist(), [2, 0])
        torch.testing.assert_close(decoded.scores, torch.tensor([0.9, 0.6]))
        torch.testing.assert_close(decoded.points[0], prediction[0, 0, 1].reshape(4, 2))
