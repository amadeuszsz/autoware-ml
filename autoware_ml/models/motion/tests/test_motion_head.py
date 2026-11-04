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

"""Tests for the SparseDrive agent motion head."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from autoware_ml.anchors.tests.fixtures import FixedAnchorRecipe, fill_anchors
from autoware_ml.dataclasses.batch.motion import MotionGTBatch
from autoware_ml.dataclasses.models.motion.head_outputs import MotionHeadOutputs
from autoware_ml.models.detection3d.heads.sparse4d import (
    ANCHOR_DIM,
    InstanceBank,
    SparseBox3DEncoder,
)
from autoware_ml.models.motion.motion_head import MotionHead
from autoware_ml.models.multi.ptv3_sparsedrive_motion import PTv3SparseDriveMotionModel

DIMS, QUERIES, CLASSES = 32, 12, 3


def build_head(tmp_path: Path, **overrides: object) -> MotionHead:
    """A small head with random motion anchors and the given settings."""
    rng = np.random.default_rng(0)
    motion_anchors = FixedAnchorRecipe({"anchors": rng.normal(size=(CLASSES, 6, 4, 2))})
    torch.manual_seed(0)
    head = MotionHead(
        num_classes=CLASSES,
        embed_dims=DIMS,
        anchor_recipe=motion_anchors,
        num_heads=4,
        agent_steps=4,
        num_agent_keys=5,
        queue_length=2,
        queue_stride=1,
        **overrides,
    )
    return fill_anchors(head)


def random_anchor(batch_size: int) -> torch.Tensor:
    """Random anchors with a unit heading."""
    anchor = torch.randn(batch_size, QUERIES, 10)
    anchor[..., 6:8] = torch.nn.functional.normalize(anchor[..., 6:8], dim=-1)
    return anchor


def run(
    head: MotionHead,
    batch_size: int = 2,
    feature: torch.Tensor | None = None,
    encoder: SparseBox3DEncoder | None = None,
) -> MotionHeadOutputs:
    """One frame of random detections."""
    return head(
        torch.randn(batch_size, QUERIES, DIMS) if feature is None else feature,
        random_anchor(batch_size),
        torch.randn(batch_size, QUERIES, CLASSES),
        SparseBox3DEncoder(DIMS) if encoder is None else encoder,
        torch.zeros(batch_size, dtype=torch.bool),
        None,
        torch.eye(4).repeat(batch_size, 1, 1),
    )


class TestMotionHead(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp_path = Path(tmp.name)

    def test_the_loss_is_finite_and_reaches_every_parameter(self) -> None:
        """
        Input: two frames, queries 0 and 1 matched to boxes with futures.
        Expected: finite mode and trajectory terms, two motion targets, a gradient for every
          parameter.
        """
        head = build_head(self.tmp_path)
        outputs = run(head)
        gt = MotionGTBatch(
            agent_future_positions=torch.randn(2, 3, 4, 2),
            agent_future_valid=torch.ones(2, 3, 4, dtype=torch.bool),
        )
        assigned = torch.full((2, QUERIES), -1)
        assigned[0, 0], assigned[1, 1] = 11, 12
        ids = torch.tensor([[11, -1, -1], [5, 12, -1]])
        losses = head.loss(outputs, assigned, torch.zeros(2, 3, 10), ids, gt)
        self.assertEqual(
            set(losses), {"loss_motion_cls", "loss_motion_reg", "num_motion_targets", "loss"}
        )
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        self.assertEqual(losses["num_motion_targets"].item(), 2)
        losses["loss"].backward()
        self.assertTrue(all(p.grad is not None for p in head.parameters() if p.requires_grad))

    def test_a_loss_over_no_frame_reports_every_term_as_zero(self) -> None:
        """
        Input: the outputs of two frames, neither of them supervised.
        Expected: the terms of a supervised loss, zero and still in the graph.
        """
        head = build_head(self.tmp_path)
        none = torch.zeros(2, dtype=torch.bool)
        gt = MotionGTBatch(
            agent_future_positions=torch.randn(2, 3, 4, 2),
            agent_future_valid=torch.ones(2, 3, 4, dtype=torch.bool),
        )

        losses = head.loss(
            run(head).select_samples(none),
            torch.full((0, QUERIES), -1),
            torch.zeros(0, 3, 10),
            torch.zeros(0, 3, dtype=torch.long),
            gt.select_samples(none),
        )

        self.assertEqual(
            list(losses), ["loss_motion_cls", "loss_motion_reg", "num_motion_targets", "loss"]
        )
        self.assertTrue(all(value.item() == 0.0 for value in losses.values()))
        losses["loss"].backward()

    def test_a_detached_head_trains_without_touching_the_detection(self) -> None:
        """
        Input: a head built with detach_detection run on detection features and an anchor encoder
          that track gradients, then its outputs backpropagated.
        Expected: the head parameters get gradients, the detection features and the anchor encoder
          none.
        """
        head = build_head(self.tmp_path, detach_detection=True)
        encoder = SparseBox3DEncoder(DIMS)
        feature = torch.randn(1, QUERIES, DIMS, requires_grad=True)
        run(
            head, batch_size=1, feature=feature, encoder=encoder
        ).agent_trajectories.sum().backward()

        self.assertIsNone(feature.grad)
        self.assertTrue(all(parameter.grad is None for parameter in encoder.parameters()))
        self.assertIsNotNone(head.motion_reg[-1].weight.grad)

    def test_a_query_carried_into_the_next_frame_reads_its_history(self) -> None:
        """
        Input: two frames of one lane, the second carrying its first temporal slot from query 0, run
          once after a first frame with other features.
        Expected: the trajectories of the carried query depend on the first frame, those of the
          fresh queries do not.
        """
        head = build_head(self.tmp_path).eval()
        anchor = random_anchor(1)
        classes = torch.randn(1, QUERIES, CLASSES)
        encoder = SparseBox3DEncoder(DIMS)
        second = torch.randn(1, QUERIES, DIMS)
        trajectories = []
        for seed in range(2):
            head.reset()
            first = torch.randn(1, QUERIES, DIMS, generator=torch.Generator().manual_seed(seed))
            with torch.no_grad():
                for feature, continues in ((first, False), (second, True)):
                    outputs = head(
                        feature,
                        anchor,
                        classes,
                        encoder,
                        torch.tensor([continues]),
                        torch.zeros(1, 1, dtype=torch.long),
                        torch.eye(4)[None],
                    )
            trajectories.append(outputs.agent_trajectories)

        self.assertFalse(torch.allclose(trajectories[0][:, 0], trajectories[1][:, 0]))
        torch.testing.assert_close(trajectories[0][:, 1:], trajectories[1][:, 1:])

    def test_a_new_batch_size_drops_the_carried_slots(self) -> None:
        """
        Input: frames of 8, 8 and 1 lanes in the order of the motion model, the slot index of the
          previous frame first, then the detection bank and the motion head.
        Expected: every frame runs, the one lane frame reads no slot of the eight lane batch.
        """
        anchors = np.zeros((QUERIES, ANCHOR_DIM), dtype=np.float32)
        anchors[:, 7] = 1.0
        bank = fill_anchors(
            InstanceBank(
                num_anchor=QUERIES,
                embed_dims=DIMS,
                num_temp_instances=4,
                anchor_recipe=FixedAnchorRecipe({"anchors": anchors}),
            )
        )
        head = build_head(self.tmp_path)
        encoder = SparseBox3DEncoder(DIMS)
        for step, batch_size in enumerate((8, 8, 1)):
            carried_index = bank.cached_query_index
            timestamps = torch.full((batch_size,), 0.1 * step, dtype=torch.float64)
            poses = torch.eye(4).repeat(batch_size, 1, 1)
            bank.get(batch_size, timestamps, poses, [f"scene{lane}" for lane in range(batch_size)])
            feature = torch.randn(batch_size, QUERIES, DIMS)
            anchor = bank.anchor.detach()[None].expand(batch_size, -1, -1).clone()
            logits = torch.randn(batch_size, QUERIES, CLASSES)
            bank.cache(feature, anchor, logits, poses)
            outputs = head(
                feature, anchor, logits, encoder, bank.lane_continues, carried_index, poses
            )

            self.assertEqual(outputs.agent_trajectories.shape[0], batch_size)

    def test_only_the_angle_of_the_heading_turns_the_mode_anchors(self) -> None:
        """
        Input: the same queries with unit headings and with the headings shrunk to 60 %.
        Expected: the regressed heading is not unit length, only its angle turns the anchors.
        Check: equal mode logits and trajectories.
        """
        head = build_head(self.tmp_path).eval()
        context = torch.randn(2, QUERIES, DIMS)
        anchor = random_anchor(2)
        shrunk = anchor.clone()
        shrunk[..., 6:8] *= 0.6
        cls_logits = torch.randn(2, QUERIES, CLASSES)

        with torch.no_grad():
            unit_logits, unit_trajectories = head._predict_agents(context, anchor, cls_logits)
            shrunk_logits, shrunk_trajectories = head._predict_agents(context, shrunk, cls_logits)

        torch.testing.assert_close(shrunk_logits, unit_logits)
        torch.testing.assert_close(shrunk_trajectories, unit_trajectories)

    def test_the_model_refuses_futures_of_other_steps_than_its_head(self) -> None:
        """
        Input: a batch with 12 future steps for a head that predicts 6, and a batch without futures.
        Expected: a ValueError for both, a gather over the steps would truncate the futures.
        """
        model = SimpleNamespace(motion_head=SimpleNamespace(agent_steps=6))
        futures = MotionGTBatch(
            agent_future_positions=torch.zeros(1, 2, 12, 2),
            agent_future_valid=torch.ones(1, 2, 12, dtype=torch.bool),
        )

        def batch(motion_gt: MotionGTBatch | None) -> SimpleNamespace:
            return SimpleNamespace(multi_task_gt_batch=SimpleNamespace(motion_gt_batch=motion_gt))

        with self.assertRaisesRegex(ValueError, "futures of 12 steps, the motion head predicts 6"):
            PTv3SparseDriveMotionModel.motion_gt(model, batch(futures))
        with self.assertRaisesRegex(ValueError, "futures of every sample"):
            PTv3SparseDriveMotionModel.motion_gt(model, batch(None))
