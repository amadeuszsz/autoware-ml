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

"""Tests for the Sparse4D query detection head."""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch
import torch.nn as nn

from autoware_ml.anchors.tests.fixtures import FixedAnchorRecipe, fill_anchors
from autoware_ml.dataclasses.models.detection3d.head_outputs import (
    Detection3DHeadOutputs,
    Sparse4DHeadOutputs,
)
from autoware_ml.models.detection3d.heads.sparse4d import (
    ANCHOR_DIM,
    InstanceBank,
    Sparse4DHead,
    SparseBox3DKeyPointsGenerator,
    SparseDeformableAggregation,
    SparseLevelGrid,
    encode_ground_truth,
)
from autoware_ml.models.detection3d.task_modules.bbox_coders import denormalize_boxes3d
from autoware_ml.models.detection3d.tests.sparse4d_fixtures import (
    build_sparse4d_head,
    make_anchor_recipe,
    random_stage,
)

POINT_CLOUD_RANGE = [-51.2, -51.2, -3.0, 51.2, 51.2, 5.0]
NUM_ANCHOR = 16
EMBED_DIMS = 32
FEATURE_STRIDE = 2


def _build_head() -> Sparse4DHead:
    """Build a small Sparse4D head with synthetic anchors."""
    rng = np.random.default_rng(0)
    anchors = np.zeros((NUM_ANCHOR, ANCHOR_DIM), dtype=np.float32)
    anchors[:, :2] = rng.uniform(-40.0, 40.0, size=(NUM_ANCHOR, 2))
    anchors[:, 2] = rng.uniform(-1.0, 2.0, size=NUM_ANCHOR)
    anchors[:, 3:6] = np.log(rng.uniform(0.5, 5.0, size=(NUM_ANCHOR, 3)))
    anchors[:, 7] = 1.0
    head = Sparse4DHead(
        num_classes=3,
        embed_dims=EMBED_DIMS,
        anchor_recipe=FixedAnchorRecipe({"anchors": anchors}),
        num_anchor=NUM_ANCHOR,
        point_cloud_range=POINT_CLOUD_RANGE,
        grid_size=1.0,
        feature_strides=(FEATURE_STRIDE,),
        num_decoder=2,
        num_heads=4,
        num_groups=4,
        feedforward_channels=64,
        num_learnable_pts=2,
        num_output=8,
    )
    return fill_anchors(head)


def _sparse_stage_inputs(
    batch_size: int, num_voxels: int = 128
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a random sparse stage covering the fixture's pooled grid."""
    generator = torch.Generator().manual_seed(0)
    dims = (51, 51, 4)  # ceil(round(102.4 / 1.0) / 2), the same for y, and ceil(8 / 2) for z
    feat = torch.randn((batch_size * num_voxels, EMBED_DIMS), generator=generator)
    grid_coord = torch.stack(
        [
            torch.randint(0, dims[0], (batch_size * num_voxels,), generator=generator),
            torch.randint(0, dims[1], (batch_size * num_voxels,), generator=generator),
            torch.randint(0, dims[2], (batch_size * num_voxels,), generator=generator),
        ],
        dim=1,
    )
    offset = torch.arange(1, batch_size + 1, dtype=torch.long) * num_voxels
    return feat, grid_coord, offset


def _query_facing_away(class_index: int) -> tuple[torch.Tensor, list, torch.Tensor]:
    """A box of the class and the predictions, query 3 on the box with the opposite heading."""
    gt_boxes = torch.tensor([[5.0, -2.0, 0.5, 2.0, 0.5, 1.0, 0.4, 0.0, 0.0]])
    ground_truth = encode_ground_truth(
        gt_boxes, torch.tensor([class_index]), torch.device("cpu"), torch.float32
    )
    pred_anchor = torch.randn(1, NUM_ANCHOR, ANCHOR_DIM) * 20.0
    pred_anchor[0, 3] = ground_truth.anchor[0]
    pred_anchor[0, 3, 6:8] = -ground_truth.anchor[0, 6:8]
    cls_logits = torch.full((1, NUM_ANCHOR, 2), -6.0)
    cls_logits[0, 3, class_index] = 6.0
    return cls_logits, [ground_truth], pred_anchor


def _decode(head: Sparse4DHead, outputs: Sparse4DHeadOutputs) -> list:
    """Decode raw Sparse4D outputs through the head interface every detection head shares."""
    return head.decode_outputs(
        Detection3DHeadOutputs(
            center_head_outputs=None, transfusion_head_outputs=None, sparse4d_head_outputs=outputs
        )
    ).detection3d_predictions


NUM_TEMP = 4
BANK_ANCHORS = 8


def _make_bank(num_temp_instances: int = NUM_TEMP) -> InstanceBank:
    """Build a small temporal instance bank with synthetic anchors."""
    anchors = np.zeros((BANK_ANCHORS, ANCHOR_DIM), dtype=np.float32)
    anchors[:, 7] = 1.0
    bank = InstanceBank(
        num_anchor=BANK_ANCHORS,
        embed_dims=8,
        anchor_recipe=FixedAnchorRecipe({"anchors": anchors}),
        num_temp_instances=num_temp_instances,
    )
    return fill_anchors(bank)


def _pose(yaw: float, translation: tuple[float, float, float]) -> torch.Tensor:
    """Build a rigid (1, 4, 4) lidar-to-global pose."""
    pose = torch.eye(4, dtype=torch.float64)
    pose[0, 0] = np.cos(yaw)
    pose[0, 1] = -np.sin(yaw)
    pose[1, 0] = np.sin(yaw)
    pose[1, 1] = np.cos(yaw)
    pose[:3, 3] = torch.tensor(translation, dtype=torch.float64)
    return pose[None]


def _cache_frame(
    bank: InstanceBank,
    anchor: torch.Tensor,
    timestamp: torch.Tensor,
    pose: torch.Tensor,
    scene: str,
) -> None:
    """Run one get + cache cycle so the bank holds the given anchors."""
    bank.get(1, timestamp, pose, [scene])
    feature = torch.full((1, BANK_ANCHORS, 8), 7.0)
    logits = torch.full((1, BANK_ANCHORS, 3), 4.0)
    bank.cache(feature, anchor, logits, pose)


def _build_temporal_head(**overrides: object) -> Sparse4DHead:
    """Build a small temporal Sparse4D head with synthetic anchors."""
    rng = np.random.default_rng(0)
    anchors = np.zeros((NUM_ANCHOR, ANCHOR_DIM), dtype=np.float32)
    anchors[:, :2] = rng.uniform(-40.0, 40.0, size=(NUM_ANCHOR, 2))
    anchors[:, 7] = 1.0
    head = Sparse4DHead(
        num_classes=3,
        embed_dims=EMBED_DIMS,
        anchor_recipe=FixedAnchorRecipe({"anchors": anchors}),
        num_anchor=NUM_ANCHOR,
        point_cloud_range=POINT_CLOUD_RANGE,
        grid_size=1.0,
        feature_strides=(FEATURE_STRIDE,),
        num_decoder=2,
        num_heads=4,
        num_groups=4,
        feedforward_channels=64,
        num_learnable_pts=2,
        num_output=8,
        num_temp_instances=8,
        min_observed_points=1,
        track_birth_score=0.0,
        **overrides,
    )
    return fill_anchors(head)


class TestSparse4DHead(unittest.TestCase):
    def test_keypoints_follow_anchor_yaw(self) -> None:
        generator = SparseBox3DKeyPointsGenerator(
            embed_dims=EMBED_DIMS,
            fix_scale=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
            num_learnable_pts=0,
        )
        anchor = torch.zeros((1, 1, ANCHOR_DIM))
        anchor[..., 0] = 10.0  # x
        anchor[..., 3:6] = torch.log(torch.tensor([4.0, 2.0, 1.5]))
        anchor[..., 6] = 1.0  # sin yaw (90 degrees)
        anchor[..., 7] = 0.0  # cos yaw
        key_points = generator(anchor, None)
        center, front = key_points[0, 0]
        self.assertTrue(torch.allclose(center, torch.tensor([10.0, 0.0, 0.0]), atol=1e-5))
        # +x half-length offset rotated by 90 degrees points along +y.
        self.assertTrue(torch.allclose(front, torch.tensor([10.0, 2.0, 0.0]), atol=1e-5))

    def test_assignment_matches_exact_prediction(self) -> None:
        head = _build_head()
        gt_boxes = torch.tensor([[5.0, -2.0, 0.5, 4.5, 1.9, 1.7, 0.4, 0.0, 0.0]])
        ground_truth = encode_ground_truth(
            gt_boxes, torch.tensor([1]), torch.device("cpu"), torch.float32
        )

        pred_anchor = torch.randn(1, NUM_ANCHOR, ANCHOR_DIM) * 20.0
        pred_anchor[0, 3] = ground_truth.anchor[0]
        cls_logits = torch.full((1, NUM_ANCHOR, 3), -6.0)
        cls_logits[0, 3, 1] = 6.0

        targets = head._assign_targets(
            cls_logits, pred_anchor, [ground_truth], [torch.tensor([7])], None
        )
        self.assertEqual(targets.num_pos, 1)
        self.assertEqual(targets.assigned_instance[0, 3], 7)
        self.assertEqual(targets.cls_targets[0, 3, 1], 1.0)
        self.assertTrue(torch.allclose(targets.box_targets[0, 3], ground_truth.anchor[0]))
        self.assertTrue(torch.allclose(targets.box_weights[0, 3], head.reg_weights))
        self.assertEqual(targets.box_weights[0, :3].sum(), 0)
        self.assertTrue(targets.matched[0].tolist() == [index == 3 for index in range(NUM_ANCHOR)])

    def test_a_yaw_free_class_takes_its_heading_out_of_the_regression(self) -> None:
        """
        Input: a head with class 1 yaw free, a box of class 1 and a query on it facing away.
        Expected: the query is matched and its heading channels get no regression weight.
        """
        head = build_sparse4d_head(
            make_anchor_recipe(), class_names=["car", "cone"], yaw_free_classes=["cone"]
        )
        cls_logits, ground_truth, pred_anchor = _query_facing_away(1)
        targets = head._assign_targets(
            cls_logits, pred_anchor, ground_truth, [torch.tensor([7])], None
        )
        self.assertEqual(targets.assigned_instance[0, 3], 7)
        self.assertEqual(targets.box_weights[0, 3, 6:8].tolist(), [0.0, 0.0])
        self.assertTrue(bool((targets.box_weights[0, 3, :6] > 0).all()))

    def test_a_yaw_free_class_gets_no_yawness_target(self) -> None:
        """
        Input: a head with class 1 yaw free, a query on a box of class 0 and one on a box of class 1.
        Expected: only the query on the class 0 box supervises its yawness.
        """
        head = build_sparse4d_head(
            make_anchor_recipe(), class_names=["car", "cone"], yaw_free_classes=["cone"]
        )
        reg_mask = torch.zeros(1, NUM_ANCHOR, dtype=torch.bool)
        reg_mask[0, 3] = True
        losses = []
        for class_index in (0, 1):
            cls_logits, ground_truth, pred_anchor = _query_facing_away(class_index)
            targets = head._assign_targets(
                cls_logits, pred_anchor, ground_truth, [torch.tensor([7])], None
            )
            quality = torch.zeros(1, NUM_ANCHOR, 2)
            _, loss_yns = head._quality_losses(quality, pred_anchor, targets, reg_mask, 1.0)
            losses.append(float(loss_yns))
        self.assertGreater(losses[0], 0.0)
        self.assertEqual(losses[1], 0.0)

    def test_a_reversible_query_facing_away_regresses_the_reversed_heading(self) -> None:
        """
        Input: a head with class 1 yaw reversible, a query facing away from a box of class 1 and from
          a box of class 0.
        Expected: the class 1 target takes the query's heading, the class 0 target keeps its own.
        """
        head = build_sparse4d_head(
            make_anchor_recipe(),
            class_names=["car", "barrier"],
            yaw_reversible_classes=["barrier"],
        )
        for class_index, reversed_heading in ((1, True), (0, False)):
            cls_logits, ground_truth, pred_anchor = _query_facing_away(class_index)
            targets = head._assign_targets(
                cls_logits, pred_anchor, ground_truth, [torch.tensor([7])], None
            )
            box_targets = head._reverse_facing_away(
                targets.cls_targets, targets.box_targets, pred_anchor
            )
            heading = box_targets[0, 3, 6:8]
            expected = (
                pred_anchor[0, 3, 6:8] if reversed_heading else ground_truth[0].anchor[0, 6:8]
            )
            torch.testing.assert_close(heading, expected)

    def test_a_diverged_prediction_stops_the_matching(self) -> None:
        """
        Input: the predictions of one sample with a NaN anchor.
        Expected: the matching raises instead of assigning the queries.
        """
        head = _build_head()
        cls_logits, ground_truth, pred_anchor = _query_facing_away(0)
        pred_anchor[0, 3, 0] = float("nan")

        with self.assertRaisesRegex(ValueError, "not finite"):
            head._assign_targets(cls_logits, pred_anchor, ground_truth, [torch.tensor([7])], None)

    def test_class_names_must_list_every_class(self) -> None:
        """
        Input: a head with two classes and three class names.
        Expected: the head refuses the class names.
        """
        with self.assertRaisesRegex(ValueError, "lists 3 classes, num_classes is 2"):
            build_sparse4d_head(make_anchor_recipe(), class_names=["car", "cone", "barrier"])

    def test_unknown_velocity_is_masked_out_of_the_regression(self) -> None:
        head = _build_head()
        gt_boxes = torch.tensor([[5.0, -2.0, 0.5, 4.5, 1.9, 1.7, 0.4, float("nan"), float("nan")]])
        ground_truth = encode_ground_truth(
            gt_boxes, torch.tensor([0]), torch.device("cpu"), torch.float32
        )
        self.assertTrue(torch.isfinite(ground_truth.anchor).all())
        self.assertEqual(ground_truth.channel_valid[0].tolist(), [True] * 8 + [False] * 2)

        outputs = head([_sparse_stage_inputs(batch_size=1)])
        losses = head.loss(outputs, [gt_boxes], [torch.tensor([0])], [torch.tensor([0])], [0])
        self.assertTrue(torch.isfinite(losses["loss"]))

    def test_unknown_box_center_is_rejected(self) -> None:
        gt_boxes = torch.tensor([[float("nan"), -2.0, 0.5, 4.5, 1.9, 1.7, 0.4, 0.0, 0.0]])
        with self.assertRaisesRegex(ValueError, "finite center"):
            encode_ground_truth(gt_boxes, torch.tensor([0]), torch.device("cpu"), torch.float32)

    def test_loss_is_finite_without_ground_truth(self) -> None:
        head = _build_head()
        outputs = head([_sparse_stage_inputs(batch_size=2)])
        losses = head.loss(
            outputs,
            gt_boxes=[torch.zeros((0, 9)), torch.zeros((0, 9))],
            gt_labels=[torch.zeros((0,), dtype=torch.long), torch.zeros((0,), dtype=torch.long)],
            gt_instance_ids=[
                torch.zeros((0,), dtype=torch.long),
                torch.zeros((0,), dtype=torch.long),
            ],
            lanes=[0, 1],
        )
        self.assertEqual(outputs.classification.shape, (2, head.num_decoder, NUM_ANCHOR, 3))
        self.assertEqual(outputs.prediction.shape, (2, head.num_decoder, NUM_ANCHOR, ANCHOR_DIM))
        self.assertEqual(outputs.quality.shape, (2, head.num_decoder, NUM_ANCHOR, 2))
        layer_keys = [key for key in losses if key.startswith("layer_")]
        self.assertEqual(len(layer_keys), 4 * head.num_decoder)
        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertEqual(float(losses["layer_-1_loss_bbox"]), 0.0)
        losses["loss"].backward()

    def test_every_parameter_gets_a_gradient_before_any_query_is_confident(self) -> None:
        """DDP needs a gradient on every parameter before any query passes cls_threshold_to_reg."""
        head = _build_head()
        gt_boxes = [torch.tensor([[5.0, -2.0, 0.5, 4.5, 1.9, 1.7, 0.4, 0.0, 0.0]])]
        gt_labels = [torch.tensor([1])]
        head.cls_threshold_to_reg = 1.0

        losses = head.loss(
            head([_sparse_stage_inputs(batch_size=1)]),
            gt_boxes,
            gt_labels,
            [torch.tensor([0])],
            [0],
        )
        losses["loss"].backward()

        missing = [
            name
            for name, parameter in head.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        self.assertEqual(missing, [])
        self.assertFalse(head.instance_bank.instance_feature.requires_grad)

    def test_quality_losses_follow_the_matched_box(self) -> None:
        head = _build_head()
        gt_boxes = [torch.tensor([[5.0, -2.0, 0.5, 4.5, 1.9, 1.7, 0.4, 0.0, 0.0]])]
        gt_labels = [torch.tensor([1])]
        head.cls_threshold_to_reg = 0.0

        losses = head.loss(
            head([_sparse_stage_inputs(batch_size=1)]),
            gt_boxes,
            gt_labels,
            [torch.tensor([0])],
            [0],
        )
        for prefix in ("layer_0", "layer_-1"):
            self.assertGreater(losses[f"{prefix}_loss_cns"], 0.0)
            self.assertGreaterEqual(losses[f"{prefix}_loss_yns"], 0.0)

    def test_an_unannotated_class_gets_no_background_target(self) -> None:
        """
        Input: one car box in a sample whose cones, class 2, are not annotated.
        Expected: the unmatched queries are pushed off the annotated classes but not off the cone,
        the matched query is supervised on every class, and without the mask the cone is background.
        """
        head = _build_head()
        outputs = head([_sparse_stage_inputs(batch_size=1)])
        gt_boxes = [torch.tensor([[5.0, -2.0, 0.5, 4.5, 1.9, 1.7, 0.4, 0.0, 0.0]])]
        targets = head.assign(outputs, gt_boxes, [torch.tensor([0])], [torch.tensor([0])], [0])
        matched = targets[-1].matched[0]

        def cone_gradient(mask: torch.Tensor | None) -> torch.Tensor:
            loss = head.loss_from_targets(outputs, targets, [0], mask)["layer_-1_loss_cls"]
            return torch.autograd.grad(loss, outputs.classification, retain_graph=True)[0][0, -1]

        gradient = cone_gradient(torch.tensor([[True, True, False]]))
        self.assertTrue((gradient[~matched, 2] == 0).all())
        self.assertTrue((gradient[~matched, :2] != 0).all())
        self.assertTrue((gradient[matched] != 0).all())
        self.assertTrue((cone_gradient(None)[~matched, 2] != 0).all())

    def test_decode_decodes_and_filters(self) -> None:
        head = _build_head()
        head.eval()
        head.score_threshold = None
        with torch.no_grad():
            outputs = head([_sparse_stage_inputs(batch_size=1)])
        predictions = _decode(head, outputs)
        # One box per query
        self.assertTrue(torch.unique(predictions[0].bboxes_3d, dim=0).shape[0] == head.num_output)
        self.assertEqual(predictions[0].bboxes_3d.shape, (head.num_output, 9))
        self.assertEqual(predictions[0].scores_3d.shape, (head.num_output,))
        self.assertLess(predictions[0].labels_3d.max(), head.num_classes)
        self.assertTrue((predictions[0].bboxes_3d[:, 3:6] > 0).all())

        head.score_threshold = 1.1  # impossible threshold removes everything
        filtered = _decode(head, outputs)
        self.assertEqual(filtered[0].bboxes_3d.shape[0], 0)

    def test_decode_gives_one_box_per_query_of_its_best_class(self) -> None:
        """
        Input: query 5 with a high logit for class 2 and a lower one for class 0, all others low.
        Expected: one box of query 5, class 2, scored by its class probability times centerness,
          and no box of class 0 for it.
        """
        head = _build_head()
        head.eval()
        with torch.no_grad():
            outputs = head([_sparse_stage_inputs(batch_size=1)])
        classification = torch.full_like(outputs.classification, -10.0)
        classification[0, -1, 5, 2] = 2.0
        classification[0, -1, 5, 0] = 1.0
        quality = torch.zeros_like(outputs.quality)
        quality[0, -1, 5, 0] = 1.0
        outputs = type(outputs)(
            classification=classification,
            prediction=outputs.prediction,
            quality=quality,
            instance_ids=None,
            denoising=None,
        )

        prediction = _decode(head, outputs)[0]
        box = denormalize_boxes3d(outputs.prediction[0, -1, 5])
        expected = torch.sigmoid(torch.tensor(2.0)) * torch.sigmoid(torch.tensor(1.0))

        self.assertEqual(prediction.labels_3d[0].item(), 2)
        self.assertTrue(torch.allclose(prediction.scores_3d[0], expected))
        self.assertTrue(torch.allclose(prediction.bboxes_3d[0], box))
        self.assertTrue(int(torch.isclose(prediction.bboxes_3d, box).all(dim=1).sum()) == 1)

    def test_instance_bank_warps_static_object_across_frames(self) -> None:
        """A static object's cached anchor must land on the object in the next frame."""
        bank = _make_bank()
        global_point = torch.tensor([10.0, 4.0, 1.0, 1.0], dtype=torch.float64)
        pose1 = _pose(yaw=0.3, translation=(5.0, 2.0, 0.0))
        pose2 = _pose(yaw=-0.2, translation=(6.0, 2.5, 0.0))
        timestamp1 = torch.tensor([100.0], dtype=torch.float64)
        timestamp2 = torch.tensor([100.1], dtype=torch.float64)

        anchor1 = torch.zeros((1, BANK_ANCHORS, ANCHOR_DIM))
        anchor1[0, :, :3] = (pose1[0].inverse() @ global_point)[:3].float()
        anchor1[0, :, 7] = 1.0
        _cache_frame(bank, anchor1, timestamp1, pose1, "scene")

        _, _, time_interval, _, temp_anchor = bank.get(1, timestamp2, pose2, ["scene"])
        expected = (pose2[0].inverse() @ global_point)[:3].float()
        self.assertTrue(torch.allclose(time_interval, torch.tensor([0.1]), atol=1e-5))
        self.assertTrue(torch.allclose(temp_anchor[0, 0, :3], expected, atol=1e-4))

    @unittest.skipUnless(torch.cuda.is_available(), "TF32 matmuls run on CUDA only")
    def test_the_anchor_warp_ignores_the_tf32_matmul_precision(self) -> None:
        """
        Input: anchors 100 m away, warped by a rotated transform under the medium matmul precision.
        Expected: the float32 warp matches the float64 one to a tenth of a millimeter.
        """
        anchor = torch.zeros((1, 4, ANCHOR_DIM), dtype=torch.float64)
        anchor[0, :, 0] = torch.tensor([100.0, -100.0, 70.0, 0.5])
        anchor[0, :, 1] = torch.tensor([3.0, 40.0, -90.0, 99.0])
        anchor[0, :, 7] = 1.0
        anchor[0, :, 8] = 2.0
        transform = _pose(yaw=0.37, translation=(1.3, -0.4, 0.1))
        interval = torch.tensor([0.1], dtype=torch.float64)
        reference = InstanceBank.project_anchor(anchor, interval, transform)
        previous = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("medium")
        try:
            warped = InstanceBank.project_anchor(
                anchor.float().cuda(), interval.float().cuda(), transform.float().cuda()
            )
        finally:
            torch.set_float32_matmul_precision(previous)

        torch.testing.assert_close(warped.double().cpu(), reference, atol=1e-4, rtol=0.0)

    def test_instance_bank_warp_advances_by_instance_velocity(self) -> None:
        bank = _make_bank()
        pose = _pose(yaw=0.0, translation=(0.0, 0.0, 0.0))
        timestamp1 = torch.tensor([10.0], dtype=torch.float64)
        timestamp2 = torch.tensor([10.5], dtype=torch.float64)

        anchor1 = torch.zeros((1, BANK_ANCHORS, ANCHOR_DIM))
        anchor1[0, :, 0] = 2.0
        anchor1[0, :, 7] = 1.0
        anchor1[0, :, 8] = 3.0  # vx
        _cache_frame(bank, anchor1, timestamp1, pose, "scene")

        _, _, _, _, temp_anchor = bank.get(1, timestamp2, pose, ["scene"])
        self.assertTrue(
            torch.allclose(temp_anchor[0, 0, 0], torch.tensor(2.0 + 3.0 * 0.5), atol=1e-5)
        )

    def test_instance_bank_resets_on_scene_change_and_time_gap(self) -> None:
        bank = _make_bank()
        pose = _pose(yaw=0.0, translation=(0.0, 0.0, 0.0))
        timestamp = torch.tensor([10.0], dtype=torch.float64)
        anchor = torch.ones((1, BANK_ANCHORS, ANCHOR_DIM))
        _cache_frame(bank, anchor, timestamp, pose, "scene")

        # Scene change resets the slot.
        _, _, interval, temp_feature, _ = bank.get(1, timestamp + 0.1, pose, ["other"])
        self.assertTrue(torch.all(temp_feature == 0))
        self.assertTrue(torch.allclose(interval, torch.tensor([bank.default_time_interval])))

        # Excessive time gap resets the slot.
        _cache_frame(bank, anchor, timestamp, pose, "scene")
        _, _, interval, temp_feature, _ = bank.get(
            1, timestamp + bank.max_time_interval + 1.0, pose, ["scene"]
        )
        self.assertTrue(torch.all(temp_feature == 0))
        self.assertTrue(torch.allclose(interval, torch.tensor([bank.default_time_interval])))

        # A gap shorter than a frame period resets the slot.
        _cache_frame(bank, anchor, timestamp, pose, "scene")
        _, _, interval, temp_feature, _ = bank.get(1, timestamp + 1e-4, pose, ["scene"])
        self.assertTrue(torch.all(temp_feature == 0))
        self.assertTrue(torch.allclose(interval, torch.tensor([bank.default_time_interval])))

    def test_a_scene_change_clears_a_non_finite_cache(self) -> None:
        """
        Input: a bank that cached NaN features and anchors, then a frame of another scene.
        Expected: the restarted lane drops the cache and the merge keeps the fresh queries.
        Check: the propagated state is zero and the merged queries are finite.
        """
        bank = _make_bank()
        pose = _pose(yaw=0.0, translation=(0.0, 0.0, 0.0))
        timestamp = torch.tensor([10.0], dtype=torch.float64)
        bank.get(1, timestamp, pose, ["scene"])
        bank.cache(
            torch.full((1, BANK_ANCHORS, 8), float("nan")),
            torch.full((1, BANK_ANCHORS, ANCHOR_DIM), float("nan")),
            torch.full((1, BANK_ANCHORS, 3), 4.0),
            pose,
        )

        _, _, _, temp_feature, temp_anchor = bank.get(1, timestamp + 0.1, pose, ["other"])
        merged_feature, merged_anchor = bank.update(
            torch.ones((1, BANK_ANCHORS, 8)),
            torch.zeros((1, BANK_ANCHORS, ANCHOR_DIM)),
            torch.zeros((1, BANK_ANCHORS, 3)),
        )

        self.assertTrue(torch.all(temp_feature == 0) and torch.all(temp_anchor == 0))
        self.assertTrue(
            torch.isfinite(merged_feature).all() and torch.isfinite(merged_anchor).all()
        )

    def test_instance_bank_drops_the_query_index_of_another_batch_size(self) -> None:
        bank = _make_bank()
        pose = _pose(yaw=0.0, translation=(0.0, 0.0, 0.0))
        timestamp = torch.tensor([10.0], dtype=torch.float64)
        _cache_frame(bank, torch.ones((1, BANK_ANCHORS, ANCHOR_DIM)), timestamp, pose, "scene")
        self.assertIsNotNone(bank.cached_query_index)

        # Two lanes index no query of the one lane batch, so the carried index goes with it
        bank.get(2, (timestamp + 0.1).repeat(2), pose.repeat(2, 1, 1), ["scene", "scene"])

        self.assertIsNone(bank.cached_query_index)
        self.assertIsNone(bank.carried_query_index)
        self.assertEqual(bank.lane_continues.tolist(), [False, False])

    def test_instance_bank_hands_out_the_query_index_its_slots_carry(self) -> None:
        """
        Input: a cached frame, then the next frame of its scene, cached in turn.
        Expected: the bank keeps the query index of the first frame as the one the slots of the
          second frame carry, after the second frame is cached as well.
        """
        bank = _make_bank()
        pose = _pose(yaw=0.0, translation=(0.0, 0.0, 0.0))
        timestamp = torch.tensor([10.0], dtype=torch.float64)
        _cache_frame(bank, torch.ones((1, BANK_ANCHORS, ANCHOR_DIM)), timestamp, pose, "scene")
        first = bank.cached_query_index

        _cache_frame(
            bank, torch.ones((1, BANK_ANCHORS, ANCHOR_DIM)), timestamp + 0.1, pose, "scene"
        )

        self.assertIs(bank.carried_query_index, first)

    def test_instance_bank_rejects_an_empty_time_gate(self) -> None:
        anchors = np.zeros((BANK_ANCHORS, ANCHOR_DIM), dtype=np.float32)
        for min_interval, max_interval in [(0.0, 2.0), (2.0, 2.0)]:
            with self.subTest(min_interval=min_interval, max_interval=max_interval):
                with self.assertRaisesRegex(ValueError, "time gate"):
                    InstanceBank(
                        num_anchor=BANK_ANCHORS,
                        embed_dims=8,
                        anchor_recipe=FixedAnchorRecipe({"anchors": anchors}),
                        min_time_interval=min_interval,
                        max_time_interval=max_interval,
                    )

    def test_instance_bank_update_mixes_cached_and_top_fresh(self) -> None:
        bank = _make_bank()
        pose = _pose(yaw=0.0, translation=(0.0, 0.0, 0.0))
        timestamp1 = torch.tensor([10.0], dtype=torch.float64)
        anchor = torch.zeros((1, BANK_ANCHORS, ANCHOR_DIM))
        anchor[0, :, 7] = 1.0
        _cache_frame(bank, anchor, timestamp1, pose, "scene")
        bank.get(1, timestamp1 + 0.1, pose, ["scene"])

        fresh_feature = torch.full((1, BANK_ANCHORS, 8), 1.0)
        fresh_anchor = torch.zeros((1, BANK_ANCHORS, ANCHOR_DIM))
        logits = torch.arange(BANK_ANCHORS, dtype=torch.float32).flip(0)[None, :, None]
        logits = logits.expand(-1, -1, 3).contiguous()

        merged_feature, _ = bank.update(fresh_feature, fresh_anchor, logits)
        # Leading block holds the propagated (cached) features, tail the top fresh.
        self.assertTrue(torch.all(merged_feature[0, :NUM_TEMP] == 7.0))
        self.assertTrue(torch.all(merged_feature[0, NUM_TEMP:] == 1.0))

    def test_instance_bank_cache_applies_confidence_decay(self) -> None:
        bank = _make_bank()
        pose = _pose(yaw=0.0, translation=(0.0, 0.0, 0.0))
        timestamp1 = torch.tensor([10.0], dtype=torch.float64)
        anchor = torch.zeros((1, BANK_ANCHORS, ANCHOR_DIM))
        anchor[0, :, 7] = 1.0
        _cache_frame(bank, anchor, timestamp1, pose, "scene")  # sigmoid(4) ~ 0.982
        previous_confidence = torch.sigmoid(torch.tensor(4.0))

        bank.get(1, timestamp1 + 0.1, pose, ["scene"])
        low_logits = torch.full((1, BANK_ANCHORS, 3), -10.0)
        bank.cache(torch.zeros((1, BANK_ANCHORS, 8)), anchor, low_logits, pose)
        # Decay is per second: 0.1 s frame gap applies confidence_decay ** 0.1.
        expected = previous_confidence * bank.confidence_decay**0.1
        self.assertTrue(torch.allclose(bank._cached_confidence[0, 0], expected, atol=1e-5))

    def test_temporal_head_runs_two_frames_and_backprops(self) -> None:
        head = _build_temporal_head()
        pose = _pose(yaw=0.1, translation=(1.0, 0.0, 0.0))
        gt_boxes = [torch.tensor([[5.0, -2.0, 0.5, 4.5, 1.9, 1.7, 0.4, 0.0, 0.0]])]
        gt_labels = [torch.tensor([1])]

        for frame_index in range(2):
            outputs = head(
                [_sparse_stage_inputs(batch_size=1)],
                timestamps_seconds=torch.tensor([100.0 + 0.1 * frame_index], dtype=torch.float64),
                lidar2global=pose,
                scene_tokens=["scene"],
                gt_boxes=gt_boxes,
                gt_labels=gt_labels,
                gt_instance_ids=[torch.tensor([3])],
                gt_num_points=[torch.tensor([5])],
            )
            losses = head.loss(outputs, gt_boxes, gt_labels, [torch.tensor([3])], [0])
            self.assertTrue(torch.isfinite(losses["loss"]))
            losses["loss"].backward()

    def test_the_slots_follow_an_assignment_only_once_it_is_committed(self) -> None:
        """
        Input: a temporal head, one frame with a box of instance 3, its assignment and loss
          computed twice.
        Expected: assign and loss leave the stream state alone, follow_assignment gives the
          instance to the query it was assigned to.
        """
        head = _build_temporal_head()
        gt_boxes = [torch.tensor([[5.0, -2.0, 0.5, 4.5, 1.9, 1.7, 0.4, 0.0, 0.0]])]
        gt_labels = [torch.tensor([1])]
        gt_ids = [torch.tensor([3])]
        outputs = head(
            [_sparse_stage_inputs(batch_size=1)],
            timestamps_seconds=torch.tensor([100.0], dtype=torch.float64),
            lidar2global=_pose(yaw=0.1, translation=(1.0, 0.0, 0.0)),
            scene_tokens=["scene"],
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            gt_instance_ids=gt_ids,
            gt_num_points=[torch.tensor([5])],
        )
        before = head.tracker._query_owner.clone()

        targets = head.assign(outputs, gt_boxes, gt_labels, gt_ids, [0])
        head.loss(outputs, gt_boxes, gt_labels, gt_ids, [0])
        unchanged = torch.equal(head.tracker._query_owner, before)
        head.follow_assignment(outputs, targets, [0])

        self.assertTrue(unchanged)
        self.assertTrue(int((head.tracker._query_owner == 3).sum()) == 1)

    def test_a_step_without_a_supervised_frame_reports_every_loss_term(self) -> None:
        """
        Input: a temporal training step with denoising, its losses over its frame and over no frame.
        Expected: the same terms in the same order, zero over no frame and still in the graph, so
          every rank logs the same keys whatever frames it trains on.
        """
        head = _build_temporal_head(dn_query_budget=8)
        gt_boxes = [torch.tensor([[5.0, -2.0, 0.5, 4.5, 1.9, 1.7, 0.4, 0.0, 0.0]])]
        gt_labels = [torch.tensor([1])]
        gt_ids = [torch.tensor([3])]
        outputs = head(
            [_sparse_stage_inputs(batch_size=1)],
            timestamps_seconds=torch.tensor([100.0], dtype=torch.float64),
            lidar2global=_pose(yaw=0.1, translation=(1.0, 0.0, 0.0)),
            scene_tokens=["scene"],
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
            gt_instance_ids=gt_ids,
            gt_num_points=[torch.tensor([5])],
        )

        supervised = head.loss(outputs, gt_boxes, gt_labels, gt_ids, [0])
        unsupervised = head.loss(outputs.select_samples(torch.tensor([False])), [], [], [], [])

        self.assertEqual(list(unsupervised), list(supervised))
        self.assertTrue(
            all(value.item() == 0.0 for key, value in unsupervised.items() if "loss" in key)
        )
        unsupervised["loss"].backward()
        self.assertIsNotNone(head.instance_bank.anchor.grad)

    def test_reset_forgets_the_cached_instances_and_the_tracks(self) -> None:
        """
        Input: a temporal head after one frame, reset as a new loop starts.
        Expected: no cached instance, no observed instance and no track stays, so the next frame
          starts a stream whatever its scene and time.
        """
        head = _build_temporal_head()
        head(
            [_sparse_stage_inputs(batch_size=1)],
            timestamps_seconds=torch.tensor([100.0], dtype=torch.float64),
            lidar2global=_pose(yaw=0.1, translation=(1.0, 0.0, 0.0)),
            scene_tokens=["scene"],
            gt_boxes=[torch.tensor([[5.0, -2.0, 0.5, 4.5, 1.9, 1.7, 0.4, 0.0, 0.0]])],
            gt_labels=[torch.tensor([1])],
            gt_instance_ids=[torch.tensor([3])],
            gt_num_points=[torch.tensor([5])],
        )
        self.assertIsNotNone(head.instance_bank._cached_anchor)

        head.reset()

        self.assertIsNone(head.instance_bank._cached_anchor)
        self.assertIsNone(head.tracker.stream_ids)
        with self.assertRaisesRegex(ValueError, "instance ids"):
            head.tracker.observed_masks()

    def test_temporal_export_matches_training_across_two_frames(self) -> None:
        """The export state loop must reproduce the training-side instance bank."""
        head = _build_temporal_head().eval()
        export_head = head.prepare_for_export()
        stage_inputs = _sparse_stage_inputs(batch_size=1)
        pose1 = _pose(yaw=0.3, translation=(5.0, 2.0, 0.0))
        pose2 = _pose(yaw=-0.2, translation=(6.0, 2.5, 0.0))
        timestamp1 = torch.tensor([100.0], dtype=torch.float64)
        timestamp2 = torch.tensor([100.5], dtype=torch.float64)

        with torch.no_grad():
            train1 = head(
                [stage_inputs],
                timestamps_seconds=timestamp1,
                lidar2global=pose1,
                scene_tokens=["scene"],
            )
            export1 = export_head.forward_temporal_export(
                [stage_inputs], *export_head.temporal_state_export_inputs(torch.device("cpu"))
            )
            train2 = head(
                [stage_inputs],
                timestamps_seconds=timestamp2,
                lidar2global=pose2,
                scene_tokens=["scene"],
            )
            export2 = export_head.forward_temporal_export(
                [stage_inputs],
                export1["temp_instance_feature_out"],
                export1["temp_anchor_out"],
                export1["temp_confidence_out"],
                torch.ones(1),
                torch.tensor([0.5]),
                torch.linalg.solve(pose2, pose1).to(torch.float32),
            )

        bank = head.instance_bank
        for train_outputs, export_outputs in ((train1, export1), (train2, export2)):
            self.assertTrue(
                torch.allclose(
                    export_outputs["classification"], train_outputs.classification[:, -1], atol=1e-5
                )
            )
            self.assertTrue(
                torch.allclose(
                    export_outputs["prediction"], train_outputs.prediction[:, -1], atol=1e-5
                )
            )
            self.assertTrue(
                torch.allclose(export_outputs["quality"], train_outputs.quality[:, -1], atol=1e-5)
            )
        self.assertTrue(
            torch.allclose(export2["temp_instance_feature_out"], bank._cached_feature, atol=1e-5)
        )
        self.assertTrue(torch.allclose(export2["temp_anchor_out"], bank._cached_anchor, atol=1e-4))
        self.assertTrue(
            torch.allclose(export2["temp_confidence_out"], bank._cached_confidence, atol=1e-5)
        )
        self.assertTrue(torch.equal(export2["temp_query_index_out"], bank.cached_query_index))

    def test_export_folds_grown_attention_outputs_into_the_fp16_range(self) -> None:
        head = _build_temporal_head()
        # A LayerNorm follows every attention block, so training can grow its value path freely
        with torch.no_grad():
            for block in head.gnn:
                block.fc_value.weight.mul_(100.0)
                block.attn.out_proj.weight.mul_(100.0)
                block.fc_out.weight.mul_(100.0)
        head.eval()
        export_head = head.prepare_for_export()
        block_outputs: dict[str, list[torch.Tensor]] = {"train": [], "export": []}
        for name, model in (("train", head), ("export", export_head)):
            for block in model.gnn:
                block.register_forward_hook(
                    lambda _module, _inputs, output, name=name: block_outputs[name].append(output)
                )
        stage_inputs = _sparse_stage_inputs(batch_size=1)
        with torch.no_grad():
            train_outputs = head(
                [stage_inputs],
                timestamps_seconds=torch.tensor([100.0], dtype=torch.float64),
                lidar2global=_pose(yaw=0.3, translation=(5.0, 2.0, 0.0)),
                scene_tokens=["scene"],
            )
            export_outputs = export_head.forward_temporal_export(
                [stage_inputs], *export_head.temporal_state_export_inputs(torch.device("cpu"))
            )
        fp16_max = torch.finfo(torch.float16).max
        self.assertTrue(max(output.abs().max() for output in block_outputs["train"]) > fp16_max)
        self.assertTrue(max(output.abs().max() for output in block_outputs["export"]) < fp16_max)
        self.assertTrue(
            torch.allclose(
                export_outputs["classification"], train_outputs.classification[:, -1], atol=1e-5
            )
        )
        self.assertTrue(
            torch.allclose(export_outputs["prediction"], train_outputs.prediction[:, -1], atol=1e-5)
        )

    def test_every_sampled_stage_reaches_the_queries(self) -> None:
        head = build_sparse4d_head(make_anchor_recipe(), feature_strides=(1, 2)).eval()
        stages = [
            random_stage(grid.grid_dims, 64 // (index + 1))
            for index, grid in enumerate(head.level_grids)
        ]

        outputs = head(stages)
        outputs.classification.sum().backward()
        with torch.no_grad():
            plain = head(stages)

        self.assertTrue(all(feat.grad.abs().sum() > 0 for feat, _, _ in stages))
        # The checkpointed sampling of the gradient pass matches the plain one
        self.assertTrue(
            torch.allclose(plain.classification, outputs.classification.detach(), atol=1e-6)
        )

    def test_level_sampling_interpolates_and_reads_zeros_off_the_voxels(self) -> None:
        grid = SparseLevelGrid([0.0, 0.0, 0.0, 4.0, 4.0, 4.0], grid_size=1.0, feature_stride=1)
        feat = torch.tensor([[2.0, -4.0]], requires_grad=True)
        level = grid(feat, torch.tensor([[1, 1, 1]]), torch.tensor([1]))
        # The voxel center, halfway to the empty neighbour, an empty cell and outside the grid
        key_points = torch.tensor(
            [[[[1.5, 1.5, 1.5], [2.0, 1.5, 1.5], [3.5, 3.5, 3.5], [-5.0, 0, 0]]]]
        )

        sampled, occupancy = level.sample(key_points)
        sampled.sum().backward()

        self.assertEqual(level.index_grid.dtype, torch.int32)
        expected = torch.tensor([[2.0, -4.0], [1.0, -2.0], [0.0, 0.0], [0.0, 0.0]])
        self.assertTrue(torch.allclose(sampled[0, 0], expected))
        self.assertTrue(torch.allclose(occupancy[0, 0], torch.tensor([1.0, 0.5, 0.0, 0.0])))
        self.assertTrue(torch.allclose(feat.grad, torch.tensor([[1.5, 1.5]])))

    def test_keypoints_in_empty_space_do_not_dilute_the_aggregated_features(self) -> None:
        aggregation = SparseDeformableAggregation(
            embed_dims=4,
            num_groups=1,
            num_levels=2,
            kps_generator=SparseBox3DKeyPointsGenerator(
                embed_dims=4, fix_scale=((0.0, 0.0, 0.0), (0.45, 0.0, 0.0)), num_learnable_pts=0
            ),
        ).eval()
        nn.init.eye_(aggregation.output_proj.weight)
        nn.init.zeros_(aggregation.output_proj.bias)
        grid = SparseLevelGrid([0.0, 0.0, 0.0, 8.0, 8.0, 8.0], grid_size=1.0, feature_stride=1)
        feat = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        occupied = grid(feat, torch.tensor([[4, 4, 4]]), torch.tensor([1]))
        empty = grid(feat, torch.tensor([[0, 0, 0]]), torch.tensor([1]))
        # A tiny box at the occupied voxel center, every keypoint reads that voxel
        anchor = torch.zeros((1, 1, ANCHOR_DIM))
        anchor[..., :3] = 4.5
        anchor[..., 3:6] = math.log(0.01)
        anchor[..., 7] = 1.0
        zeros = torch.zeros((1, 1, 4))

        with torch.no_grad():
            both = aggregation(zeros, anchor, zeros, [occupied, empty])

        # Half the keypoints sample the empty stage, the feature still reads at full strength
        self.assertTrue(torch.allclose(both[0, 0], feat[0], atol=1e-5))

    def test_a_lidar_block_rejects_a_frame_context(self) -> None:
        head = build_sparse4d_head(make_anchor_recipe())
        levels = head.levels([random_stage(grid.grid_dims, 64) for grid in head.level_grids])
        anchor = head.instance_bank.anchor[None].detach()

        with self.assertRaisesRegex(ValueError, "no frame context"):
            head.deformable[0](
                torch.zeros(1, 8, 32), anchor, head.anchor_encoder(anchor), levels, context=object()
            )
