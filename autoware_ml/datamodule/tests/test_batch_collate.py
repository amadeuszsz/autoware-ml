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

"""Tests for the typed batch collation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from autoware_ml.datamodule.samples.batch import Batch
from autoware_ml.testing.factories import make_boxes3d, make_point_cloud, make_sample


def test_collate_keeps_per_sample_tensors_and_inclusive_offsets() -> None:
    samples = [
        make_sample(points=make_point_cloud(num_points=4), with_segment=True),
        make_sample(points=make_point_cloud(num_points=2, seed=1), with_segment=True, seed=1),
    ]

    batch = Batch.collate(samples)

    assert batch.batch_size == 2
    assert batch.point_cloud.lengths == (4, 2)
    assert batch.offset.tolist() == [4, 6]
    assert batch.points[0].shape == (4, 5)
    assert batch.points[1].shape == (2, 5)
    assert np.array_equal(batch.points[0].numpy(), samples[0].points.features)
    assert batch.num_current_points == (4, 2)


def test_collate_concatenates_segmentation_labels_along_the_point_dimension() -> None:
    samples = [
        make_sample(points=make_point_cloud(num_points=3), with_segment=True),
        make_sample(points=make_point_cloud(num_points=2, seed=1), with_segment=True, seed=1),
    ]

    batch = Batch.collate(samples)

    expected = np.concatenate([samples[0].segment.labels, samples[1].segment.labels])
    assert batch.segment.shape == (5,)
    assert np.array_equal(batch.segment.numpy(), expected)


def test_collate_rejects_mixed_point_cloud_presence() -> None:
    samples = [make_sample(points=make_point_cloud(num_points=3)), make_sample()]

    with pytest.raises(ValueError, match="present in every sample"):
        Batch.collate(samples)


def test_collate_distinguishes_empty_boxes_from_absent_boxes() -> None:
    with_empty = Batch.collate([make_sample(boxes=make_boxes3d(num_boxes=0))] * 2)
    without = Batch.collate([make_sample()] * 2)

    assert with_empty.boxes is not None
    assert with_empty.gt_boxes[0].shape == (0, 9)
    assert with_empty.gt_labels[0].shape == (0,)
    assert without.boxes is None
    assert without.gt_boxes is None

    with pytest.raises(ValueError, match="present in every sample"):
        Batch.collate([make_sample(boxes=make_boxes3d(num_boxes=0)), make_sample()])


def test_collate_rejects_point_feature_name_mismatch() -> None:
    samples = [
        make_sample(points=make_point_cloud(num_points=3, with_time_lag=True)),
        make_sample(points=make_point_cloud(num_points=3, with_time_lag=False)),
    ]

    with pytest.raises(ValueError, match="feature names must match"):
        Batch.collate(samples)


def test_collate_drops_current_point_counts_when_one_sample_lost_them() -> None:
    tracked = make_point_cloud(num_points=3)
    untracked = make_point_cloud(num_points=3, seed=1).reorder(np.arange(3, dtype=np.int64))
    samples = [make_sample(points=tracked), make_sample(points=untracked)]

    batch = Batch.collate(samples)

    assert batch.num_current_points is None


def test_collate_rejects_mixed_scene_token_presence() -> None:
    samples = [make_sample(), make_sample(scene_token=None)]

    with pytest.raises(ValueError, match="Scene tokens"):
        Batch.collate(samples)


def test_collate_rejects_an_empty_sample_sequence() -> None:
    with pytest.raises(ValueError, match="empty sequence"):
        Batch.collate([])


def test_collated_boxes_expose_the_flat_ground_truth_properties() -> None:
    boxes = make_boxes3d(num_boxes=3)
    batch = Batch.collate([make_sample(boxes=boxes)])

    assert np.array_equal(batch.gt_boxes[0].numpy(), boxes.params)
    assert np.array_equal(batch.gt_labels[0].numpy(), boxes.labels)
    assert np.array_equal(batch.gt_num_points[0].numpy(), boxes.num_lidar_points)
    assert batch.boxes.names[0] == boxes.names


def test_collated_meta_exposes_sample_ids_timestamps_and_ego_poses() -> None:
    batch = Batch.collate([make_sample()])

    assert batch.sample_token == ("sample-0",)
    assert batch.scene_token == ("db/scene/0",)
    assert batch.timestamp == (100.0,)
    assert torch.equal(batch.ego2global[0], torch.eye(4, dtype=torch.float64))


def test_to_cpu_is_an_identity_for_values() -> None:
    batch = Batch.collate([make_sample(points=make_point_cloud(num_points=3))])

    moved = batch.to(torch.device("cpu"))

    assert torch.equal(moved.points[0], batch.points[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_pin_memory_and_device_round_trip() -> None:
    batch = Batch.collate(
        [
            make_sample(
                points=make_point_cloud(num_points=3),
                boxes=make_boxes3d(num_boxes=2),
                with_segment=True,
            )
        ]
    )

    pinned = batch.pin_memory()
    assert pinned.points[0].is_pinned()
    assert pinned.gt_labels[0].is_pinned()
    assert pinned.segmentation.labels[0].is_pinned()

    moved = pinned.to(torch.device("cuda"))
    assert moved.points[0].device.type == "cuda"
    assert moved.ego2global[0].device.type == "cuda"

    back = moved.to(torch.device("cpu"))
    assert torch.equal(back.points[0], batch.points[0])
    assert torch.equal(back.segment, batch.segment)
