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

"""Tests for the point in box containment check."""

from __future__ import annotations

import math

import torch

from autoware_ml.geometry.utils import points_in_boxes_3d


def _box(x, y, z, length, width, height, yaw) -> torch.Tensor:
    return torch.tensor([[x, y, z, length, width, height, yaw, 0.0, 0.0]], dtype=torch.float32)


def test_axis_aligned_box_contains_only_its_interior_points() -> None:
    box = _box(0.0, 0.0, 0.0, 4.0, 2.0, 2.0, 0.0)
    points = torch.tensor(
        [
            [1.9, 0.9, 0.9],
            [2.1, 0.0, 0.0],
            [0.0, 1.1, 0.0],
            [0.0, 0.0, -1.1],
        ],
        dtype=torch.float32,
    )

    mask = points_in_boxes_3d(points, box)

    assert mask.shape == (1, 4)
    assert mask[0].tolist() == [True, False, False, False]


def test_rotated_box_contains_points_in_its_rotated_frame() -> None:
    box = _box(0.0, 0.0, 0.0, 4.0, 2.0, 2.0, math.pi / 2)
    points = torch.tensor(
        [
            [0.0, 1.9, 0.0],
            [0.9, 0.0, 0.0],
            [1.9, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    mask = points_in_boxes_3d(points, box)

    # The 4 m length axis points along y after the 90 degree yaw, so only 1 m of width
    # remains along x.
    assert mask[0].tolist() == [True, True, False]


def test_off_center_box_follows_its_gravity_center() -> None:
    box = _box(10.0, -4.0, 2.0, 2.0, 2.0, 2.0, 0.0)
    points = torch.tensor(
        [
            [10.5, -4.5, 2.5],
            [10.5, -4.5, 0.5],
        ],
        dtype=torch.float32,
    )

    mask = points_in_boxes_3d(points, box)

    assert mask[0].tolist() == [True, False]


def test_empty_inputs_produce_all_false_masks_with_the_right_shape() -> None:
    boxes = _box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0)
    points = torch.tensor([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=torch.float32)
    no_boxes = torch.zeros((0, 9), dtype=torch.float32)
    no_points = torch.zeros((0, 3), dtype=torch.float32)

    assert points_in_boxes_3d(points, no_boxes).shape == (0, 2)
    assert points_in_boxes_3d(no_points, boxes).shape == (1, 0)
    assert points_in_boxes_3d(no_points, no_boxes).shape == (0, 0)
