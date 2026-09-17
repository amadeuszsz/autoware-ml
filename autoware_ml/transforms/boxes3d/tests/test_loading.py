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

"""Tests for the box filters applied while loading a sample."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch

from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.geometry.bbox_3d.lidar_bbox3d import LidarBBoxes3D
from autoware_ml.transforms.boxes3d.loading import BBoxesAttributeFilter
from autoware_ml.types.geometry import Box3DCenterCoordinateType, Box3DFieldIndex

RULES = [["bicycle", "vehicle_state.parked"], ["motorcycle", "cycle_state.without_rider"]]


def sample(label_names: Sequence[str], attributes: Sequence[Sequence[str]] | None) -> ModelGTSample:
    """
    Sample carrying one unit box per label, with the given attributes.

    Args:
      label_names: Label name of every box.
      attributes: Attribute names of every box, None when the dataset carries none.

    Returns:
      ModelGTSample: Sample whose detection boxes carry those labels and attributes.
    """
    count = len(label_names)
    bbox_params = torch.zeros((count, len(Box3DFieldIndex)), dtype=torch.float32)
    bbox_params[:, Box3DFieldIndex.LENGTH : Box3DFieldIndex.YAW] = 1.0
    bbox_params[:, Box3DFieldIndex.X] = torch.arange(count, dtype=torch.float32)
    boxes = LidarBBoxes3D(
        bbox_params=bbox_params,
        bbox_labels=torch.arange(count, dtype=torch.int32),
        bbox_label_names=list(label_names),
        bbox_num_lidar_points=torch.full((count,), 10, dtype=torch.int32),
        bbox_center_coordinate_type=Box3DCenterCoordinateType.GRAVITY_CENTER,
        bbox_attributes=attributes,
    )
    return ModelGTSample(
        lidar_point_cloud_samples=None,
        image_samples=None,
        point_cloud_data=None,
        camera_image_data=None,
        detection3d_gt_bboxes_3d=boxes,
        segmentation3d_gt_sample=None,
    )


def test_drops_only_the_boxes_matching_a_rule() -> None:
    # The parked car keeps its box, the rule names bicycles. The riding motorcycle stays too.
    result = BBoxesAttributeFilter(RULES)(
        sample(
            ["bicycle", "bicycle", "car", "motorcycle", "motorcycle"],
            [
                ["vehicle_state.parked"],
                ["vehicle_state.moving"],
                ["vehicle_state.parked"],
                ["cycle_state.without_rider", "vehicle_state.parked"],
                ["cycle_state.with_rider"],
            ],
        )
    )

    boxes = result.detection3d_gt_bboxes_3d
    assert boxes.bbox_label_names == ["bicycle", "car", "motorcycle"]
    assert boxes.bbox_params[:, Box3DFieldIndex.X].tolist() == [1.0, 2.0, 4.0]
    assert boxes.bbox_attributes == [
        ["vehicle_state.moving"],
        ["vehicle_state.parked"],
        ["cycle_state.with_rider"],
    ]


def test_without_rules_every_box_stays() -> None:
    result = BBoxesAttributeFilter([])(sample(["bicycle"], [["vehicle_state.parked"]]))

    assert len(result.detection3d_gt_bboxes_3d) == 1


def test_rejects_boxes_without_attributes() -> None:
    with pytest.raises(ValueError, match="attributes of every box"):
        BBoxesAttributeFilter(RULES)(sample(["bicycle"], None))


@pytest.mark.parametrize("rules", [[["bicycle"]], ["bicycle", "vehicle_state.parked"]])
def test_rejects_a_rule_that_is_not_a_pair(rules: list) -> None:
    with pytest.raises(ValueError, match="pair"):
        BBoxesAttributeFilter(rules)
