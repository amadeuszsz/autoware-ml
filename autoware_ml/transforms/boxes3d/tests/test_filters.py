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

"""Tests for the 3D box filters."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch

from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.geometry.bbox_3d.lidar_bbox3d import LidarBBoxes3D
from autoware_ml.geometry.points.lidar_points import LiDARPoints
from autoware_ml.transforms.boxes3d.filters import (
    BBoxesAttributeFilter,
    BBoxesLabelNameFilter,
    BBoxesMinPointsFilter,
    BBoxesPointCount,
    BBoxesRangeFilter,
)

from autoware_ml.types.geometry import Box3DCenterCoordinateType, Box3DFieldIndex, PointFeatureName

POINT_CLOUD_RANGE = [-10.0, -10.0, -2.0, 10.0, 10.0, 4.0]
CLASS_NAMES = ("car", "truck", "pedestrian", "debris")
RULES = [["bicycle", "vehicle_state.parked"], ["motorcycle", "cycle_state.without_rider"]]


def sample(
    label_names: Sequence[str],
    labels: Sequence[int] | None = None,
    centers: Sequence[Sequence[float]] | None = None,
    attributes: Sequence[Sequence[str]] | None = None,
) -> ModelGTSample:
    """
    Sample carrying one unit box per label name.

    Args:
      label_names: Label name of every box.
      labels: Label index of every box, the box position when omitted.
      centers: Gravity center (x, y, z) of every box, the box position along x when omitted.
      attributes: Attribute names of every box, None when the dataset carries none.

    Returns:
      ModelGTSample: Sample whose detection boxes carry those labels, centers and attributes.
    """
    count = len(label_names)
    bbox_params = torch.zeros((count, len(Box3DFieldIndex)), dtype=torch.float32)
    bbox_params[:, Box3DFieldIndex.LENGTH : Box3DFieldIndex.YAW] = 1.0
    bbox_params[:, Box3DFieldIndex.X] = torch.arange(count, dtype=torch.float32)
    if centers is not None:
        bbox_params[:, [Box3DFieldIndex.X, Box3DFieldIndex.Y, Box3DFieldIndex.Z]] = torch.tensor(
            centers, dtype=torch.float32
        )
    boxes = LidarBBoxes3D(
        bbox_params=bbox_params,
        bbox_labels=torch.tensor(range(count) if labels is None else labels, dtype=torch.int32),
        bbox_label_names=list(label_names),
        bbox_num_lidar_points=torch.full((count,), 10, dtype=torch.int32),
        bbox_instance_ids=torch.arange(count),
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


def test_drops_the_boxes_outside_every_axis_of_the_range() -> None:
    # box_0 is inside, box_1 leaves the range along x, box_2 below z_min, box_3 above z_max.
    # box_4 sits on the boundary and stays.
    result = BBoxesRangeFilter(POINT_CLOUD_RANGE)(
        sample(
            [f"box_{index}" for index in range(5)],
            centers=[
                [0.0, 0.0, 0.0],
                [11.0, 0.0, 0.0],
                [0.0, 0.0, -3.0],
                [0.0, 0.0, 5.0],
                [10.0, -10.0, 4.0],
            ],
        )
    )

    assert result.detection3d_gt_bboxes_3d.bbox_label_names == ["box_0", "box_4"]
    assert result.detection3d_gt_bboxes_3d.bbox_instance_ids.tolist() == [0, 4]


def _with_points(model_gt_sample: ModelGTSample, points: torch.Tensor) -> ModelGTSample:
    feature_names = [PointFeatureName.X, PointFeatureName.Y, PointFeatureName.Z]
    return model_gt_sample._replace(
        point_cloud_data=LiDARPoints(points, feature_names, timestamp=0.0)
    )


def test_point_count_matches_the_dense_inside_test() -> None:
    torch.manual_seed(0)
    boxes_sample = sample([f"box_{index}" for index in range(6)])
    boxes = boxes_sample.detection3d_gt_bboxes_3d
    boxes.bbox_params[:, Box3DFieldIndex.LENGTH : Box3DFieldIndex.YAW] = torch.rand(6, 3) * 3 + 0.5
    boxes.bbox_params[:, Box3DFieldIndex.YAW] = torch.rand(6) * 6.0 - 3.0
    points = torch.rand(20000, 3) * torch.tensor([12.0, 6.0, 4.0]) - torch.tensor([3.0, 3.0, 2.0])
    expected = boxes.compute_points_in_bboxes(points).sum(dim=1).int()

    result = BBoxesPointCount()(_with_points(boxes_sample, points))

    assert torch.equal(result.detection3d_gt_bboxes_3d.bbox_num_lidar_points, expected)
    assert int(expected.sum()) > 0


def test_min_points_filter_drops_the_boxes_below_the_count() -> None:
    boxes_sample = sample(["near", "far"], centers=[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    points = torch.tensor([[0.1, 0.0, 0.0], [-0.1, 0.1, 0.0], [5.0, 0.0, 0.0]])

    result = BBoxesMinPointsFilter(min_points=2, bev_range=[-10.0, -10.0, 10.0, 10.0])(
        _with_points(boxes_sample, points)
    )

    assert result.detection3d_gt_bboxes_3d.bbox_label_names == ["near"]


def test_label_name_filter_keeps_fine_names_mapped_to_kept_classes() -> None:
    # Boxes are kept by the class of their label index, not by the name they were annotated
    # with. Only the ignored box is dropped.
    result = BBoxesLabelNameFilter(label_names_to_keep=CLASS_NAMES, class_names=CLASS_NAMES)(
        sample(
            ["ambulance", "construction_vehicle", "stroller", "pushable_pullable", "bollard"],
            labels=[0, 1, 2, 3, -1],
        )
    )

    assert result.detection3d_gt_bboxes_3d.bbox_label_names == [
        "ambulance",
        "construction_vehicle",
        "stroller",
        "pushable_pullable",
    ]


def test_label_name_filter_drops_classes_left_out_of_the_kept_names() -> None:
    result = BBoxesLabelNameFilter(label_names_to_keep=["car"], class_names=CLASS_NAMES)(
        sample(["police_car", "pedestrian"], labels=[0, 2])
    )

    assert result.detection3d_gt_bboxes_3d.bbox_labels.tolist() == [0]


def test_label_name_filter_rejects_names_outside_the_classes() -> None:
    # Input: a kept name that is not a class. Expected: construction fails, so a typo cannot
    # drop every box.
    with pytest.raises(ValueError, match="busses"):
        BBoxesLabelNameFilter(label_names_to_keep=["busses"], class_names=CLASS_NAMES)


def test_attribute_filter_drops_only_the_boxes_matching_a_rule() -> None:
    # The parked car keeps its box, the rule names bicycles. The riding motorcycle stays too.
    result = BBoxesAttributeFilter(RULES)(
        sample(
            ["bicycle", "bicycle", "car", "motorcycle", "motorcycle"],
            attributes=[
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


def test_attribute_filter_without_rules_keeps_every_box() -> None:
    result = BBoxesAttributeFilter([])(sample(["bicycle"], attributes=[["vehicle_state.parked"]]))

    assert len(result.detection3d_gt_bboxes_3d) == 1


def test_attribute_filter_rejects_boxes_without_attributes() -> None:
    with pytest.raises(ValueError, match="attributes of every box"):
        BBoxesAttributeFilter(RULES)(sample(["bicycle"]))


@pytest.mark.parametrize("rules", [[["bicycle"]], ["bicycle", "vehicle_state.parked"]])
def test_attribute_filter_rejects_a_rule_that_is_not_a_pair(rules: list) -> None:
    with pytest.raises(ValueError, match="pair"):
        BBoxesAttributeFilter(rules)
