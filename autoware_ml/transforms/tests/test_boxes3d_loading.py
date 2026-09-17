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

"""Unit tests for the label name filter of the 3D boxes."""

from __future__ import annotations

import unittest

import torch

from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.geometry.bbox_3d.lidar_bbox3d import LidarBBoxes3D
from autoware_ml.transforms.boxes3d.loading import BBoxesLabelNameFilter
from autoware_ml.types.geometry import Box3DCenterCoordinateType

CLASS_NAMES = ("car", "truck", "pedestrian", "debris")


def build_sample(label_names: list[str], labels: list[int]) -> ModelGTSample:
    """Sample holding one box per given name, trained as the class of its label index."""
    num_bboxes = len(labels)
    boxes = LidarBBoxes3D(
        bbox_params=torch.zeros((num_bboxes, 10), dtype=torch.float32),
        bbox_labels=torch.tensor(labels, dtype=torch.int32),
        bbox_label_names=label_names,
        bbox_num_lidar_points=torch.ones(num_bboxes, dtype=torch.int32),
        bbox_center_coordinate_type=Box3DCenterCoordinateType.GRAVITY_CENTER,
    )
    return ModelGTSample(
        lidar_point_cloud_samples=None,
        image_samples=None,
        point_cloud_data=None,
        camera_image_data=None,
        detection3d_gt_bboxes_3d=boxes,
        segmentation3d_gt_sample=None,
    )


class TestBBoxesLabelNameFilter(unittest.TestCase):
    """Boxes are kept by the class they train as, not by the name they were annotated with."""

    def test_keeps_fine_names_mapped_to_kept_classes(self) -> None:
        """
        Input: boxes named ambulance, construction_vehicle, stroller and pushable_pullable,
        mapped to car, truck, pedestrian and debris, plus one ignored box.
        Expected: every mapped box stays and only the ignored one is dropped.
        Check: the label names left after the filter.
        """
        sample = build_sample(
            ["ambulance", "construction_vehicle", "stroller", "pushable_pullable", "bollard"],
            [0, 1, 2, 3, -1],
        )

        filtered = BBoxesLabelNameFilter(label_names_to_keep=CLASS_NAMES, class_names=CLASS_NAMES)(
            sample
        )

        assert filtered.detection3d_gt_bboxes_3d is not None
        self.assertEqual(
            list(filtered.detection3d_gt_bboxes_3d.bbox_label_names),
            ["ambulance", "construction_vehicle", "stroller", "pushable_pullable"],
        )

    def test_drops_classes_left_out_of_the_kept_names(self) -> None:
        """
        Input: a car box and a pedestrian box, keeping only car.
        Expected: the pedestrian box is dropped.
        Check: the label indices left after the filter.
        """
        sample = build_sample(["police_car", "pedestrian"], [0, 2])

        filtered = BBoxesLabelNameFilter(label_names_to_keep=["car"], class_names=CLASS_NAMES)(
            sample
        )

        assert filtered.detection3d_gt_bboxes_3d is not None
        self.assertEqual(filtered.detection3d_gt_bboxes_3d.bbox_labels.tolist(), [0])

    def test_rejects_names_outside_the_classes(self) -> None:
        """
        Input: a kept name that is no class.
        Expected: the filter refuses to be built, so a typo cannot silently drop every box.
        Check: ValueError naming the unknown class.
        """
        with self.assertRaisesRegex(ValueError, "busses"):
            BBoxesLabelNameFilter(label_names_to_keep=["busses"], class_names=CLASS_NAMES)


if __name__ == "__main__":
    unittest.main()
